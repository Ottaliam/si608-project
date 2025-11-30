#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a PyPI dependency network from a very large CSV (e.g., 19.1 GB),
then compute standard graph measures.

Pipeline:
  CSV --(DuckDB filter)--> NCOL edgelist --(igraph)--> metrics CSVs

Usage:
  python build_pypi_graph.py --csv path/to/dependencies.csv --outdir pypi_graph_artifacts
"""

import argparse
import os
from pathlib import Path
import sys
import duckdb
import igraph as ig
import numpy as np

# -------------------------
# Helpers
# -------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def write_array_csv(path: Path, header: str, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(header.strip() + "\n")
        for i, v in enumerate(arr):
            f.write(f"{i},{v}\n")

def write_hist_csv(path: Path, data, bins="auto"):
    counts, bin_edges = np.histogram(data, bins=bins)
    with path.open("w", encoding="utf-8") as f:
        f.write("bin_left,bin_right,count\n")
        for i in range(len(counts)):
            f.write(f"{bin_edges[i]},{bin_edges[i+1]},{counts[i]}\n")

def export_top_n(path: Path, vertex_names, values, n=100, colname="score"):
    idx = np.argpartition(values, -n)[-n:]
    idx = idx[np.argsort(np.array(values)[idx])[::-1]]
    with path.open("w", encoding="utf-8") as f:
        f.write("rank,vertex_id,project_id,{}\n".format(colname))
        for r, vid in enumerate(idx, 1):
            f.write(f"{r},{vid},{vertex_names[vid]},{values[vid]}\n")

# -------------------------
# Core
# -------------------------
def build_edge_ncol(
    csv_path: Path,
    ncol_path: Path,
    allow_cross_platform: bool,
    dedup_edges: bool,
):
    """
    Use DuckDB to stream-filter the giant CSV and write a whitespace-separated NCOL file (no header).
    Keeps two TEXT columns: src(Project ID) dst(Dependency Project ID).
    """
    ncol_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    base_where = """
      "Platform" = 'Pypi'
      AND "Project ID" IS NOT NULL
      AND "Dependency Project ID" IS NOT NULL
    """
    if not allow_cross_platform:
        base_where += ' AND "Dependency Platform" = \'Pypi\''

    select_stmt = f"""
    SELECT {"DISTINCT" if dedup_edges else ""}
           CAST("Project ID" AS TEXT) AS src,
           CAST("Dependency Project ID" AS TEXT) AS dst
    FROM read_csv_auto('{csv_path.as_posix()}', header=True)
    WHERE {base_where}
      AND TRIM(CAST("Project ID" AS TEXT)) <> ''
      AND TRIM(CAST("Dependency Project ID" AS TEXT)) <> ''
      AND CAST("Project ID" AS TEXT) <> CAST("Dependency Project ID" AS TEXT)
    """

    # Write as NCOL: space-delimited, no header
    con.execute(f"""
      COPY ({select_stmt})
      TO '{ncol_path.as_posix()}'
      (DELIMITER ' ', HEADER FALSE);
    """)
    con.close()
    return ncol_path

def load_graph_from_ncol(ncol_path: Path, simplify: bool):
    """
    Read NCOL into igraph (directed). Vertex 'name' attribute stores the Project ID strings.
    """
    g = ig.Graph.Read_Ncol(ncol_path.as_posix(), directed=True, names=True)
    if simplify:
        # remove parallel edges & loops (loops were already filtered, but harmless)
        g.simplify(multiple=True, loops=True)
    return g

def compute_and_save_metrics(g: ig.Graph, outdir: Path, compute_betweenness: bool):
    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save vertex_id -> project_id map
    names = g.vs["name"]
    with (outdir / "node_id_map.csv").open("w", encoding="utf-8") as f:
        f.write("vertex_id,project_id\n")
        for vid, pid in enumerate(names):
            f.write(f"{vid},{pid}\n")

    # Summary
    n, m = g.vcount(), g.ecount()
    is_weak = g.components(mode="WEAK").giant().vcount() == n if n > 0 else True
    with (metrics_dir / "graph_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"nodes,{n}\n")
        f.write(f"edges,{m}\n")
        f.write(f"weakly_connected,{is_weak}\n")
        dens = (m / (n * (n - 1))) if n > 1 else 0.0
        f.write(f"density,{dens}\n")

    # Degrees
    indeg = g.indegree()
    outdeg = g.outdegree()
    write_array_csv(metrics_dir / "in_degree.csv", "vertex_id,in_degree", indeg)
    write_array_csv(metrics_dir / "out_degree.csv", "vertex_id,out_degree", outdeg)
    write_hist_csv(metrics_dir / "in_degree_hist.csv", indeg)
    write_hist_csv(metrics_dir / "out_degree_hist.csv", outdeg)
    export_top_n(metrics_dir / "top_out_degree.csv", names, outdeg, n=100, colname="out_degree")
    export_top_n(metrics_dir / "top_in_degree.csv", names, indeg, n=100, colname="in_degree")

    # PageRank (fast)
    pr = g.pagerank(directed=True)
    write_array_csv(metrics_dir / "pagerank.csv", "vertex_id,pagerank", pr)
    export_top_n(metrics_dir / "top_pagerank.csv", names, pr, n=100, colname="pagerank")

    # Harmonic closeness (works on disconnected graphs); direction: OUT (project -> dependency)
    harm = g.harmonic_centrality(mode="OUT")
    write_array_csv(metrics_dir / "harmonic_closeness.csv", "vertex_id,harmonic_closeness", harm)
    export_top_n(metrics_dir / "top_harmonic_closeness.csv", names, harm, n=100, colname="harmonic_closeness")

    # Degree assortativity (directed degree mixing by default)
    deg_assort = g.assortativity_degree(directed=True)
    with (metrics_dir / "assortativity.txt").open("w", encoding="utf-8") as f:
        f.write(f"degree_assortativity,{deg_assort}\n")

    # (Optional) Betweenness – WARNING: exact is expensive on very large graphs
    if compute_betweenness:
        eprint("Computing exact betweenness (can be very slow on large graphs)...")
        betw = g.betweenness(directed=True)
        write_array_csv(metrics_dir / "betweenness.csv", "vertex_id,betweenness", betw)
        export_top_n(metrics_dir / "top_betweenness.csv", names, betw, n=100, colname="betweenness")

def main():
    parser = argparse.ArgumentParser(
        description="Build a PyPI dependency graph from a large dependencies CSV and compute metrics."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to the big dependencies CSV")
    parser.add_argument("--outdir", default=Path("pypi_graph_artifacts"), type=Path, help="Output directory")
    parser.add_argument("--allow-cross-platform", action="store_true",
                        help="Keep edges where source is PyPI but dependency platform can be anything "
                             "(default off → keep only PyPI→PyPI edges)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Do NOT deduplicate edges (default is to deduplicate).")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Do NOT simplify the igraph (keep parallel edges).")
    parser.add_argument("--betweenness", action="store_true",
                        help="Compute exact betweenness (can be very slow on large graphs).")
    parser.add_argument("--ncol", type=Path, default=None,
                        help="Optional path to write/read NCOL edges. If exists, skip CSV filtering and just load it.")

    args = parser.parse_args()
    csv_path: Path = args.csv
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ncol_path = args.ncol or (outdir / "edges_pypi_intra.ncol")

    if args.ncol and ncol_path.exists():
        eprint(f"[skip filter] Using existing NCOL: {ncol_path}")
    else:
        eprint("[step] Filtering CSV with DuckDB → NCOL ...")
        build_edge_ncol(
            csv_path=csv_path,
            ncol_path=ncol_path,
            allow_cross_platform=args.allow_cross_platform,
            dedup_edges=not args.no_dedup,
        )
        eprint(f"[ok] Wrote NCOL: {ncol_path}")

    eprint("[step] Loading igraph from NCOL ...")
    g = load_graph_from_ncol(ncol_path, simplify=not args.no_simplify)
    eprint(f"[ok] Graph: |V|={g.vcount():,} |E|={g.ecount():,}")

    eprint("[step] Computing metrics ...")
    compute_and_save_metrics(g, outdir, compute_betweenness=args.betweenness)
    eprint(f"[done] Artifacts written to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
