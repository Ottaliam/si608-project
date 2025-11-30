#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot PyPI dependency graph metrics produced by build_pypi_graph.py.

Inputs (from --artifacts):
  metrics/
    - in_degree.csv (vertex_id,in_degree)
    - out_degree.csv (vertex_id,out_degree)
    - pagerank.csv (vertex_id,pagerank)
    - assortativity.txt (degree_assortativity,<float>)
    - graph_summary.txt (nodes,edges,density,weakly_connected,...)
    - top_in_degree.csv, top_out_degree.csv, top_pagerank.csv (rank,vertex_id,project_id,score)

Outputs (into --out):
  - degree_pdf_loglog.png         (PDF of in/out-degree, log-log)
  - degree_ccdf_loglog.png        (CCDF of in/out-degree, log-log)
  - pagerank_pdf_loglog.png       (PDF of PageRank values, log-log)
  - top_out_degree.png            (Top-N by out-degree)
  - top_in_degree.png             (Top-N by in-degree)
  - top_pagerank.png              (Top-N by PageRank)
  - summary.txt                   (echo of key stats + assortativity)
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

try:
    import seaborn as sns
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False


# -------------------------
# IO helpers
# -------------------------
def read_metric_series(path: Path, value_col: str) -> np.ndarray:
    df = pl.read_csv(path)
    return df[value_col].to_numpy()

def read_top_table(path: Path, n: int, score_col: str):
    df = pl.read_csv(path)
    df = df.sort(score_col, descending=True).head(n)
    return df

def read_scalar_from_txt(path: Path, key_prefix: str):
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    for ln in lines:
        if ln.startswith(key_prefix):
            _, val = ln.split(",", 1)
            try:
                return float(val)
            except Exception:
                return val
    return None

def read_summary_dict(path: Path):
    d = {}
    if not path.exists():
        return d
    for ln in path.read_text(encoding="utf-8").strip().splitlines():
        if "," in ln:
            k, v = ln.split(",", 1)
            d[k] = v
    return d


# -------------------------
# Plot helpers
# -------------------------
def _setup_style(use_seaborn: bool):
    if use_seaborn and HAVE_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("default")

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_degree_pdf_loglog(out_path: Path, indeg: np.ndarray, outdeg: np.ndarray, title_suffix=""):
    # Log-spaced bins
    max_k = int(max(indeg.max() if indeg.size else 1, outdeg.max() if outdeg.size else 1))
    if max_k < 1:
        max_k = 1
    bins = np.unique(np.logspace(0, np.log10(max_k), num=min(60, max_k)).astype(int))
    bins = bins[bins > 0]
    if len(bins) < 2:
        bins = np.array([1, 2])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # PDF for in-degree
    if indeg.size:
        counts_in, edges_in = np.histogram(indeg[indeg > 0], bins=bins, density=True)
        centers_in = np.sqrt(edges_in[:-1] * edges_in[1:])
        ax.plot(centers_in, counts_in, marker="o", linestyle="-", label="In-degree PDF")

    # PDF for out-degree
    if outdeg.size:
        counts_out, edges_out = np.histogram(outdeg[outdeg > 0], bins=bins, density=True)
        centers_out = np.sqrt(edges_out[:-1] * edges_out[1:])
        ax.plot(centers_out, counts_out, marker="s", linestyle="-", label="Out-degree PDF")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("P(k)")
    ax.set_title(f"Degree PDF (log-log){title_suffix}")
    ax.legend()
    _save(fig, out_path)

def plot_degree_ccdf_loglog(out_path: Path, indeg: np.ndarray, outdeg: np.ndarray, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    def ccdf(arr):
        arr = arr[arr > 0]
        if arr.size == 0:
            return None, None
        vals = np.sort(arr)
        uniq = np.unique(vals)
        # CCDF(k) = P(X >= k)
        ccdf_y = 1.0 - np.searchsorted(vals, uniq, side="left") / vals.size
        return uniq, ccdf_y

    x_in, y_in = ccdf(indeg)
    x_out, y_out = ccdf(outdeg)

    if x_in is not None:
        ax.plot(x_in, y_in, marker="o", linestyle="-", label="In-degree CCDF")
    if x_out is not None:
        ax.plot(x_out, y_out, marker="s", linestyle="-", label="Out-degree CCDF")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("P(K ≥ k)")
    ax.set_title(f"Degree CCDF (log-log){title_suffix}")
    ax.legend()
    _save(fig, out_path)

def plot_pagerank_pdf_loglog(out_path: Path, pr: np.ndarray, title_suffix=""):
    pr = pr[pr > 0]
    if pr.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No positive PageRank values", ha="center", va="center")
        _save(fig, out_path)
        return

    # Log-spaced bins across observed PR range
    pr_min, pr_max = pr.min(), pr.max()
    # guard against identical values
    if pr_max <= pr_min:
        pr_max = pr_min * 1.01
    bins = np.logspace(np.log10(pr_min), np.log10(pr_max), num=60)

    counts, edges = np.histogram(pr, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(centers, counts, marker="o", linestyle="-", label="PageRank PDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PageRank")
    ax.set_ylabel("Density")
    ax.set_title(f"PageRank PDF (log-log){title_suffix}")
    ax.legend()
    _save(fig, out_path)

def plot_top_bar(out_path: Path, top_df: pl.DataFrame, score_col: str, title: str):
    if top_df.height == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        _save(fig, out_path)
        return
    # Use readable labels: project_id (or truncate)
    labels = top_df["project_id"].to_list()
    vals = top_df[score_col].to_numpy()

    # Truncate long labels
    def trunc(s, n=28):
        s = str(s)
        return s if len(s) <= n else s[:n-1] + "…"

    labels = [trunc(s) for s in labels]
    y_pos = np.arange(len(labels))[::-1]  # highest on top

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labels))))
    ax.barh(y_pos, vals)
    ax.set_yticks(y_pos, labels)
    ax.set_xlabel(score_col.replace("_", " ").title())
    ax.set_title(title)
    if "degree" in score_col:
        ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    _save(fig, out_path)


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Plot PyPI graph metrics")
    p.add_argument("--artifacts", type=Path, required=True, help="Path to pypi_graph_artifacts")
    p.add_argument("--out", type=Path, default=Path("plots"), help="Directory to save plots")
    p.add_argument("--topn", type=int, default=20, help="Top-N for bar charts")
    p.add_argument("--use-seaborn", action="store_true", help="Use seaborn styling if available")
    args = p.parse_args()

    _setup_style(args.use_seaborn)

    metrics = args.artifacts / "metrics"
    metrics.mkdir(exist_ok=True, parents=True)

    # Load series
    in_deg_path = metrics / "in_degree.csv"
    out_deg_path = metrics / "out_degree.csv"
    pr_path = metrics / "pagerank.csv"

    indeg = read_metric_series(in_deg_path, "in_degree") if in_deg_path.exists() else np.array([])
    outdeg = read_metric_series(out_deg_path, "out_degree") if out_deg_path.exists() else np.array([])
    pagerank = read_metric_series(pr_path, "pagerank") if pr_path.exists() else np.array([])

    title_suffix = ""
    summary = read_summary_dict(metrics / "graph_summary.txt")
    if summary:
        n = summary.get("nodes", "?")
        m = summary.get("edges", "?")
        title_suffix = f"  |V|={n}, |E|={m}"

    # Degree plots
    plot_degree_pdf_loglog(args.out / "degree_pdf_loglog.png", indeg, outdeg, title_suffix)
    plot_degree_ccdf_loglog(args.out / "degree_ccdf_loglog.png", indeg, outdeg, title_suffix)

    # PageRank distribution
    if pagerank.size:
        plot_pagerank_pdf_loglog(args.out / "pagerank_pdf_loglog.png", pagerank, title_suffix)

    # Top-N bar charts
    top_out = metrics / "top_out_degree.csv"
    top_in = metrics / "top_in_degree.csv"
    top_pr = metrics / "top_pagerank.csv"

    if top_out.exists():
        df_top_out = read_top_table(top_out, args.topn, "out_degree")
        plot_top_bar(args.out / "top_out_degree.png", df_top_out, "out_degree",
                     f"Top {args.topn} by Out-degree (dependencies required)")

    if top_in.exists():
        df_top_in = read_top_table(top_in, args.topn, "in_degree")
        plot_top_bar(args.out / "top_in_degree.png", df_top_in, "in_degree",
                     f"Top {args.topn} by In-degree (depended upon)")

    if top_pr.exists():
        df_top_pr = read_top_table(top_pr, args.topn, "pagerank")
        plot_top_bar(args.out / "top_pagerank.png", df_top_pr, "pagerank",
                     f"Top {args.topn} by PageRank")

    # Echo summary & assortativity to a simple text file
    assort = read_scalar_from_txt(metrics / "assortativity.txt", "degree_assortativity")
    with (args.out / "summary.txt").open("w", encoding="utf-8") as f:
        if summary:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        if assort is not None:
            f.write(f"degree_assortativity: {assort}\n")

    print(f"Saved plots to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
