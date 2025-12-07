#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick script to check what platforms are available in the dependencies CSV.
"""

import duckdb
from pathlib import Path

csv_path = Path("data/dependencies-1.6.0-2020-01-12.csv")

con = duckdb.connect()

# Get distinct platforms from the "Platform" column
print("=== Distinct Platforms (from 'Platform' column) ===")
platforms = con.execute(f"""
    SELECT DISTINCT "Platform", COUNT(*) as count
    FROM read_csv_auto('{csv_path.as_posix()}', header=True)
    WHERE "Platform" IS NOT NULL
    GROUP BY "Platform"
    ORDER BY count DESC
""").fetchall()

for platform, count in platforms:
    print(f"{platform}: {count:,} rows")

# Also check dependency platforms
print("\n=== Distinct Dependency Platforms (from 'Dependency Platform' column) ===")
dep_platforms = con.execute(f"""
    SELECT DISTINCT "Dependency Platform", COUNT(*) as count
    FROM read_csv_auto('{csv_path.as_posix()}', header=True)
    WHERE "Dependency Platform" IS NOT NULL
    GROUP BY "Dependency Platform"
    ORDER BY count DESC
""").fetchall()

for platform, count in dep_platforms:
    print(f"{platform}: {count:,} rows")

# Show platform combinations
print("\n=== Platform → Dependency Platform Combinations ===")
combinations = con.execute(f"""
    SELECT 
        "Platform" as source_platform,
        "Dependency Platform" as dep_platform,
        COUNT(*) as count
    FROM read_csv_auto('{csv_path.as_posix()}', header=True)
    WHERE "Platform" IS NOT NULL 
      AND "Dependency Platform" IS NOT NULL
    GROUP BY "Platform", "Dependency Platform"
    ORDER BY count DESC
    LIMIT 20
""").fetchall()

for src, dep, count in combinations:
    print(f"{src} → {dep}: {count:,} rows")

con.close()

