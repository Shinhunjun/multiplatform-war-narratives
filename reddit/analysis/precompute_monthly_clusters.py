"""
Pre-compute monthly cluster counts from cluster_assignments.parquet.

Output: outputs/clusters/clusters_monthly.parquet
Schema: year_month, cluster_id, count, proportion, keywords
"""

import pandas as pd
from pathlib import Path


def main():
    base = Path(__file__).parent
    assignments_path = base / "outputs" / "clusters" / "cluster_assignments.parquet"
    keywords_path = base / "outputs" / "clusters" / "cluster_keywords.csv"
    output_path = base / "outputs" / "clusters" / "clusters_monthly.parquet"

    print(f"Loading {assignments_path} ...")
    assignments = pd.read_parquet(assignments_path)
    print(f"  {len(assignments):,} rows")

    # Filter out noise cluster -1
    assignments = assignments[assignments["cluster_id"] >= 0]

    # Load cluster keywords for labels
    kw_df = pd.read_csv(keywords_path)
    kw_map = {}
    for _, row in kw_df.iterrows():
        kws = row["keywords"]
        if isinstance(kws, str):
            kw_map[row["cluster_id"]] = ", ".join(kws.split(", ")[:3])

    # Group by month + cluster
    monthly = (
        assignments.groupby(["year_month", "cluster_id"])
        .size()
        .reset_index(name="count")
    )

    # Compute proportion within each month
    month_totals = monthly.groupby("year_month")["count"].transform("sum")
    monthly["proportion"] = (monthly["count"] / month_totals).round(6)

    # Add keywords
    monthly["keywords"] = monthly["cluster_id"].map(kw_map).fillna("")

    # Sort
    monthly = monthly.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)

    print(f"Output: {len(monthly):,} rows across {monthly['year_month'].nunique()} months")
    monthly.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
