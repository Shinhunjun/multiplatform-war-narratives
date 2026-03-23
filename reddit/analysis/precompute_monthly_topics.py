"""
Pre-compute monthly topic counts for the monthly slider UI.

Reddit: uses topic_assignments.parquet (per-document topic_id) → group by year_month.
News:   uses topics_over_time.csv (already aggregated Frequency per month) → reshape.

Output:
  outputs/topics/topics_monthly.parquet
  outputs_news/topics/topics_monthly.parquet
Schema: year_month, topic_id, count, proportion, name, keywords
"""

import pandas as pd
from pathlib import Path


def extract_keywords(name: str) -> str:
    if not name:
        return ""
    parts = name.split("_")
    return ", ".join(parts[1:4]) if len(parts) > 1 else name


def build_reddit(base: Path):
    assignments_path = base / "outputs" / "topics" / "topic_assignments.parquet"
    topic_info_path = base / "outputs" / "topics" / "topic_info.csv"
    output_path = base / "outputs" / "topics" / "topics_monthly.parquet"

    print(f"[Reddit] Loading {assignments_path} ...")
    assignments = pd.read_parquet(assignments_path)
    print(f"  {len(assignments):,} rows")

    assignments = assignments[assignments["topic_id"] >= 0]

    topic_info = pd.read_csv(topic_info_path)
    topic_info = topic_info[topic_info["Topic"] >= 0]
    name_map = topic_info.set_index("Topic")["Name"].to_dict()

    monthly = (
        assignments.groupby(["year_month", "topic_id"])
        .size()
        .reset_index(name="count")
    )

    month_totals = monthly.groupby("year_month")["count"].transform("sum")
    monthly["proportion"] = (monthly["count"] / month_totals).round(6)
    monthly["name"] = monthly["topic_id"].map(name_map).fillna("")
    monthly["keywords"] = monthly["name"].apply(extract_keywords)
    monthly = monthly.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)

    print(f"[Reddit] Output: {len(monthly):,} rows across {monthly['year_month'].nunique()} months")
    monthly.to_parquet(output_path, index=False)
    print(f"[Reddit] Saved to {output_path}")


def build_news(base: Path):
    over_time_path = base / "outputs_news" / "topics" / "topics_over_time.csv"
    topic_info_path = base / "outputs_news" / "topics" / "topic_info.csv"
    output_path = base / "outputs_news" / "topics" / "topics_monthly.parquet"

    if not over_time_path.exists():
        print("[News] topics_over_time.csv not found, skipping.")
        return

    print(f"[News] Loading {over_time_path} ...")
    tot = pd.read_csv(over_time_path)
    print(f"  {len(tot):,} rows")

    # Filter outlier topics (Topic 0 is Spanish-language noise for news)
    topic_info = pd.read_csv(topic_info_path)
    topic_info = topic_info[topic_info["Topic"] >= 1]
    name_map = topic_info.set_index("Topic")["Name"].to_dict()

    tot = tot[tot["Topic"] >= 1].copy()
    tot["year_month"] = tot["Timestamp"].str[:7]

    monthly = (
        tot.groupby(["year_month", "Topic"])["Frequency"]
        .sum()
        .reset_index()
        .rename(columns={"Topic": "topic_id", "Frequency": "count"})
    )

    month_totals = monthly.groupby("year_month")["count"].transform("sum")
    monthly["proportion"] = (monthly["count"] / month_totals).round(6)
    monthly["name"] = monthly["topic_id"].map(name_map).fillna("")
    monthly["keywords"] = monthly["name"].apply(extract_keywords)
    monthly = monthly.sort_values(["year_month", "count"], ascending=[True, False]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[News] Output: {len(monthly):,} rows across {monthly['year_month'].nunique()} months")
    monthly.to_parquet(output_path, index=False)
    print(f"[News] Saved to {output_path}")


def main():
    base = Path(__file__).parent
    build_reddit(base)
    build_news(base)


if __name__ == "__main__":
    main()
