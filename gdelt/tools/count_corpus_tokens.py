"""Count token frequencies across the analysis-ready URL content parquet.

Reads the Tokens column (JSON arrays) from analysis_url_content.parquet,
counts how often each token appears across documents, and writes the result
to a CSV sorted by descending frequency.

Usage:
    python tools/count_corpus_tokens.py
    python tools/count_corpus_tokens.py --input data/analysis_ready/analysis_url_content.parquet
    python tools/count_corpus_tokens.py --analysis-include-only
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = PROJECT_DIR / "data" / "analysis_ready" / "analysis_url_content.parquet"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "analysis_ready" / "corpus_token_frequencies.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count token frequencies in the analysis-ready corpus.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PARQUET, help="Path to analysis_url_content.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--analysis-include-only",
        action="store_true",
        help="Only count tokens from rows where analysis_include=True",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_parquet(args.input, columns=["url_id", "Tokens", "analysis_include"])
    print(f"  {len(df):,} rows loaded")

    if args.analysis_include_only:
        df = df[df["analysis_include"] == True]
        print(f"  {len(df):,} rows after filtering to analysis_include=True")

    print("Counting token frequencies...")
    counter: Counter[str] = Counter()
    skipped = 0
    for raw in df["Tokens"].dropna():
        # Tokens are stored as numpy arrays in the parquet file.
        try:
            tokens = list(raw)
        except TypeError:
            skipped += 1
            continue
        if not tokens:
            skipped += 1
            continue
        counter.update(tokens)

    print(f"  {skipped:,} rows skipped (empty or unparseable Tokens)")
    print(f"  {len(counter):,} unique tokens counted")

    out_df = pd.DataFrame(
        [(token, count) for token, count in counter.most_common()],
        columns=["token", "doc_count"],
    )
    out_df["rank"] = range(1, len(out_df) + 1)
    out_df = out_df[["rank", "token", "doc_count"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\nTop 20 tokens:")
    print(out_df.head(20).to_string(index=False))
    print(f"\nOutput written: {args.output} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
