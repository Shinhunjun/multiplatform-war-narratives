from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from build_text_relevance_tokens import build_stopword_set, tokenize


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_lookup = Path(__file__).resolve().parent / "url_lookup.csv"
    parser = argparse.ArgumentParser(
        description="Tokenize url_lookup Text column and store token arrays in Tokens."
    )
    parser.add_argument("--lookup", type=Path, default=default_lookup, help="Path to url_lookup.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write updated lookup (default: overwrite --lookup)",
    )
    parser.add_argument("--text-col", default="Text", help="Text column name")
    parser.add_argument("--tokens-col", default="Tokens", help="Tokens column name")
    parser.add_argument("--force", action="store_true", help="Retokenize all rows, including already tokenized")
    return parser.parse_args()


def is_blank(value: object) -> bool:
    """Execute is_blank."""
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def main() -> None:
    """Run the script entry point."""
    args = parse_args()
    output_path = args.output if args.output is not None else args.lookup

    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")

    df = pd.read_csv(args.lookup, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing required text column: {args.text_col}")
    if args.tokens_col not in df.columns:
        df[args.tokens_col] = ""
    # Ensure token writes are always string-compatible (not float NaN dtype).
    df[args.tokens_col] = df[args.tokens_col].astype("object")

    text_series = df[args.text_col].fillna("").astype(str)
    tokens_series = df[args.tokens_col]
    has_text_mask = text_series.str.strip() != ""

    if args.force:
        target_mask = has_text_mask
    else:
        target_mask = has_text_mask & tokens_series.apply(is_blank)

    target_indices = df.index[target_mask]

    lemmatizer = WordNetLemmatizer()
    stopword_set = build_stopword_set()

    for idx in tqdm(
        target_indices,
        total=len(target_indices),
        desc="Tokenizing url_lookup rows",
        unit="row",
        file=sys.stdout,
    ):
        text = text_series.at[idx]
        tokens = sorted(tokenize(text, lemmatizer, stopword_set))
        df.at[idx, args.tokens_col] = json.dumps(tokens, ensure_ascii=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Lookup rows: {len(df):,}")
    print(f"Rows with non-empty text: {int(has_text_mask.sum()):,}")
    print(f"Rows tokenized this run: {len(target_indices):,}")
    print(f"Output written: {output_path}")


if __name__ == "__main__":
    main()
