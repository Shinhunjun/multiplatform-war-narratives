from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Score url_lookup rows using token relevance scores."
    )
    parser.add_argument("--lookup", type=Path, default=base / "url_lookup.csv", help="Path to url_lookup.csv")
    parser.add_argument(
        "--relevance",
        type=Path,
        default=base / "text_relevance_tokens.csv",
        help="Path to text_relevance_tokens.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output url_lookup with doc_relevance_score (default: overwrite --lookup)",
    )
    parser.add_argument("--tokens-col", default="Tokens", help="Token-array column name in lookup")
    parser.add_argument("--score-col", default="doc_relevance_score", help="Output score column name")
    parser.add_argument(
        "--sum-col",
        default="doc_relevance_sum",
        help="Output column for raw summed matched relevance scores",
    )
    parser.add_argument(
        "--matches-col",
        default="doc_relevance_matches",
        help="Output column for count of matched tokens with known relevance scores",
    )
    parser.add_argument(
        "--token-count-col",
        default="doc_token_count",
        help="Output column for token count used in denominator",
    )
    return parser.parse_args()


def parse_token_array(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    s = str(value).strip()
    if not s:
        return set()
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        arr = None
    if isinstance(arr, list):
        return {str(x).strip() for x in arr if str(x).strip()}
    if "," in s:
        return {p.strip() for p in s.split(",") if p.strip()}
    return {s}


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output is not None else args.lookup

    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")
    if not args.relevance.exists():
        raise FileNotFoundError(f"Relevance file not found: {args.relevance}")

    lookup_df = pd.read_csv(args.lookup, low_memory=False)
    rel_df = pd.read_csv(args.relevance, low_memory=False)

    if args.tokens_col not in lookup_df.columns:
        raise ValueError(f"Missing tokens column in lookup: {args.tokens_col}")
    if "token" not in rel_df.columns or "relevance_score" not in rel_df.columns:
        raise ValueError("Relevance CSV must contain 'token' and 'relevance_score' columns")

    token_to_score = dict(zip(rel_df["token"].astype(str), rel_df["relevance_score"].astype(float)))

    scores: list[float] = []
    sums: list[float] = []
    matches: list[int] = []
    token_counts: list[int] = []

    for value in tqdm(lookup_df[args.tokens_col], total=len(lookup_df), desc="Scoring url_lookup", unit="row"):
        tokens = parse_token_array(value)
        n_tokens = len(tokens)
        token_counts.append(n_tokens)

        if n_tokens == 0:
            sums.append(0.0)
            matches.append(0)
            scores.append(0.0)
            continue

        matched_scores = [token_to_score[t] for t in tokens if t in token_to_score]
        score_sum = float(sum(matched_scores))
        match_count = len(matched_scores)
        score = score_sum / math.sqrt(n_tokens)

        sums.append(score_sum)
        matches.append(match_count)
        scores.append(score)

    lookup_df[args.sum_col] = sums
    lookup_df[args.matches_col] = matches
    lookup_df[args.token_count_col] = token_counts
    lookup_df[args.score_col] = scores

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.to_csv(output_path, index=False)

    nonzero = sum(1 for x in scores if x > 0)
    print(f"Rows scored: {len(scores):,}")
    print(f"Rows with non-zero {args.score_col}: {nonzero:,}")
    print(f"Output written: {output_path}")


if __name__ == "__main__":
    main()
