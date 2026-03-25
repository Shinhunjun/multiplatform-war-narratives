from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from common import LOOKUP_PATH, RELEVANCE_PATH, load_changed_url_ids, write_audit_rows, bootstrap_project_paths


try:
    from nltk.stem import WordNetLemmatizer

    bootstrap_project_paths()
    from build_text_relevance_tokens import build_stopword_set, tokenize
except Exception:  # pragma: no cover - runtime fallback when nltk stack is unavailable
    WordNetLemmatizer = None  # type: ignore[assignment]
    FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'.-]*")
    FALLBACK_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }

    def build_stopword_set() -> set[str]:
        """Fallback stopword set when NLTK resources are unavailable."""
        return set(FALLBACK_STOPWORDS)

    def tokenize(text: str, lemmatizer: object, stopword_set: set[str]) -> set[str]:
        """Fallback tokenizer used only when project NLTK dependencies are unavailable."""
        tokens = {
            token.lower().strip("'.")
            for token in FALLBACK_TOKEN_RE.findall(str(text or ""))
        }
        return {token for token in tokens if token and token not in stopword_set}


DOC_SCORE_COLS = [
    "doc_relevance_sum",
    "doc_relevance_matches",
    "doc_token_count",
    "doc_relevance_score",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for frozen relevance application."""
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize and score only changed url_lookup rows using the existing frozen token "
            "relevance table in data/preprocessing/text_relevance_tokens.csv."
        )
    )
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH, help="Path to data/preprocessing/url_lookup.csv")
    parser.add_argument("--relevance", type=Path, default=RELEVANCE_PATH, help="Path to frozen relevance CSV")
    parser.add_argument("--changed-url-ids", type=Path, required=True, help="CSV of changed url_ids")
    parser.add_argument("--summary-output", type=Path, required=True, help="Weekly score summary CSV path")
    return parser.parse_args()


def parse_changed_ids(path: Path) -> set[int]:
    """Load the changed-url-id worklist as a plain integer set."""
    df = load_changed_url_ids(path)
    if df.empty or "url_id" not in df.columns:
        return set()
    return {int(value) for value in df["url_id"].dropna().astype(int).tolist()}


def score_token_set(tokens: set[str], token_to_score: dict[str, float]) -> tuple[float, int, int, float]:
    """Score one token set using the frozen relevance table."""
    token_count = len(tokens)
    if token_count == 0:
        return 0.0, 0, 0, 0.0
    matched_scores = [token_to_score[token] for token in tokens if token in token_to_score]
    score_sum = float(sum(matched_scores))
    match_count = len(matched_scores)
    score = score_sum / math.sqrt(token_count)
    return score_sum, match_count, token_count, score


def main() -> None:
    """Apply frozen token relevance to changed lookup rows and update url_lookup.csv."""
    args = parse_args()
    lookup_df = pd.read_csv(args.lookup, low_memory=False)
    rel_df = pd.read_csv(args.relevance, low_memory=False)
    changed_ids = parse_changed_ids(args.changed_url_ids)

    if not {"token", "relevance_score"}.issubset(rel_df.columns):
        raise ValueError("Frozen relevance CSV must contain 'token' and 'relevance_score' columns")

    for col in ["Tokens"] + DOC_SCORE_COLS:
        if col not in lookup_df.columns:
            lookup_df[col] = "" if col == "Tokens" else 0.0
    lookup_df["Tokens"] = lookup_df["Tokens"].astype("object")

    if not changed_ids:
        lookup_df.to_csv(args.lookup, index=False)
        write_audit_rows(
            args.summary_output,
            [
                {"metric": "changed_url_ids", "value": 0},
                {"metric": "rows_scored", "value": 0},
                {"metric": "rows_with_nonzero_score", "value": 0},
            ],
        )
        print(f"No changed url_ids to score. Lookup rewritten unchanged: {args.lookup}")
        return

    token_to_score = dict(zip(rel_df["token"].astype(str), rel_df["relevance_score"].astype(float)))
    lemmatizer = WordNetLemmatizer() if WordNetLemmatizer is not None else None
    stopword_set = build_stopword_set()

    mask = pd.to_numeric(lookup_df["url_id"], errors="coerce").isin(changed_ids)
    changed_indices = lookup_df.index[mask].tolist()

    nonzero_scores = 0
    for idx in changed_indices:
        text = str(lookup_df.at[idx, "Text"] or "")
        token_set = tokenize(text, lemmatizer, stopword_set) if text.strip() else set()
        lookup_df.at[idx, "Tokens"] = json.dumps(sorted(token_set), ensure_ascii=True)
        score_sum, match_count, token_count, score = score_token_set(token_set, token_to_score)
        lookup_df.at[idx, "doc_relevance_sum"] = score_sum
        lookup_df.at[idx, "doc_relevance_matches"] = match_count
        lookup_df.at[idx, "doc_token_count"] = token_count
        lookup_df.at[idx, "doc_relevance_score"] = score
        if score > 0:
            nonzero_scores += 1

    args.lookup.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.to_csv(args.lookup, index=False)
    write_audit_rows(
        args.summary_output,
        [
            {"metric": "changed_url_ids", "value": int(len(changed_ids))},
            {"metric": "rows_scored", "value": int(len(changed_indices))},
            {"metric": "rows_with_nonzero_score", "value": int(nonzero_scores)},
        ],
    )

    print(f"Changed url_ids scored: {len(changed_indices):,}")
    print(f"Rows with non-zero score: {nonzero_scores:,}")
    print(f"Lookup written: {args.lookup}")
    print(f"Summary written: {args.summary_output}")


if __name__ == "__main__":
    main()
