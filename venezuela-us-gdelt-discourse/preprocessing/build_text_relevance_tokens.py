from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


DEFAULT_SEED_TERMS = [
    "venezuela",
    "venezuelan",
    "maduro",
    "caracas",
    "pdvsa",
    "chavez",
    "guaido",
    "united",
    "state",
    "us",
    "usa",
    "washington",
    "white",
    "house",
    "department",
    "treasury",
    "sanction",
]

DOMAIN_STOPWORDS = {
    "news",
    "latest",
    "breaking",
    "update",
    "report",
    "reports",
    "said",
    "says",
    "say",
    "also",
    "people",
    "person",
    "time",
    "day",
    "week",
    "month",
    "year",
    "today",
    "yesterday",
    "tomorrow",
    "first",
    "last",
    "new",
    "old",
    "house",
    "call",
    "called",
    "include",
    "including",
    "according",
    "statement",
    "statements",
    "former",
    "later",
    "early",
    "go",
    "going",
    "made",
    "make",
    "well",
    "like",
    "two",
    "three",
    "million",
}

TOKEN_RE = re.compile(r"[a-z][a-z']{2,}")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    default_lookup = Path(__file__).resolve().parent / "url_lookup.csv"
    default_out = Path(__file__).resolve().parent / "text_relevance_tokens.csv"

    parser = argparse.ArgumentParser(
        description="Build token relevance scores from pre-tokenized url_lookup.csv Tokens."
    )
    parser.add_argument("--lookup", type=Path, default=default_lookup, help="Path to url_lookup.csv")
    parser.add_argument("--output", type=Path, default=default_out, help="Path to output token score CSV")
    parser.add_argument("--tokens-col", default="Tokens", help="Name of token-array column")
    parser.add_argument("--status-col", default="Scrape_Status", help="Name of scrape status column")
    parser.add_argument(
        "--seed-terms",
        nargs="+",
        default=DEFAULT_SEED_TERMS,
        help="Seed terms used to define likely-relevant documents",
    )
    parser.add_argument("--min-doc-frac", type=float, default=0.005, help="Minimum document frequency fraction")
    parser.add_argument("--max-doc-frac", type=float, default=0.40, help="Maximum document frequency fraction")
    parser.add_argument(
        "--min-seed-doc-freq",
        type=int,
        default=25,
        help="Minimum number of seed documents containing a token",
    )
    parser.add_argument(
        "--min-nonseed-doc-freq",
        type=int,
        default=25,
        help="Minimum number of non-seed documents containing a token",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Smoothing for probability/lift estimates")
    parser.add_argument(
        "--require-success-status",
        action="store_true",
        help="Only use rows where Scrape_Status contains 'success'",
    )
    return parser.parse_args()


def ensure_nltk_resources() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            nltk.download("averaged_perceptron_tagger", quiet=True)


def build_stopword_set() -> set[str]:
    ensure_nltk_resources()
    return set(stopwords.words("english")) | DOMAIN_STOPWORDS


def penn_to_wordnet(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize(text: str, lemmatizer: WordNetLemmatizer, stopword_set: set[str]) -> set[str]:
    raw_tokens = TOKEN_RE.findall(text.lower())
    if not raw_tokens:
        return set()

    tagged_tokens = pos_tag(raw_tokens)
    return {
        lemma
        for token, pos in tagged_tokens
        for lemma in [lemmatizer.lemmatize(token, pos=penn_to_wordnet(pos)).strip("'")]
        if lemma and lemma not in stopword_set and not lemma.isdigit()
    }


def normalize_seed_terms(seed_terms: list[str], lemmatizer: WordNetLemmatizer) -> set[str]:
    normalized: set[str] = set()
    for term in seed_terms:
        text = (
            str(term)
            .lower()
            .replace("u.s.", " us ")
            .replace("u.s", " us ")
            .replace("united states", " united state ")
        )
        tokens = re.findall(r"[a-z][a-z']*", text)
        for token in tokens:
            lemma = lemmatizer.lemmatize(token, pos=wordnet.NOUN).strip("'")
            if lemma:
                normalized.add(lemma)
    return normalized


def parse_token_array(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    s = str(value).strip()
    if not s:
        return set()

    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return {str(x).strip() for x in parsed if str(x).strip()}

    if "," in s:
        return {p.strip() for p in s.split(",") if p.strip()}

    return {s}


def main() -> None:
    args = parse_args()

    if not args.lookup.exists():
        raise FileNotFoundError(f"Lookup file not found: {args.lookup}")

    df = pd.read_csv(args.lookup, low_memory=False)
    required_cols = {args.tokens_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Lookup missing required columns: {sorted(missing)}")

    if args.require_success_status:
        if args.status_col not in df.columns:
            raise ValueError(f"--require-success-status requested but missing status column: {args.status_col}")
        success_mask = df[args.status_col].fillna("").str.contains("success", case=False)
        df = df[success_mask].copy()

    token_sets: list[set[str]] = []
    for token_value in tqdm(df[args.tokens_col], total=len(df), desc="Loading token arrays", unit="row"):
        token_set = parse_token_array(token_value)
        if token_set:
            token_sets.append(token_set)

    total_docs = len(token_sets)
    if total_docs == 0:
        raise ValueError("No tokenized documents found in lookup. Run tokenize_url_lookup.py first.")

    df_counter: Counter[str] = Counter()
    seed_df_counter: Counter[str] = Counter()
    nonseed_df_counter: Counter[str] = Counter()

    seed_docs = 0
    nonseed_docs = 0

    lemmatizer = WordNetLemmatizer()
    seed_term_set = normalize_seed_terms(args.seed_terms, lemmatizer)

    for token_set in tqdm(token_sets, total=total_docs, desc="Building document frequencies", unit="doc"):
        is_seed_doc = bool(token_set & seed_term_set)
        if is_seed_doc:
            seed_docs += 1
            seed_df_counter.update(token_set)
        else:
            nonseed_docs += 1
            nonseed_df_counter.update(token_set)
        df_counter.update(token_set)

    usable_docs = seed_docs + nonseed_docs
    if usable_docs == 0:
        raise ValueError("No documents available after filtering.")
    if seed_docs == 0 or nonseed_docs == 0:
        raise ValueError(
            f"Need both seed and non-seed documents for scoring (seed_docs={seed_docs}, nonseed_docs={nonseed_docs})"
        )

    rows: list[dict[str, float | int | str | bool]] = []
    alpha = args.alpha

    for token, doc_freq in tqdm(df_counter.items(), total=len(df_counter), desc="Scoring tokens", unit="token"):
        is_protected_seed_token = token in seed_term_set
        doc_frac = doc_freq / usable_docs
        if (doc_frac < args.min_doc_frac or doc_frac > args.max_doc_frac) and not is_protected_seed_token:
            continue

        seed_freq = seed_df_counter.get(token, 0)
        nonseed_freq = nonseed_df_counter.get(token, 0)
        if seed_freq < args.min_seed_doc_freq and not is_protected_seed_token:
            continue
        if nonseed_freq < args.min_nonseed_doc_freq and not is_protected_seed_token:
            continue

        p_seed = (seed_freq + alpha) / (seed_docs + 2 * alpha)
        p_nonseed = (nonseed_freq + alpha) / (nonseed_docs + 2 * alpha)
        lift = p_seed / p_nonseed
        log2_lift = math.log2(lift)

        idf = math.log((1 + usable_docs) / (1 + doc_freq)) + 1
        relevance_score = log2_lift * idf

        rows.append(
            {
                "token": token,
                "doc_freq": int(doc_freq),
                "doc_frac": doc_frac,
                "seed_doc_freq": int(seed_freq),
                "nonseed_doc_freq": int(nonseed_freq),
                "p_seed": p_seed,
                "p_nonseed": p_nonseed,
                "lift": lift,
                "log2_lift": log2_lift,
                "idf": idf,
                "relevance_score": relevance_score,
                "is_protected_seed_token": is_protected_seed_token,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("No tokens survived frequency filtering; lower min-doc-frac or raise max-doc-frac")

    out_df = out_df.sort_values(["relevance_score", "doc_freq"], ascending=[False, False]).reset_index(drop=True)
    out_df["rank"] = out_df.index + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"Lookup rows after filters: {len(df):,}")
    print(f"Tokenized docs used: {total_docs:,}")
    print(f"Seed docs: {seed_docs:,} | Non-seed docs: {nonseed_docs:,}")
    print(f"Scored tokens written: {len(out_df):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
