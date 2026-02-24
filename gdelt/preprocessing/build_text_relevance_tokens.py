from __future__ import annotations

import argparse
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
    "member",
    "hold",
    "back",
    "add",
}

TOKEN_RE = re.compile(r"[a-z][a-z']{2,}")


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parents[1] / "data" / "gdelt_scraped.csv"
    default_out = Path(__file__).resolve().parent / "text_relevance_tokens.csv"

    parser = argparse.ArgumentParser(
        description="Build token relevance scores from gdelt_scraped Text field."
    )
    parser.add_argument("--input", type=Path, default=default_data, help="Path to gdelt_scraped.csv")
    parser.add_argument("--output", type=Path, default=default_out, help="Path to output token score CSV")
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

    # NLTK tagger path differs by version.
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


def normalize_seed_terms(seed_terms: list[str], lemmatizer: WordNetLemmatizer) -> set[str]:
    normalized: set[str] = set()
    for term in seed_terms:
        lowered = term.lower().strip()
        tokens = TOKEN_RE.findall(lowered)
        if tokens:
            tagged = pos_tag(tokens)
            for token, pos in tagged:
                lemma = lemmatizer.lemmatize(token, pos=penn_to_wordnet(pos)).strip("'")
                if lemma:
                    normalized.add(lemma)
        else:
            collapsed = re.sub(r"[^a-z]+", "", lowered)
            if collapsed:
                normalized.add(collapsed)
    return normalized


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


def seed_tokens_in_text(text: str, lemmatizer: WordNetLemmatizer) -> set[str]:
    text_lower = text.lower().replace("u.s.", " usa ").replace("u.s", " usa ")
    tokens = TOKEN_RE.findall(text_lower)
    if not tokens:
        return set()
    tagged = pos_tag(tokens)
    return {
        lemmatizer.lemmatize(token, pos=penn_to_wordnet(pos)).strip("'")
        for token, pos in tagged
        if token
    }


def has_seed_term(text: str, seed_term_set: set[str], lemmatizer: WordNetLemmatizer) -> bool:
    text_seed_tokens = seed_tokens_in_text(text, lemmatizer)
    return bool(text_seed_tokens & seed_term_set)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    if "Text" not in df.columns:
        raise ValueError("Expected a 'Text' column in input CSV")

    if args.require_success_status and "Scrape_Status" in df.columns:
        success_mask = df["Scrape_Status"].fillna("").str.contains("success", case=False)
        df = df[success_mask].copy()

    texts = df["Text"].dropna().astype(str)
    texts = texts[texts.str.strip() != ""]

    total_docs = int(len(texts))
    if total_docs == 0:
        raise ValueError("No non-empty text rows available after filtering")

    df_counter: Counter[str] = Counter()
    seed_df_counter: Counter[str] = Counter()
    nonseed_df_counter: Counter[str] = Counter()

    seed_docs = 0
    nonseed_docs = 0

    lemmatizer = WordNetLemmatizer()
    stopword_set = build_stopword_set()
    seed_term_set = normalize_seed_terms(args.seed_terms, lemmatizer)

    for text in tqdm(texts, total=total_docs, desc="Tokenizing documents", unit="doc"):
        tokens = tokenize(text, lemmatizer, stopword_set)
        if not tokens:
            continue

        is_seed_doc = has_seed_term(text, seed_term_set, lemmatizer)
        if is_seed_doc:
            seed_docs += 1
            seed_df_counter.update(tokens)
        else:
            nonseed_docs += 1
            nonseed_df_counter.update(tokens)

        df_counter.update(tokens)

    usable_docs = seed_docs + nonseed_docs
    if usable_docs == 0:
        raise ValueError("No tokenized documents available")
    if seed_docs == 0 or nonseed_docs == 0:
        raise ValueError(
            f"Need both seed and non-seed documents for scoring (seed_docs={seed_docs}, nonseed_docs={nonseed_docs})"
        )

    rows: list[dict[str, float | int | str]] = []
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

    print(f"Input rows after filters: {len(df):,}")
    print(f"Non-empty text docs: {total_docs:,}")
    print(f"Tokenized docs used: {usable_docs:,}")
    print(f"Seed docs: {seed_docs:,} | Non-seed docs: {nonseed_docs:,}")
    print(f"Scored tokens written: {len(out_df):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
