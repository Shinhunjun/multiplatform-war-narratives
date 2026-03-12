from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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
    "u.s.",
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

SPECIAL_KEEP_TOKENS = {"us", "u.s.", "usa"}
CONTRACTION_FRAGMENTS = {"n't", "'re", "'ve", "'ll", "'d", "'m", "'s"}
US_DOTTED_RE = re.compile(r"(?<!\w)u\.s\.(?!\w)", re.IGNORECASE)
LETTER_TOKEN_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
EDGE_CLEAN_RE = re.compile(r"^[^a-z0-9'.]+|[^a-z0-9'.]+$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for token relevance scoring.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    default_lookup = Path(__file__).resolve().parent / "url_lookup.csv"
    default_out = Path(__file__).resolve().parent / "text_relevance_tokens.csv"

    parser = argparse.ArgumentParser(
        description="Build token relevance scores from pre-tokenized url_lookup.csv Tokens."
    )
    parser.add_argument("--lookup", type=Path, default=default_lookup, help="Path to url_lookup.csv")
    parser.add_argument("--output", type=Path, default=default_out, help="Path to output token score CSV")
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Optional path to url_filter_eval.csv for filtering training rows",
    )
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
    parser.add_argument(
        "--exclude-duplicate-drops",
        action="store_true",
        help="Exclude rows marked duplicate-drop in --eval (or rows where used_for_token_training is false)",
    )
    return parser.parse_args()


def ensure_nltk_resources() -> None:
    """Ensure required NLTK tokenization, tagging, and lexical resources are available locally.
    
    Returns:
        None: No return value.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)

    # Newer NLTK builds may require punkt_tab separately.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            nltk.download("averaged_perceptron_tagger", quiet=True)


def build_stopword_set() -> set[str]:
    """Build the stopword vocabulary by combining NLTK English stopwords and domain stopwords.
    
    Returns:
        set[str]: Combined stopword set.
    """
    ensure_nltk_resources()
    return set(stopwords.words("english")) | DOMAIN_STOPWORDS


def penn_to_wordnet(tag: str) -> str:
    """Map a Penn Treebank POS tag to the closest WordNet POS category.
    
    Args:
        tag (str): Penn Treebank tag from NLTK POS tagging.
    
    Returns:
        str: WordNet POS constant used by the lemmatizer.
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def normalize_raw_token(token: str) -> str:
    """Normalize a raw token by lowercasing and cleaning punctuation/contractions.
    
    Args:
        token (str): Raw token produced by tokenization.
    
    Returns:
        str: Normalized token string, or an empty string if nothing usable remains.
    """
    token = token.lower().replace("’", "'").replace("`", "'")
    token = EDGE_CLEAN_RE.sub("", token)
    if not token:
        return ""

    if token in {"u.s.", "u.s"}:
        return "u.s."

    if token.endswith("'s"):
        token = token[:-2]
    elif token.endswith("s'"):
        token = token[:-1]

    token = token.strip("'")
    return token


def parse_text_tokens(text: str) -> list[str]:
    """Tokenize free text and apply project-specific token normalization rules.
    
    Args:
        text (str): Raw document text.
    
    Returns:
        list[str]: Ordered list of normalized tokens.
    """
    raw_tokens = word_tokenize(text)
    parsed: list[str] = []
    for raw in raw_tokens:
        token = normalize_raw_token(raw)
        if not token:
            continue
        if token in CONTRACTION_FRAGMENTS:
            continue
        parsed.append(token)

    # Explicit project rule: if "U.S." appears in text, force-add "u.s." token.
    if US_DOTTED_RE.search(text):
        parsed.append("u.s.")

    return parsed


def tokenize(text: str, lemmatizer: WordNetLemmatizer, stopword_set: set[str]) -> set[str]:
    """Convert document text into a cleaned, lemmatized set of lexical tokens.
    
    Args:
        text (str): Raw document text.
        lemmatizer (WordNetLemmatizer): Initialized WordNet lemmatizer.
        stopword_set (set[str]): Stopword set used to remove low-information tokens.
    
    Returns:
        set[str]: Unique normalized token set for the document.
    """
    if text is None:
        return set()

    parsed_tokens = parse_text_tokens(str(text))
    if not parsed_tokens:
        return set()

    special = {tok for tok in parsed_tokens if tok in SPECIAL_KEEP_TOKENS}
    lexical = [tok for tok in parsed_tokens if tok not in special and LETTER_TOKEN_RE.fullmatch(tok)]

    lemmas: set[str] = set()
    if lexical:
        tagged_tokens = pos_tag(lexical)
        for token, pos in tagged_tokens:
            lemma = lemmatizer.lemmatize(token, pos=penn_to_wordnet(pos)).strip("'")
            if not lemma:
                continue
            if lemma in CONTRACTION_FRAGMENTS:
                continue
            if lemma.isdigit():
                continue
            if lemma in stopword_set and lemma not in SPECIAL_KEEP_TOKENS:
                continue
            lemmas.add(lemma)

    for token in special:
        if token not in stopword_set or token in SPECIAL_KEEP_TOKENS:
            lemmas.add(token)

    return lemmas


def normalize_seed_terms(seed_terms: list[str], lemmatizer: WordNetLemmatizer) -> set[str]:
    """Normalize seed terms using the same tokenization and lemmatization logic as documents.
    
    Args:
        seed_terms (list[str]): Seed term list that defines likely in-scope documents.
        lemmatizer (WordNetLemmatizer): Initialized WordNet lemmatizer.
    
    Returns:
        set[str]: Normalized seed-term token set.
    """
    normalized: set[str] = set()
    for term in seed_terms:
        parsed = parse_text_tokens(str(term))
        for token in parsed:
            if token in SPECIAL_KEEP_TOKENS:
                normalized.add(token)
                continue
            if not LETTER_TOKEN_RE.fullmatch(token):
                continue
            lemma = lemmatizer.lemmatize(token, pos=wordnet.NOUN).strip("'")
            if lemma:
                normalized.add(lemma)
    return normalized


def parse_token_array(value: object) -> set[str]:
    """Parse token arrays stored as JSON/list-like strings in CSV fields.
    
    Args:
        value (object): Serialized token field value from the lookup table.
    
    Returns:
        set[str]: Normalized token set parsed from the input value.
    """
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
        candidates = [str(x).strip() for x in parsed if str(x).strip()]
    elif "," in s:
        candidates = [p.strip() for p in s.split(",") if p.strip()]
    else:
        candidates = [s]

    normalized: set[str] = set()
    for token in candidates:
        t = normalize_raw_token(token)
        if not t or t in CONTRACTION_FRAGMENTS:
            continue
        if t in SPECIAL_KEEP_TOKENS:
            normalized.add(t)
            continue
        if LETTER_TOKEN_RE.fullmatch(t):
            normalized.add(t)
    return normalized


def main() -> None:
    """Compute per-token relevance scores from tokenized lookup rows and write ranked output.
    
    Returns:
        None: No return value.
    """
    args = parse_args()
    ensure_nltk_resources()

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

    if args.exclude_duplicate_drops:
        if args.eval is None:
            raise ValueError("--exclude-duplicate-drops requires --eval <url_filter_eval.csv>")
        if not args.eval.exists():
            raise FileNotFoundError(f"Eval file not found: {args.eval}")
        if "url_id" not in df.columns:
            raise ValueError("--exclude-duplicate-drops requires 'url_id' in lookup CSV")

        eval_df = pd.read_csv(args.eval, low_memory=False)
        if "url_id" not in eval_df.columns:
            raise ValueError(f"Eval file missing required column: url_id ({args.eval})")

        keep_col = None
        if "used_for_token_training" in eval_df.columns:
            keep_col = "used_for_token_training"
            eval_keep = eval_df[["url_id", keep_col]].copy()
            eval_keep[keep_col] = (
                eval_keep[keep_col]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
        elif "filter_duplicate_decision" in eval_df.columns:
            keep_col = "filter_duplicate_decision"
            eval_keep = eval_df[["url_id", keep_col]].copy()
            eval_keep[keep_col] = eval_keep[keep_col].fillna("out_of_scope").astype(str)
        else:
            raise ValueError(
                "Eval file must contain either 'used_for_token_training' or 'filter_duplicate_decision' "
                "to support --exclude-duplicate-drops"
            )

        eval_keep["url_id"] = pd.to_numeric(eval_keep["url_id"], errors="coerce").astype("Int64")
        df["url_id"] = pd.to_numeric(df["url_id"], errors="coerce").astype("Int64")
        merged = df.merge(eval_keep, on="url_id", how="left")

        if keep_col == "used_for_token_training":
            keep_mask = merged[keep_col] == True
        else:
            keep_mask = merged[keep_col] != "drop"
            keep_mask = keep_mask & merged[keep_col].notna()

        missing_eval = int(merged[keep_col].isna().sum())
        before_dedup = len(merged)
        df = merged[keep_mask].copy()
        df = df.drop(columns=[keep_col], errors="ignore")
        after_dedup = len(df)
        print(
            f"Duplicate-filter training gate: kept {after_dedup:,}/{before_dedup:,} rows "
            f"(excluded {before_dedup - after_dedup:,}; missing eval rows={missing_eval:,})",
            flush=True,
        )

    token_sets: list[set[str]] = []
    for token_value in tqdm(
        df[args.tokens_col],
        total=len(df),
        desc="Loading token arrays",
        unit="row",
        file=sys.stdout,
    ):
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

    for token_set in tqdm(
        token_sets,
        total=total_docs,
        desc="Building document frequencies",
        unit="doc",
        file=sys.stdout,
    ):
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

    for token, doc_freq in tqdm(
        df_counter.items(),
        total=len(df_counter),
        desc="Scoring tokens",
        unit="token",
        file=sys.stdout,
    ):
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
