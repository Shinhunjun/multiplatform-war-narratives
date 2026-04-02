from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def sanitize_tag(value: str) -> str:
    """Convert an input filename into a safe suffix tag for derived output files.
    
    Args:
        value (str): Raw file stem or tag candidate.
    
    Returns:
        str: Filesystem-safe lowercase tag string.
    """
    tag = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return tag or "input"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing orchestrator.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    base = Path(__file__).resolve().parent
    default_input = base.parent / "data" / "gdelt_scraped.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Run the preprocessing pipeline end-to-end on a specified input CSV "
            "(URL indexing, tokenization, token scoring, document scoring, filtering, and histograms)."
        )
    )
    parser.add_argument("input", type=Path, help="Input gdelt_scraped-style CSV to process")
    parser.add_argument("--anchors", type=Path, default=base / "anchor_token_sets.json", help="Anchor token JSON")
    parser.add_argument(
        "--filter-rules",
        type=Path,
        default=base / "filter_rule_config.json",
        help="Filter rule config JSON",
    )

    # Optional explicit output paths. If omitted, defaults are derived from input file name.
    parser.add_argument("--lookup", type=Path, default=None, help="Output path for url_lookup CSV")
    parser.add_argument(
        "--token-scores",
        type=Path,
        default=None,
        help="Output path for text_relevance_tokens CSV",
    )
    parser.add_argument("--eval-output", type=Path, default=None, help="Output path for url_filter_eval CSV")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output path for url_filter_summary_counts CSV",
    )
    parser.add_argument("--sample-dir", type=Path, default=None, help="Directory for filter sample CSVs")
    parser.add_argument("--hist-output", type=Path, default=None, help="Output PNG path for filter-stage histograms")

    parser.add_argument(
        "--indexed-output",
        type=Path,
        default=None,
        help="Optional path to write a gdelt_scraped copy with url_id added (input is never overwritten by default).",
    )
    parser.add_argument("--force-retokenize", action="store_true", help="Retokenize all rows in url_lookup")
    parser.add_argument(
        "--require-success-status",
        action="store_true",
        help="Only use success-status rows when building token relevance scores",
    )
    parser.add_argument("--sample-size", type=int, default=25, help="Sample size per decision bucket")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bin count")
    parser.add_argument(
        "--hist-mode",
        choices=["stacked", "overlay"],
        default="stacked",
        help="Histogram rendering mode",
    )
    return parser.parse_args()


def derive_default_paths(input_path: Path, base: Path) -> dict[str, Path]:
    """Derive standard output paths for all preprocessing artifacts from an input file.
    
    Args:
        input_path (Path): Input scraped CSV path.
        base (Path): Preprocessing directory used to store generated outputs.
    
    Returns:
        dict[str, Path]: Dictionary of output artifact paths keyed by artifact name.
    """
    data_dir = base.parent / "data"
    artifact_dir = data_dir / "preprocessing"
    default_input = (data_dir / "gdelt_scraped.csv").resolve()
    resolved_input = input_path.resolve()

    if resolved_input == default_input:
        return {
            "lookup": artifact_dir / "url_lookup.csv",
            "token_scores": artifact_dir / "text_relevance_tokens.csv",
            "eval_output": artifact_dir / "url_filter_eval.csv",
            "summary_output": artifact_dir / "url_filter_summary_counts.csv",
            "sample_dir": artifact_dir / "filter_samples",
            "hist_output": artifact_dir / "filter_stage_score_histograms.png",
        }

    tag = sanitize_tag(input_path.stem)
    return {
        "lookup": artifact_dir / f"url_lookup_{tag}.csv",
        "token_scores": artifact_dir / f"text_relevance_tokens_{tag}.csv",
        "eval_output": artifact_dir / f"url_filter_eval_{tag}.csv",
        "summary_output": artifact_dir / f"url_filter_summary_counts_{tag}.csv",
        "sample_dir": artifact_dir / f"filter_samples_{tag}",
        "hist_output": artifact_dir / f"filter_stage_score_histograms_{tag}.png",
    }


def run_step(label: str, command: list[str]) -> None:
    """Run one pipeline subprocess step and fail fast on non-zero exit status.
    
    Args:
        label (str): Human-readable step label for terminal logs.
        command (list[str]): Subprocess command list to execute.
    
    Returns:
        None: No return value.
    """
    print("", flush=True)
    print("=" * 72, flush=True)
    print(label, flush=True)
    print("=" * 72, flush=True)
    print("Command:", " ".join(command), flush=True)
    start = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, check=True, env=env)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s", flush=True)


def main() -> None:
    """Execute the full weekly preprocessing pipeline in deterministic step order.
    
    Returns:
        None: No return value.
    """
    args = parse_args()
    base = Path(__file__).resolve().parent

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.anchors.exists():
        raise FileNotFoundError(f"Anchor file not found: {args.anchors}")
    if not args.filter_rules.exists():
        raise FileNotFoundError(f"Filter rule config file not found: {args.filter_rules}")

    defaults = derive_default_paths(args.input, base)
    lookup = args.lookup if args.lookup is not None else defaults["lookup"]
    token_scores = args.token_scores if args.token_scores is not None else defaults["token_scores"]
    eval_output = args.eval_output if args.eval_output is not None else defaults["eval_output"]
    summary_output = args.summary_output if args.summary_output is not None else defaults["summary_output"]
    sample_dir = args.sample_dir if args.sample_dir is not None else defaults["sample_dir"]
    hist_output = args.hist_output if args.hist_output is not None else defaults["hist_output"]

    lookup.parent.mkdir(parents=True, exist_ok=True)
    token_scores.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.parent.mkdir(parents=True, exist_ok=True)
    hist_output.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    cmd1 = [
        py,
        str(base / "build_url_index.py"),
        "--input",
        str(args.input),
        "--lookup",
        str(lookup),
    ]
    if args.indexed_output is not None:
        cmd1 += ["--output", str(args.indexed_output)]
    run_step("Step 1/7: Build URL Index", cmd1)

    cmd2 = [
        py,
        str(base / "tokenize_url_lookup.py"),
        "--lookup",
        str(lookup),
    ]
    if args.force_retokenize:
        cmd2.append("--force")
    run_step("Step 2/7: Tokenize URL Lookup", cmd2)

    cmd3 = [
        py,
        str(base / "build_duplicate_filter_eval.py"),
        "--lookup",
        str(lookup),
        "--output",
        str(eval_output),
        "--summary-output",
        str(summary_output),
    ]
    run_step("Step 3/7: Build Duplicate Filter Eval (Early)", cmd3)

    cmd4 = [
        py,
        str(base / "build_text_relevance_tokens.py"),
        "--lookup",
        str(lookup),
        "--output",
        str(token_scores),
        "--eval",
        str(eval_output),
        "--exclude-duplicate-drops",
    ]
    if args.require_success_status:
        cmd4.append("--require-success-status")
    run_step("Step 4/7: Build Token Relevance Scores", cmd4)

    cmd5 = [
        py,
        str(base / "score_url_relevance.py"),
        "--lookup",
        str(lookup),
        "--relevance",
        str(token_scores),
    ]
    run_step("Step 5/7: Score URL Relevance", cmd5)

    cmd6 = [
        py,
        str(base / "evaluate_filter_strategy.py"),
        "--lookup",
        str(lookup),
        "--anchors",
        str(args.anchors),
        "--filter-rules",
        str(args.filter_rules),
        "--output",
        str(eval_output),
        "--summary-output",
        str(summary_output),
        "--sample-dir",
        str(sample_dir),
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
    ]
    run_step("Step 6/7: Evaluate Filter Strategy (Full)", cmd6)

    cmd7 = [
        py,
        str(base / "plot_filter_stage_score_histograms.py"),
        "--eval-csv",
        str(eval_output),
        "--output",
        str(hist_output),
        "--bins",
        str(args.bins),
        "--mode",
        args.hist_mode,
    ]
    run_step("Step 7/7: Plot Filter Stage Histograms", cmd7)

    print("", flush=True)
    print("Pipeline complete.", flush=True)
    print(f"Input processed: {args.input}", flush=True)
    print(f"url_lookup: {lookup}", flush=True)
    print(f"token scores: {token_scores}", flush=True)
    print(f"filter eval: {eval_output}", flush=True)
    print(f"summary counts: {summary_output}", flush=True)
    print(f"sample dir: {sample_dir}", flush=True)
    print(f"histogram PNG: {hist_output}", flush=True)
    if args.indexed_output is not None:
        print(f"indexed input copy: {args.indexed_output}", flush=True)


if __name__ == "__main__":
    main()
