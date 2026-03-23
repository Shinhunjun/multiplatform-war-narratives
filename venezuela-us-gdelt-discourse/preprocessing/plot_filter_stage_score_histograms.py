from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DECISION_ORDER = ["drop", "review", "keep"]
DECISION_COLORS = {
    # Colorblind-safe palette (Okabe-Ito inspired)
    "drop": "#D55E00",
    "review": "#0072B2",
    "keep": "#009E73",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for filter-stage histogram plotting.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    project_dir = Path(__file__).resolve().parents[1]
    artifact_dir = project_dir / "data" / "preprocessing"
    parser = argparse.ArgumentParser(
        description=(
            "Create 3-panel histograms of doc_relevance_score split by filter stage "
            "decisions (length, score, anchor)."
        )
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=artifact_dir / "url_filter_eval.csv",
        help="Path to url_filter_eval.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "filter_stage_score_histograms.png",
        help="Output PNG path",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bin count")
    parser.add_argument(
        "--mode",
        choices=["overlay", "stacked"],
        default="stacked",
        help="Histogram rendering mode (default: stacked for readability)",
    )
    return parser.parse_args()


def annotate_counts(ax: plt.Axes, counts: dict[str, int], total: int) -> None:
    """Annotate an axis with drop/review/keep counts and total rows.
    
    Args:
        ax (plt.Axes): Matplotlib axis to annotate.
        counts (dict[str, int]): Decision-count mapping for the plotted stage.
        total (int): Total number of rows represented in the stage.
    
    Returns:
        None: No return value.
    """
    if total <= 0:
        text = "n=0"
    else:
        lines = [f"n={total:,}"]
        for decision in DECISION_ORDER:
            n = counts.get(decision, 0)
            pct = n / total
            lines.append(f"{decision}: {n:,} ({pct:.1%})")
        text = "\n".join(lines)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "#aaaaaa"},
    )


def plot_stage(
    ax: plt.Axes,
    frame: pd.DataFrame,
    decision_col: str,
    title: str,
    bins: int,
    x_min: float,
    x_max: float,
    mode: str,
) -> dict[str, int]:
    """Plot one filter-stage relevance-score distribution split by decision label.
    
    Args:
        ax (plt.Axes): Matplotlib axis for this stage.
        frame (pd.DataFrame): Input DataFrame containing evaluation rows.
        decision_col (str): Decision column for the stage (drop/review/keep).
        title (str): Subplot title.
        bins (int): Histogram bin count.
        x_min (float): Minimum x-axis score limit.
        x_max (float): Maximum x-axis score limit.
        mode (str): Histogram rendering mode (stacked or overlap).
    
    Returns:
        dict[str, int]: Decision counts used in this plot.
    """
    counts: dict[str, int] = {}
    series_by_decision: list[pd.Series] = []
    labels: list[str] = []
    colors: list[str] = []
    for decision in DECISION_ORDER:
        values = frame.loc[frame[decision_col] == decision, "doc_relevance_score"].dropna()
        n = int(len(values))
        counts[decision] = n
        series_by_decision.append(values)
        labels.append(f"{decision} (n={n:,})")
        colors.append(DECISION_COLORS[decision])

    if mode == "stacked":
        ax.hist(
            series_by_decision,
            bins=bins,
            stacked=True,
            alpha=0.85,
            color=colors,
            label=labels,
            range=(x_min, x_max),
            edgecolor="white",
            linewidth=0.25,
        )
    else:
        hatches = {"drop": "///", "review": "\\\\\\", "keep": "xxx"}
        for decision, values, label, color in zip(DECISION_ORDER, series_by_decision, labels, colors):
            ax.hist(
                values,
                bins=bins,
                alpha=0.45,
                color=color,
                label=label,
                range=(x_min, x_max),
                edgecolor="black",
                linewidth=0.35,
                hatch=hatches[decision],
            )

    total = sum(counts.values())
    annotate_counts(ax, counts, total)
    ax.set_title(title)
    ax.set_xlabel("doc_relevance_score")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    return counts


def main() -> None:
    """Render and save multi-stage relevance-score histograms from url_filter_eval.csv.
    
    Returns:
        None: No return value.
    """
    args = parse_args()

    if not args.eval_csv.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {args.eval_csv}")

    df = pd.read_csv(args.eval_csv, low_memory=False)
    required = {
        "in_filter_scope",
        "doc_relevance_score",
        "filter_duplicate_decision",
        "filter_length_decision",
        "filter_score_decision",
        "filter_anchor_decision",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in eval CSV: {sorted(missing)}")

    df["doc_relevance_score"] = pd.to_numeric(df["doc_relevance_score"], errors="coerce")
    in_scope = df[df["in_filter_scope"] == True].copy()
    in_scope = in_scope[in_scope["doc_relevance_score"].notna()].copy()
    if in_scope.empty:
        raise ValueError("No in-scope rows with numeric doc_relevance_score were found.")

    step0 = in_scope.copy()
    step1 = in_scope[in_scope["filter_duplicate_decision"] != "drop"].copy()
    step2 = step1[step1["filter_length_decision"] != "drop"].copy()
    step3 = step2[step2["filter_score_decision"] != "drop"].copy()

    x_min = float(in_scope["doc_relevance_score"].min())
    x_max = float(in_scope["doc_relevance_score"].max())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    ax = axes.ravel()
    fig.suptitle("Score Distributions by Filtering Stage Decisions", fontsize=20, fontweight="bold")

    c0 = plot_stage(
        ax[0],
        step0,
        "filter_duplicate_decision",
        f"Duplicate Text Decision Split\n(all in-scope rows, {args.mode})",
        args.bins,
        x_min,
        x_max,
        args.mode,
    )
    c1 = plot_stage(
        ax[1],
        step1,
        "filter_length_decision",
        f"Length Filter Decision Split\n(excluding duplicate-drop rows, {args.mode})",
        args.bins,
        x_min,
        x_max,
        args.mode,
    )
    c2 = plot_stage(
        ax[2],
        step2,
        "filter_score_decision",
        f"Score Filter Decision Split\n(excluding duplicate-drop and length-drop rows, {args.mode})",
        args.bins,
        x_min,
        x_max,
        args.mode,
    )
    c3 = plot_stage(
        ax[3],
        step3,
        "filter_anchor_decision",
        f"Anchor Filter Decision Split\n(excluding duplicate-drop, length-drop, and score-drop rows, {args.mode})",
        args.bins,
        x_min,
        x_max,
        args.mode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Saved: {args.output}")
    print("Stage counts:")
    print(
        "  Duplicate -> "
        f"drop={c0['drop']:,}, review={c0['review']:,}, keep={c0['keep']:,}, total={sum(c0.values()):,}"
    )
    print(
        "  Length -> "
        f"drop={c1['drop']:,}, review={c1['review']:,}, keep={c1['keep']:,}, total={sum(c1.values()):,}"
    )
    print(
        "  Score  -> "
        f"drop={c2['drop']:,}, review={c2['review']:,}, keep={c2['keep']:,}, total={sum(c2.values()):,}"
    )
    print(
        "  Anchor -> "
        f"drop={c3['drop']:,}, review={c3['review']:,}, keep={c3['keep']:,}, total={sum(c3.values()):,}"
    )


if __name__ == "__main__":
    main()
