import argparse
from pathlib import Path
import sys
import pandas as pd

KEY_COLS = [
    "Date",
    "Year",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor2Name",
    "Actor2CountryCode",
    "EventCode",
    "QuadClass",
    "GoldsteinScale",
    "AvgTone",
    "SourceURL",
]

CORE_COLS = KEY_COLS + ["Title", "Text", "Scrape_Status", "Error_Details"]
SUCCESS_STATUS = {"Success", "Success (Archived)"}


def print_progress(current: int, total: int, year: int | None = None, width: int = 36) -> None:
    """Render a simple in-place progress bar for yearly processing."""
    if total <= 0:
        return

    ratio = max(0.0, min(1.0, current / total))
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    pct = ratio * 100
    year_label = f" | year {year}" if year is not None else ""

    sys.stdout.write(f"\rProgress{year_label}: [{bar}] {current}/{total} ({pct:5.1f}%)")
    sys.stdout.flush()

    if current >= total:
        sys.stdout.write("\n")


def print_rotated_audit(audit_df: pd.DataFrame) -> None:
    """Print audit diagnostics in a readable year-by-year vertical layout."""
    if audit_df.empty:
        return

    metric_cols = [col for col in audit_df.columns if col != "Year"]
    label_width = max(len(col) for col in metric_cols)

    for _, row in audit_df.iterrows():
        print(f"\nYear {int(row['Year'])}")
        for col in metric_cols:
            print(f"  {col:<{label_width}} : {row[col]}")


def parse_years(expr: str) -> list[int]:
    """Parse a year expression into an explicit sorted year list.
    
    Args:
        expr (str): Year expression string to parse.
    
    Returns:
        list[int]: List result produced by this function.
    """
    years: list[int] = []
    for chunk in expr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            step = 1 if end >= start else -1
            years.extend(list(range(start, end + step, step)))
        else:
            years.append(int(chunk))
    return sorted(set(years))


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve a potentially relative path against the consolidation base directory.
    
    Args:
        base_dir (Path): Base directory used for path resolution.
        value (str): Input value to process.
    
    Returns:
        Path: Filesystem path value.
    """
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Ensure required columns exist in a DataFrame, creating missing columns when needed.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
        columns (list[str]): List of columns that must exist in the DataFrame.
    
    Returns:
        None: No return value.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA


def present(series: pd.Series) -> pd.Series:
    """Compute a boolean mask for non-empty values in a Series.
    
    Args:
        series (pd.Series): Input pandas Series.
    
    Returns:
        pd.Series: Computed result for this function.
    """
    return series.notna() & (series.astype(str).str.strip() != "")


def normalize(series: pd.Series) -> pd.Series:
    """Normalize text values for deterministic key generation.
    
    Args:
        series (pd.Series): Input pandas Series.
    
    Returns:
        pd.Series: Computed result for this function.
    """
    return series.fillna("<NA>").astype(str).str.strip()


def key_with_occurrence(df: pd.DataFrame) -> pd.Series:
    """Build a stable record key with occurrence indexing for duplicate handling.
    
    Args:
        df (pd.DataFrame): Input DataFrame to process.
    
    Returns:
        pd.Series: Computed result for this function.
    """
    key_df = pd.DataFrame({col: normalize(df[col]) for col in KEY_COLS})
    key = key_df.agg("|".join, axis=1)
    occ = key.groupby(key, sort=False).cumcount()
    return key + "||" + occ.astype(str)


def issue_type(title_ok: bool, text_ok: bool) -> str:
    """Classify merge quality based on title/text completeness.
    
    Args:
        title_ok (bool): Whether title content is available.
        text_ok (bool): Whether body text content is available.
    
    Returns:
        str: Processed string value.
    """
    if title_ok and not text_ok:
        return "Title_Only"
    return "Text_Only"


def consolidate_year(base_dir: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Consolidate yearly primary and rescued scrape files and produce QA diagnostics.
    
    Args:
        base_dir (Path): Base directory used for path resolution.
        year (int): Year value being processed.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: Processed pandas DataFrame.
    """
    orig_path = base_dir / f"ven_usa_{year}.csv"
    resc_path = base_dir / f"ven_usa_{year}_rescued.csv"

    if not orig_path.exists():
        raise FileNotFoundError(f"Missing original file: {orig_path}")

    odf = pd.read_csv(orig_path, low_memory=False)
    ensure_columns(odf, CORE_COLS)

    original_key_dupes = int(odf.duplicated(subset=KEY_COLS, keep=False).sum())

    title_ok = present(odf["Title"])
    text_ok = present(odf["Text"])
    status_ok = odf["Scrape_Status"].fillna("").astype(str).isin(SUCCESS_STATUS)

    orig_good = title_ok & text_ok & status_ok
    orig_partial = title_ok ^ text_ok
    orig_both_missing = (~title_ok) & (~text_ok)

    problematic = odf.loc[orig_partial, CORE_COLS].copy()
    if not problematic.empty:
        problematic.insert(0, "RowInYear", problematic.index)
        problematic.insert(0, "YearFile", year)
        title_flags = title_ok.loc[problematic.index]
        text_flags = text_ok.loc[problematic.index]
        problematic["Issue_Type"] = [
            issue_type(bool(t), bool(x)) for t, x in zip(title_flags.tolist(), text_flags.tolist())
        ]

    replaced_count = 0
    overwrite_risk_count = 0
    key_mismatch_count = 0

    if resc_path.exists():
        rdf = pd.read_csv(resc_path, low_memory=False)
        ensure_columns(rdf, CORE_COLS)

        okey = key_with_occurrence(odf)
        rkey = key_with_occurrence(rdf)

        r_title_ok = present(rdf["Title"])
        r_text_ok = present(rdf["Text"])
        r_status_ok = rdf["Scrape_Status"].fillna("").astype(str).isin(SUCCESS_STATUS)
        resc_good = r_title_ok & r_text_ok & r_status_ok

        r_lookup = rdf.loc[resc_good, ["Title", "Text", "Scrape_Status", "Error_Details"]].copy()
        r_lookup["__key__"] = rkey[resc_good].values
        r_lookup = r_lookup.drop_duplicates(subset="__key__", keep="first").set_index("__key__")

        can_replace = orig_both_missing & okey.isin(r_lookup.index)
        if can_replace.any():
            replacements = r_lookup.loc[okey[can_replace], ["Title", "Text", "Scrape_Status", "Error_Details"]]
            odf.loc[can_replace, ["Title", "Text", "Scrape_Status", "Error_Details"]] = replacements.to_numpy()
            replaced_count = int(can_replace.sum())

        o_good_comp = pd.DataFrame(
            {
                "__key__": okey[orig_good].values,
                "Title_orig": normalize(odf.loc[orig_good, "Title"]).values,
                "Text_orig": normalize(odf.loc[orig_good, "Text"]).values,
            }
        )
        r_good_comp = pd.DataFrame(
            {
                "__key__": rkey[resc_good].values,
                "Title_resc": normalize(rdf.loc[resc_good, "Title"]).values,
                "Text_resc": normalize(rdf.loc[resc_good, "Text"]).values,
            }
        )

        if not o_good_comp.empty and not r_good_comp.empty:
            comp = o_good_comp.merge(r_good_comp, on="__key__", how="inner")
            overwrite_risk_count = int(
                ((comp["Title_orig"] != comp["Title_resc"]) | (comp["Text_orig"] != comp["Text_resc"])).sum()
            )

        n = min(len(odf), len(rdf))
        if n > 0:
            on = pd.DataFrame({col: normalize(odf[col].iloc[:n].reset_index(drop=True)) for col in KEY_COLS})
            rn = pd.DataFrame({col: normalize(rdf[col].iloc[:n].reset_index(drop=True)) for col in KEY_COLS})
            key_mismatch_count = int((on != rn).any(axis=1).sum())

    audit = {
        "Year": year,
        "OriginalRows": len(odf),
        "RescuedFileExists": resc_path.exists(),
        "OriginalGoodRows": int(orig_good.sum()),
        "OriginalPartialRows": int(orig_partial.sum()),
        "OriginalBothMissingRows": int(orig_both_missing.sum()),
        "RowsReplacedFromRescue": replaced_count,
        "PotentialOverwriteRiskRows": overwrite_risk_count,
        "IndexKeyMismatchRows": key_mismatch_count,
        "OriginalKeyDuplicateRows": original_key_dupes,
    }

    return odf[CORE_COLS].copy(), problematic, audit


def main() -> None:
    """Run the script entry point.
    
    Returns:
        None: No return value.
    """
    parser = argparse.ArgumentParser(
        description="Consolidate yearly scraped and rescued CSV files with original-first data priority."
    )
    parser.add_argument("--base-dir", default="data", help="Directory containing yearly CSV files.")
    parser.add_argument("--years", default="2013-2026", help="Years to process, e.g. 2013-2026 or 2013,2015,2017.")
    parser.add_argument("--output", default="gdelt_scraped.csv", help="Consolidated CSV filename or absolute path.")
    parser.add_argument(
        "--problematic",
        default="problematic_rows.csv",
        help="CSV filename or absolute path for rows with only one of Title/Text present in original files.",
    )
    parser.add_argument(
        "--audit",
        default="consolidation_audit.csv",
        help="CSV filename or absolute path for year-level audit stats.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run audit and counts without writing output files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    years = parse_years(args.years)

    output_path = resolve_path(base_dir, args.output)
    problematic_path = resolve_path(base_dir, args.problematic)
    audit_path = resolve_path(base_dir, args.audit)

    if not years:
        raise ValueError("No valid years provided.")

    print(f"Base directory: {base_dir}")
    print(f"Years: {years}")

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        problematic_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_output = False
    wrote_problematic = False
    audit_rows: list[dict] = []

    total_rows = 0
    total_problematic = 0

    if not args.no_progress:
        print_progress(0, len(years))

    for i, year in enumerate(years, start=1):
        if not args.no_progress:
            sys.stdout.write("\n")
        orig_file = base_dir / f"ven_usa_{year}.csv"
        if not orig_file.exists():
            print(f"Skipping {year}: missing {orig_file.name}")
            if not args.no_progress:
                print_progress(i, len(years), year)
            continue

        print(f"Processing year {year}...")
        consolidated_df, problematic_df, audit = consolidate_year(base_dir, year)
        audit_rows.append(audit)

        total_rows += len(consolidated_df)
        total_problematic += len(problematic_df)

        if not args.dry_run:
            consolidated_df.to_csv(output_path, mode="a" if wrote_output else "w", header=not wrote_output, index=False)
            wrote_output = True

            if not problematic_df.empty:
                problematic_df.to_csv(
                    problematic_path,
                    mode="a" if wrote_problematic else "w",
                    header=not wrote_problematic,
                    index=False,
                )
                wrote_problematic = True

        if not args.no_progress:
            print_progress(i, len(years), year)

    audit_df = pd.DataFrame(audit_rows)

    if audit_df.empty:
        raise RuntimeError("No years were processed. Check --base-dir and --years.")

    if not args.dry_run:
        if not wrote_output:
            pd.DataFrame(columns=CORE_COLS).to_csv(output_path, index=False)
        if not wrote_problematic:
            pd.DataFrame(columns=["YearFile", "RowInYear"] + CORE_COLS + ["Issue_Type"]).to_csv(
                problematic_path, index=False
            )
        audit_df.to_csv(audit_path, index=False)

    print("\nAudit summary by year (rotated):")
    print_rotated_audit(audit_df)
    print("\nTotals")
    print(f"Consolidated rows: {total_rows}")
    print(f"Problematic rows:  {total_problematic}")
    print(f"Rows replaced from rescue: {int(audit_df['RowsReplacedFromRescue'].sum())}")
    print(f"Potential overwrite risk rows: {int(audit_df['PotentialOverwriteRiskRows'].sum())}")
    print(f"Index key mismatch rows: {int(audit_df['IndexKeyMismatchRows'].sum())}")

    if not args.dry_run:
        print(f"\nWrote consolidated CSV: {output_path}")
        print(f"Wrote problematic rows CSV: {problematic_path}")
        print(f"Wrote audit CSV: {audit_path}")


if __name__ == "__main__":
    main()
