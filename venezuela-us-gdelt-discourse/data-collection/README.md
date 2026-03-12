# Data Collection Workflow

This folder contains scripts used to build and consolidate yearly Venezuela-US GDELT scrape files.

## Scripts

- `scrape_by_year.py`: scrapes article `Title`/`Text` from `SourceURL` for one year into `ven_usa_{YEAR}.csv`.
- `rescue_by_year.py`: retries failed rows through Wayback and writes `ven_usa_{YEAR}_rescued.csv`.
- `consolidate_yearly.py`: merges yearly original + rescued files into one consolidated output.

## Expected Files

For consolidation of a year `Y`:

- Required: `ven_usa_Y.csv`
- Optional: `ven_usa_Y_rescued.csv` (needed to apply rescue replacements)

`consolidate_yearly.py` processes years independently, then appends them into a single output file.

## Consolidation Behavior (Important)

The merge is original-first.

- Rows with complete original scrape (`Title` + `Text` + success status) are kept.
- Rescue data is only copied when original row has both `Title` and `Text` missing.
- Rescue rows must also be complete and successful (`Success` or `Success (Archived)`).
- Matching is done by a deterministic key built from event columns + duplicate occurrence index.
- Audit diagnostics are computed per year:
  - `RowsReplacedFromRescue`
  - `PotentialOverwriteRiskRows`
  - `IndexKeyMismatchRows`
  - `OriginalKeyDuplicateRows`

Note: this script does not do incremental in-place updates against an already-existing `gdelt_scraped.csv`. It rebuilds output from the selected yearly files.

## Run Commands

Run from the project root (`venezuela-us-gdelt-discourse`) unless noted.

Scrape one year (run from `data/` because script paths are relative):

```powershell
cd .\data
python ..\data-collection\scrape_by_year.py 2020
```

Rescue one year (run from `data/`):

```powershell
cd .\data
python ..\data-collection\rescue_by_year.py 2020
```

Consolidate one year:

```powershell
python .\data-collection\consolidate_yearly.py --base-dir .\data --years 2020
```

Dry run (same logic, no files written):

```powershell
python .\data-collection\consolidate_yearly.py --base-dir .\data --years 2020 --dry-run
```

Multiple years:

```powershell
python .\data-collection\consolidate_yearly.py --base-dir .\data --years 2020-2026
```

Disable progress bar:

```powershell
python .\data-collection\consolidate_yearly.py --base-dir .\data --years 2020-2026 --no-progress
```

## Output Files

By default, consolidation writes into `--base-dir`:

- `gdelt_scraped.csv`
- `problematic_rows.csv`
- `consolidation_audit.csv`

## Example: 2020 Consolidation

Example run summary:

- `OriginalRows`: `17881`
- `RowsReplacedFromRescue`: `5353`
- `PotentialOverwriteRiskRows`: `0`
- `IndexKeyMismatchRows`: `0`

Files written:

- `data/gdelt_scraped.csv`
- `data/problematic_rows.csv`
- `data/consolidation_audit.csv`
