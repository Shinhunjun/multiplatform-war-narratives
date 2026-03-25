# Weekly Update

This folder contains the incremental weekly refresh workflow for the Venezuela-USA GDELT corpus.

The purpose of `weekly_update/` is different from `data_collection/`:

- `data_collection/` is for historical dataset construction.
- `weekly_update/` is for ongoing maintenance of the existing corpus.

The weekly workflow optimizes for predictable automation, frozen scoring artifacts, static filter rules, and lightweight QA. It does not depend on Wayback rescue by default.

## Design Principles

- Treat `data/gdelt_scraped.csv` as the canonical master dataset.
- Ingest only new weekly event rows instead of rebuilding the full historical collection workflow.
- Preserve existing token relevance in `data/preprocessing/text_relevance_tokens.csv`.
- Preserve static filter rules in `preprocessing/filter_rule_config.json`.
- Update only new or changed URLs when possible.
- Write weekly logs, manifests, and QA snapshots under `data/weekly_runs/`.

## Folder Layout

```text
weekly_update/
├── README.md
├── run_weekly_update.py
├── fetch_weekly_events.py
├── scrape_weekly_urls.py
├── append_master_dataset.py
├── update_lookup_incremental.py
├── apply_frozen_relevance.py
├── apply_static_filters.py
├── refresh_analysis_exports.py
└── common.py
```

## Workflow

The intended weekly sequence is:

1. Read the latest event date already present in `data/gdelt_scraped.csv`.
2. Fetch official GDELT 2 export files from that date forward, inclusive, so same-date overlap is intentionally refetched.
3. Filter the fetched rows down to the bidirectional `VEN`/`USA` dyad and normalize them into the local raw-event schema.
4. Reuse existing lookup content where possible and scrape only new canonical URLs.
5. Append only genuinely new event rows into `data/gdelt_scraped.csv`, while ignoring overlap already present in the master dataset.
6. Incrementally update `data/preprocessing/url_lookup.csv`.
7. Apply the frozen token relevance table from `data/preprocessing/text_relevance_tokens.csv`.
8. Apply the static filter rules from `preprocessing/filter_rule_config.json` only to changed `url_id` values.
9. Rebuild analysis-ready parquet exports and optionally rerun EDA and downstream analysis.

## Scripts

### `run_weekly_update.py`

Primary orchestrator for the weekly pipeline.

Responsibilities:
- Create a run directory under `data/weekly_runs/`.
- Default the fetch start date from the latest event date already present in `data/gdelt_scraped.csv`.
- Execute each stage in order.
- Stop on failure and write a run manifest with status, timestamps, row counts, and file outputs.
- Support `--from-date`, `--to-date`, `--run-eda`, `--run-analysis`, and `--max-fetch-files`.

This should be the one command you run each week.

### `fetch_weekly_events.py`

Collect new GDELT event rows for the weekly window.

Responsibilities:
- Read the latest available official export timestamp from `http://data.gdeltproject.org/gdeltv2/lastupdate.txt`.
- Discover relevant `*.export.CSV.zip` files from `http://data.gdeltproject.org/gdeltv2/masterfilelist.txt`.
- Pull from the current master dataset's latest event date forward, inclusive, so same-date overlap is intentionally captured.
- Filter to the inferred historical corpus rule:
  `(Actor1CountryCode = 'VEN' AND Actor2CountryCode = 'USA') OR (Actor1CountryCode = 'USA' AND Actor2CountryCode = 'VEN')`.
- Normalize the incoming event schema to match `data/gdelt_scraped.csv` expectations.
- Write a weekly staging file such as `data/weekly_runs/<run_id>/weekly_events_raw.csv`.
- Deduplicate incoming event rows before scraping.

This script should not scrape article text. It should only gather the structured event rows that define the weekly update scope.

### `scrape_weekly_urls.py`

Scrape article content for the new weekly event rows.

Responsibilities:
- Extract candidate `SourceURL` values from the weekly event staging file.
- Reuse content already present in `data/preprocessing/url_lookup.csv` when the canonical URL already exists.
- Scrape only new canonical URLs.
- Write a weekly scrape result file such as `data/weekly_runs/<run_id>/weekly_scraped.csv`.
- Capture scrape status, retry counts, and basic QA counts.

This is the weekly replacement for the yearly scraping flow. It is optimized for short windows and quick reruns, not for full historical reconstruction.

### `append_master_dataset.py`

Merge the weekly scrape results into the canonical master dataset.

Responsibilities:
- Validate that the weekly scrape output matches the master schema.
- Append only genuinely new rows into `data/gdelt_scraped.csv`.
- Prevent duplicate insertion caused by same-date overlap.
- Write the actual appended subset into `data/weekly_runs/<run_id>/weekly_appended.csv`.
- Write a run-specific audit file such as `data/weekly_runs/<run_id>/append_audit.csv`.
- Preserve the existing historical rows unchanged.

This script replaces the role that yearly consolidation played for the bootstrap workflow, but for incremental weekly appends.

### `update_lookup_incremental.py`

Incrementally update URL-level lookup state from the refreshed master dataset.

Responsibilities:
- Reuse the canonicalization and stable `url_id` rules from `preprocessing/build_url_index.py`.
- Add new canonical URLs to `data/preprocessing/url_lookup.csv`.
- Preserve existing `url_id` assignments and existing lookup state.
- Update cumulative `row_count` values for touched canonical URLs.
- Identify which `url_id` values are new or content-changed in the current weekly run.
- Emit a small worklist such as `data/weekly_runs/<run_id>/changed_url_ids.csv`.

This should be a weekly-oriented wrapper around the existing URL-index logic, not a brand-new URL identity system.

### `apply_frozen_relevance.py`

Apply existing token relevance scores to weekly additions.

Responsibilities:
- Read frozen token weights from `data/preprocessing/text_relevance_tokens.csv`.
- Tokenize and score only new or changed lookup rows.
- Update the `doc_relevance_*` fields in `data/preprocessing/url_lookup.csv`.
- Write a QA snapshot such as `data/weekly_runs/<run_id>/weekly_score_summary.csv`.

This script should not rebuild token weights weekly.

### `apply_static_filters.py`

Apply the current filter policy to weekly additions.

Responsibilities:
- Read static rules from `preprocessing/filter_rule_config.json`.
- Recompute duplicate cluster sizes against the full lookup table.
- Evaluate only new or changed rows against the current thresholds and anchor logic.
- Update `data/preprocessing/url_filter_eval.csv`.
- Refresh `data/preprocessing/url_filter_summary_counts.csv`.
- Write weekly-only QA samples under `data/weekly_runs/<run_id>/filter_samples/`.
- Refresh the canonical histogram at `data/preprocessing/filter_stage_score_histograms.png`.

This script should preserve historical decisions unless a row is explicitly reprocessed.

### `refresh_analysis_exports.py`

Refresh downstream data products after weekly ingestion.

Responsibilities:
- Rebuild `data/analysis_ready/analysis_events.parquet`.
- Rebuild `data/analysis_ready/analysis_url_content.parquet`.
- Optionally rerun `eda/run_eda.py` when `--run-eda` is provided.
- Optionally rerun `python -m analysis.main --all` when `--run-analysis` is provided.

This stage is where the weekly pipeline reconnects with the existing EDA and analysis layers.

### `common.py`

Shared helpers for the weekly pipeline.

Responsibilities:
- Resolve project paths consistently.
- Create per-run directories under `data/weekly_runs/`.
- Read/write run manifests.
- Standardize logging, date-window parsing, and audit summaries.

This keeps orchestration code small and avoids repeating path logic across scripts.

## Proposed Data Outputs

The weekly workflow should use the existing canonical outputs plus a new run-history area:

```text
data/
├── gdelt_scraped.csv
├── preprocessing/
│   ├── url_lookup.csv
│   ├── text_relevance_tokens.csv
│   ├── url_filter_eval.csv
│   └── url_filter_summary_counts.csv
├── analysis_ready/
│   ├── analysis_events.parquet
│   └── analysis_url_content.parquet
└── weekly_runs/
    └── <run_id>/
        ├── manifest.json
        ├── weekly_events_raw.csv
        ├── weekly_scraped.csv
        ├── weekly_appended.csv
        ├── append_audit.csv
        ├── changed_url_ids.csv
        ├── weekly_score_summary.csv
        └── filter_samples/
```

## Boundary With Existing Folders

- `data_collection/` remains responsible for initial historical corpus construction.
- `weekly_update/` becomes responsible for incremental weekly maintenance.
- `preprocessing/` remains the home of source code and static configs for the preprocessing stage.
- `data/preprocessing/` remains the home of canonical lookup, scoring, filter, and QA artifacts.
- `eda/` and `analysis/` remain downstream consumers.

## Run Command

Typical weekly run:

```bash
python weekly_update/run_weekly_update.py --run-eda
```

If you want to override the fetch window:

```bash
python weekly_update/run_weekly_update.py --from-date 20260320 --to-date 20260323 --run-eda
```

## Current Tradeoffs

- Overlap handling is append-safe by raw event identity over the same-date window, but it does not rewrite historical master rows in place.
- Weekly filtering preserves historical decisions for unchanged URLs; only changed `url_id` values are reevaluated.
- EDA and downstream analysis are optional flags because they can be slower and more expensive than the core weekly ingest path.
