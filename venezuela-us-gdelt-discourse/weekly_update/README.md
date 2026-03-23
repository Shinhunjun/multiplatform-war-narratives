# Weekly Update Design

This folder is the proposed home for the incremental weekly refresh workflow.

The purpose of `weekly_update/` is different from `data_collection/`:

- `data_collection/` is for historical dataset construction.
- `weekly_update/` is for ongoing maintenance of the existing corpus.

The weekly workflow should optimize for predictable automation, frozen scoring artifacts, static filter rules, and lightweight QA. It should not depend on Wayback rescue by default.

## Design Principles

- Treat `data/gdelt_scraped.csv` as the canonical master dataset.
- Ingest only new weekly event rows instead of rebuilding the full historical collection workflow.
- Preserve existing token relevance in `data/preprocessing/text_relevance_tokens.csv`.
- Preserve static filter rules in `preprocessing/filter_rule_config.json`.
- Update only new or changed URLs when possible.
- Write weekly logs, manifests, and QA snapshots under `data/weekly_runs/`.

## Proposed Folder Layout

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

## Proposed Responsibilities

### `run_weekly_update.py`

Primary orchestrator for the weekly pipeline.

Responsibilities:
- Read the last successful run metadata from `data/weekly_runs/`.
- Determine the weekly date window or input file for the current run.
- Execute each stage in order.
- Stop on failure and write a run manifest with status, timestamps, row counts, and file outputs.
- Optionally support flags like `--dry-run`, `--from-date`, `--to-date`, and `--skip-analysis`.

This should be the one command you run each week.

### `fetch_weekly_events.py`

Collect new GDELT event rows for the weekly window.

Responsibilities:
- Pull only events newer than the last successful weekly run.
- Normalize the incoming event schema to match `data/gdelt_scraped.csv` expectations.
- Write a weekly staging file such as `data/weekly_runs/<run_id>/weekly_events_raw.csv`.
- Deduplicate incoming event rows before scraping.
- Record source metadata for the run manifest.

This script should not scrape article text. It should only gather the structured event rows that define the weekly update scope.

### `scrape_weekly_urls.py`

Scrape article content for the new weekly event rows.

Responsibilities:
- Extract candidate `SourceURL` values from the weekly event staging file.
- Skip URLs already covered by the existing lookup when possible.
- Scrape only new or changed URLs.
- Write a weekly scrape result file such as `data/weekly_runs/<run_id>/weekly_scraped.csv`.
- Capture scrape status, retry counts, and basic QA counts.

This is the weekly replacement for the yearly scraping flow. It should be optimized for short windows and quick reruns, not for full historical reconstruction.

### `append_master_dataset.py`

Merge the weekly scrape results into the canonical master dataset.

Responsibilities:
- Validate that the weekly scrape output matches the master schema.
- Append new rows into `data/gdelt_scraped.csv`.
- Prevent duplicate event-row insertion.
- Write a run-specific audit file such as `data/weekly_runs/<run_id>/append_audit.csv`.
- Preserve the existing historical rows unchanged.

This script replaces the role that yearly consolidation played for the bootstrap workflow, but for incremental weekly appends.

### `update_lookup_incremental.py`

Incrementally update URL-level lookup state from the refreshed master dataset.

Responsibilities:
- Reuse the canonicalization and stable `url_id` rules from `preprocessing/build_url_index.py`.
- Add new canonical URLs to `data/preprocessing/url_lookup.csv`.
- Preserve existing `url_id` assignments and existing lookup state.
- Identify which `url_id` values are new or changed in the current weekly run.
- Emit a small worklist such as `data/weekly_runs/<run_id>/changed_url_ids.csv`.

This should be a weekly-oriented wrapper around the existing URL-index logic, not a brand-new URL identity system.

### `apply_frozen_relevance.py`

Apply existing token relevance scores to weekly additions.

Responsibilities:
- Read frozen token weights from `data/preprocessing/text_relevance_tokens.csv`.
- Score only new or changed lookup rows.
- Update the `doc_relevance_*` fields in `data/preprocessing/url_lookup.csv`.
- Write a QA snapshot such as `data/weekly_runs/<run_id>/weekly_score_summary.csv`.

This script should not rebuild token weights weekly.

### `apply_static_filters.py`

Apply the current filter policy to weekly additions.

Responsibilities:
- Read static rules from `preprocessing/filter_rule_config.json`.
- Evaluate only new or changed rows against the current thresholds and anchor logic.
- Update `data/preprocessing/url_filter_eval.csv`.
- Refresh `data/preprocessing/url_filter_summary_counts.csv`.
- Optionally write weekly-only QA samples under `data/weekly_runs/<run_id>/filter_samples/`.

This script should preserve historical decisions unless a row is explicitly reprocessed.

### `refresh_analysis_exports.py`

Refresh downstream data products after weekly ingestion.

Responsibilities:
- Rebuild `data/analysis_ready/analysis_events.parquet`.
- Rebuild `data/analysis_ready/analysis_url_content.parquet`.
- Optionally rerun `eda/run_eda.py`.
- Optionally rerun `python -m analysis.main --all`.
- Capture output timestamps and key file paths in the run manifest.

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

## Implementation Order

Recommended order for building this:

1. Create `common.py` and `run_weekly_update.py`.
2. Implement `fetch_weekly_events.py`.
3. Implement `scrape_weekly_urls.py`.
4. Implement `append_master_dataset.py`.
5. Implement `update_lookup_incremental.py`.
6. Implement `apply_frozen_relevance.py`.
7. Implement `apply_static_filters.py`.
8. Implement `refresh_analysis_exports.py`.

## Open Design Choices

- What should define a weekly event-row primary key for append safety?
- Should changed rows be detected only by new `SourceURL`, or also by changed `Title` and `Text`?
- Should weekly runs rerun full EDA and full analysis by default, or make them optional flags?
- Should weekly QA samples live only under `data/weekly_runs/`, or also refresh the canonical preprocessing QA outputs every run?
