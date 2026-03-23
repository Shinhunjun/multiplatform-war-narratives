# Data Folder

This folder is the working area for the GDELT pipeline. It holds the raw collection artifacts, the consolidated master CSV, preprocessing artifacts, QA outputs from consolidation, and the analysis-ready parquet exports used by preprocessing, EDA, and analysis.

## Intended Structure

```text
data/
├── raw/
│   ├── bq-results-*.csv
│   ├── ven_usa_<YEAR>.csv
│   ├── ven_usa_<YEAR>_rescued.csv
│   └── duration_log_<YEAR>.txt
├── gdelt_scraped.csv
├── problematic_rows.csv
├── consolidation_audit.csv
├── preprocessing/
│   ├── url_lookup.csv
│   ├── text_relevance_tokens.csv
│   ├── url_filter_eval.csv
│   ├── url_filter_summary_counts.csv
│   ├── filter_samples/
│   ├── filter_stage_score_histograms.png
│   └── redirect_duplicate_clusters.csv
└── analysis_ready/
    ├── analysis_events.parquet
    └── analysis_url_content.parquet
```

## What Belongs Where

`data/raw/`
- Upstream source exports, such as the BigQuery extract used by collection.
- Yearly scrape outputs from `scrape_by_year.py`.
- Yearly rescue outputs from `rescue_by_year.py`.
- Collection-stage logs, if you want to keep them alongside the source files.

`data/`
- The canonical consolidated dataset: `gdelt_scraped.csv`.
- Consolidation QA outputs: `problematic_rows.csv` and `consolidation_audit.csv`.

`data/preprocessing/`
- Generated preprocessing artifacts such as `url_lookup.csv`, `text_relevance_tokens.csv`, and `url_filter_eval.csv`.
- QA outputs such as `url_filter_summary_counts.csv`, `filter_samples/`, and `filter_stage_score_histograms.png`.
- Additional preprocessing diagnostics such as `redirect_duplicate_clusters.csv`.

`data/analysis_ready/`
- `analysis_events.parquet`: one row per scraped event row, enriched for downstream analysis.
- `analysis_url_content.parquet`: one row per URL, with representative title, text, tokens, and metadata.

## Inputs Versus Outputs

Inputs:
- `data/raw/bq-results-*.csv`
- `data/raw/ven_usa_<YEAR>.csv`
- `data/raw/ven_usa_<YEAR>_rescued.csv`

Generated outputs:
- `data/gdelt_scraped.csv`
- `data/problematic_rows.csv`
- `data/consolidation_audit.csv`
- `data/preprocessing/url_lookup.csv`
- `data/preprocessing/text_relevance_tokens.csv`
- `data/preprocessing/url_filter_eval.csv`
- `data/preprocessing/url_filter_summary_counts.csv`
- `data/preprocessing/filter_samples/`
- `data/preprocessing/filter_stage_score_histograms.png`
- `data/analysis_ready/analysis_events.parquet`
- `data/analysis_ready/analysis_url_content.parquet`

## How The Code Uses This

- `data_collection/scrape_by_year.py` expects the raw source export to be available in the current working directory when run from `data/`.
- `data_collection/rescue_by_year.py` expects the yearly scrape files in the current working directory.
- `data_collection/consolidate_yearly.py` reads the yearly files and writes the consolidated CSV plus audit outputs.
- The preprocessing scripts read from and write to `data/preprocessing/` by default while keeping source code and configs under `preprocessing/`.
- `preprocessing/build_analysis_ready_datasets.py` writes the parquet exports under `data/analysis_ready/`.
- `eda/run_eda.py` and `analysis/` consume the analysis-ready parquet files.

## Practical Note

`data/raw/` is the cleanest long-term convention for keeping source and stage-1 collection files together, but the collection scripts do not yet consistently assume that layout. For now, treat `data/raw/` as the preferred organization target and adjust the working directory or script paths when running the collection stage.
