# EDA

This folder contains the exploratory data analysis stage for the GDELT corpus after analysis-ready parquet export.

## Purpose

- Inspect event volume, categories, tone, and conflict metrics over time.
- Inspect scrape-quality distributions and URL uniqueness.
- Inspect title and article-text token patterns.
- Produce a reusable markdown report plus figure files for review.

## Main Entry Point

Run from the project root:

```bash
python eda/run_eda.py
```

## Inputs

`run_eda.py` reads:

- `data/analysis_ready/analysis_events.parquet`
- `data/analysis_ready/analysis_url_content.parquet`

It also imports token-processing helpers from `preprocessing/build_text_relevance_tokens.py` for word normalization and token parsing.

## Outputs

The script writes into this folder:

- `01_gdelt_timeline.png`
- `02_gdelt_yearly_stats.png`
- `03_gdelt_categories.png`
- `04_gdelt_intensity.png`
- `05_gdelt_tone_trend.png`
- `06_scraped_status.png`
- `07_scraped_url_uniqueness.png`
- `08_title_wordcloud.png`
- `09_text_wordcloud.png`
- `GDELT_EDA_Report.md`

## What The Script Assumes

- `analysis_events.parquet` contains event-level GDELT columns such as `Date`, `Actor*`, `QuadClass`, `GoldsteinScale`, `AvgTone`, `SourceURL`, `Scrape_Status`, and `url_id`.
- `analysis_url_content.parquet` contains URL-level fields such as `SourceURL`, `Title`, `Text`, and `Tokens`.
- The two parquet files can be joined on `url_id`.

## Typical Use

Run EDA after:

1. `data/gdelt_scraped.csv` exists.
2. preprocessing has rebuilt `data/analysis_ready/analysis_events.parquet`.
3. preprocessing has rebuilt `data/analysis_ready/analysis_url_content.parquet`.

This stage is intended to be cheap to rerun whenever the canonical dataset changes.

## Notes

- Tests for this stage live under `tests/eda`.
- The current script adjusts `sys.path` to reuse preprocessing token helpers. That works, but it is a code-smell worth revisiting later if you refactor toward cleaner shared utilities.
