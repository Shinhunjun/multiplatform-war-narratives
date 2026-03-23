# Venezuela-US GDELT Discourse

This project contains the GDELT-specific pipeline for collecting, preprocessing, exploring, and analyzing discourse about Venezuela-US relations. It is organized as a staged workflow so each part of the research process can be run independently or as part of a larger end-to-end pipeline.

## Project Layout

- `data_collection/`: yearly scraping, rescue, and consolidation scripts for raw GDELT article content.
- `preprocessing/`: preprocessing source code, configs, dictionaries, and supporting documentation.
- `eda/`: exploratory data analysis over the analysis-ready parquet outputs.
- `analysis/`: sentiment analysis, topic modeling, clustering, summarization, and visualization workflows.
- `tests/`: pytest coverage for collection, preprocessing, EDA, and analysis modules.

## Pipeline Order

The typical workflow runs in four stages:

1. Collect yearly scrape outputs and consolidate them into `data/gdelt_scraped.csv`.
2. Run preprocessing to build URL-level features, filtering decisions, and analysis-ready parquet datasets.
3. Run EDA to inspect event patterns, scrape quality, and text content characteristics.
4. Run downstream analysis for sentiment, topics, clustering, and visualizations.

## Quickstart

Install dependencies from the GDELT project root:

```bash
python -m pip install -r requirements-dev.txt
```

Run the main stages:

```bash
# 1) Build or refresh scraped yearly data
python data_collection/consolidate_yearly.py --base-dir data --years 2020-2026

# 2) Generate preprocessing artifacts and analysis-ready parquet files
python preprocessing/run_preprocessing_pipeline.py data/gdelt_scraped.csv --force-retokenize --require-success-status
python preprocessing/build_analysis_ready_datasets.py

# 3) Produce exploratory analysis outputs
python eda/run_eda.py

# 4) Run the analysis pipeline
python -m analysis.main --all
```

## Key Inputs And Outputs

- Primary input: `data/gdelt_scraped.csv`
- Analysis-ready outputs: `data/analysis_ready/analysis_events.parquet` and `data/analysis_ready/analysis_url_content.parquet`
- Preprocessing artifacts: files such as `data/preprocessing/url_lookup*.csv`, `data/preprocessing/text_relevance_tokens*.csv`, and `data/preprocessing/url_filter_eval*.csv`
- EDA outputs: plots and `eda/GDELT_EDA_Report.md`
- Analysis outputs: saved artifacts under `analysis/outputs/`

## Running Tests

From this folder:

```bash
pytest
```

## Notes

- Stage-specific details live in [data_collection/README.md](/home/rich/Desktop/capstone/multiplatform-war-narratives/venezuela-us-gdelt-discourse/data_collection/README.md) and [preprocessing/README.md](/home/rich/Desktop/capstone/multiplatform-war-narratives/venezuela-us-gdelt-discourse/preprocessing/README.md).
- The codebase currently assumes a local `data/` directory for scrape outputs, generated preprocessing artifacts, and analysis-ready exports.
- `analysis/` is the cleanest module-style entry point and supports `python -m analysis.main`.
