# Analysis

This folder contains the downstream discourse-analysis pipeline that runs after preprocessing and analysis-ready export.

## Purpose

- Load the analysis-ready GDELT datasets.
- Run sentiment analysis, topic modeling, clustering, visualization, and optional cluster summarization.
- Save outputs under `analysis/outputs/`.

## Main Entry Point

Run from the project root:

```bash
python -m analysis.main --all
```

Common modes:

```bash
python -m analysis.main --sentiment
python -m analysis.main --topics
python -m analysis.main --clusters
python -m analysis.main --visualize
python -m analysis.main --summarize --llm anthropic
```

Useful optional flags:

- `--sample N`: run on a smaller subset for quick checks.
- `--min-relevance X`: require `doc_relevance_score >= X`.
- `--exclude-suspect-redirect`: drop rows flagged in preprocessing as suspect redirect duplicates.
- `--include-non-success`: include non-success scrape rows if they still have usable text.

## Inputs

Primary runtime inputs are resolved by `config.py`:

- `data/analysis_ready/analysis_events.parquet`
- `data/analysis_ready/analysis_url_content.parquet`
- `data/preprocessing/url_lookup.csv`
- `data/preprocessing/text_relevance_tokens.csv`
- Optional: `data/preprocessing/relevant_terms.csv`
- Optional: `data/preprocessing/redirect_duplicate_clusters.csv`

By default the pipeline prefers the analysis-ready parquet exports over legacy direct loading from `data/gdelt_scraped.csv`.

## Folder Layout

- `config.py`: central path and runtime configuration.
- `data_loader.py`: loads analysis-ready data and preprocessing artifacts.
- `main.py`: top-level pipeline runner.
- `sentiment/`: sentiment model wrappers and aggregation helpers.
- `topic/`: BERTopic modeling helpers.
- `clustering/`: embedding, clustering, temporal visualization, and summarization helpers.
- `visualize_temporal.py`: additional temporal plotting utilities.
- `outputs/`: saved artifacts from analysis runs.

## Outputs

The pipeline writes into `analysis/outputs/`:

- `sentiment/`
  - `sentiment_full.parquet`
  - `sentiment_by_month.csv`
  - `sentiment_by_source_domain*.csv`
  - `sentiment_by_event_category.csv`
- `topics/`
  - `bertopic_model/`
  - `topic_assignments.parquet`
  - `topic_info.csv`
  - `topics_over_time.csv`
  - `topics_by_source_domain.csv`
  - `document_embeddings.npy`
- `clusters/`
  - `cluster_assignments.parquet`
  - `cluster_summary.csv`
  - `temporal_clusters.csv`
  - `embeddings.npy`
  - `embeddings_2d.npy`
  - `cluster_keywords.csv`
  - optional `cluster_summaries.csv`
- `visualizations/`
  - `umap_clusters.png`
  - `umap_source_domains.png`
  - `umap_animation.gif`
  - `cluster_river.png`
  - `cluster_heatmap.png`
  - `interactive_clusters.html`

## Notes

- The analysis stack is dependency-heavy because it includes `torch`, `transformers`, BERTopic, and clustering libraries.
- Cluster summarization may require API credentials depending on whether you use `--llm anthropic` or `--llm openai`.
- If you want to change default paths or filtering behavior, start in `config.py`.
