# Preprocessing Artifacts

This folder stores generated preprocessing outputs. It is the runtime-data companion to the source code in `preprocessing/`.

## Purpose

- Keep generated CSVs, QA outputs, and other preprocessing artifacts under `data/`.
- Separate code and static configs from rebuildable runtime products.
- Provide canonical inputs for analysis-ready export, EDA, and downstream analysis.

## Canonical Files

- `url_lookup.csv`
  - One row per canonical URL.
  - Produced and updated by `build_url_index.py`.
  - Enriched by tokenization, scoring, and redirect-duplicate utilities.
- `text_relevance_tokens.csv`
  - Token-level relevance weights used for document scoring.
  - Produced by `build_text_relevance_tokens.py`.
- `url_filter_eval.csv`
  - URL-level filtering decisions and diagnostic flags.
  - Produced by `build_duplicate_filter_eval.py` and `evaluate_filter_strategy.py`.
- `url_filter_summary_counts.csv`
  - Aggregate counts summarizing filter decisions.
  - Produced by the filtering scripts.

## Common Optional Outputs

These may exist after a full preprocessing run or utility pass:

- `filter_samples/`
  - Stratified QA samples from `evaluate_filter_strategy.py`.
- `filter_stage_score_histograms.png`
  - Filter-stage histogram plot from `plot_filter_stage_score_histograms.py`.
- `redirect_duplicate_clusters.csv`
  - Redirect/fallback duplicate review output from `flag_redirect_duplicates.py`.
- `relevant_terms.csv`
  - Optional curated or derived term list used by some downstream analysis utilities.

## How These Files Are Used

- `build_analysis_ready_datasets.py` reads `url_lookup.csv` and `url_filter_eval.csv`.
- `eda/run_eda.py` consumes the downstream parquet exports built from these artifacts.
- `analysis/config.py` points downstream analysis modules here for preprocessing-side CSV inputs.

## Regeneration

From the project root, the standard rebuild is:

```bash
python preprocessing/run_preprocessing_pipeline.py data/gdelt_scraped.csv --force-retokenize --require-success-status
python preprocessing/build_analysis_ready_datasets.py
```

The first command refreshes the preprocessing artifacts in this folder. The second command refreshes `data/analysis_ready/`.

## Safe To Rebuild

These files are generated artifacts, not hand-edited source files. In normal workflow they should be treated as rebuildable outputs derived from:

- `data/gdelt_scraped.csv`
- preprocessing source code under `preprocessing/`
- static configs such as `filter_rule_config.json` and `anchor_token_sets.json`

If you are cleaning the workspace, be careful not to remove artifacts you still want to compare historically before rerunning the pipeline.
