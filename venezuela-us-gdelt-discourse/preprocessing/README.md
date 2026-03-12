# Preprocessing Pipeline

This folder contains the full preprocessing workflow for `data/gdelt_scraped.csv`.

The orchestrator script is:

`run_preprocessing_pipeline.py`

## What The Pipeline Produces

- `url_lookup*.csv`: one row per canonical URL (with scraped text, tokens, and relevance score).
- `text_relevance_tokens*.csv`: ranked token relevance table.
- `url_filter_eval*.csv`: row-level filtering decisions and reasons.
- `url_filter_summary_counts*.csv`: aggregated decision counts by stage.
- `filter_samples*/`: QA samples from each filtering stage.
- `filter_stage_score_histograms*.png`: score distributions by filter stage decisions.

## Run

From this folder:

```powershell
python .\run_preprocessing_pipeline.py ..\data\gdelt_scraped.csv --force-retokenize --require-success-status
```



## Notes

- The pipeline is designed to be rerun weekly as `gdelt_scraped.csv` grows.
- URL indexing and lookup updates are incremental.
- Tokenization can be forced with `--force-retokenize`.
- `--require-success-status` limits token-relevance training to successful scrapes.

## Stage Histogram Example

The default output figure is:

`filter_stage_score_histograms.png`

![Filter Stage Score Histograms](filter_stage_score_histograms.png)
