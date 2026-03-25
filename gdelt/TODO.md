# TODO

## Pressing Tasks

### Create a weekly update pipeline

Goal: refresh the project weekly without redoing completed preprocessing work for unchanged URLs.

1. Download new GDELT event rows since the last successful update and stage the new raw inputs under `data/raw/`.
2. Scrape and rescue only the new period, then refresh `data/gdelt_scraped.csv`, `data/problematic_rows.csv`, and `data/consolidation_audit.csv`.
   Current caveat: `data_collection/consolidate_yearly.py` rebuilds the consolidated output from selected yearly files rather than incrementally patching `data/gdelt_scraped.csv` in place.
3. Rebuild the URL index with `preprocessing/build_url_index.py` so `data/preprocessing/url_lookup.csv` adds new canonical URLs while preserving existing `url_id` values.
4. Tokenize only newly added or changed lookup rows with `preprocessing/tokenize_url_lookup.py` so the `Tokens` column in `data/preprocessing/url_lookup.csv` stays current.
5. Apply the existing token relevance table in `data/preprocessing/text_relevance_tokens.csv` to new or changed lookup rows so the `doc_relevance_*` columns in `data/preprocessing/url_lookup.csv` stay current without recalculating token weights each week.
6. Apply the existing static filter rules in `preprocessing/filter_rule_config.json` to new or changed rows, then refresh the downstream exports:
   - `data/preprocessing/url_filter_eval.csv`
   - `data/preprocessing/url_filter_summary_counts.csv`
   - `data/preprocessing/filter_samples/`
   - `data/preprocessing/filter_stage_score_histograms.png`
   - `data/analysis_ready/analysis_events.parquet`
   - `data/analysis_ready/analysis_url_content.parquet`
7. Re-run downstream outputs so `eda/GDELT_EDA_Report.md`, the EDA figures, and `analysis/outputs/` all reflect the refreshed dataset while preserving historical scoring and filtering decisions.

## Future Tasks

- Refactor `data_collection/` into a pipeline.
- Add version pins to the requirements files.
- Add a separate recalibration workflow for periodically regenerating `data/preprocessing/text_relevance_tokens.csv` and revisiting `preprocessing/filter_rule_config.json`.
- Add drift monitoring for weekly runs so score distributions, keep/drop rates, anchor-hit rates, and QA samples can signal when recalibration is due.

## Questions

1. What is `sys.path` and why is it unprofessional?
2. What does adding CI for `pytest` look like?
3. How would you improve `run_eda.py`?
4. Since `build_text_relevance_tokens.py` is used for `Title` as well in `run_eda.py`, should it be renamed?
