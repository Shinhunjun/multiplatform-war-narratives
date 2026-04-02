# TODO

## Pressing Tasks

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
