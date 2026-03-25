# Tests

This folder contains the automated test suite for the GDELT project.

## Purpose

- Verify collection, preprocessing, EDA, and analysis behavior.
- Protect path conventions and file-format expectations.
- Catch regressions when the pipeline structure or defaults change.

## Layout

- `analysis/`: tests for configuration, data loading, clustering/model helpers, visualization, and the main analysis runner.
- `data_collection/`: tests for yearly scraping, rescue, and consolidation logic.
- `eda/`: tests for analysis-ready loading and report/plot generation behavior.
- `preprocessing/`: tests for URL indexing, tokenization, relevance scoring, filtering, plotting, and analysis-ready export.
- [conftest.py](/home/rich/Desktop/capstone/multiplatform-war-narratives/venezuela-us-gdelt-discourse/tests/conftest.py): shared test bootstrap. It sets a non-interactive matplotlib backend and adjusts import paths for the current project layout.

## Running Tests

Run the full suite from the project root:

```bash
.venv/bin/pytest
```

Run a single module:

```bash
.venv/bin/pytest tests/preprocessing/test_relevance_and_filtering.py
```

Run one test file with verbose output:

```bash
.venv/bin/pytest -v tests/analysis/test_config_data_loader.py
```

## Current Coverage Areas

- Path and config resolution for `data/`, `data/preprocessing/`, and `data/analysis_ready/`
- Preprocessing artifact generation and update behavior
- Analysis-ready parquet export behavior
- EDA loading, plotting, and markdown reporting
- Analysis pipeline helpers, loaders, visualization, and orchestration
- Data-collection edge cases for scraping and consolidation logic

## Test Style

- Most tests use `tmp_path` and write small synthetic fixtures instead of relying on large real datasets.
- Many tests call scripts through their CLI argument parsing to verify default-path behavior as well as pure helper functions.
- The suite is designed to stay fast enough for regular reruns during refactors.

## Practical Note

Because the codebase still includes some script-style modules, [conftest.py](/home/rich/Desktop/capstone/multiplatform-war-narratives/venezuela-us-gdelt-discourse/tests/conftest.py) adds the project directories to `sys.path` during testing. That keeps the current layout testable, but it is also a sign that future packaging cleanup could simplify imports.
