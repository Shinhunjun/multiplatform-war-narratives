# Venezuela-US Reddit Discourse Analysis

Reddit data collection and analysis pipeline for studying online discourse about Venezuela-US relations during the Maduro era (2013-2026).

## Project Overview

This project collects and analyzes Reddit discussions related to Venezuela-US relations, focusing on:
- Political discourse across different ideological communities
- Crisis period narratives and sentiment shifts
- Cross-community engagement patterns

## Data Statistics

| Metric | Raw Data | After Preprocessing |
|--------|----------|---------------------|
| **Time Period** | 2013-01-01 ~ 2026-01-29 | - |
| **Total Submissions** | 101,960 | 86,809 |
| **Total Comments** | 431,981 | 339,626 |
| **Total Data Points** | 533,941 | 426,435 |
| **Unique Submission Authors** | 26,363 | 23,497 |
| **Unique Comment Authors** | 129,740 | 119,021 |
| **Subreddits** | 11 | 11 |

### Regional Distribution

| Region | Submissions | Comments |
|--------|-------------|----------|
| Venezuela (r/vzla, r/venezuela) | 64.9% | 54.3% |
| US/English | 35.1% | 45.7% |

## Target Subreddits (11)

| Category | Subreddits |
|----------|------------|
| Venezuela-focused | r/venezuela, r/vzla |
| US Politics | r/politics, r/news, r/worldnews |
| Ideological | r/Conservative, r/neoliberal, r/socialism, r/Libertarian |
| Regional | r/LatinAmerica, r/geopolitics |

## Key Crisis Periods (Flashpoints)

| Date | Event |
|------|-------|
| 2013-04 | Maduro Inauguration |
| 2014-02 | Venezuelan Protests |
| 2014-11 | Oil Price Crash |
| 2017-08 | Trump Administration Sanctions |
| 2018-05 | Disputed Election |
| 2019-01 | Guaido Recognition Crisis |
| 2019-04 | April Uprising |
| 2021-01 | Biden Policy Shift |
| 2024-07 | 2024 Election Crisis |
| 2024-09 | Gonzalez Exile |
| 2026-01 | Maduro Captured by US Forces |

## Data Access

**Full dataset available on Google Drive**: [Download Here](https://drive.google.com/drive/folders/1MV2-ktL-OsiT4cDmoGWwlmt9l-OY_j-U?usp=sharing)

## Project Structure

```
capstone/
├── README.md
├── .gitignore
└── venezuela-us-reddit-discourse/
    ├── EDA/
    │   ├── EDA_Report.md              # Analysis report
    │   ├── 01_timeline.png            # Timeline visualization
    │   ├── 02_yearly_distribution.png
    │   ├── 03_subreddit_distribution.png
    │   ├── 04_region_comparison.png
    │   ├── 05_engagement_metrics.png
    │   ├── 06_top_authors.png
    │   └── run_eda_clean.py           # EDA script
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── config.py                  # Preprocessing config
    │   ├── filters.py                 # Bot/deleted filters
    │   ├── text_cleaner.py            # Text cleaning functions
    │   └── preprocessor.py            # Main preprocessing class
    └── data-collection/
        ├── main.py                    # CLI entry point
        ├── pyproject.toml             # Dependencies
        ├── scripts/
        │   ├── config.py              # Collection config
        │   ├── collectors.py          # Data collection
        │   ├── processors.py          # Data processing
        │   ├── analyzers.py           # Analysis functions
        │   └── visualizers.py         # Visualization
        └── data/
            ├── submissions/           # Raw submission JSON
            ├── comments/              # Raw comment JSON
            └── preprocessed/          # Cleaned Parquet files
```

## Usage

### Setup
```bash
cd venezuela-us-reddit-discourse/data-collection
uv sync
```

### Data Collection
```bash
# Collect historical posts
uv run main.py historical --start 2013-01-01 --end 2026-01-29

# Collect comments for all posts
uv run main.py comments

# Collect crisis period data
uv run main.py crisis --all
```

### Preprocessing
```bash
cd venezuela-us-reddit-discourse
python -m preprocessing.preprocessor
```

**Preprocessing removes:**
- `[deleted]` and `[removed]` content
- Bot accounts (AutoModerator, autotldr, etc.)
- Comments with < 5 words
- URLs, markdown formatting, edit markers

**Output:** `data/preprocessed/submissions_clean.parquet`, `comments_clean.parquet`

### EDA
```bash
python EDA/run_eda_clean.py
```

## Technical Details

- **Data Source**: [Arctic Shift API](https://arctic-shift.photon-reddit.com/) (Reddit historical archive)
- **Collection Tool**: [BAScraper](https://github.com/Jython1415/BAScraper) (async Python wrapper)
- **Rate Limiting**: 1.0s between requests
- **Output Format**: Parquet (preprocessed), JSON (raw)

## License

For academic research purposes only.

## Author

Hunjun Shin
