# Venezuela-US Reddit Discourse Analysis

Reddit data collection and analysis pipeline for studying online discourse about Venezuela-US relations during the Maduro era (2013-2025).

## Project Overview

This project collects and analyzes Reddit discussions related to Venezuela-US relations, focusing on:
- Political discourse across different ideological communities
- Crisis period narratives and sentiment shifts
- Cross-community engagement patterns

## Data Collection

### Time Period
- **Start**: January 1, 2013 (Maduro era begins)
- **End**: January 27, 2025

### Target Subreddits (11)

| Category | Subreddits |
|----------|------------|
| Venezuela-focused | r/venezuela, r/vzla |
| US Politics | r/politics, r/news, r/worldnews |
| Ideological | r/Conservative, r/neoliberal, r/socialism, r/Libertarian |
| Regional | r/LatinAmerica, r/geopolitics |

### Search Queries
```
Venezuela, Maduro, Venezuela US, Venezuela sanctions, Guaido,
Venezuelan crisis, Venezuela oil, Caracas, Venezuela election,
Venezuela humanitarian
```

### Key Crisis Periods (Flashpoints)
- **2013**: Maduro Inauguration
- **2014**: Venezuelan Protests, Oil Price Crash
- **2017**: Trump Administration Sanctions
- **2018**: Disputed Election
- **2019**: Guaido Recognition Crisis, April Uprising
- **2021**: Biden Policy Shift
- **2024**: Election Crisis, González Exile

## Data Access

**Full dataset available on Google Drive**: [Download Here](https://drive.google.com/drive/folders/1MV2-ktL-OsiT4cDmoGWwlmt9l-OY_j-U?usp=sharing)

```
venezuela/
└── reddit/
    ├── post/             # Reddit posts (.json)
    └── comment/          # Comment threads (.json)
```

## Data Structure

### Submission Example
```json
{
  "18gnc5": {
    "author": "username",
    "created_utc": 1360783918,
    "id": "18gnc5",
    "num_comments": 20,
    "score": 4,
    "selftext": "Post content here...",
    "subreddit": "Libertarian",
    "title": "Post title here",
    "url": "https://reddit.com/r/...",
    "_matched_queries": ["Venezuela", "Venezuela US"],
    "_window_start": "2013-02-01",
    "_window_end": "2013-02-28",
    "_granularity": "monthly",
    "_collection_type": "historical"
  }
}
```

### Comment Example
```json
{
  "dh8pguz": {
    "author": "username",
    "body": "Comment text here...",
    "controversiality": 0,
    "created_utc": 1494161422,
    "id": "dh8pguz",
    "link_id": "t3_69r06f",
    "parent_id": "t3_69r06f",
    "score": 46,
    "subreddit": "Libertarian",
    "_submission_id": "69r06f",
    "_submission_title": "Original post title",
    "_depth": 0,
    "_is_top_level": true,
    "_parent_comment_id": null
  }
}
```

## Usage

### Setup
```bash
cd venezuela-us-reddit-discourse/data-collection
uv sync
```

### Data Collection Commands
```bash
# Collect historical posts (2013-2025)
uv run main.py historical

# Collect comments for all posts
uv run main.py comments

# Collect crisis period data
uv run main.py crisis --all

# Export to Parquet format
uv run main.py export

# Run analysis
uv run main.py analyze
```

### List Available Crisis Periods
```bash
uv run main.py list-flashpoints
```

## Data Statistics

| Metric | Count |
|--------|-------|
| Subreddits | 11 |
| Time Period | 12 years |
| Total Posts | ~90,000 |
| Total Comments | TBD |
| Crisis Periods | 10 |

## Technical Details

- **Data Source**: [Arctic Shift API](https://arctic-shift.photon-reddit.com/) (Reddit historical archive)
- **Collection Tool**: [BAScraper](https://github.com/Jython1415/BAScraper) (async Python wrapper)
- **Rate Limiting**: 2.0s between requests
- **Resume Support**: Collections can be interrupted and resumed

## Project Structure

```
capstone/
├── README.md                          # This file
├── .gitignore
└── venezuela-us-reddit-discourse/
    └── data-collection/
        ├── main.py                    # CLI entry point
        ├── pyproject.toml             # Dependencies
        └── scripts/
            ├── config.py              # Configuration
            ├── collectors.py          # Data collection
            ├── processors.py          # Data processing
            ├── analyzers.py           # Analysis functions
            └── visualizers.py         # Visualization
```

## License

For academic research purposes only.

## Author

Hunjun Shin
