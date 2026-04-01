"""
Configuration for Venezuela-US Reddit Data Collection Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PipelineConfig:
    """Central configuration for the data collection pipeline."""

    base_dir: Path = field(default_factory=lambda: Path("./data"))

    @property
    def submissions_dir(self) -> Path:
        return self.base_dir / "submissions"

    @property
    def comments_dir(self) -> Path:
        return self.base_dir / "comments"

    @property
    def analysis_dir(self) -> Path:
        return self.base_dir / "analysis"

    @property
    def exports_dir(self) -> Path:
        return self.base_dir / "exports"

    # Arctic Shift settings
    sleep_sec: float = 1.0      # 요청 간 대기 시간 (rate limit 방지)
    backoff_sec: float = 10.0   # 실패 시 백오프
    max_retries: int = 5        # 재시도 횟수
    timeout: int = 60           # 타임아웃
    task_num: int = 1           # 동시 요청 1개만!
    default_limit: int = 100

    def ensure_directories(self) -> "PipelineConfig":
        """Create all required directories."""
        for d in [
            self.base_dir,
            self.submissions_dir,
            self.comments_dir,
            self.analysis_dir,
            self.exports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"Directories created at: {self.base_dir}")
        return self


# =============================================================================
# DEFAULT DATE RANGES
# =============================================================================

# Historical collection default range (Maduro era: 2013-present)
HISTORICAL_DEFAULT_START: str = "2025-01-27"
HISTORICAL_DEFAULT_END: str = "2026-01-29"

# Comments collection default date range
COMMENTS_DEFAULT_START: str = "2005-01-01"
COMMENTS_DEFAULT_END: str = "2025-12-31"


# =============================================================================
# FLASHPOINTS / CRISIS PERIODS
# =============================================================================

FLASHPOINTS: Dict[str, Dict] = {
    "maduro_inauguration_2013": {
        "name": "Maduro Inauguration 2013",
        "start": "2013-04-14T00:00:00",
        "end": "2013-04-30T23:59:59",
        "priority": "high",
        "color": "#E63946",
        "description": "Maduro presidential inauguration and initial controversies",
    },
    "protests_2014": {
        "name": "2014 Venezuelan Protests",
        "start": "2014-02-01T00:00:00",
        "end": "2014-05-31T23:59:59",
        "priority": "critical",
        "color": "#F4A261",
        "description": "Large-scale anti-government protests",
    },
    "oil_price_crash_2014": {
        "name": "Oil Price Crash Impact",
        "start": "2014-11-01T00:00:00",
        "end": "2015-02-28T23:59:59",
        "priority": "high",
        "color": "#2A9D8F",
        "description": "Economic crisis deepening due to oil price crash",
    },
    "trump_sanctions_2017": {
        "name": "Trump Administration Sanctions",
        "start": "2017-08-01T00:00:00",
        "end": "2017-09-30T23:59:59",
        "priority": "critical",
        "color": "#E76F51",
        "description": "Trump administration strengthens sanctions against Venezuela",
    },
    "maduro_reelection_2018": {
        "name": "2018 Disputed Election",
        "start": "2018-05-15T00:00:00",
        "end": "2018-05-31T23:59:59",
        "priority": "high",
        "color": "#F4A261",
        "description": "Disputed Maduro reelection amid international criticism",
    },
    "guaido_recognition_2019": {
        "name": "Guaido Recognition Crisis",
        "start": "2019-01-20T00:00:00",
        "end": "2019-02-28T23:59:59",
        "priority": "critical",
        "color": "#264653",
        "description": "US recognition of Guaido as interim president",
    },
    "failed_uprising_2019": {
        "name": "April 2019 Uprising Attempt",
        "start": "2019-04-28T00:00:00",
        "end": "2019-05-05T23:59:59",
        "priority": "high",
        "color": "#E9C46A",
        "description": "Guaido's failed military uprising attempt",
    },
    "biden_policy_2021": {
        "name": "Biden Administration Policy Shift",
        "start": "2021-01-20T00:00:00",
        "end": "2021-03-31T23:59:59",
        "priority": "medium",
        "color": "#457B9D",
        "description": "Biden administration Venezuela policy changes",
    },
    "election_2024": {
        "name": "2024 Venezuelan Election Crisis",
        "start": "2024-07-20T00:00:00",
        "end": "2024-08-15T23:59:59",
        "priority": "critical",
        "color": "#D62828",
        "description": "2024 presidential election and fraud allegations",
    },
    "gonzalez_exile_2024": {
        "name": "Gonzalez Urrutia Exile",
        "start": "2024-09-01T00:00:00",
        "end": "2024-09-15T23:59:59",
        "priority": "high",
        "color": "#F77F00",
        "description": "Opposition presidential candidate exile to Spain",
    },
}

# Crisis periods for analysis (extracted from FLASHPOINTS)
CRISIS_PERIODS: Dict[str, tuple] = {
    fp["name"]: (fp["start"][:10], fp["end"][:10], fp["color"])
    for fp in FLASHPOINTS.values()
}


# =============================================================================
# TARGET SUBREDDITS
# =============================================================================

TARGET_SUBREDDITS: Dict[str, List[str]] = {
    "us_perspective": ["politics", "news", "worldnews", "Conservative", "neoliberal"],
    "venezuela_perspective": ["venezuela", "vzla"],
    "regional": ["LatinAmerica", "southamerica"],
    "ideological": ["socialism", "Libertarian", "geopolitics"],
}

ALL_SUBREDDITS: List[str] = [
    "venezuela",
    "vzla",
    "politics",
    "news",
    "worldnews",
    "geopolitics",
    "LatinAmerica",
    "Conservative",
    "neoliberal",
    "socialism",
    "Libertarian",
]

CRISIS_SUBREDDITS: List[str] = [
    "venezuela",
    "vzla",
    "worldnews",
    "politics",
    "news",
    "geopolitics",
    "LatinAmerica",
]

# Subreddit colors for visualizations
SUBREDDIT_COLORS: Dict[str, str] = {
    "venezuela": "#FCDD09",         # Venezuelan Yellow
    "vzla": "#CF142B",              # Venezuelan Red
    "politics": "#3C3B6E",          # US Blue
    "news": "#B22234",              # US Red
    "worldnews": "#1f77b4",         # Blue
    "geopolitics": "#2ca02c",       # Green
    "LatinAmerica": "#FF6B35",      # Orange
    "Conservative": "#d62728",      # Red
    "neoliberal": "#17becf",        # Cyan
    "socialism": "#e377c2",         # Pink
    "Libertarian": "#bcbd22",       # Yellow-green
    "southamerica": "#9edae5",      # Light blue
}

# Crisis period colors for visualizations
CRISIS_COLORS: Dict[str, str] = {
    "Maduro Inauguration 2013": "#E63946",
    "2014 Venezuelan Protests": "#F4A261",
    "Oil Price Crash Impact": "#2A9D8F",
    "Trump Administration Sanctions": "#E76F51",
    "2018 Disputed Election": "#9467bd",
    "Guaido Recognition Crisis": "#264653",
    "April 2019 Uprising Attempt": "#E9C46A",
    "Biden Administration Policy Shift": "#457B9D",
    "2024 Venezuelan Election Crisis": "#D62828",
    "Gonzalez Urrutia Exile": "#F77F00",
}

# Flashpoint windows for timeline visualizations
FLASHPOINT_WINDOWS: Dict[str, Dict] = {
    "Guaido Crisis 2019": {
        "start": "2019-01-01",
        "end": "2019-05-31",
        "events": {
            "2019-01-23": {"name": "Guaido Declares Presidency", "y_pct": 0.85, "x_offset": 20, "y_offset": 0},
            "2019-02-23": {"name": "Aid Standoff", "y_pct": 0.55, "x_offset": 25, "y_offset": -5},
            "2019-04-30": {"name": "Failed Uprising", "y_pct": 0.70, "x_offset": 20, "y_offset": 0},
        },
        "line_color": "#1f77b4",
        "label_color": "#d62728",
    },
    "2024 Election Crisis": {
        "start": "2024-07-01",
        "end": "2024-09-30",
        "events": {
            "2024-07-28": {"name": "Election Day", "y_pct": 0.90, "x_offset": 20, "y_offset": 0},
            "2024-08-02": {"name": "Protests Erupt", "y_pct": 0.55, "x_offset": 20, "y_offset": 0},
            "2024-09-08": {"name": "Gonzalez Exile", "y_pct": 0.70, "x_offset": 20, "y_offset": 0},
        },
        "line_color": "#d62728",
        "label_color": "#264653",
    },
    "Trump Sanctions 2017": {
        "start": "2017-07-01",
        "end": "2017-10-31",
        "events": {
            "2017-08-11": {"name": "Trump Threat", "y_pct": 0.50, "x_offset": 20, "y_offset": 0},
            "2017-08-25": {"name": "Financial Sanctions", "y_pct": 0.85, "x_offset": 20, "y_offset": 0},
        },
        "line_color": "#E76F51",
        "label_color": "#d62728",
    },
}

# Topic categories for NLP analysis
TOPIC_CATEGORIES: Dict[str, List[str]] = {
    "Political": ["maduro", "guaido", "chavez", "opposition", "government", "regime", "dictator", "democracy"],
    "Economic": ["oil", "sanctions", "economy", "inflation", "crisis", "pdvsa", "currency", "petro"],
    "Humanitarian": ["humanitarian", "refugees", "migration", "food", "medicine", "shortage", "aid"],
    "Military": ["military", "army", "intervention", "coup", "colectivos", "national guard"],
    "Diplomatic": ["diplomatic", "recognition", "embassy", "relations", "lima group", "negotiations"],
}

# Topic colors for visualization
TOPIC_COLORS: Dict[str, str] = {
    "Political": "#8B4513",      # Brown
    "Economic": "#DC143C",       # Crimson
    "Humanitarian": "#228B22",   # Forest Green
    "Military": "#4B0082",       # Indigo
    "Diplomatic": "#1E90FF",     # Dodger Blue
}


# =============================================================================
# SEARCH QUERIES
# =============================================================================

SEARCH_QUERIES: Dict[str, List[str]] = {
    "bilateral": [
        "Venezuela US",
        "Venezuela United States",
        "US Venezuela relations",
        "American Venezuela",
        "Venezuela Washington",
    ],
    "political": [
        "Maduro",
        "Guaido",
        "Venezuela election",
        "Venezuela democracy",
        "Venezuela opposition",
        "Venezuela government",
        "Chavismo",
    ],
    "economic": [
        "Venezuela sanctions",
        "Venezuela oil",
        "PDVSA",
        "Venezuela economy",
        "Venezuela inflation",
        "Venezuela currency",
        "petro cryptocurrency",
    ],
    "humanitarian": [
        "Venezuela crisis",
        "Venezuela humanitarian",
        "Venezuela refugees",
        "Venezuela migration",
        "Venezuela food shortage",
        "Venezuela medicine",
    ],
    "military": [
        "Venezuela military",
        "Venezuela intervention",
        "Venezuela coup",
        "Venezuela army",
        "colectivos",
    ],
    "diplomatic": [
        "Venezuela diplomacy",
        "Venezuela embassy",
        "Venezuela recognition",
        "Lima Group",
        "Venezuela UN",
    ],
}

ALL_SEARCH_QUERIES: List[str] = [
    q for queries in SEARCH_QUERIES.values() for q in queries
]

PRIORITY_QUERIES: List[str] = [
    "Venezuela",
    "Maduro",
    "Venezuela US",
    "Venezuela sanctions",
    "Guaido",
    "Venezuelan crisis",
    "Venezuela oil",
    "Caracas",
    "Venezuela election",
    "Venezuela humanitarian",
]

CRISIS_QUERIES: List[str] = [
    "Venezuela",
    "Maduro",
    "Guaido",
    "Venezuela election",
    "Venezuela sanctions",
    "Venezuela crisis",
    "Venezuela coup",
    "Venezuela intervention",
    "Venezuela oil",
    "Venezuelan",
]


# =============================================================================
# FIELD DEFINITIONS
# =============================================================================

SUBMISSION_FIELDS: List[str] = [
    "id",
    "title",
    "selftext",
    "author",
    "subreddit",
    "created_utc",
    "score",
    "num_comments",
    "url",
]

COMMENT_FIELDS: List[str] = [
    "id",
    "body",
    "author",
    "subreddit",
    "created_utc",
    "score",
    "link_id",
    "parent_id",
    "controversiality",
    "is_submitter",
]

COMMENT_FIELDS_TO_KEEP: List[str] = [
    "id",
    "body",
    "author",
    "subreddit",
    "created_utc",
    "score",
    "link_id",
    "parent_id",
    "controversiality",
    "is_submitter",
    "distinguished",
    "edited",
    "permalink",
]
