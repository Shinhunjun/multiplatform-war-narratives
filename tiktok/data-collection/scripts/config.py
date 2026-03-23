"""
Configuration for Venezuela TikTok Data Collection Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PipelineConfig:
    """Central configuration for the TikTok data collection pipeline."""

    base_dir: Path = field(default_factory=lambda: Path("./data"))

    @property
    def videos_dir(self) -> Path:
        return self.base_dir / "videos"

    @property
    def comments_dir(self) -> Path:
        return self.base_dir / "comments"

    @property
    def exports_dir(self) -> Path:
        return self.base_dir / "exports"

    @property
    def checkpoints_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    # TikTok API settings
    max_count: int = 100          # Max records per request
    daily_request_limit: int = 1000
    daily_record_limit: int = 100_000
    qps: int = 1                  # Queries per second (rate limiter)

    def ensure_directories(self) -> "PipelineConfig":
        """Create all required directories."""
        for d in [
            self.base_dir,
            self.videos_dir,
            self.comments_dir,
            self.exports_dir,
            self.checkpoints_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"Directories created at: {self.base_dir}")
        return self


# =============================================================================
# DEFAULT DATE RANGES
# =============================================================================

# TikTok launched internationally ~2017, but Research API may have data from 2016+
HISTORICAL_DEFAULT_START: str = "20160801"
HISTORICAL_DEFAULT_END: str = "20260214"


# =============================================================================
# FLASHPOINTS / CRISIS PERIODS (same as Reddit pipeline)
# =============================================================================

FLASHPOINTS: Dict[str, Dict] = {
    "maduro_inauguration_2013": {
        "name": "Maduro Inauguration 2013",
        "start": "20130414",
        "end": "20130430",
        "priority": "high",
        "description": "Maduro presidential inauguration and initial controversies",
    },
    "protests_2014": {
        "name": "2014 Venezuelan Protests",
        "start": "20140201",
        "end": "20140531",
        "priority": "critical",
        "description": "Large-scale anti-government protests",
    },
    "oil_price_crash_2014": {
        "name": "Oil Price Crash Impact",
        "start": "20141101",
        "end": "20150228",
        "priority": "high",
        "description": "Economic crisis deepening due to oil price crash",
    },
    "trump_sanctions_2017": {
        "name": "Trump Administration Sanctions",
        "start": "20170801",
        "end": "20170930",
        "priority": "critical",
        "description": "Trump administration strengthens sanctions against Venezuela",
    },
    "maduro_reelection_2018": {
        "name": "2018 Disputed Election",
        "start": "20180515",
        "end": "20180531",
        "priority": "high",
        "description": "Disputed Maduro reelection amid international criticism",
    },
    "guaido_recognition_2019": {
        "name": "Guaido Recognition Crisis",
        "start": "20190120",
        "end": "20190228",
        "priority": "critical",
        "description": "US recognition of Guaido as interim president",
    },
    "failed_uprising_2019": {
        "name": "April 2019 Uprising Attempt",
        "start": "20190428",
        "end": "20190505",
        "priority": "high",
        "description": "Guaido's failed military uprising attempt",
    },
    "biden_policy_2021": {
        "name": "Biden Administration Policy Shift",
        "start": "20210120",
        "end": "20210331",
        "priority": "medium",
        "description": "Biden administration Venezuela policy changes",
    },
    "election_2024": {
        "name": "2024 Venezuelan Election Crisis",
        "start": "20240720",
        "end": "20240815",
        "priority": "critical",
        "description": "2024 presidential election and fraud allegations",
    },
    "gonzalez_exile_2024": {
        "name": "Gonzalez Urrutia Exile",
        "start": "20240901",
        "end": "20240915",
        "priority": "high",
        "description": "Opposition presidential candidate exile to Spain",
    },
}


# =============================================================================
# SEARCH QUERIES (Keywords for TikTok video search)
# =============================================================================

SEARCH_QUERIES: Dict[str, List[str]] = {
    "bilateral": [
        "Venezuela US",
        "Venezuela United States",
        "Venezuela sanctions",
        "Venezuela Washington",
    ],
    "political": [
        "Maduro",
        "Guaido",
        "Venezuela election",
        "Venezuela opposition",
        "Venezuela government",
        "Chavismo",
    ],
    "economic": [
        "Venezuela oil",
        "PDVSA",
        "Venezuela economy",
        "Venezuela inflation",
    ],
    "humanitarian": [
        "Venezuela crisis",
        "Venezuelan refugees",
        "Venezuela migration",
        "Venezuelan migrants",
    ],
    "military": [
        "Venezuela military",
        "Venezuela intervention",
        "Venezuela coup",
    ],
    "diplomatic": [
        "Venezuela diplomacy",
        "Venezuela UN",
        "Lima Group Venezuela",
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
    "Venezuela crisis",
    "Venezuela oil",
    "Venezuela election",
    "Venezuelan",
]

CRISIS_QUERIES: List[str] = [
    "Venezuela",
    "Maduro",
    "Guaido",
    "Venezuela election",
    "Venezuela sanctions",
    "Venezuela crisis",
    "Venezuela coup",
    "Venezuela oil",
    "Venezuelan",
]


# =============================================================================
# HASHTAGS (TikTok-specific)
# =============================================================================

HASHTAGS: Dict[str, List[str]] = {
    "general": ["venezuela", "venezuelan", "vzla"],
    "political": ["maduro", "guaido", "sosvenezuela", "venezuelalibre"],
    "crisis": ["venezuelacrisis", "crisisvenezuela", "venezuelahumanitaria"],
    "migration": ["venezolanosporelmundo", "venezuelansinusa", "venezolanosenelexterior"],
    "election": ["eleccionesvenezuela", "venezuelaelection", "28julio"],
}

ALL_HASHTAGS: List[str] = [
    h for hashtags in HASHTAGS.values() for h in hashtags
]

PRIORITY_HASHTAGS: List[str] = [
    "venezuela",
    "maduro",
    "venezuelan",
    "sosvenezuela",
    "venezuelalibre",
    "vzla",
]


# =============================================================================
# FIELD DEFINITIONS (TikTok API fields)
# =============================================================================

VIDEO_FIELDS: str = (
    "id,video_description,create_time,region_code,"
    "share_count,view_count,like_count,comment_count,"
    "music_id,hashtag_names,username,video_duration"
)

COMMENT_FIELDS: str = (
    "id,video_id,text,parent_comment_id,"
    "like_count,reply_count,create_time"
)
