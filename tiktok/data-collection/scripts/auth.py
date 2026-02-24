"""
TikTok Research API authentication module.
Handles OAuth2 client credentials flow with automatic token refresh.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from tiktok_research_api import TikTokResearchAPI

from .config import PipelineConfig


def load_credentials() -> tuple[str, str]:
    """Load TikTok API credentials from .env file."""
    # Try project root .env first, then parent directories
    env_paths = [
        Path(__file__).parent.parent.parent / ".env",  # project root
        Path(__file__).parent.parent / ".env",          # data-collection/
        Path.cwd() / ".env",                            # current directory
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break

    client_key = os.getenv("TIKTOK_CLIENT_KEY")
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET")

    if not client_key or not client_secret:
        raise ValueError(
            "TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in .env file"
        )

    return client_key, client_secret


def create_client(config: PipelineConfig | None = None) -> TikTokResearchAPI:
    """
    Create authenticated TikTok Research API client.

    The SDK handles token generation and refresh internally.

    Args:
        config: Optional pipeline config for rate limit settings.

    Returns:
        Authenticated TikTokResearchAPI instance.
    """
    client_key, client_secret = load_credentials()
    qps = config.qps if config else 1

    api = TikTokResearchAPI(client_key, client_secret, qps)
    print(f"TikTok Research API client initialized (QPS: {qps})")
    return api
