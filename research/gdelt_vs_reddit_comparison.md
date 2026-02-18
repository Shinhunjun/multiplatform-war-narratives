# GDELT News Source vs Reddit Social Media Data

This document compares the two core data sources used in this repository's daily pipeline:
- GDELT news/event data (plus scraped article text)
- Reddit social discussion data

## 1) Data nature and signal type

| Dimension | GDELT (News/Event) | Reddit (Social Media) |
|---|---|---|
| Primary signal | Structured event metadata + global news coverage | User-generated discussions, opinions, and reactions |
| Content style | Institutional/media framing | Community framing, informal language, debate |
| Typical unit | Event rows, GKG records, article URLs/text | Submissions and comments |
| Strength | Macro-level geopolitical/event monitoring | Micro-level public discourse and sentiment |
| Limitation | Media selection/framing bias, incomplete full text without scraping | Community bias, moderation effects, noisy/bot-like content |

## 2) How collection is implemented in this repo

### GDELT collection
- Implemented in `webapp/pipeline/collectors/gdelt.py` via `GDELTCollector`.
- Pulls:
  - Events table (`gdelt-bq.gdeltv2.events`) through `collect_events(...)`.
  - GKG table (`gdelt-bq.gdeltv2.gkg`) through `collect_gkg(...)`.
- Query logic uses Venezuela-related actor filters plus keyword matching from `PipelineConfig.gdelt_keywords` in `webapp/pipeline/config.py`.
- Returns article URLs (`SOURCEURL`, `DocumentIdentifier`) for downstream scraping.

### Reddit collection
- Implemented in `webapp/pipeline/collectors/reddit.py` via `RedditCollector`.
- Uses Arctic Shift API endpoints:
  - `posts/search` for submissions.
  - `comments/search` for comment retrieval per post.
- Applies strategy from config (`webapp/pipeline/config.py`):
  - Venezuela-focused subs collected without keyword filtering.
  - General subs queried with `reddit_queries`.
- Includes retry/backoff/rate-limit handling and deduplication by post ID.

## 3) Text acquisition differences

- GDELT does not always provide rich article body text directly. This repo adds a scraping stage:
  - `webapp/pipeline/collectors/scraper.py` (`ArticleScraper`) fetches and extracts article text from URLs.
- Reddit already provides native text fields:
  - Submission text from `title + selftext`.
  - Comment text from `body`.
- Practical result:
  - GDELT path is two-step (metadata -> URL scraping -> text corpus).
  - Reddit path is mostly direct API-to-text.

## 4) Preprocessing and normalization in this repo

- Implemented in `webapp/pipeline/processing/preprocessor.py`.
- Reddit preprocessing (`preprocess_reddit(...)`):
  - Cleans markdown/URLs, removes low-information content.
  - Filters bots and deleted/removed content.
  - Normalizes both submissions and comments to one schema with time fields.
- GDELT preprocessing (`preprocess_gdelt(...)`):
  - Creates rows for scraped `news_article` text.
  - Also preserves event-level metadata rows (`gdelt_event`) with actors, tone, mentions, and event code.

This means the project keeps both:
- discourse-level language data (Reddit), and
- institutional event/media context (GDELT).

## 5) Pipeline orchestration and analysis implications

- Main orchestrator: `webapp/pipeline/main.py`.
- Collection stages are split:
  - `stage_collect_reddit(...)`
  - `stage_collect_gdelt(...)`
- GDELT article URLs flow into `stage_scrape(...)`.
- Incremental analysis is currently run on preprocessed Reddit documents in `stage_analyze(...)` (loading `processed/reddit/reddit_<date>.parquet`), using `IncrementalAnalyzer` in `webapp/pipeline/processing/analyzer.py`.

Implication:
- Reddit is the primary source for regular sentiment/topic refresh in the incremental analysis stage.
- GDELT contributes event/news context and can be analyzed in parallel or integrated further in future iterations.

## 6) Recommended interpretation of combined use

- Use GDELT for: "what happened, where, and how institutional media records it."
- Use Reddit for: "how communities interpret, debate, and emotionally react."
- Use both together for triangulation:
  - Event timeline and actor context from GDELT.
  - Public narrative/sentiment shifts from Reddit.

In short, this repository already implements a complementary dual-source architecture where GDELT provides structured external context and Reddit provides social narrative depth.
