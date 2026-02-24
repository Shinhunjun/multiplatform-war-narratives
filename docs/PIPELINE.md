# Multiplatform War Narratives — Pipeline & Algorithm Documentation

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Sources & Collection](#2-data-sources--collection)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Analysis Algorithms](#4-analysis-algorithms)
5. [Daily ETL Pipeline](#5-daily-etl-pipeline)
6. [Web Dashboard](#6-web-dashboard)
7. [Deployment](#7-deployment)
8. [Project Directory Structure](#8-project-directory-structure)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐     │
│  │  Reddit   │    │    GDELT     │    │       TikTok          │     │
│  │ (Arctic   │    │  (BigQuery)  │    │  (Research API)       │     │
│  │  Shift)   │    │              │    │                       │     │
│  └─────┬─────┘    └──────┬───────┘    └───────────┬───────────┘     │
│        │                 │                        │                 │
└────────┼─────────────────┼────────────────────────┼─────────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     COLLECTION LAYER                                │
│  Arctic Shift API ──→ Submissions + Comments (JSON)                │
│  GDELT BigQuery   ──→ Events + GKG records (Parquet)               │
│  News Scraper     ──→ Article full text (JSON)                     │
│  TikTok SDK       ──→ Videos + Comments (JSON)                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                   │
│  Filter bots/deleted ──→ Clean text ──→ Normalize ──→ Parquet      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ANALYSIS ALGORITHMS                             │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐      │
│  │   Sentiment   │  │    Topic      │  │    Clustering      │      │
│  │   (RoBERTa)   │  │  (BERTopic)   │  │ (HDBSCAN + UMAP)  │      │
│  └───────┬───────┘  └───────┬───────┘  └─────────┬──────────┘      │
│          │                  │                     │                 │
│          ▼                  ▼                     ▼                 │
│     sentiment_*.csv    topic_*.csv          cluster_*.csv           │
│                    + embeddings.npy                                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WEB APPLICATION                                 │
│  FastAPI (Backend)  ◄───────────────►  React + Recharts (Frontend) │
│  Google Cloud Run                      Vercel                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Sources & Collection

### 2.1 Reddit — Arctic Shift API

Reddit data is collected through the [Arctic Shift](https://arctic-shift.photon-reddit.com/) API. This API provides search access to the full Reddit archive (2005–present) and requires **no authentication**.

**API Endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /api/posts/search` | Search submissions by subreddit, keyword, and date |
| `GET /api/comments/search` | Search comments (filterable by `link_id` for specific posts) |

**Collection Strategy:**

- **Venezuela-focused subreddits** (`r/venezuela`, `r/vzla`): Collect all posts without keyword filtering
- **General subreddits** (`r/politics`, `r/worldnews`, etc.): Multi-query search using 10 keywords with ID-based deduplication

**Search Keywords (10):**
```
Venezuela, Maduro, Venezuela US, Venezuela sanctions,
Guaido, Venezuelan crisis, Venezuela oil, Caracas,
Venezuela election, Venezuela humanitarian
```

**Monitored Subreddits (11):**

| Category | Subreddits |
|----------|------------|
| Venezuela-focused | `r/venezuela`, `r/vzla` |
| US Mainstream | `r/politics`, `r/news`, `r/worldnews` |
| US Conservative | `r/Conservative`, `r/Libertarian` |
| US Progressive | `r/neoliberal`, `r/socialism` |
| Regional/Academic | `r/LatinAmerica`, `r/geopolitics` |

**Rate Limiting:** 1-second interval between requests; exponential backoff on 429 responses (10s, 20s, 30s...)

**Data Scale (Historical Collection):**

| Metric | Count |
|--------|-------|
| Submissions | 101,960 |
| Comments | 431,981 |
| Period | 2013-01 – 2026-01 |
| After Preprocessing | 426,435 |

### 2.2 GDELT — BigQuery

Venezuela-related news events are collected from the [GDELT Project](https://www.gdeltproject.org/) public dataset on BigQuery.

**Tables:**

| Table | Contents |
|-------|----------|
| `gdelt-bq.gdeltv2.events` | Global events (Actor, EventCode, GoldsteinScale, AvgTone) |
| `gdelt-bq.gdeltv2.gkg` | Global Knowledge Graph (themes, persons, organizations, tone) |

**Query Conditions:**
- `Actor1CountryCode = 'VEN'` or `Actor2CountryCode = 'VEN'`
- Or keyword matching: `venezuela`, `maduro`, `caracas`, `guaido`, `pdvsa`, `citgo`

**After extraction, the News Scraper visits article URLs referenced in GDELT records to extract full-text content.**

### 2.3 TikTok — Research API

Videos and comments are collected via the TikTok Research API (OAuth2 authentication, daily quota limits).

- **Daily Limit:** 1,000 requests / 100,000 records
- **Query Window:** Maximum 30 days
- **Collection Targets:** Hashtags (`#venezuela`, `#maduro`, etc.) + keyword search

---

## 3. Preprocessing Pipeline

Raw collected data is transformed into an analysis-ready format through the following process.

### 3.1 Filtering

```
Raw Data
  │
  ├── [removed] content → Removed
  ├── [deleted] content → Removed
  ├── Bot accounts → Removed
  │     AutoModerator, autotldr, empleadoEstatalBot,
  │     RemindMeBot, WikiTextBot, TotesMessenger,
  │     RepostSleuthBot, SaveVideo, VisualMod
  ├── Fewer than 5 words → Removed
  └── Media-only (URL only) → Removed
```

### 3.2 Text Cleaning

| Step | Example |
|------|---------|
| Remove URLs | `Check https://example.com` → `Check` |
| Remove Markdown | `**bold** text` → `bold text` |
| Remove Reddit quotes | `> quoted text` → (removed) |
| Remove code blocks | `` `code` `` → (removed) |
| Remove edit markers | `Edit: added more` → `added more` |
| Normalize whitespace | multiple spaces → single space |

### 3.3 Output Schema

```
DataFrame Columns:
  id              : str    — Reddit post/comment ID
  type            : str    — "submission" or "comment"
  subreddit       : str    — e.g. "politics"
  author          : str    — Reddit username
  text            : str    — Cleaned text
  score           : int    — Reddit score (upvotes - downvotes)
  created_utc     : float  — Unix timestamp
  created_datetime: datetime — UTC datetime
  year            : int
  month           : int
  year_month      : str    — "YYYY-MM" format
```

**Output Format:** Apache Parquet (columnar, compressed)

---

## 4. Analysis Algorithms

### 4.1 Sentiment Analysis — RoBERTa

Analyzes the sentiment of social media text.

**Model:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

This model is a fine-tuned version of RoBERTa-base, pre-trained on ~124M tweets, for the sentiment analysis task. It effectively handles the informal language, abbreviations, and emojis typical of social media.

**How It Works:**

```
Input Text (max 512 tokens)
      │
      ▼
┌─────────────────────────┐
│ RoBERTa Tokenizer       │  WordPiece tokenization
│ (twitter-roberta-base)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ RoBERTa Encoder         │  12 layers, 768 hidden dim
│ (124M parameters)       │  Self-attention mechanism
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Classification Head     │  Linear → Softmax
│ 3 classes               │
└────────────┬────────────┘
             │
             ▼
    ┌────────┼────────┐
    │        │        │
 Negative  Neutral  Positive
  (0~1)    (0~1)    (0~1)
```

**Output:**

| Field | Type | Description |
|-------|------|-------------|
| `sentiment_label` | str | `"positive"`, `"negative"`, `"neutral"` |
| `sentiment_confidence` | float | 0.0–1.0 (model confidence) |
| `sentiment_score` | float | -1.0–+1.0 (direction × confidence) |

**Aggregation Methods:**

- **By month (by_month):** Mean sentiment score, positive/negative ratios for each YYYY-MM period
- **By subreddit (by_subreddit):** Sentiment comparison across communities
- **By subreddit × month (by_subreddit_month):** Tracking sentiment changes per community over time

**GPU Acceleration:** Apple Silicon (MPS), NVIDIA (CUDA), or CPU fallback. Batch size: 64.

---

### 4.2 Topic Modeling — BERTopic

Automatically discovers semantically coherent topics from the document corpus.

**BERTopic Pipeline:**

```
Documents (426K texts)
      │
      ▼
┌───────────────────────────────┐
│ 1. Sentence-BERT Embedding    │
│    all-MiniLM-L6-v2           │
│    Output: 384-dim vectors    │
│    (22M params, fast)         │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 2. UMAP Dimensionality       │
│    Reduction                  │
│    384-dim → 5-dim            │
│    n_neighbors=15             │
│    metric=cosine              │
│    min_dist=0.0               │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 3. HDBSCAN Clustering        │
│    min_cluster_size=50        │
│    metric=euclidean           │
│    method=eom (excess of mass)│
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 4. Topic Representation       │
│    CountVectorizer (1-2 gram) │
│    + KeyBERTInspired          │
│    → Top keywords per topic   │
└──────────────┬────────────────┘
               │
               ▼
     15 Topics Discovered
     (+ outlier topic -1)
```

**Role of Each Step:**

| Step | Algorithm | Purpose |
|------|-----------|---------|
| Embedding | Sentence-BERT (all-MiniLM-L6-v2) | Converts text into 384-dimensional dense vectors. Semantically similar texts produce nearby vectors |
| Reduction | UMAP | Reduces high-dimensional embeddings to 5 dimensions while preserving local structure. Improves clustering efficiency |
| Clustering | HDBSCAN | Density-based hierarchical clustering. Naturally separates noise (topic -1) and automatically determines the number of clusters |
| Representation | c-TF-IDF + KeyBERT | Extracts representative keywords for each topic. c-TF-IDF identifies important words per class, then KeyBERT refines them semantically |

**Output Files:**

| File | Contents |
|------|----------|
| `topic_info.csv` | Topic ID, name (keywords), document count |
| `topic_assignments.parquet` | Document → topic mapping (ID, probability) |
| `topics_over_time.csv` | Topic frequency changes over time |
| `topics_by_subreddit.csv` | Topic distribution by subreddit |
| `document_embeddings.npy` | Full document 384-dim embeddings (648MB) |
| `bertopic_model` | Trained BERTopic model (2.9GB) |

**Incremental Topic Assignment (Daily Pipeline):**

When new documents arrive, instead of retraining the BERTopic model, they are assigned to existing topics using **embedding cosine similarity**:

1. Compute **per-topic centroids** from existing `document_embeddings.npy` + `topic_assignments.parquet`
2. Embed new documents using Sentence-BERT
3. Calculate cosine similarity between each new document and all centroids
4. Assign to the most similar topic (threshold > 0.25; below threshold → outlier -1)

---

### 4.3 Clustering — HDBSCAN + UMAP

Discovers semantic clusters in the document embedding space and provides visualizations.

This is a global clustering performed **separately from BERTopic's topic modeling**, designed to discover finer-grained subgroups.

**Pipeline:**

```
Preprocessed Documents
      │
      ▼
┌───────────────────────────────┐
│ Sentence-BERT Embedding       │
│ all-MiniLM-L6-v2              │
│ 426K docs → 426K × 384       │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ UMAP → 50-dim (for HDBSCAN)  │
│ metric=cosine                 │
│ random_state=42               │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ HDBSCAN Clustering            │
│ min_cluster_size=50           │
│ min_samples=10                │
│ metric=euclidean              │
│ selection=eom                 │
│ → 3,406 clusters found       │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ UMAP → 2-dim (for viz)       │
│ n_neighbors=15                │
│ min_dist=0.1                  │
│ metric=cosine                 │
│ → Scatter plot coordinates    │
└──────────────┬────────────────┘
               │
               ▼
  Cluster summaries, keywords,
  temporal evolution, heatmaps
```

**HDBSCAN Parameter Description:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `min_cluster_size` | 50 | Minimum of 50 documents required to form a cluster |
| `min_samples` | 10 | Minimum of 10 neighbors within radius required to be a core point |
| `metric` | euclidean | Euclidean distance used after UMAP reduction |
| `cluster_selection_method` | eom | Excess of Mass — effectively detects clusters with varying densities |

**Cluster Analysis Outputs:**

| File | Contents |
|------|----------|
| `cluster_assignments.parquet` | Per-document cluster_id, probability, UMAP coordinates |
| `cluster_summaries.csv` | Per-cluster statistics (document count, top subreddits, sentiment, time period) |
| `cluster_keywords.csv` | TF-IDF-based top keywords per cluster |
| `temporal_clusters.csv` | Cluster size changes over time |
| `embeddings.npy` | Full 384-dim embeddings |
| `embeddings_2d.npy` | 2D UMAP coordinates for visualization |

**Visualizations:**

- **UMAP Scatter Plot:** Projects all documents onto a 2D plane, colored by cluster/subreddit
- **Animated UMAP:** GIF animation showing cluster changes over time
- **River Plot:** Sankey-style cluster flow visualization
- **Heatmap:** Subreddit × cluster matrix

---

## 5. Daily ETL Pipeline

An automated pipeline that runs daily to collect, preprocess, analyze, and update data.

### 5.1 Architecture

```
┌─────────────────────┐
│  Cloud Scheduler     │  Cron: 0 6 * * * (daily at 06:00 UTC)
│  (GCP)              │
└──────────┬──────────┘
           │ HTTP POST
           ▼
┌─────────────────────┐
│  Cloud Run Job       │  Container: pipeline-etl-daily
│  (4GB RAM, 2 CPU)   │  Timeout: 1 hour
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│              ETL STAGES                      │
│                                              │
│  Stage 1: COLLECT                            │
│  ├── Reddit: Arctic Shift API (11 subs)     │
│  └── GDELT: BigQuery (events + GKG)         │
│                                              │
│  Stage 2: SCRAPE                             │
│  └── News articles from GDELT URLs          │
│      (aiohttp, 5 concurrent, 15s timeout)   │
│                                              │
│  Stage 3: PREPROCESS                         │
│  ├── Filter bots, deleted, low-quality      │
│  ├── Clean text (URLs, markdown, etc.)      │
│  └── Output: Parquet files                  │
│                                              │
│  Stage 4: ANALYZE                            │
│  ├── Sentiment: RoBERTa pipeline            │
│  ├── Topics: Embedding → cosine similarity  │
│  └── Merge results with existing CSVs       │
│                                              │
└─────────────────────────────────────────────┘
```

### 5.2 CLI Usage

```bash
# Run the full pipeline (based on today's date)
python -m webapp.pipeline.main

# Collect Reddit only
python -m webapp.pipeline.main --reddit-only

# Collect GDELT only
python -m webapp.pipeline.main --gdelt-only

# Specify a particular date
python -m webapp.pipeline.main --date 2026-02-14

# Run a specific stage only
python -m webapp.pipeline.main --stage collect
python -m webapp.pipeline.main --stage preprocess
python -m webapp.pipeline.main --stage analyze

# Change lookback period (default: 1 day)
python -m webapp.pipeline.main --lookback 7
```

### 5.3 Incremental Update Strategy

The pipeline **merges new data with existing results** without requiring full re-analysis:

**Sentiment:**
1. Run RoBERTa sentiment analysis on new documents
2. Load existing `sentiment_by_month.csv`
3. Concatenate with new monthly aggregations → re-aggregate overlapping months
4. Save updated CSV

**Topics:**
1. Compute topic centroids from existing `document_embeddings.npy` + `topic_assignments.parquet`
2. Embed new documents using Sentence-BERT
3. Assign to nearest topic by cosine similarity (threshold 0.25)
4. Update `topics_over_time.csv` and `topics_by_subreddit.csv`

### 5.4 Sample Pipeline Execution Result

```json
{
  "run_date": "2026-02-14",
  "stages": {
    "collect_reddit": {
      "submissions_count": 108,
      "comments_count": 289
    },
    "preprocess": {
      "reddit": { "count": 276 }
    },
    "analyze": {
      "sentiment": { "mean_score": -0.168, "count": 276 },
      "topics": { "assigned": 251, "count": 276 }
    }
  }
}
```

---

## 6. Web Dashboard

### 6.1 Backend — FastAPI

Serves analysis result CSVs/Parquets via a REST API.

**Tech Stack:** Python 3.11, FastAPI, pandas, LRU cache

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/overview/stats` | GET | Overall statistics (document count, subreddit count, topic count, mean sentiment) |
| `/api/sentiment/by-month` | GET | Monthly sentiment trends (start, end parameters) |
| `/api/sentiment/by-subreddit` | GET | Sentiment comparison by subreddit |
| `/api/sentiment/by-subreddit-month` | GET | Subreddit × month sentiment matrix |
| `/api/topics/info` | GET | Topic list (ID, keywords, document count) |
| `/api/topics/over-time` | GET | Topic trends over time |
| `/api/topics/by-subreddit` | GET | Topic distribution by subreddit |
| `/api/clusters/summaries` | GET | Cluster summaries (limit, min_count parameters) |
| `/api/clusters/keywords` | GET | Keywords per cluster |
| `/api/clusters/temporal` | GET | Cluster changes over time |
| `/health` | GET | Health check |

**Data Loading:** CSVs/Parquets are loaded once and cached in memory using `@functools.lru_cache`.

### 6.2 Frontend — React Dashboard

**Tech Stack:** React 19, TypeScript, Vite, TailwindCSS, Recharts, React Router, Axios

**Pages:**

| Page | Visualizations |
|------|----------------|
| **Dashboard** | StatCards (total documents/subreddits/topics/mean sentiment), Sentiment Timeline (LineChart), Sentiment by Subreddit (BarChart), Volume Distribution (StackedBarChart) |
| **Sentiment** | Multi-subreddit sentiment comparison (toggle buttons), Composite Timeline (LineChart), Subreddit Sentiment Overview Table |
| **Topics** | Topic distribution (Horizontal BarChart), Topic Evolution Top 8 (Stacked AreaChart), Topic Detail Table |
| **Clusters** | Top 20 cluster sizes (sentiment-colored BarChart), Custom Tooltip (subreddit, sentiment, time period), Cluster Detail Table (50 rows) |

---

## 7. Deployment

### 7.1 Backend → Google Cloud Run

```bash
# Build & Deploy
gcloud builds submit --config webapp/pipeline/deploy/cloudbuild.yaml .

# Or manual
docker build -t gcr.io/PROJECT/backend-api .
docker push gcr.io/PROJECT/backend-api
gcloud run deploy backend-api --image gcr.io/PROJECT/backend-api --region us-central1
```

### 7.2 Frontend → Vercel

```bash
cd webapp/frontend
npm run build     # → dist/
# Vercel auto-deploys from GitHub
```

**Environment Variable:** `VITE_API_URL` → Cloud Run backend URL

### 7.3 Daily Pipeline → Cloud Run Jobs + Cloud Scheduler

```bash
# 1. Deploy Cloud Run Job
gcloud builds submit --config webapp/pipeline/deploy/cloudbuild.yaml .

# 2. Setup Cloud Scheduler (daily at 06:00 UTC)
chmod +x webapp/pipeline/deploy/setup_scheduler.sh
./webapp/pipeline/deploy/setup_scheduler.sh PROJECT_ID us-central1

# 3. Set secrets
gcloud run jobs update pipeline-etl-daily --region=us-central1 \
  --set-env-vars GCP_PROJECT=xxx,GCS_BUCKET=xxx
```

---

## 8. Project Directory Structure

```
capstone/
├── README.md                              # Project overview
├── docs/
│   ├── PIPELINE.md                        # This document (pipeline & algorithms)
│   ├── CODE_WALK.md                       # Code walk presentation guide
│   ├── research-questions.md              # Research questions
│   └── RelatedWork_Dataset.pdf            # Related work reference
│
├── reddit/                                # Reddit analysis project
│   ├── data-collection/
│   │   ├── main.py                        # Collection CLI (historical/crisis/comments)
│   │   ├── pyproject.toml
│   │   └── scripts/
│   │       ├── config.py                  # Subreddit, keyword, flashpoint settings
│   │       ├── collectors.py              # Arctic Shift API calls
│   │       └── processors.py              # JSON load/save/merge/dedup
│   │
│   ├── preprocessing/
│   │   ├── config.py                      # Preprocessing settings
│   │   ├── filters.py                     # Bot/deleted/low-quality filters
│   │   ├── text_cleaner.py                # URL/markdown/emoji cleaning
│   │   └── preprocessor.py                # RedditPreprocessor main class
│   │
│   ├── analysis/
│   │   ├── main.py                        # Analysis pipeline orchestrator
│   │   ├── config.py                      # AnalysisConfig (models, parameters)
│   │   ├── data_loader.py                 # Parquet loader
│   │   ├── sentiment/
│   │   │   └── roberta_analyzer.py        # RoBERTa sentiment analysis
│   │   ├── topic/
│   │   │   └── bertopic_model.py          # BERTopic topic modeling
│   │   ├── clustering/
│   │   │   ├── embedder.py                # Sentence-BERT embedder
│   │   │   ├── cluster.py                 # TemporalClusterer (HDBSCAN)
│   │   │   ├── summarizer.py              # TF-IDF + LLM cluster summarization
│   │   │   └── temporal_viz.py            # UMAP scatter, animation, heatmap
│   │   └── outputs/                       # Analysis results (CSV, Parquet, npy)
│   │
│   └── eda/                               # Exploratory data analysis
│
├── gdelt/                                 # GDELT news data collection + analysis
│   ├── data-collection/
│   │   ├── scrape_by_year.py              # Year-by-year GDELT scraping
│   │   ├── rescue_by_year.py              # Retry failed scrapes
│   │   └── consolidate_yearly.py          # Merge yearly CSVs
│   ├── preprocessing/
│   │   └── build_text_relevance_tokens.py # Text relevance filtering
│   └── analysis/
│       ├── run_eda.py                     # BigQuery export EDA
│       ├── run_eda_scraped.py             # Scraped data EDA
│       └── analyze_gdelt.py              # GDELT analysis script
│
├── tiktok/                                # TikTok collection pipeline
│   ├── data-collection/
│   │   ├── main.py
│   │   └── scripts/
│   │       ├── config.py                  # TikTok API settings
│   │       ├── auth.py                    # OAuth2 authentication
│   │       └── collectors.py              # Video/comment collection
│   └── preprocessing/
│       ├── filters.py                     # TikTok-specific filters
│       └── preprocessor.py
│
├── graphrag/                              # GraphRAG knowledge graph instance
│   ├── settings.yaml                      # Config (Ollama LLM + embeddings)
│   ├── input/                             # 200 Reddit thread documents (.txt)
│   ├── output/                            # Indexed entities, communities
│   └── prompts/                           # Custom extraction prompts
│
├── webapp/
│   ├── backend/                           # FastAPI backend
│   │   ├── main.py                        # FastAPI app, CORS, router registration
│   │   ├── routers/
│   │   │   ├── overview.py                # /api/overview/*
│   │   │   ├── sentiment.py               # /api/sentiment/*
│   │   │   ├── topics.py                  # /api/topics/*
│   │   │   └── clusters.py                # /api/clusters/*
│   │   └── services/
│   │       └── data_service.py            # CSV/Parquet loading + GCS download
│   │
│   ├── frontend/                          # React dashboard
│   │   ├── src/
│   │   │   ├── App.tsx                    # Router + Layout
│   │   │   ├── lib/api.ts                 # API client (Axios)
│   │   │   ├── components/layout/
│   │   │   │   └── Sidebar.tsx            # Navigation
│   │   │   └── pages/
│   │   │       ├── Dashboard.tsx           # Main dashboard
│   │   │       ├── SentimentPage.tsx       # Sentiment analysis page
│   │   │       ├── TopicsPage.tsx          # Topic modeling page
│   │   │       └── ClustersPage.tsx        # Cluster analysis page
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── tsconfig.json
│   │
│   └── pipeline/                          # Daily ETL pipeline
│       ├── main.py                        # CLI orchestrator
│       ├── config.py                      # PipelineConfig
│       ├── collectors/
│       │   ├── reddit.py                  # Arctic Shift API collection
│       │   ├── gdelt.py                   # BigQuery collection
│       │   └── scraper.py                 # News article scraper
│       ├── processing/
│       │   ├── preprocessor.py            # Preprocessing (filter + cleaning)
│       │   └── analyzer.py                # Incremental analysis (sentiment + topics)
│       ├── deploy/
│       │   ├── cloudbuild.yaml            # Cloud Build configuration
│       │   └── setup_scheduler.sh         # Cloud Scheduler setup
│       ├── Dockerfile
│       └── requirements.txt
│
└── data/                                  # Raw data files (gitignored)
    └── gdelt/                             # GDELT BigQuery exports + scraped CSVs
```
