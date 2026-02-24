# Code Walk — Venezuela-US Multiplatform Narrative Analysis

**Date:** Feb 24, 2026 | **Team:** Hunjun Shin, Rich Goodier, Ameir El Ouadi

---

## Project at a Glance

| Metric | Value |
|--------|-------|
| **Time Span** | 2013-01 ~ 2026-02 (13 years) |
| **Total Data Points** | 533,941 raw → 426,435 after cleaning |
| **Platforms** | Reddit (11 subreddits), GDELT News, TikTok |
| **NLP Models** | RoBERTa (sentiment), BERTopic (topics), HDBSCAN+UMAP (clusters) |
| **Infra** | FastAPI on Cloud Run, React on Vercel, GCS storage |
| **Daily ETL** | Cloud Run Jobs + Cloud Scheduler |

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    DATA SOURCES                            │
│   Reddit (Arctic Shift)  │  GDELT (BigQuery)  │  TikTok  │
└──────────┬───────────────┴─────────┬──────────┴────┬─────┘
           ▼                         ▼               ▼
┌───────────────────────────────────────────────────────────┐
│  COLLECTION → PREPROCESSING → ANALYSIS                    │
│  reddit/data-collection/   reddit/preprocessing/          │
│  gdelt/data-collection/    reddit/analysis/               │
│  tiktok/data-collection/   (sentiment, topics, clusters)  │
└───────────────────┬───────────────────────────────────────┘
                    ▼
┌───────────────────────────────────────────────────────────┐
│  WEB APPLICATION                                          │
│  webapp/backend/  → FastAPI on Cloud Run (GCS → /tmp)     │
│  webapp/frontend/ → React + Recharts on Vercel            │
│  webapp/pipeline/ → Daily ETL (Cloud Run Jobs)            │
└───────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
capstone/
├── reddit/              # Reddit: collection → preprocessing → NLP analysis
│   ├── data-collection/ #   Arctic Shift API scraper
│   ├── preprocessing/   #   Bot filtering, text cleaning
│   ├── analysis/        #   Sentiment, BERTopic, HDBSCAN clustering
│   └── eda/             #   Exploratory data analysis
├── gdelt/               # GDELT: BigQuery export + news scraping + EDA
├── tiktok/              # TikTok: Research API + preprocessing
├── graphrag/            # GraphRAG knowledge graph (Ollama + LanceDB)
├── webapp/
│   ├── backend/         # FastAPI REST API (Cloud Run)
│   ├── frontend/        # React dashboard (Vercel)
│   └── pipeline/        # Daily ETL job (Cloud Run Jobs)
├── data/                # Raw data files (gitignored)
└── docs/                # Documentation
```

---

## Presenter 1: Project Overview + Data Pipeline (2–3 min)

**Speaker:** _______________

### What to Cover

1. **Problem Statement** (30 sec)
   - How do online narratives about Venezuela-US relations differ across platforms, communities, and time?
   - 13-year longitudinal study spanning 6 major crisis events

2. **Data Collection Architecture** (1 min)
   - Show: `reddit/data-collection/main.py` — entry point for Arctic Shift API
   - Show: `reddit/data-collection/scripts/collectors.py` — multi-subreddit fetching with rate limiting
   - Key numbers: 11 subreddits, 101K submissions, 431K comments

3. **Preprocessing Pipeline** (1 min)
   - Show: `reddit/preprocessing/filters.py` — bot detection, deleted-content removal
   - Show: `reddit/preprocessing/text_cleaner.py` — URL/emoji stripping, unicode normalization
   - Result: 533K → 426K data points (20% filtered)

### Key Files to Open

| File | What to Show |
|------|-------------|
| `reddit/data-collection/scripts/collectors.py` | `fetch_submissions()` — Arctic Shift API calls |
| `reddit/data-collection/scripts/config.py` | `SUBREDDITS` list — all 11 target subreddits |
| `reddit/preprocessing/filters.py` | Bot detection heuristics |
| `reddit/preprocessing/text_cleaner.py` | Text normalization pipeline |

### Talking Points
- Arctic Shift API = no-auth access to full Reddit archive (2005–present)
- Preprocessing removes ~20% of noise (bots, deleted, duplicates)
- All outputs as Parquet for efficient downstream analysis

---

## Presenter 2: NLP Analysis Algorithms (2–3 min)

**Speaker:** _______________

### What to Cover

1. **Sentiment Analysis — RoBERTa** (1 min)
   - Show: `reddit/analysis/sentiment/roberta_analyzer.py`
   - Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` (124M params)
   - Output: score (-1 to +1), label, confidence per document
   - Aggregation: by month, by subreddit, by subreddit×month

2. **Topic Modeling — BERTopic** (1 min)
   - Show: `reddit/analysis/topic/bertopic_model.py`
   - Pipeline: Sentence-BERT → UMAP (384→5 dim) → HDBSCAN → c-TF-IDF
   - Result: 15 discovered topics + outlier topic -1

3. **Semantic Clustering — HDBSCAN + UMAP** (30 sec)
   - Show: `reddit/analysis/clustering/cluster.py`
   - Separate from BERTopic — finer-grained, 3,406 clusters
   - 2D UMAP visualization with keyword labels

### Key Files to Open

| File | What to Show |
|------|-------------|
| `reddit/analysis/main.py` | `run_sentiment_analysis()` — pipeline orchestration |
| `reddit/analysis/sentiment/roberta_analyzer.py` | `analyze_batch()` — RoBERTa inference with batching |
| `reddit/analysis/topic/bertopic_model.py` | `fit_topics()` — BERTopic training |
| `reddit/analysis/clustering/cluster.py` | `TemporalClusterer` class — HDBSCAN with temporal tracking |
| `reddit/analysis/config.py` | `AnalysisConfig` — model names, hyperparameters |

### Talking Points
- Three complementary NLP approaches: sentiment (what?), topics (about what?), clusters (how grouped?)
- All use Sentence-BERT embeddings (384-dim) as shared representation
- Clustering tracks temporal evolution — how discourse clusters shift over crisis events

---

## Presenter 3: Dashboard + Deployment + Next Steps (2–3 min)

**Speaker:** _______________

### What to Cover

1. **Backend API** (30 sec)
   - Show: `webapp/backend/main.py` — FastAPI app with GCS lifespan handler
   - Endpoints: `/api/sentiment/*`, `/api/topics/*`, `/api/clusters/*`
   - Data loaded from GCS on startup → served from memory

2. **Frontend Dashboard** (1 min)
   - **Live demo**: open the [deployed dashboard](https://capstone-dashboard.vercel.app)
   - 4 pages: Dashboard, Sentiment, Topics, Clusters
   - Show: interactive time-range filtering, subreddit comparison, cluster scatter plot

3. **Daily ETL Pipeline** (30 sec)
   - Show: `webapp/pipeline/main.py` — 5-stage pipeline (collect → scrape → preprocess → analyze → update)
   - Runs daily via Cloud Scheduler → Cloud Run Jobs
   - Incrementally updates analysis with new data

4. **Deployment Architecture** (30 sec)
   - Backend: Cloud Run (auto-scaling, ~29MB data from GCS)
   - Frontend: Vercel (CDN, auto-deploy from git)
   - Pipeline: Cloud Run Jobs + Cloud Scheduler (daily 06:00 UTC)

### Key Files to Open

| File | What to Show |
|------|-------------|
| `webapp/backend/main.py` | `lifespan()` — GCS download on cold start |
| `webapp/backend/services/data_service.py` | `download_from_gcs()` — GCS → local cache |
| `webapp/backend/routers/sentiment.py` | API endpoint example |
| `webapp/frontend/src/pages/Dashboard.tsx` | Main dashboard page |
| `webapp/frontend/src/lib/api.ts` | API client with endpoint definitions |
| `webapp/pipeline/main.py` | Daily ETL orchestrator |

### Talking Points
- Backend is stateless — data downloaded fresh from GCS on each cold start
- Frontend uses Recharts for all visualizations, TailwindCSS for styling
- Pipeline designed for incremental updates (not full reprocessing)

---

## Key Crisis Events (Reference Timeline)

| Date | Event | Expected Signal |
|------|-------|-----------------|
| 2013-04 | Maduro Inauguration | Volume spike, polarized sentiment |
| 2014-02 | Venezuelan Protests | High negative sentiment |
| 2017-08 | Trump Sanctions | Political subreddit activation |
| 2019-01 | Guaido Recognition | Largest volume peak, topic shift |
| 2024-07 | 2024 Election Crisis | Cross-platform activity |
| 2026-01 | Maduro Captured | Most recent event |

---

## Tech Stack Summary

| Layer | Technology |
|-------|------------|
| Data Collection | Arctic Shift API, BigQuery, TikTok Research API |
| Preprocessing | pandas, regex, Parquet |
| Sentiment | RoBERTa (`twitter-roberta-base-sentiment-latest`) |
| Topics | BERTopic + Sentence-BERT + UMAP + HDBSCAN |
| Clustering | HDBSCAN, UMAP, TF-IDF |
| Knowledge Graph | Microsoft GraphRAG, Ollama (llama3.1:8b), LanceDB |
| Backend | FastAPI, Google Cloud Storage, Cloud Run |
| Frontend | React 19, TypeScript, Vite, TailwindCSS, Recharts |
| Pipeline | Cloud Run Jobs, Cloud Scheduler |
| Deployment | GCP Cloud Run, Vercel |
