# Venezuela-US Multiplatform Narrative Analysis

End-to-end pipeline for collecting, analyzing, and visualizing online discourse about Venezuela-US relations during the Maduro era (2013–2026). Covers **Reddit**, **GDELT News**, and **TikTok** with AI-powered reports and interactive chat.

## Live Demo

| Service | URL | Status |
|---------|-----|--------|
| **Dashboard** | [capstone-dashboard-iota.vercel.app](https://capstone-dashboard-iota.vercel.app) | Deployed |
| **API** | [backend-api-318799600047.us-central1.run.app](https://backend-api-318799600047.us-central1.run.app/docs) | Deployed |
| **API Health** | [/health](https://backend-api-318799600047.us-central1.run.app/health) | `{"status":"ok"}` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                               │
│  Reddit (Arctic Shift)  │  GDELT (BigQuery)  │  TikTok (API)   │
└────────┬────────────────┴──────────┬─────────┴────────┬────────┘
         │                           │                  │
         ▼                           ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  COLLECTION → PREPROCESSING → ANALYSIS                          │
│  Arctic Shift API    Filter bots     Sentiment (RoBERTa)        │
│  BigQuery            Clean text      Topics (BERTopic)          │
│  TikTok Research     Normalize       Clusters (HDBSCAN+UMAP)   │
│  + Playwright                        TikTok-specific analytics  │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────────┐
│  KNOWLEDGE GRAPH (GraphRAG)  │ │  WEB APPLICATION               │
│  Microsoft GraphRAG          │ │  FastAPI (Cloud Run)  ◄──►     │
│  Ollama (llama3.1:8b)        │ │  React + Recharts (Vercel)     │
│  LanceDB vector store        │ │  Gemini LLM (Reports + Chat)   │
└──────────────────────────────┘ │  GCS bucket (data storage)     │
                                 └────────────────────────────────┘
```

## Data Statistics

### Reddit

| Metric | Raw Data | After Preprocessing |
|--------|----------|---------------------|
| **Time Period** | 2013-01 ~ 2026-02 | — |
| **Total Submissions** | 101,960 | 86,809 |
| **Total Comments** | 431,981 | 339,626 |
| **Total Data Points** | 533,941 | 426,435 |
| **Unique Submission Authors** | 26,363 | 23,497 |
| **Unique Comment Authors** | 129,740 | 119,021 |
| **Subreddits** | 11 | 11 |

**Target Subreddits (11):**

| Category | Subreddits |
|----------|------------|
| Venezuela-focused | r/venezuela, r/vzla |
| US Politics | r/politics, r/news, r/worldnews |
| Ideological | r/Conservative, r/neoliberal, r/socialism, r/Libertarian |
| Regional | r/LatinAmerica, r/geopolitics |

### GDELT News

| Metric | Value |
|--------|-------|
| **Data Period** | 2013-01 ~ 2026-01 |
| **Total Events** | 292,566 |
| **Successful Scrapes** | 211,071 (72.1% success rate) |
| **Unique URLs** | 105,095 |
| **Avg Goldstein Scale** | 0.04 |
| **Avg Tone** | -3.08 |
| **Initiator Split (VEN / USA)** | 136,614 / 155,952 |

### TikTok

| Metric | Value |
|--------|-------|
| **Data Period** | 2016-09 ~ 2026-03 |
| **Total Videos** | 3,399 |
| **Total Comments** | 1,309 (Playwright browser collection) |
| **Total Documents** | 4,656 |
| **Unique Creators** | 1,013 |
| **Unique Hashtags** | 3,147 |
| **Top Region** | Venezuela (50.6%), Spain (12.5%), USA (11.7%) |
| **Collection Method** | TikTok Research API (videos) + Playwright (comments) |

## Analysis Pipeline

### Sentiment Analysis — RoBERTa
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (124M params)
- **Output:** sentiment score (-1 to +1), label, confidence per document
- **Applied to:** Reddit posts/comments, GDELT news articles, TikTok videos/comments
- **Aggregation:** by month, by subreddit/source/creator, cross-tabulated

### Topic Modeling — BERTopic
- **Embedding:** Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim)
- **Reduction:** UMAP (384→5 dim)
- **Clustering:** HDBSCAN (min_cluster_size=50)
- **Stopwords:** Combined EN+ES (504 words) for bilingual topic representation
- **Monthly fitting:** Independent BERTopic models per month for all 3 platforms
- **Result:** Reddit 647 topics, News 1,169 topics, TikTok 19 topics

### Semantic Clustering — HDBSCAN + UMAP
- **Separate from BERTopic** — finer-grained global clustering
- **Result:** 3,406 clusters with keywords, temporal evolution, and 2D UMAP visualization

### TikTok-Specific Analytics
- **Hashtag trends:** 3,147 unique hashtags with frequency and sentiment over time
- **Engagement metrics:** Views, likes, shares, comments aggregated monthly
- **Region distribution:** Video origin by country code
- **Voice-to-text:** Auto-caption text included in analysis

### Knowledge Graph — Microsoft GraphRAG
- **Framework:** [Microsoft GraphRAG](https://github.com/microsoft/graphrag) with local LLM
- **LLM:** Ollama `llama3.1:8b` (entity extraction, community reports)
- **Embedding:** `nomic-embed-text` (768-dim, via Ollama)
- **Vector Store:** LanceDB
- **Input:** 200 high-engagement Reddit threads from the 2019 Guaido Recognition Crisis

### LLM Features — Gemini via Vertex AI
- **Intelligence Reports:** AI-generated cross-platform analysis for any time period
- **Data Chat:** Natural language Q&A over all platform data with auto context retrieval
- **Model:** `gemini-2.0-flash` via Vertex AI (project: `theta-bliss-486220-s1`)

## Web Dashboard

### Backend — FastAPI on Cloud Run

| Endpoint | Description |
|----------|-------------|
| `GET /api/overview/stats` | Summary statistics (platform=reddit\|news\|tiktok) |
| `GET /api/sentiment/by-month` | Monthly sentiment trends |
| `GET /api/sentiment/by-subreddit` | Per-subreddit/source/creator sentiment |
| `GET /api/sentiment/by-subreddit-month` | Subreddit×month matrix |
| `GET /api/sentiment/boxplot` | Box plot statistics |
| `GET /api/topics/info` | Topic list with keywords |
| `GET /api/topics/over-time` | Topic temporal evolution |
| `GET /api/topics/monthly-fitted` | Independent monthly BERTopic results |
| `GET /api/clusters/summaries` | Cluster summaries |
| `GET /api/clusters/scatter` | UMAP scatter (30K points) |
| `GET /api/clusters/temporal` | Cluster temporal changes |
| `GET /api/tiktok/hashtags` | TikTok hashtag trends |
| `GET /api/tiktok/engagement` | TikTok engagement metrics |
| `GET /api/tiktok/regions` | TikTok region distribution |
| `GET /api/reports/generate` | AI-generated intelligence report |
| `POST /api/chat` | Natural language data chat |

### Frontend — React on Vercel

| Page | Visualizations |
|------|----------------|
| **Dashboard** | 3-platform StatCards, Cross-platform Sentiment Timeline, Volume Distribution (Reddit/News/TikTok) |
| **Sentiment** | Multi-source comparison, Composite Timeline, Box Plots |
| **Topics** | 3-column monthly BERTopic (Reddit/News/TikTok), Topic Evolution AreaCharts |
| **TikTok** | Hashtag trends, Engagement metrics, Region distribution, Monthly topics |
| **Clusters** | UMAP scatter (30K points), Top 20 clusters, Temporal bar charts |
| **Reports** | AI intelligence report generation with period selector, platform stats cards |
| **Chat** | Natural language Q&A with suggested questions, conversation history |

## Deployment

### Backend (GCP Cloud Run)

```bash
cd webapp/backend
gcloud builds submit --config=cloudbuild.yaml --project=theta-bliss-486220-s1
```

### Frontend (Vercel)

```bash
cd webapp/frontend
npm run build && npx vercel --prod
```

**Environment variable:** `VITE_API_URL` → Cloud Run backend URL (set in `.env.production`)

## Key Crisis Periods

| Date | Event |
|------|-------|
| 2013-04 | Maduro Inauguration |
| 2014-02 | Venezuelan Protests |
| 2017-08 | Trump Administration Sanctions |
| 2019-01 | Guaido Recognition Crisis |
| 2024-07 | 2024 Election Crisis |
| 2026-01 | Maduro Captured by US Forces |

## Project Structure

```
capstone/
├── README.md
├── reddit/                                # Reddit data + analysis
│   ├── data-collection/                   # Arctic Shift API collection
│   ├── preprocessing/                     # Text cleaning pipeline
│   ├── analysis/                          # Sentiment, topics, clustering
│   │   ├── outputs/                       # Reddit analysis results
│   │   ├── outputs_news/                  # GDELT analysis results
│   │   └── outputs_tiktok/                # TikTok analysis results
│   └── eda/                               # Exploratory data analysis
├── gdelt/                                 # GDELT news data
│   ├── data-collection/                   # BigQuery export + web scraping
│   ├── preprocessing/                     # Text relevance filtering
│   └── analysis/                          # EDA + analyze_gdelt.py
├── tiktok/                                # TikTok data pipeline
│   ├── data-collection/                   # TikTok Research API + Playwright comments
│   ├── analysis/                          # run_analysis.py (sentiment + topics + specific)
│   └── preprocessing/                     # Video/comment filtering
├── comment_scrape/                        # Playwright-based TikTok comment scraper
├── graphrag/                              # GraphRAG knowledge graph instance
├── webapp/
│   ├── backend/                           # FastAPI + GCS + Gemini (Cloud Run)
│   │   ├── routers/                       # overview, sentiment, topics, clusters, tiktok, reports, chat
│   │   └── services/                      # data_service, llm_service
│   ├── frontend/                          # React + Vite + Recharts + TailwindCSS (Vercel)
│   │   └── src/pages/                     # Dashboard, Sentiment, Topics, TikTok, Clusters, Reports, Chat
│   └── pipeline/                          # Daily ETL (Cloud Run Jobs)
└── data/                                  # Raw data files (gitignored)
```

## Local Development

```bash
# Backend
cd webapp
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend
cd webapp/frontend
npm install
npm run dev    # → http://localhost:5173 (proxies /api to localhost:8000)
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| Data Collection | Arctic Shift API, BigQuery, TikTok Research API, Playwright |
| Preprocessing | pandas, regex, Parquet |
| Sentiment | RoBERTa (`twitter-roberta-base-sentiment-latest`) |
| Topic Modeling | BERTopic, Sentence-BERT, UMAP, HDBSCAN |
| Clustering | HDBSCAN, UMAP, TF-IDF |
| Knowledge Graph | Microsoft GraphRAG, Ollama (llama3.1:8b), LanceDB |
| LLM | Gemini 2.0 Flash via Vertex AI |
| Backend | FastAPI, Google Cloud Storage, Cloud Run |
| Frontend | React 19, TypeScript, Vite, TailwindCSS, Recharts |
| Deployment | GCP Cloud Run, Vercel, Cloud Build |

## Data Access

**Full dataset on Google Drive:** [Download Here](https://drive.google.com/drive/folders/1MV2-ktL-OsiT4cDmoGWwlmt9l-OY_j-U?usp=sharing)

## License

For academic research purposes only.

## Authors

Hunjun Shin, Rich Goodier, Ameir El Ouadi
