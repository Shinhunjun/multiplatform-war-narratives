# Venezuela-US Multiplatform Narrative Analysis

End-to-end pipeline for collecting, analyzing, and visualizing online discourse about Venezuela-US relations during the Maduro era (2013–2026). Covers Reddit, GDELT news, and TikTok across 11 subreddits and multiple platforms.

## Live Demo

| Service | URL | Status |
|---------|-----|--------|
| **Dashboard** | [frontend-shinhunjuns-projects.vercel.app](https://frontend-a26f9qr47-shinhunjuns-projects.vercel.app) | Deployed |
| **API** | [backend-api-762303020827.us-central1.run.app](https://backend-api-762303020827.us-central1.run.app/docs) | Deployed |
| **API Health** | [/health](https://backend-api-762303020827.us-central1.run.app/health) | `{"status":"ok"}` |

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
│  TikTok SDK          Normalize       Clusters (HDBSCAN+UMAP)   │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────────┐
│  KNOWLEDGE GRAPH (GraphRAG)  │ │  WEB APPLICATION               │
│  Microsoft GraphRAG          │ │  FastAPI (Cloud Run)  ◄──►     │
│  Ollama (llama3.1:8b)        │ │  React + Recharts (Vercel)     │
│  LanceDB vector store        │ │  GCS bucket (29MB)             │
└──────────────────────────────┘ └────────────────────────────────┘
```

## Data Statistics

| Metric | Raw Data | After Preprocessing |
|--------|----------|---------------------|
| **Time Period** | 2013-01 ~ 2026-02 | — |
| **Total Submissions** | 101,960 | 86,809 |
| **Total Comments** | 431,981 | 339,626 |
| **Total Data Points** | 533,941 | 426,435 |
| **Unique Submission Authors** | 26,363 | 23,497 |
| **Unique Comment Authors** | 129,740 | 119,021 |
| **Subreddits** | 11 | 11 |

### Target Subreddits (11)

| Category | Subreddits |
|----------|------------|
| Venezuela-focused | r/venezuela, r/vzla |
| US Politics | r/politics, r/news, r/worldnews |
| Ideological | r/Conservative, r/neoliberal, r/socialism, r/Libertarian |
| Regional | r/LatinAmerica, r/geopolitics |

## Analysis Pipeline

### Sentiment Analysis — RoBERTa
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (124M params)
- **Output:** sentiment score (-1 to +1), label, confidence per document
- **Aggregation:** by month, by subreddit, by subreddit×month

### Topic Modeling — BERTopic
- **Embedding:** Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim)
- **Reduction:** UMAP (384→5 dim)
- **Clustering:** HDBSCAN (min_cluster_size=50)
- **Result:** 15 topics discovered (+ outlier topic -1)

### Semantic Clustering — HDBSCAN + UMAP
- **Separate from BERTopic** — finer-grained global clustering
- **Result:** 3,406 clusters with keywords, temporal evolution, and 2D UMAP visualization

### Knowledge Graph — Microsoft GraphRAG
- **Framework:** [Microsoft GraphRAG](https://github.com/microsoft/graphrag) with local LLM
- **LLM:** Ollama `llama3.1:8b` (entity extraction, community reports)
- **Embedding:** `nomic-embed-text` (768-dim, via Ollama)
- **Vector Store:** LanceDB
- **Entity Types:** PERSON, ORGANIZATION, EVENT, POLICY, LOCATION, TOPIC
- **Input:** 200 high-engagement Reddit threads from the 2019 Guaido Recognition Crisis
- **Query Modes:** Local search, Global search, Drift search

## Web Dashboard

### Backend — FastAPI on Cloud Run

| Endpoint | Description |
|----------|-------------|
| `GET /api/overview/stats` | Summary statistics |
| `GET /api/sentiment/by-month` | Monthly sentiment trends |
| `GET /api/sentiment/by-subreddit` | Per-subreddit sentiment |
| `GET /api/sentiment/by-subreddit-month` | Subreddit×month matrix |
| `GET /api/topics/info` | Topic list with keywords |
| `GET /api/topics/over-time` | Topic temporal evolution |
| `GET /api/topics/by-subreddit` | Topic distribution by subreddit |
| `GET /api/clusters/summaries` | Cluster summaries |
| `GET /api/clusters/keywords` | Cluster keywords |
| `GET /api/clusters/temporal` | Cluster temporal changes |

### Frontend — React on Vercel

| Page | Visualizations |
|------|----------------|
| **Dashboard** | StatCards, Sentiment Timeline, Subreddit BarChart, Volume Distribution |
| **Sentiment** | Multi-subreddit comparison, Composite Timeline, Overview Table |
| **Topics** | Topic distribution, Top 8 Stacked AreaChart, Detail Table |
| **Clusters** | Top 20 clusters (sentiment-colored), Custom Tooltips, Detail Table |

## Deployment

### Backend (GCP Cloud Run)

```bash
# 1. Upload data to GCS (~29MB)
bash webapp/backend/upload_data_to_gcs.sh

# 2. Deploy to Cloud Run
gcloud run deploy backend-api \
  --source webapp/backend \
  --project mlops-compute-lab \
  --region us-central1 \
  --set-env-vars GCS_BUCKET=mlops-compute-lab-analysis-data,DATA_DIR=/tmp/analysis_data \
  --allow-unauthenticated \
  --memory 1Gi
```

The backend downloads data from GCS on startup (`download_from_gcs()` in lifespan handler).

### Frontend (Vercel)

```bash
cd webapp/frontend
vercel --prod
```

**Environment variable:** `VITE_API_URL` → Cloud Run backend URL (set in `.env.production`)

### Daily ETL Pipeline (Cloud Run Jobs)

```bash
# Deploy pipeline job
gcloud builds submit --config webapp/pipeline/deploy/cloudbuild.yaml .

# Setup Cloud Scheduler (daily 06:00 UTC)
./webapp/pipeline/deploy/setup_scheduler.sh mlops-compute-lab us-central1
```

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
├── README.md                              # This file
├── PIPELINE.md                            # Detailed pipeline & algorithm docs
├── venezuela-us-reddit-discourse/         # Reddit data + analysis
│   ├── data-collection/                   # Arctic Shift API collection
│   ├── preprocessing/                     # Text cleaning pipeline
│   ├── analysis/                          # Sentiment, topics, clustering
│   │   └── outputs/                       # CSV/Parquet/npy results
│   └── EDA/                               # Exploratory data analysis
├── venezuela-tiktok-discourse/            # TikTok collection pipeline
├── graphrag/                              # Microsoft GraphRAG framework (source)
├── venezuela-graphrag/                    # GraphRAG knowledge graph for Venezuela data
│   ├── settings.yaml                      # GraphRAG config (Ollama LLM + embeddings)
│   ├── input/                             # 200 Reddit thread documents (.txt)
│   ├── output/                            # Indexed entities, communities, text units
│   ├── prompts/                           # Custom extraction & search prompts
│   └── cache/                             # LLM response cache
├── gdelt/                                 # GDELT BigQuery export data
├── webapp/
│   ├── backend/                           # FastAPI + GCS data loading
│   │   ├── main.py                        # App entry, CORS, lifespan
│   │   ├── routers/                       # API route handlers
│   │   ├── services/data_service.py       # Data loading + GCS download
│   │   ├── Dockerfile                     # Cloud Run container
│   │   ├── cloudbuild.yaml                # Cloud Build config
│   │   └── requirements.txt
│   ├── frontend/                          # React + Vite + TailwindCSS
│   │   ├── src/
│   │   │   ├── pages/                     # Dashboard, Sentiment, Topics, Clusters
│   │   │   ├── components/                # Layout, charts
│   │   │   └── lib/api.ts                 # API client
│   │   ├── vercel.json                    # Vercel SPA config
│   │   └── .env.production                # VITE_API_URL
│   └── pipeline/                          # Daily ETL (Cloud Run Jobs)
│       ├── collectors/                    # Reddit, GDELT, news scraper
│       ├── processing/                    # Preprocessing + analysis
│       └── deploy/                        # Cloud Build + Scheduler
└── pipeline_data/                         # Pipeline runtime data
```

## Local Development

```bash
# Backend
cd webapp/backend
pip install -r requirements.txt
uvicorn webapp.backend.main:app --reload

# Frontend
cd webapp/frontend
npm install
npm run dev    # → http://localhost:5173 (proxies /api to localhost:8000)
```

## Data Access

**Full dataset on Google Drive:** [Download Here](https://drive.google.com/drive/folders/1MV2-ktL-OsiT4cDmoGWwlmt9l-OY_j-U?usp=sharing)

## Technical Stack

| Component | Technology |
|-----------|------------|
| Data Collection | Arctic Shift API, BigQuery, TikTok Research API |
| Preprocessing | pandas, regex, Parquet |
| Sentiment | RoBERTa (`twitter-roberta-base-sentiment-latest`) |
| Topic Modeling | BERTopic, Sentence-BERT, UMAP, HDBSCAN |
| Clustering | HDBSCAN, UMAP, TF-IDF |
| Knowledge Graph | Microsoft GraphRAG, Ollama (llama3.1:8b), LanceDB |
| Backend | FastAPI, Google Cloud Storage, Cloud Run |
| Frontend | React 19, TypeScript, Vite, TailwindCSS, Recharts |
| Deployment | GCP Cloud Run, Vercel, Cloud Scheduler |

## License

For academic research purposes only.

## Author

Hunjun Shin
