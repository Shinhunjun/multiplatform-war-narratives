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

Reddit 데이터는 [Arctic Shift](https://arctic-shift.photon-reddit.com/) API를 통해 수집합니다. 이 API는 Reddit의 전체 아카이브(2005~현재)에 대한 검색을 제공하며, **인증 없이** 사용 가능합니다.

**API Endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /api/posts/search` | 서브레딧/키워드/날짜 기반 submission 검색 |
| `GET /api/comments/search` | 댓글 검색 (link_id로 특정 게시글 필터) |

**수집 전략:**

- **Venezuela 전용 서브레딧** (`r/venezuela`, `r/vzla`): 키워드 필터 없이 전체 수집
- **일반 서브레딧** (`r/politics`, `r/worldnews` 등): 10개 키워드 쿼리로 멀티-쿼리 검색 후 ID 기반 중복 제거

**검색 키워드 (10개):**
```
Venezuela, Maduro, Venezuela US, Venezuela sanctions,
Guaido, Venezuelan crisis, Venezuela oil, Caracas,
Venezuela election, Venezuela humanitarian
```

**모니터링 서브레딧 (11개):**

| Category | Subreddits |
|----------|------------|
| Venezuela-focused | `r/venezuela`, `r/vzla` |
| US Mainstream | `r/politics`, `r/news`, `r/worldnews` |
| US Conservative | `r/Conservative`, `r/Libertarian` |
| US Progressive | `r/neoliberal`, `r/socialism` |
| Regional/Academic | `r/LatinAmerica`, `r/geopolitics` |

**Rate Limiting:** 1초 간격, 429 응답 시 exponential backoff (10s, 20s, 30s...)

**데이터 규모 (역사적 수집):**

| Metric | Count |
|--------|-------|
| Submissions | 101,960 |
| Comments | 431,981 |
| Period | 2013-01 ~ 2026-01 |
| After Preprocessing | 426,435 |

### 2.2 GDELT — BigQuery

[GDELT Project](https://www.gdeltproject.org/) 의 BigQuery 공개 데이터셋에서 Venezuela 관련 뉴스 이벤트를 수집합니다.

**테이블:**

| Table | Contents |
|-------|----------|
| `gdelt-bq.gdeltv2.events` | 글로벌 이벤트 (Actor, EventCode, GoldsteinScale, AvgTone) |
| `gdelt-bq.gdeltv2.gkg` | Global Knowledge Graph (themes, persons, organizations, tone) |

**쿼리 조건:**
- `Actor1CountryCode = 'VEN'` 또는 `Actor2CountryCode = 'VEN'`
- 또는 키워드 매칭: `venezuela`, `maduro`, `caracas`, `guaido`, `pdvsa`, `citgo`

**추출 후 News Scraper가 GDELT에서 참조하는 기사 URL을 방문하여 본문 텍스트 추출.**

### 2.3 TikTok — Research API

TikTok Research API를 통해 비디오 및 댓글을 수집합니다 (OAuth2 인증, 일일 할당량 제한).

- **일일 제한:** 1,000 requests / 100,000 records
- **쿼리 윈도우:** 최대 30일
- **수집 대상:** 해시태그 (`#venezuela`, `#maduro` 등) + 키워드 검색

---

## 3. Preprocessing Pipeline

수집된 원시 데이터를 분석에 적합한 형태로 변환하는 과정입니다.

### 3.1 필터링

```
Raw Data
  │
  ├── [removed] 콘텐츠 → 제거
  ├── [deleted] 콘텐츠 → 제거
  ├── Bot 계정 → 제거
  │     AutoModerator, autotldr, empleadoEstatalBot,
  │     RemindMeBot, WikiTextBot, TotesMessenger,
  │     RepostSleuthBot, SaveVideo, VisualMod
  ├── 5단어 미만 → 제거
  └── 미디어 전용 (URL만 있는) → 제거
```

### 3.2 텍스트 클리닝

| Step | Example |
|------|---------|
| URL 제거 | `Check https://example.com` → `Check` |
| Markdown 제거 | `**bold** text` → `bold text` |
| Reddit 인용 제거 | `> quoted text` → (removed) |
| 코드 블록 제거 | `` `code` `` → (removed) |
| Edit 마커 제거 | `Edit: added more` → `added more` |
| 공백 정규화 | multiple spaces → single space |

### 3.3 출력 스키마

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

**출력 포맷:** Apache Parquet (columnar, compressed)

---

## 4. Analysis Algorithms

### 4.1 Sentiment Analysis — RoBERTa

소셜 미디어 텍스트의 감성을 분석합니다.

**모델:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

이 모델은 ~124M 트윗으로 사전학습된 RoBERTa-base를 감성 분석 태스크로 fine-tuning한 것입니다. 소셜 미디어 특유의 비공식적 언어, 약어, 이모지 등을 잘 처리합니다.

**작동 방식:**

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

**출력:**

| Field | Type | Description |
|-------|------|-------------|
| `sentiment_label` | str | `"positive"`, `"negative"`, `"neutral"` |
| `sentiment_confidence` | float | 0.0 ~ 1.0 (모델 확신도) |
| `sentiment_score` | float | -1.0 ~ +1.0 (방향 × 확신도) |

**집계 방식:**

- **월별 (by_month):** 각 YYYY-MM 기간의 평균 감성 점수, positive/negative 비율
- **서브레딧별 (by_subreddit):** 커뮤니티 간 감성 비교
- **서브레딧×월 (by_subreddit_month):** 시간에 따른 커뮤니티별 감성 변화 추적

**GPU 가속:** Apple Silicon (MPS), NVIDIA (CUDA), 또는 CPU fallback. 배치 크기 64.

---

### 4.2 Topic Modeling — BERTopic

문서 코퍼스에서 의미적으로 일관된 토픽을 자동 발견합니다.

**BERTopic 파이프라인:**

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

**각 단계의 역할:**

| Step | Algorithm | Purpose |
|------|-----------|---------|
| Embedding | Sentence-BERT (all-MiniLM-L6-v2) | 텍스트를 384차원 밀집 벡터로 변환. 의미적으로 유사한 텍스트가 가까운 벡터를 가짐 |
| Reduction | UMAP | 고차원 임베딩을 5차원으로 축소하면서 지역적 구조 보존. 클러스터링 효율성 향상 |
| Clustering | HDBSCAN | 밀도 기반 계층적 클러스터링. 노이즈(topic -1)를 자연스럽게 분리, 클러스터 수 자동 결정 |
| Representation | c-TF-IDF + KeyBERT | 각 토픽을 대표하는 키워드 추출. c-TF-IDF로 클래스별 중요 단어 찾고 KeyBERT로 의미적 정제 |

**출력 파일:**

| File | Contents |
|------|----------|
| `topic_info.csv` | 토픽 ID, 이름(키워드), 문서 수 |
| `topic_assignments.parquet` | 각 문서 → 토픽 매핑 (ID, probability) |
| `topics_over_time.csv` | 시간별 토픽 빈도 변화 |
| `topics_by_subreddit.csv` | 서브레딧별 토픽 분포 |
| `document_embeddings.npy` | 전체 문서 384-dim 임베딩 (648MB) |
| `bertopic_model` | 학습된 BERTopic 모델 (2.9GB) |

**Incremental Topic Assignment (일일 파이프라인):**

새 문서가 들어오면 BERTopic 모델을 재학습하지 않고, **embedding cosine similarity** 기반으로 기존 토픽에 할당합니다:

1. 기존 `document_embeddings.npy` + `topic_assignments.parquet`에서 **토픽별 centroid** 계산
2. 새 문서를 Sentence-BERT로 임베딩
3. 각 새 문서와 모든 centroid 간 cosine similarity 계산
4. 가장 유사한 토픽에 할당 (threshold > 0.25, 이하는 outlier -1)

---

### 4.3 Clustering — HDBSCAN + UMAP

문서 임베딩 공간에서 의미적 클러스터를 발견하고 시각화합니다.

**BERTopic의 토픽 모델링과 별도로** 수행되는 글로벌 클러스터링으로, 더 세밀한 하위 그룹을 발견합니다.

**파이프라인:**

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

**HDBSCAN 파라미터 설명:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `min_cluster_size` | 50 | 클러스터로 인정되려면 최소 50개 문서 필요 |
| `min_samples` | 10 | core point가 되려면 반경 내 최소 10개 이웃 필요 |
| `metric` | euclidean | UMAP 축소 후 유클리드 거리 사용 |
| `cluster_selection_method` | eom | Excess of Mass — 다양한 밀도의 클러스터를 잘 감지 |

**클러스터 분석 출력:**

| File | Contents |
|------|----------|
| `cluster_assignments.parquet` | 문서별 cluster_id, probability, UMAP 좌표 |
| `cluster_summaries.csv` | 클러스터별 통계 (문서 수, 주요 서브레딧, 감성, 기간) |
| `cluster_keywords.csv` | TF-IDF 기반 클러스터별 상위 키워드 |
| `temporal_clusters.csv` | 시간별 클러스터 크기 변화 |
| `embeddings.npy` | 전체 384-dim 임베딩 |
| `embeddings_2d.npy` | 시각화용 2D UMAP 좌표 |

**시각화:**

- **UMAP Scatter Plot:** 전체 문서를 2D 평면에 투사, 클러스터/서브레딧 별 색상
- **Animated UMAP:** 시간에 따른 클러스터 변화 GIF 애니메이션
- **River Plot:** Sankey 스타일 클러스터 흐름 시각화
- **Heatmap:** 서브레딧 × 클러스터 매트릭스

---

## 5. Daily ETL Pipeline

매일 자동으로 실행되는 데이터 수집 → 전처리 → 분석 → 업데이트 파이프라인입니다.

### 5.1 아키텍처

```
┌─────────────────────┐
│  Cloud Scheduler     │  Cron: 0 6 * * * (매일 06:00 UTC)
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

### 5.2 CLI 사용법

```bash
# 전체 파이프라인 실행 (오늘 날짜 기준)
python -m webapp.pipeline.main

# Reddit만 수집
python -m webapp.pipeline.main --reddit-only

# GDELT만 수집
python -m webapp.pipeline.main --gdelt-only

# 특정 날짜 지정
python -m webapp.pipeline.main --date 2026-02-14

# 특정 단계만 실행
python -m webapp.pipeline.main --stage collect
python -m webapp.pipeline.main --stage preprocess
python -m webapp.pipeline.main --stage analyze

# Lookback 기간 변경 (기본 1일)
python -m webapp.pipeline.main --lookback 7
```

### 5.3 Incremental Update 전략

파이프라인은 전체 재분석 없이 **기존 결과에 새 데이터를 병합**합니다:

**Sentiment:**
1. 새 문서에 RoBERTa 감성 분석 실행
2. 기존 `sentiment_by_month.csv` 로드
3. 새 월별 집계와 concat → 겹치는 월은 재집계
4. 업데이트된 CSV 저장

**Topics:**
1. 기존 `document_embeddings.npy` + `topic_assignments.parquet`에서 토픽 centroid 계산
2. 새 문서를 Sentence-BERT로 임베딩
3. Cosine similarity로 가장 가까운 토픽에 할당 (threshold 0.25)
4. `topics_over_time.csv`, `topics_by_subreddit.csv` 업데이트

### 5.4 파이프라인 실행 결과 예시

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

분석 결과 CSV/Parquet를 REST API로 제공합니다.

**Tech Stack:** Python 3.11, FastAPI, pandas, LRU cache

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/overview/stats` | GET | 전체 통계 (문서 수, 서브레딧 수, 토픽 수, 평균 감성) |
| `/api/sentiment/by-month` | GET | 월별 감성 추이 (start, end 파라미터) |
| `/api/sentiment/by-subreddit` | GET | 서브레딧별 감성 비교 |
| `/api/sentiment/by-subreddit-month` | GET | 서브레딧×월 감성 매트릭스 |
| `/api/topics/info` | GET | 토픽 목록 (ID, 키워드, 문서 수) |
| `/api/topics/over-time` | GET | 토픽별 시간 추이 |
| `/api/topics/by-subreddit` | GET | 서브레딧별 토픽 분포 |
| `/api/clusters/summaries` | GET | 클러스터 요약 (limit, min_count 파라미터) |
| `/api/clusters/keywords` | GET | 클러스터별 키워드 |
| `/api/clusters/temporal` | GET | 시간별 클러스터 변화 |
| `/health` | GET | 헬스 체크 |

**데이터 로딩:** `@functools.lru_cache` 로 CSV/Parquet를 한 번만 로드하여 메모리에 캐싱.

### 6.2 Frontend — React Dashboard

**Tech Stack:** React 19, TypeScript, Vite, TailwindCSS, Recharts, React Router, Axios

**Pages:**

| Page | Visualizations |
|------|----------------|
| **Dashboard** | StatCards (총 문서/서브레딧/토픽/평균감성), 감성 타임라인 (LineChart), 서브레딧별 감성 (BarChart), 볼륨 분포 (StackedBarChart) |
| **Sentiment** | 멀티-서브레딧 감성 비교 (토글 버튼), 합성 타임라인 (LineChart), 서브레딧 감성 개요 테이블 |
| **Topics** | 토픽 분포 (Horizontal BarChart), 토픽 진화 Top 8 (Stacked AreaChart), 토픽 상세 테이블 |
| **Clusters** | Top 20 클러스터 크기 (감성별 색상 BarChart), 커스텀 Tooltip (서브레딧, 감성, 기간), 클러스터 상세 테이블 (50행) |

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

**환경 변수:** `VITE_API_URL` → Cloud Run 백엔드 URL

### 7.3 Daily Pipeline → Cloud Run Jobs + Cloud Scheduler

```bash
# 1. Deploy Cloud Run Job
gcloud builds submit --config webapp/pipeline/deploy/cloudbuild.yaml .

# 2. Setup Cloud Scheduler (daily 06:00 UTC)
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
├── README.md                              # 프로젝트 개요
├── PIPELINE.md                            # 이 문서 (파이프라인 & 알고리즘)
├── .gitignore
│
├── venezuela-us-reddit-discourse/         # Reddit 분석 프로젝트
│   ├── data-collection/
│   │   ├── main.py                        # 수집 CLI (historical/crisis/comments)
│   │   ├── pyproject.toml
│   │   └── scripts/
│   │       ├── config.py                  # 서브레딧, 키워드, 플래시포인트 설정
│   │       ├── collectors.py              # Arctic Shift API 호출
│   │       └── processors.py              # JSON 로드/저장/머지/중복제거
│   │
│   ├── preprocessing/
│   │   ├── config.py                      # 전처리 설정
│   │   ├── filters.py                     # Bot/삭제/저품질 필터
│   │   ├── text_cleaner.py                # URL/마크다운/이모지 클리닝
│   │   └── preprocessor.py                # RedditPreprocessor 메인 클래스
│   │
│   ├── analysis/
│   │   ├── main.py                        # 분석 파이프라인 오케스트레이터
│   │   ├── config.py                      # AnalysisConfig (모델, 파라미터)
│   │   ├── data_loader.py                 # Parquet 로더
│   │   ├── sentiment/
│   │   │   └── roberta_analyzer.py        # RoBERTa 감성 분석
│   │   ├── topic/
│   │   │   └── bertopic_model.py          # BERTopic 토픽 모델링
│   │   ├── clustering/
│   │   │   ├── embedder.py                # Sentence-BERT 임베더
│   │   │   ├── cluster.py                 # TemporalClusterer (HDBSCAN)
│   │   │   ├── summarizer.py              # TF-IDF + LLM 클러스터 요약
│   │   │   └── temporal_viz.py            # UMAP scatter, animation, heatmap
│   │   └── outputs/                       # 분석 결과 (CSV, Parquet, npy)
│   │       ├── sentiment/
│   │       ├── topics/
│   │       ├── clusters/
│   │       └── visualizations/
│   │
│   └── EDA/                               # 탐색적 데이터 분석
│
├── venezuela-tiktok-discourse/            # TikTok 수집 파이프라인
│   ├── data-collection/
│   │   ├── main.py
│   │   └── scripts/
│   │       ├── config.py                  # TikTok API 설정
│   │       ├── auth.py                    # OAuth2 인증
│   │       └── collectors.py              # 비디오/댓글 수집
│   └── preprocessing/
│       ├── filters.py                     # TikTok 특화 필터
│       └── preprocessor.py
│
├── webapp/
│   ├── backend/                           # FastAPI 백엔드
│   │   ├── main.py                        # FastAPI app, CORS, 라우터 등록
│   │   ├── routers/
│   │   │   ├── overview.py                # /api/overview/*
│   │   │   ├── sentiment.py               # /api/sentiment/*
│   │   │   ├── topics.py                  # /api/topics/*
│   │   │   └── clusters.py                # /api/clusters/*
│   │   └── services/
│   │       └── data_service.py            # CSV/Parquet 로드 + LRU 캐시
│   │
│   ├── frontend/                          # React 대시보드
│   │   ├── src/
│   │   │   ├── App.tsx                    # Router + Layout
│   │   │   ├── lib/api.ts                 # API 클라이언트 (Axios)
│   │   │   ├── components/layout/
│   │   │   │   └── Sidebar.tsx            # 네비게이션
│   │   │   └── pages/
│   │   │       ├── Dashboard.tsx           # 메인 대시보드
│   │   │       ├── SentimentPage.tsx       # 감성 분석 페이지
│   │   │       ├── TopicsPage.tsx          # 토픽 모델링 페이지
│   │   │       └── ClustersPage.tsx        # 클러스터 분석 페이지
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── tsconfig.json
│   │
│   └── pipeline/                          # 일일 ETL 파이프라인
│       ├── main.py                        # CLI 오케스트레이터
│       ├── config.py                      # PipelineConfig
│       ├── collectors/
│       │   ├── reddit.py                  # Arctic Shift API 수집
│       │   ├── gdelt.py                   # BigQuery 수집
│       │   └── scraper.py                 # 뉴스 기사 스크래퍼
│       ├── processing/
│       │   ├── preprocessor.py            # 전처리 (필터 + 클리닝)
│       │   └── analyzer.py                # 증분 분석 (감성 + 토픽)
│       ├── deploy/
│       │   ├── cloudbuild.yaml            # Cloud Build 설정
│       │   └── setup_scheduler.sh         # Cloud Scheduler 설정
│       ├── Dockerfile
│       └── requirements.txt
│
└── pipeline_data/                         # 파이프라인 런타임 데이터
    ├── raw/
    │   ├── reddit/submissions/            # 일별 수집 JSON
    │   ├── reddit/comments/
    │   ├── gdelt/                         # GDELT Parquet
    │   └── news/                          # 스크랩 기사 JSON
    ├── processed/
    │   ├── reddit/                        # 전처리된 Parquet
    │   └── gdelt/
    └── reports/                           # 파이프라인 실행 리포트 JSON
```
