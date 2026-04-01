# Final Presentation — March 31, 2026
## Multiplatform Narrative Analysis of Venezuela-US Relations
### Hunjun Shin, Rich Goodier, Ameir El Ouadi | Northeastern University

**Total: 15 minutes**

---

## Slide 1 — Motivation & Problem Statement (~1.5 min)

### Why This Project?
- The same geopolitical event is framed completely differently across platforms
  - 2019 Guaidó Crisis: U.S. media → "defending democracy" / Venezuelan state media → "coup attempt" / Reddit → varies by subreddit ideology
  - 2024 Election: news coverage focused on fraud allegations; TikTok amplified emotional, visual narratives
- Existing research is limited to single-platform, static analysis
- **Our goal:** Build a unified, real-time analysis system across 3 platforms (Reddit, GDELT News, TikTok) spanning 13 years of Venezuela-US discourse

### Key Crisis Periods Covered
| Period | Event | Why It Matters |
|--------|-------|----------------|
| 2013-04 | Maduro Inauguration | Start of the Maduro era |
| 2014-02 | Venezuelan Protests | First major civil unrest |
| 2017-08 | Trump Sanctions | U.S. policy escalation |
| 2019-01 | Guaidó Recognition Crisis | International legitimacy battle |
| 2024-07 | 2024 Election Crisis | Fraud allegations, global response |
| 2026-01 | Maduro Captured | Unprecedented event, massive volume spike |

### Research Questions
1. How do narratives differ across platforms and political communities?
2. Which discourse communities emerge around Venezuela-US relations?
3. How do sentiments shift during major political crises?
4. What entities and relationships dominate the discourse?

---

## Slide 2 — Data Collection (~2 min)

### Three Platforms, 690K+ Documents, 13 Years

| Platform | Scale | Collection Method | Period |
|----------|-------|-------------------|--------|
| Reddit | 426,435 docs | Arctic Shift API — 11 subreddits (r/vzla, r/politics, r/worldnews, r/Conservative, etc.) | 2013–2026 |
| GDELT News | 211,071 articles | BigQuery event filtering + newspaper scraping + Wayback Machine rescue (72.1% success rate) | 2013–2026 |
| TikTok | 18,632 videos + 34,263 comments | Research API (daily quota 1,000 req) + Playwright browser automation for comments | 2016–2026 |

### Data Growth Over the Project

| Platform | Iteration 1–3 (Mar 10) | Final (Mar 31) | Change |
|----------|------------------------|----------------|--------|
| Reddit | 426,435 docs | 426,435 | — (complete) |
| GDELT News | 211,071 articles | 226,165 | +7.3% (Wayback rescue) |
| TikTok Videos | 3,641 | 18,632 | **5.1x increase** |
| TikTok Comments | 1,309 | 34,263 | **26x increase** |
| **Total** | **642,456** | **705,495** | **+9.8%** |

### Key Challenges & Fixes
- TikTok API: ~62.7% of public videos inaccessible (documented API limitation); persistent gap Oct 2017–Jan 2018
  - Removed `voice_to_text` field causing API errors on every request
  - Reduced SDK retry from 60 → 5 (prevented quota waste on server errors)
  - Added checkpoint-based resumption for multi-day collection (quota: 1,000 req/day → 4+ days)
- News scraping: domain blocking, dead links → Wayback Machine fallback rescued archived articles
- Reddit: bot accounts (AutoModerator, autotldr, etc.), deleted/removed content filtering
- Bilingual corpus: English + Spanish content across all platforms

---

## Slide 3 — Preprocessing Pipeline (~1.5 min)

### Platform-Specific Preprocessing, Unified Output

**Reddit (426K docs):**
- Bot detection: known bot account list (AutoModerator, autotldr, empleadoEstatalBot, RemindMeBot, etc.)
- Remove `[deleted]` / `[removed]` posts
- Strip Markdown formatting (`**bold**`), URLs, Reddit quotes (`> quoted text`), edit markers (`Edit: ...`)
- Remove code blocks and media-only posts (URL-only)
- Minimum length filter (< 5 words)
- Unicode normalization + whitespace normalization

**GDELT News (211K articles):**
- Domain shuffling (interleave sort) to minimize rate-limit collisions
- User-agent rotation, 10s timeout per request
- Wayback Machine fallback for dead/archived URLs (rescued ~15K articles)
- Relevance filtering by scrape status (Success, Error, Empty_Content, Archived)

**TikTok (52K docs):**
- Duplicate removal, spam comment filtering
- Video description + auto-caption text extraction
- Language detection for bilingual corpus

### Result
- 533,941 raw → 426,435 cleaned documents (20% filtered)
- **Identical downstream pipeline applied to all 3 platforms** — only preprocessing differs
- Output format: standardized Parquet (id, type, source, author, text, score, created_utc, year_month)

---

## Slide 4 — Analysis Pipeline & Methods (~2 min)

### Unified Architecture

```
                                              ┌─→ RoBERTa Sentiment (3-class)  ──┐
Reddit  ──┐                                   │                                   │
GDELT   ──┼→ Per-platform   →  S-BERT       ──┼─→ BERTopic Monthly Topics      ──┼→ Gemini 2.0  → Dashboard
TikTok  ──┘   Preprocessing    Embed (384d)   │                                   │   Reports      (React +
                                              ├─→ HDBSCAN Monthly Clusters     ──┘   + Chat        FastAPI)
                                              └─→ Gemini Entity Extraction
```

### Sentiment Analysis — RoBERTa
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` (124M params)
- 3-class output: positive / neutral / negative with confidence score
- Validated: **89% accuracy** on 200 politically sarcastic samples vs VADER's 66%
- GPU-accelerated batch processing (Apple MPS, CUDA, CPU fallback)

### Topic Modeling — BERTopic
- Embedding: Sentence-BERT (multilingual MiniLM, 384-dim, EN+ES)
- Dimensionality reduction: UMAP (384 → 5 dims)
- Clustering: HDBSCAN with adaptive parameters
- Representation: c-TF-IDF + KeyBERT (top 10 keywords per topic)
- **Monthly independent fitting** — separate model per month per platform
  - Reddit: 6,266 topics (157 months)
  - News: 9,119 topics (157 months)
  - TikTok: 510 topics (24 months)
- Captures event-specific discourse (e.g., "Guaidó interim president" only in Jan–Feb 2019)

### Semantic Clustering — HDBSCAN + UMAP

**Problem:** Fixed min_cluster_size=50 doesn't work across varying monthly data volumes (750 to 13,000 docs)

**Adaptive clustering experiment:** 1,944 parameter combinations tested
- 2 platforms × 3 density periods (low / medium / high)
- Grid: UMAP (n_components, n_neighbors, min_dist) × HDBSCAN (min_cluster_size, min_samples)
- Evaluation: composite score = silhouette × (1 − noise_ratio), minimum 5 clusters

**Experiment Results by Platform & Density:**

| Platform | Density | n | Best mcs | Silhouette | Noise |
|----------|---------|---|----------|------------|-------|
| Reddit | Low (1.3K) | 1,309 | 10 | 0.665 | 22.0% |
| Reddit | High (10K) | 10,000 | 25 | 0.669 | 25.4% |
| News | Low (3K) | 3,022 | 10 | 0.856 | 6.8% |
| News | High (8K) | 8,010 | 10 | 0.856 | 7.3% |

**Derived rule:** `min_cluster_size = max(10, ⌊n / 400⌋)` — applied uniformly across all platforms

**Global Clustering Results:**

| Platform | Clusters | Silhouette | Noise Ratio |
|----------|----------|------------|-------------|
| Reddit | 3,406 | 0.669 | 25.4% (was 29.1%) |
| GDELT News | 92 | 0.856 | 39.7% |
| TikTok | 78 | 0.72 | 27.8% |

### Entity Extraction — Gemini 2.0 Flash
- Cross-platform extraction: PERSON, ORG, EVENT, POLICY, LOCATION, TOPIC
- Relationship mapping: "imposed sanctions on", "recognized as president", "fled to"
- One API call per month per platform (concatenated top documents)

---

## Slide 5 — Web Dashboard & Deployment (~1.5 min)

### Tech Stack

| Component | Technology | Deployment |
|-----------|-----------|------------|
| Frontend | React 19 + Vite + TailwindCSS 4.1 + Recharts + TypeScript 5.9 | Vercel |
| Backend | FastAPI + Google Cloud Storage + pandas/numpy | Google Cloud Run |
| LLM Features | Gemini 2.0 Flash (report generation + chat) | Vertex AI |
| Daily ETL | Cloud Run Jobs + Cloud Scheduler (6 AM UTC, 4GB RAM, 2 CPU) | GCP |
| Knowledge Graph | Microsoft GraphRAG + Ollama llama3.1:8b + LanceDB | Local |

### Daily ETL Pipeline (Automated)
```
Cloud Scheduler (cron 6AM UTC)
  → Cloud Run Job
    → Collect: Arctic Shift (11 subreddits) + GDELT BigQuery
    → Scrape: News articles (aiohttp, 5 concurrent, 15s timeout)
    → Preprocess: Bot filter, text clean → Parquet
    → Analyze: RoBERTa sentiment + topic assignment (cosine similarity > 0.25)
    → Merge: Concatenate with existing CSVs on GCS
```

### Dashboard Pages (7 pages)

| Page | Visualizations |
|------|----------------|
| **Dashboard** | 3-platform stat cards, cross-platform sentiment timeline, volume distribution |
| **Sentiment** | Multi-source comparison, composite timeline, box plots, subreddit×month heatmap |
| **Topics** | Monthly BERTopic 3-column layout (Reddit/News/TikTok), topic evolution area charts, detail tables |
| **Clusters** | UMAP scatter (30K points), Top 20 per platform, temporal bar charts, side-by-side views |
| **TikTok** | Hashtag trends, engagement metrics (views/likes/shares), region distribution |
| **Reports** | AI intelligence report with period selector, platform stat cards, PDF export |
| **Chat** | Natural language Q&A with auto date extraction, suggested questions, conversation history |

### Key UI Features
- Dual range slider for time filtering (start/end month)
- Dynamic platform visibility (hide platforms with no data for selected period)
- Platform accent colors: Purple (#6366f1 Reddit), Amber (#f59e0b News), Pink (#ff0050 TikTok)
- Loading skeletons, responsive grid layouts, dark-themed UI

---

## Slide 6 — Key Insights (~2.5 min)

### Cross-Platform Narrative Divergence — Data-Backed Findings

**Finding 1: Sentiment Divergence During Crises**

The same geopolitical event produces sentiment scores that differ by 0.4–0.6 across platforms. Reddit is consistently the most negative; TikTok remains positive even during severe crises.

| Month | Event | Reddit | News | TikTok | Gap |
|-------|-------|:---:|:---:|:---:|:---:|
| **2019-01** | Guaidó interim presidency | -0.291 | -0.065 | +0.168 | 0.460 |
| **2017-05** | Constitutional crisis | -0.364 | -0.197 | +0.279 | **0.644** |
| **2019-05** | Failed uprising aftermath | -0.369 | -0.116 | +0.159 | 0.527 |
| **2026-01** | Maduro captured (simulated) | -0.426 | -0.086 | +0.105 | 0.531 |

**Finding 2: Topic Framing Differences (Jan 2019 — Guaidó Crisis)**

Each platform frames the exact same event through a completely different lens:

| Platform | Top Topics | Framing |
|----------|-----------|---------|
| **Reddit** | "crisis venezuela", "socialism, socialist", "happening venezuela" | Ideology & crisis |
| **News** | "venezuela pdvsa", "venezuelan diplomats", "situation venezuela" | Oil policy & diplomacy |
| **TikTok** | "destacame venezuela", "risas venezuela", "jajajjaja si" | Comedy & entertainment |

- Reddit: 23.8% of content in the crisis/venezolanos cluster; 4.6% in socialism debate
- News: top clusters are oil/sanctions (1.5%) and Guaidó leadership (1.3%)
- TikTok: 16.1% in comedy cluster, 6.2% in comedia, 0% political clusters

**Finding 3: Cluster Separation (May 2017 — Largest Divergence)**

| Platform | Top 3 Clusters | Combined % |
|----------|---------------|:---:|
| **Reddit** | Political crisis (maduro) · Socialism debate · Armed conflict | 33.3% |
| **News** | Caracas protests · Opposition deaths · Trump diplomacy | 9.4% |
| **TikTok** | Lip sync & lifestyle · Social engagement · Feature requests | **67.9%** |

TikTok's top 3 clusters are entirely apolitical — 67.9% of content is lip sync, dance, and "destacame" (feature me) requests, while Reddit discusses armed conflict and socialism.

**Finding 4: TikTok's Unique Role**
- TikTok remains positive (+0.1 to +0.3) even during months when Reddit drops below -0.4
- Regional distribution: 50.6% Venezuela, 12.5% Spain, 11.7% USA
- Creator-driven hashtag communities (#destacame, #comedia) dominate over political discourse
- Volume is stable during crises — TikTok audiences are less reactive to political events

---

## Slide 7 — Evaluation & Validation (~1 min)

### Sentiment Model Validation
- **RoBERTa vs VADER** on 200 politically sarcastic samples:
  - VADER: 34% misclassified sarcasm as positive
  - RoBERTa: **89% correctly identified negative sentiment**
- RoBERTa's Twitter fine-tuning handles informal language, abbreviations, and emojis

### Topic Modeling Validation
- **Monthly BERTopic vs Global LDA:** monthly fitting captures event-specific topic emergence (e.g., "Guaidó interim president" only in Jan–Feb 2019) that global LDA averaged away
- Before: 1 global model → temporally blurred topics
- After: independent model per month per platform → 15,895 total monthly topics

### Clustering Quality

| Platform | Clusters | Silhouette | Noise |
|----------|----------|------------|-------|
| Reddit | 3,406 | 0.669 | 25.4% |
| GDELT News | 92 | 0.856 | 39.7% |
| TikTok | 78 | 0.72 | 27.8% |

### Known Limitations
- TikTok API: ~62.7% of public videos inaccessible via Research API (documented in prior research)
- TikTok temporal gap: Oct 2017–Jan 2018 (persistent API server errors)
- News clustering noise ratio (39.7%) — higher due to diverse article sources
- RoBERTa trained on Twitter — may not perfectly capture all political discourse styles

---

## Slide 8 — Significance & Impact (~1.5 min)

### Policy Implications
- **Public opinion monitoring:** Policymakers can track real-time cross-platform sentiment during geopolitical crises
- **Crisis response:** Understanding how narratives spread differently across platforms enables better-informed communication strategies
- **Disinformation detection:** Cross-platform comparison reveals narrative manipulation patterns
- **Media literacy:** Demonstrates quantitatively that platform choice shapes the narrative a user receives

### Academic Contributions
- **First 3-platform integrated narrative analysis** spanning Reddit, news media, and TikTok over 13 years
- **Adaptive clustering methodology:** Reproducible hyperparameter rule derived from 1,944-combination experiment
- **Monthly independent topic fitting:** Preserves temporal resolution lost in global models
- **690K+ multilingual corpus** across 3 platforms with unified analysis pipeline
- **Replicable architecture:** Same pipeline can be applied to any geopolitical topic by changing keywords and subreddits

### Social Media Comparative Analysis
- Quantitative proof that the same event produces fundamentally different discourse across platforms
- Each platform has distinct characteristics:
  - **News:** Institutional framing, slightly negative tone (avg -3.08), policy-oriented clusters
  - **Reddit:** Community-driven polarization, ideology-based subreddit clustering, highest negativity during crises
  - **TikTok:** Emotional, visual, creator-driven narratives, entertainment-dominant even during political crises
- Answers RQ1–RQ4: platforms diverge in sentiment (0.4–0.6 gap), topic framing (ideology vs diplomacy vs entertainment), and temporal reactivity (Reddit/News surge during crises, TikTok stable)

---

## Slide 9 — Live Demo (~2.5 min)

### Demo Flow — Recommended Month Sequence

**Step 1: January 2019 (Guaidó Crisis) — Main Demo**
- Best all-around: balanced volume across all 3 platforms (Reddit 11K, News 6K, TikTok 12K)
- Show sentiment timeline → zoom into Jan 2019 → highlight the 0.460 gap
- Switch to Topics page → show 3-column layout: crisis/socialism (Reddit) vs oil/diplomacy (News) vs comedy (TikTok)
- Switch to Clusters page → compare political clusters (Reddit) vs entertainment clusters (TikTok)
- **Talking point:** "During the Guaidó crisis, Reddit debated socialism and intervention, News covered oil sanctions and diplomacy, TikTok posted comedy sketches"

**Step 2: May 2017 (Constitutional Crisis) — Largest Gap**
- Show the most extreme sentiment divergence in the entire dataset (0.644)
- TikTok: 67.9% lip-sync/dance clusters vs Reddit: 52.4% negative content
- **Talking point:** "TikTok was 67% lip-sync and dance content while Reddit was over half negative about the same country's political crisis"

**Step 3 (if time): January 2026 — Scale Demo**
- Flash the volume numbers: 47K Reddit + 25K News in a single month
- **Talking point:** "Our system processes 70K+ documents per month with the same pipeline"

**Step 4: Reports / Chat**
- Generate an AI intelligence report for Jan 2019 or ask a natural language question

### URL
- Dashboard: https://capstone-dashboard-iota.vercel.app
- API Docs: https://backend-api-318799600047.us-central1.run.app/docs

---

## Timing Summary

| Slide | Content | Time |
|-------|---------|------|
| 1 | Motivation & Problem Statement | ~1.5 min |
| 2 | Data Collection | ~2 min |
| 3 | Preprocessing Pipeline | ~1.5 min |
| 4 | Analysis Pipeline & Methods | ~2 min |
| 5 | Web Dashboard & Deployment | ~1.5 min |
| 6 | Key Insights | ~2.5 min |
| 7 | Evaluation & Validation | ~1 min |
| 8 | Significance & Impact | ~1.5 min |
| 9 | Live Demo | ~2.5 min |
| **Total** | | **~15 min** |

---

## Q&A Preparation

**Q: Why Venezuela-US relations specifically?**
> Continuous political tensions since 2013 (Maduro inauguration, sanctions, Guaidó crisis, elections) create rich, diverse narratives across platforms. Active discourse in both English and Spanish makes it ideal for cross-platform comparison.

**Q: How do you handle multilingual data?**
> We use `paraphrase-multilingual-MiniLM-L12-v2` for embeddings, which supports both English and Spanish. Combined EN+ES stopwords (504 words) for topic representation. RoBERTa handles informal multilingual text well due to Twitter fine-tuning.

**Q: Why BERTopic over LDA?**
> BERTopic with transformer embeddings captures semantic meaning far better than bag-of-words LDA. Monthly independent fitting captures event-specific topic emergence that global LDA averages away.

**Q: What are the main limitations?**
> TikTok API restricts access to ~62.7% of public videos. News scraping has 72.1% success rate. Reddit skews toward certain demographics. RoBERTa was trained on Twitter, which may not perfectly capture all political discourse styles.

**Q: How does the daily ETL pipeline work?**
> Cloud Scheduler triggers a Cloud Run Job at 6 AM UTC daily. It collects new Reddit + GDELT data, preprocesses, runs sentiment analysis, assigns topics by cosine similarity to existing centroids (threshold 0.25), and merges with existing results.

---

## Slide Generation Prompt

Copy into Gamma.app or Google Slides AI:

```
Create a 9-slide professional dark-themed presentation for a university capstone final presentation. Each slide should be information-dense with tables, diagrams, and data points. Do NOT oversimplify — this is a technical audience.

Design: Dark navy background (#0f1117), white text (#e5e7eb), subtle card borders (#2a2e3d), rounded corners on all cards/tables.
Accent colors — use consistently throughout:
  - Purple (#6366f1) for Reddit
  - Amber (#f59e0b) for GDELT News
  - Pink (#ff0050) for TikTok
  - Green (#34d399) for positive metrics / success indicators
  - Red (#ef4444) for negative metrics
Font: Inter or system sans-serif. Headings bold, body regular.

Title slide:
  Title: "Multiplatform Narrative Analysis of Venezuela-US Relations"
  Subtitle: "Cross-Platform Discourse Analysis with Real-Time Dashboard"
  Team: Hunjun Shin, Rich Goodier, Ameir El Ouadi
  Affiliation: Northeastern University
  Date: March 31, 2026
  Add 3 small colored icons/badges for Reddit (purple), GDELT News (amber), TikTok (pink)

---

Slide 1: MOTIVATION & PROBLEM STATEMENT

Left side — "The Problem" section:
  - Heading: "Same Event, Different Narratives"
  - Example box with 3 rows, each with platform color accent:
    - [Purple] Reddit: "Varies by subreddit ideology — r/socialism vs r/Conservative"
    - [Amber] U.S. News: "Defending democracy, sanctions, diplomatic framing"
    - [Pink] TikTok: "Emotional, visual, creator-driven content"
  - Below: "Example: 2019 Guaidó Recognition Crisis"
  - Note: "Existing research: limited to single-platform, static analysis"

Right side — "Key Crisis Periods" table:
  | Period | Event |
  | 2013-04 | Maduro Inauguration |
  | 2014-02 | Venezuelan Protests |
  | 2017-08 | Trump Sanctions |
  | 2019-01 | Guaidó Recognition Crisis |
  | 2024-07 | 2024 Election Crisis |
  | 2026-01 | Maduro Captured |

Bottom — 4 Research Questions as numbered cards:
  RQ1: How do narratives differ across platforms?
  RQ2: Which discourse communities emerge?
  RQ3: How do sentiments shift during crises?
  RQ4: What entities and relationships dominate?

---

Slide 2: DATA COLLECTION

Top section — 3 platform cards side by side, each with platform color header:
  [Purple card] Reddit:
    - 426,435 documents
    - Arctic Shift API
    - 11 subreddits: r/vzla, r/venezuela, r/politics, r/news, r/worldnews, r/Conservative, r/Libertarian, r/neoliberal, r/socialism, r/LatinAmerica, r/geopolitics
    - Period: 2013–2026 (13 years)

  [Amber card] GDELT News:
    - 211,071 articles (from 292,566 events)
    - BigQuery event filtering + newspaper scraping
    - Wayback Machine rescue (72.1% success rate)
    - Period: 2013–2026

  [Pink card] TikTok:
    - 18,632 videos + 34,263 comments
    - Research API (1,000 req/day quota) + Playwright browser automation
    - 3,147 unique hashtags, 1,013 creators
    - Period: 2016–2026

Middle section — "Data Growth" table with green arrow indicators:
  | Platform | Before (Mar 10) | Final (Mar 31) | Change |
  | Reddit | 426,435 | 426,435 | — (complete) |
  | GDELT News | 211,071 | 226,165 | +7.3% |
  | TikTok Videos | 3,641 | 18,632 | 5.1x ↑ (green) |
  | TikTok Comments | 1,309 | 34,263 | 26x ↑ (green) |
  | Total | 642,456 | 705,495 | +9.8% |

Bottom section — "Challenges & Fixes" as 3 small cards:
  - TikTok: Removed voice_to_text field (API errors), retry 60→5, checkpoint resumption
  - News: Domain blocking → Wayback Machine rescued ~15K archived articles
  - Reddit: Bot filtering (AutoModerator, autotldr, etc.), deleted content removal

---

Slide 3: PREPROCESSING PIPELINE

Layout: 3 columns, one per platform, with arrow flowing down to unified output.

[Purple column] Reddit (426K docs):
  - Bot detection (known bot list)
  - Remove [deleted] / [removed]
  - Strip Markdown, URLs, quotes, edit markers
  - Remove code blocks, media-only posts
  - Min length filter (< 5 words)
  - Unicode + whitespace normalization

[Amber column] GDELT News (211K articles):
  - Domain shuffling (interleave sort)
  - User-agent rotation, 10s timeout
  - Wayback Machine fallback (~15K rescued)
  - Relevance filter by scrape status

[Pink column] TikTok (52K docs):
  - Duplicate removal
  - Spam comment filtering
  - Video description + auto-caption extraction
  - Language detection (EN/ES)

Bottom — unified output box with green accent:
  - "533,941 raw → 426,435 cleaned (20% filtered)"
  - "Identical downstream pipeline for all 3 platforms — only preprocessing differs"
  - "Output: standardized Parquet (id, type, source, author, text, score, created_utc, year_month)"

---

Slide 4: ANALYSIS PIPELINE & METHODS

Top — Full-width architecture diagram (LEFT to RIGHT flow):
  [3 source boxes with platform colors] Reddit / GDELT / TikTok
    → [gray box] Per-platform Preprocessing
    → [blue box] S-BERT Embedding (384-dim, multilingual MiniLM)
    → [4 parallel branches]:
      ↗ [red/green box] RoBERTa Sentiment (3-class, 124M params)
      → [blue box] BERTopic Monthly Topics (UMAP + HDBSCAN + c-TF-IDF)
      → [orange box] HDBSCAN Monthly Clusters (adaptive min_cluster_size)
      ↘ [yellow box] Gemini 2.0 Flash Entity Extraction
    → [teal box] Gemini 2.0 Reports + Chat
    → [final box] React + FastAPI Dashboard

Middle — 4 method summary cards in a 2×2 grid:

  Card 1 "Sentiment — RoBERTa":
    - twitter-roberta-base-sentiment-latest (124M params)
    - 3-class: positive / neutral / negative
    - 89% accuracy vs VADER 66% on sarcastic political text
    - GPU batch processing (MPS, CUDA, CPU)

  Card 2 "Topics — BERTopic":
    - Monthly independent fitting per platform
    - S-BERT → UMAP (384→5d) → HDBSCAN → c-TF-IDF + KeyBERT
    - Reddit: 6,266 topics | News: 9,119 | TikTok: 510
    - Bilingual stopwords: 504 words (EN+ES)

  Card 3 "Clustering — HDBSCAN + UMAP":
    - Adaptive experiment: 1,944 combinations tested
    - Derived rule: min_cluster_size = max(10, ⌊n/400⌋)
    - Reddit noise: 29.1% → 25.4%
    - Results table:
      | Platform | Clusters | Silhouette | Noise |
      | Reddit | 3,406 | 0.669 | 25.4% |
      | News | 92 | 0.856 | 39.7% |
      | TikTok | 78 | 0.72 | 27.8% |

  Card 4 "Entities — Gemini 2.0 Flash":
    - Types: PERSON, ORG, EVENT, POLICY, LOCATION, TOPIC
    - Relationships: "imposed sanctions on", "recognized as president"
    - 1 API call per month per platform

Bottom — Experiment detail table (smaller font):
  | Platform | Density | n | Best mcs | Silhouette | Noise |
  | Reddit | Low | 1,309 | 10 | 0.665 | 22.0% |
  | Reddit | High | 10,000 | 25 | 0.669 | 25.4% |
  | News | Low | 3,022 | 10 | 0.856 | 6.8% |
  | News | High | 8,010 | 10 | 0.856 | 7.3% |

---

Slide 5: WEB DASHBOARD & DEPLOYMENT

Top — Tech stack table with logo icons:
  | Component | Technology | Deployment |
  | Frontend | React 19 + Vite + TailwindCSS 4.1 + Recharts + TypeScript 5.9 | Vercel |
  | Backend | FastAPI + Google Cloud Storage + pandas/numpy | Google Cloud Run |
  | LLM | Gemini 2.0 Flash (reports + chat) | Vertex AI |
  | ETL | Cloud Run Jobs + Cloud Scheduler (6 AM UTC, 4GB RAM) | GCP |
  | Knowledge Graph | Microsoft GraphRAG + Ollama llama3.1:8b + LanceDB | Local |

Middle — ETL pipeline flow diagram (horizontal):
  Cloud Scheduler (cron 6AM UTC)
    → Cloud Run Job container
    → Collect (Arctic Shift + GDELT BigQuery)
    → Scrape (aiohttp, 5 concurrent, 15s timeout)
    → Preprocess (bot filter, text clean → Parquet)
    → Analyze (RoBERTa sentiment + topic assignment via cosine similarity > 0.25)
    → Merge (concatenate with existing CSVs on GCS)

Bottom — 7 dashboard page cards in a grid (each with a small icon):
  1. Dashboard: 3-platform stat cards, cross-platform sentiment timeline, volume distribution
  2. Sentiment: multi-source comparison, composite timeline, box plots, subreddit×month heatmap
  3. Topics: monthly BERTopic 3-column layout (Reddit/News/TikTok), evolution area charts
  4. Clusters: UMAP scatter (30K points), Top 20 per platform, temporal bars, side-by-side
  5. TikTok: hashtag trends, engagement (views/likes/shares), region distribution
  6. Reports: AI intelligence report, period selector, platform stats, PDF export
  7. Chat: natural language Q&A, auto date extraction, suggested questions

UI callouts (small badges):
  - Dual range slider (start/end month)
  - Dynamic platform visibility
  - Dark theme with platform accent colors

---

Slide 6: KEY INSIGHTS — CROSS-PLATFORM NARRATIVE DIVERGENCE

This slide should be split into 2 sub-slides (6a and 6b) or use a scrollable layout.

Slide 6a — Findings 1 & 2:

  [Finding 1] "Sentiment Divergence During Crises" — full-width section
    - Intro text: "The same event produces sentiment gaps of 0.4–0.6 across platforms"
    - Table with color-coded cells (red for negative, green for positive):
      | Month | Event | Reddit | News | TikTok | Gap |
      | 2019-01 | Guaidó presidency | -0.291 (red) | -0.065 | +0.168 (green) | 0.460 |
      | 2017-05 | Constitutional crisis | -0.364 (red) | -0.197 | +0.279 (green) | 0.644 (bold) |
      | 2019-05 | Failed uprising | -0.369 (red) | -0.116 | +0.159 (green) | 0.527 |
      | 2026-01 | Maduro captured | -0.426 (deep red) | -0.086 | +0.105 (green) | 0.531 |
    - Callout box: "Reddit = consistently most negative | TikTok = positive even during severe crises"

  [Finding 2] "Topic Framing — Jan 2019 Guaidó Crisis" — 3-column layout with platform colors
    [Purple] Reddit — Ideology & Crisis:
      - "crisis venezuela, venezolanos" (23.8%)
      - "socialism, socialist" (4.6%)
      - "happening venezuela" (6.0%)
    [Amber] News — Oil Policy & Diplomacy:
      - "venezuela pdvsa" (2.0%)
      - "venezuelan diplomats" (1.3%)
      - "situation venezuela" (1.2%)
    [Pink] TikTok — Comedy & Entertainment:
      - "destacame venezuela, risas" (6.8%)
      - "jigneshkaviraj, ilikacruz" (7.7%)
      - "jajajjaja si" (3.5%)

Slide 6b — Findings 3 & 4:

  [Finding 3] "Cluster Separation — May 2017 (Largest Divergence)" — horizontal bar chart or table
    | Platform | Top 3 Clusters | Combined % |
    | Reddit | Political crisis · Socialism debate · Armed conflict | 33.3% |
    | News | Caracas protests · Opposition deaths · Trump diplomacy | 9.4% |
    | TikTok | Lip sync & lifestyle · Social engagement · Feature requests | 67.9% (highlighted) |
    - Callout: "67.9% of TikTok content is entirely apolitical (lip sync, dance, 'destacame') while Reddit discusses armed conflict and socialism"

  [Finding 4] "TikTok's Unique Role" — info cards
    - TikTok stays positive (+0.1 to +0.3) even when Reddit drops below -0.4
    - Regional distribution pie chart: Venezuela 50.6%, Spain 12.5%, USA 11.7%, Other 25.2%
    - Creator-driven hashtag communities (#destacame, #comedia) dominate over political discourse
    - Volume stable during crises — TikTok audiences are less reactive to political events

---

Slide 7: EVALUATION & VALIDATION

Layout: 2×2 grid of validation cards + bottom limitation section

  Card 1 "Sentiment Validation" (with checkmark icon):
    - Side-by-side comparison bar chart:
      - VADER: 66% accuracy (34% misclassified sarcasm as positive)
      - RoBERTa: 89% accuracy (correctly identified negative sentiment)
    - "Tested on 200 politically sarcastic samples"
    - RoBERTa's Twitter fine-tuning handles informal language, abbreviations, emojis

  Card 2 "Topic Modeling Validation" (with checkmark icon):
    - Before/After comparison:
      - Before: 1 global model → "temporally blurred topics"
      - After: independent model per month per platform → 15,895 total monthly topics
    - Example: "Guaidó interim president" topic only appears in Jan–Feb 2019 (monthly fitting captures this; global LDA does not)

  Card 3 "Clustering Quality" (with table):
    | Platform | Clusters | Silhouette | Noise |
    | Reddit | 3,406 | 0.669 | 25.4% |
    | GDELT News | 92 | 0.856 | 39.7% |
    | TikTok | 78 | 0.72 | 27.8% |
    - Note: "Adaptive rule reduced Reddit noise from 29.1% → 25.4%"

  Card 4 "Adaptive Clustering Experiment":
    - 1,944 parameter combinations tested
    - 2 platforms × 3 density periods × grid search
    - Derived rule: min_cluster_size = max(10, ⌊n/400⌋)
    - Composite score = silhouette × (1 − noise_ratio)

  Bottom — "Known Limitations" bar (yellow/amber warning style):
    - TikTok API: ~62.7% of public videos inaccessible (documented limitation)
    - TikTok temporal gap: Oct 2017–Jan 2018 (persistent API server errors)
    - News clustering noise: 39.7% (diverse article sources)
    - RoBERTa trained on Twitter — may not perfectly capture all political discourse

---

Slide 8: SIGNIFICANCE & IMPACT

3-column layout:

  [Left column] "Policy Implications" (with government/policy icon):
    - Public opinion monitoring: real-time cross-platform sentiment tracking during crises
    - Crisis response: platform-specific narrative spread patterns
    - Disinformation detection: cross-platform comparison reveals manipulation
    - Media literacy: platform choice shapes the narrative users receive

  [Center column] "Academic Contributions" (with graduation cap icon):
    - First 3-platform integrated narrative analysis (Reddit + News + TikTok, 13 years)
    - Adaptive clustering methodology: reproducible rule from 1,944-combination experiment
    - Monthly independent topic fitting: temporal resolution preserved
    - 690K+ multilingual corpus with unified pipeline
    - Replicable architecture: swap keywords/subreddits for any geopolitical topic

  [Right column] "Social Media Comparative Analysis" (with chart icon):
    - Quantitative proof: same event → fundamentally different discourse
    - Platform characteristics (3 mini cards with platform colors):
      - [Amber] News: institutional framing, avg tone -3.08, policy-oriented clusters
      - [Purple] Reddit: community polarization, ideology-based clustering, most negative during crises
      - [Pink] TikTok: emotional, visual, entertainment-dominant even during political crises

  Bottom — "Answers to Research Questions" summary bar:
    - RQ1–RQ4: platforms diverge in sentiment (0.4–0.6 gap), topic framing (ideology vs diplomacy vs entertainment), temporal reactivity (Reddit/News surge, TikTok stable)

---

Slide 9: LIVE DEMO

Layout: Step-by-step demo flow with numbered cards

  Step 1 (main, large card): "January 2019 — Guaidó Crisis"
    - Volume: Reddit 11K, News 6K, TikTok 12K
    - Demo: Sentiment timeline → Topics 3-column → Clusters scatter
    - Talking point: "Reddit debated socialism, News covered oil sanctions, TikTok posted comedy sketches"

  Step 2: "May 2017 — Constitutional Crisis"
    - Largest sentiment gap in entire dataset: 0.644
    - Talking point: "TikTok was 67% lip-sync content while Reddit was 52% negative"

  Step 3: "January 2026 — Scale Demo"
    - Volume: 47K Reddit + 25K News in a single month
    - Talking point: "Our system processes 70K+ documents per month"

  Step 4: "AI Reports & Chat"
    - Generate intelligence report for Jan 2019
    - Or ask natural language question with auto date extraction

  Bottom — URL badges:
    - Dashboard: capstone-dashboard-iota.vercel.app
    - API Docs: backend-api-318799600047.us-central1.run.app/docs
```
