# Iteration 4 Presentation — March 24, 2026
## Multiplatform Narrative Analysis of Venezuela-US Relations
### Hunjun Shin, Rich Goodier, Ameir El Ouadi | Northeastern University

**Total: 7 minutes**

---

## Slide 1 — Recap (~1 min) | Speaker: Rich

### Last Week (Code Walk, March 10)
- Reddit (426K docs) + GDELT News (211K articles) — sentiment & topic analysis deployed
- Basic TikTok integration started — only 3,641 videos collected
- Single global BERTopic model with fixed hyperparameters (min_cluster_size=50)
- Dashboard: single-month slider, Reddit-centric views
- No clustering experiment, no cross-platform comparison on clusters

---

## Slide 2 — System Overview Diagram (~1 min) | Speaker: Rich

### Unified Analysis Pipeline
```
                                           ┌─→ RoBERTa Sentiment (3-class) ──┐
Reddit  ──┐                                │                                  │
GDELT   ──┼→ Per-platform  →  S-BERT     ──┼─→ BERTopic Monthly Topics     ──┼→ Gemini 2.0 → Dashboard
TikTok  ──┘   Preprocessing   Embed(384d)  │                                  │   Reports     (React+
                                           └─→ HDBSCAN Monthly Clusters   ──┘   + Chat       FastAPI)
```

**Key point:** Identical pipeline applied to ALL 3 platforms
- Same models, same adaptive hyperparameters
- Only preprocessing differs (Reddit: markdown, GDELT: web scrape, TikTok: video descriptions)
- Adaptive min_cluster_size = max(10, n/400) — scales with monthly data volume

---

## Slide 3 — Data Collection Updates (~1.5 min) | Speaker: Ameir

### Data Growth

| Platform | Before (Mar 10) | Now | Change |
|----------|-----------------|-----|--------|
| Reddit | 426,435 docs | 426,435 | — (complete) |
| GDELT News | 211,071 articles | 226,165 | +7.3% (Wayback rescue) |
| TikTok Videos | 3,641 | 18,632 | **5.1x increase** |
| TikTok Comments | 1,309 | 34,263 | **26x increase** |
| **Total** | **642,456** | **705,495** | **+9.8%** |

### TikTok Collection Fixes
- Removed `voice_to_text` field causing API errors on every request
- Reduced SDK retry from 60 → 5 (prevented quota waste on server errors)
- Added checkpoint-based resumption for multi-day collection
- Quota: 1,000 requests/day — required 4+ days of collection

---

## Slide 4 — Adaptive Clustering Experiment (~1.5 min) | Speaker: Ameir

### Problem
Fixed min_cluster_size=50 doesn't work across varying monthly data volumes (750 to 13,000 docs)

### Experiment
- **1,944 parameter combinations** tested
- 2 platforms × 3 density periods (low / medium / high)
- Grid: UMAP (n_components, n_neighbors, min_dist) × HDBSCAN (min_cluster_size, min_samples)
- Evaluation: composite = silhouette × (1 − noise_ratio), minimum 5 clusters

### Results

| Platform | Density | n | Best mcs | Silhouette | Noise |
|----------|---------|---|----------|------------|-------|
| Reddit | Low (1.3K) | 1,309 | 10 | 0.665 | 22.0% |
| Reddit | High (10K) | 10,000 | 25 | 0.669 | 25.4% |
| News | Low (3K) | 3,022 | 10 | 0.856 | 6.8% |
| News | High (8K) | 8,010 | 10 | 0.856 | 7.3% |

### Derived Rule
```
min_cluster_size = max(10, ⌊n / 400⌋)
```
Applied uniformly across all platforms. Previous fixed mcs=50 → noise 29.1%, now 25.4% on Reddit.

---

## Slide 5 — Monthly Independent Fitting (~30 sec) | Speaker: Ameir

### Before: 1 global model → temporally blurred topics
### Now: Independent model per month per platform

| Platform | Monthly Topics | Monthly Clusters |
|----------|---------------|-----------------|
| Reddit | 6,266 (157 months) | 6,263 |
| GDELT News | 9,119 (157 months) | 9,144 |
| TikTok | 510 (24 months) | 510 |

Captures event-specific discourse shifts — e.g., "Guaidó interim president" appears only in Jan-Feb 2019

---

## Slide 6 — Live Demo (~2.5 min) | Speaker: Hunjun

### Dashboard Demo Flow
1. **Overview** — 3-platform stats cards, cross-platform sentiment comparison chart
2. **Sentiment** — Range slider (start/end), 3-month moving average smoothing, Reddit vs News vs TikTok
3. **Topics** — Month slider, platform-specific monthly BERTopic bar charts + details tables
4. **Clusters** — Reddit/News side-by-side UMAP scatter, Top 20 per platform, temporal volume
5. **Reports** — Select date range → Generate intelligence report → PDF export
6. **Chat** — Ask: "How did sentiment change during the 2018 Maduro reelection?" (auto date detection)

### Key UI Improvements Since Code Walk
- Dual range slider for time filtering
- Dynamic platform visibility (hide platforms with no data for selected period)
- Platform-separated cluster views (scatter, top 20, temporal — all side-by-side)
- LLM chat with natural language date extraction
- PDF export for reports

---

## Slide 7 — Evaluation Results (~45 sec) | Speaker: Rich

### Sentiment Model Validation
- **RoBERTa vs VADER** on 200 politically sarcastic samples:
  - VADER: 34% misclassified sarcasm as positive
  - RoBERTa: 89% correctly identified negative sentiment

### Clustering Quality

| Platform | Clusters | Silhouette | Noise |
|----------|----------|------------|-------|
| Reddit (global) | 3,406 | 0.669 | 25.4% |
| GDELT (global) | 92 | 0.856 | 39.7% |
| TikTok (global) | 78 | 0.72 | 27.8% |

### Topic Modeling
- Monthly BERTopic vs global LDA: monthly captures event-specific topic emergence
- Global LDA averaged away temporal shifts

### TikTok API Limitation
- ~62.7% of public videos inaccessible via Research API (documented in prior research)
- TikTok serves as supplementary source for 2016-2018 period

---

## Slide 8 — Next Steps (~30 sec) | Speaker: Rich

### Before Final Presentation (March 31)
- [ ] TikTok re-collection in progress — API now returning previously empty data (93 windows)
- [ ] Re-run full analysis pipeline with expanded TikTok data
- [ ] Update dashboard + GCS deployment with new results
- [ ] Final report with updated figures and numbers
- [ ] Presentation rehearsal

### Known Limitations to Address
- TikTok temporal gap: 2017-10 to 2018-01 (persistent API server errors)
- News clustering noise ratio (39.7%) — exploring min_samples tuning
- GraphRAG knowledge graph — implemented but not yet integrated into dashboard

---

## Speaker Assignment Summary

| Slide | Topic | Speaker | Time |
|-------|-------|---------|------|
| 1 | Recap | Rich | ~1 min |
| 2 | System Architecture | Rich | ~1 min |
| 3 | Data Updates | Ameir | ~1.5 min |
| 4 | Adaptive Clustering | Ameir | ~1.5 min |
| 5 | Monthly Fitting | Ameir | ~30 sec |
| 6 | **Live Demo** | **Hunjun** | ~2.5 min |
| 7 | Evaluation | Rich | ~45 sec |
| 8 | Next Steps | Rich | ~30 sec |

**Rich: ~3.25 min (Slides 1, 2, 7, 8)**
**Ameir: ~3.5 min (Slides 3, 4, 5)**
**Hunjun: ~2.5 min (Slide 6 — Live Demo)**

---

## AI Slide Generation Prompt

Copy this into Gamma.app, Beautiful.ai, or ChatGPT:

```
Create an 8-slide professional dark-themed presentation for a university capstone project.

Design: Dark navy background (#0f1117), white text, subtle card borders (#2a2e3d).
Accent colors: purple (#6366f1) for Reddit, amber (#f59e0b) for GDELT News, pink (#ff0050) for TikTok, green (#34d399) for positive metrics.

Title: "Multiplatform Narrative Analysis of Venezuela-US Relations"
Subtitle: "Iteration 4 — Unified Pipeline, Adaptive Clustering, Cross-Platform Dashboard"
Team: Hunjun Shin, Rich Goodier, Ameir El Ouadi — Northeastern University
Date: March 24, 2026

Slide 1: RECAP — What we had as of March 10 code walk
4 bullet points: Reddit+GDELT deployed, basic TikTok (3.6K videos), fixed clustering params, single-month dashboard

Slide 2: SYSTEM ARCHITECTURE — Unified Pipeline diagram
Show LEFT-TO-RIGHT flow: 3 data sources → preprocessing → S-BERT embedding → 3 parallel analysis boxes (RoBERTa Sentiment, BERTopic Monthly Topics, HDBSCAN Monthly Clusters) → Gemini LLM → Dashboard
Add callout: "Identical pipeline for all 3 platforms — only preprocessing differs"

Slide 3: DATA COLLECTION — Growth table
Table showing before/after: Reddit 426K→426K, GDELT 211K→226K (+7.3%), TikTok 3.6K→18.6K videos (5.1x), Comments 1.3K→34.3K. Total 642K→705K.
Note TikTok fixes: removed invalid API field, optimized retry, checkpoint resumption

Slide 4: ADAPTIVE CLUSTERING — Experiment results
1,944 combinations tested. Table of results by platform/density.
Formula: min_cluster_size = max(10, n/400). Reddit noise 29.1%→25.4%.

Slide 5: MONTHLY INDEPENDENT FITTING
Before: 1 global model. After: independent per month.
Table: Reddit 6,266 topics, News 9,119, TikTok 510. Same for clusters.

Slide 6: LIVE DEMO — placeholder with dashboard screenshot
List demo flow: Overview → Sentiment → Topics → Clusters → Reports → Chat

Slide 7: EVALUATION
Sentiment: RoBERTa 89% vs VADER 66% on sarcastic text.
Clustering silhouette table.
Monthly BERTopic captures event-specific shifts vs global LDA.

Slide 8: NEXT STEPS
TikTok re-collection in progress (API now returning data).
Re-run analysis, update dashboard, final report, rehearsal.
Known limitations: TikTok API gaps, news noise ratio.
```
