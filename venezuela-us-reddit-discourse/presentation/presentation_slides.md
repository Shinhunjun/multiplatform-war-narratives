# Cross-Platform Narrative Analysis of U.S.-Venezuela Relations
## Real-Time Knowledge Graph and Interactive Dashboard (2013-2025)

**Team:** [본인], Rich, Ameir
**Target:** ICWSM 2026 Submission

---

## Slide 1: Title

### Cross-Platform Narrative Analysis of U.S.-Venezuela Relations
#### Real-Time Knowledge Graph and Interactive Dashboard (2013-2025)

**Team Members:** [본인], Rich, Ameir

*Target: ICWSM 2026 Submission*

---

## Slide 2: Problem Statement

| Item | Description |
|------|-------------|
| **Problem** | 동일 지정학적 사건이 플랫폼/국가별로 다르게 서술됨 |
| **Gap** | 기존 연구는 정적 분석, 단일 플랫폼에 한정 |
| **Who's Affected** | 연구자, 저널리스트, 정책 입안자, 일반 시민 |
| **Our Impact** | Multi-platform 실시간 내러티브 비교 시스템 구축 |

### Example
- **2019 Guaidó Crisis**
  - US Media: "Defense of Democracy"
  - Venezuelan Gov Media: "Coup Attempt"
  - Reddit: Varies by subreddit community

---

## Slide 3: Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Do US and Venezuelan news media construct different narratives when reporting on the same geopolitical events? |
| **RQ2** | How do narratives in social media discourse (Reddit) differ from those in traditional news media? |
| **RQ3** | To what extent do social media narratives mirror, amplify, or diverge from national news media narratives? |
| **RQ4** | How do narrative differences across media systems evolve around major geopolitical events? |

---

## Slide 4: Team Objectives & Roles

### Team Objectives
1. 4개 RQ에 대한 정량적 분석 결과 도출
2. Knowledge Graph 기반 구조적 관계 분석
3. Real-time Interactive Dashboard 배포
4. **ICWSM 2026 논문 제출**

### Roles

| Member | Data Collection | Shared Work |
|--------|-----------------|-------------|
| **[본인]** | Reddit (Arctic Shift API) | Analysis, KG, Web App |
| **Rich** | GDELT (News Events) | Analysis, KG, Web App |
| **Ameir** | TikTok | Analysis, KG, Web App |

---

## Slide 5: Datasets

| Source | Size | Period | Key Features |
|--------|------|--------|--------------|
| **Reddit** | 426,435 posts | 2013-2026 | 11 subreddits, preprocessed |
| **GDELT** | TBD events | 2013-2025 | US/VEN news, event codes |
| **TikTok** | TBD | TBD | Video metadata, comments |

### Subreddit Coverage

| Region | Subreddits |
|--------|------------|
| Venezuela | r/vzla, r/venezuela |
| US/English | r/politics, r/news, r/worldnews, r/Conservative, r/Libertarian, r/neoliberal, r/socialism, r/geopolitics, r/LatinAmerica |

### Challenges
- Deleted/removed content filtering
- Cross-platform event alignment
- Multilingual text (EN/ES)
- Real-time data streaming

---

## Slide 6: Related Work

| Paper | Contribution | Gap |
|-------|--------------|-----|
| Kwak & An (2016) | GDELT dataset validation | No narrative framing |
| Olteanu et al. (2015) | News-Social media coverage gap | Frequency only, no text analysis |
| Zhao et al. (2024) | Event-centric framing | Single platform |
| Kuila et al. (2024) | Aspect-level media bias | No cross-national comparison |

### Our Contribution
- **First** event-aligned multi-platform narrative comparison (News + Reddit + TikTok)
- Knowledge Graph with temporal evolution
- Real-time interactive dashboard

---

## Slide 7: System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                             │
├───────────────┬───────────────┬───────────────┬─────────────────────┤
│     GDELT     │    Reddit     │    TikTok     │   Real-time Stream  │
│   (News API)  │ (Arctic Shift)│   (API/Scrape)│   (Cron Jobs)       │
└───────┬───────┴───────┬───────┴───────┬───────┴──────────┬──────────┘
        └───────────────┴───────────────┴──────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS PIPELINE                           │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│    Sentiment    │  Topic Modeling │    Knowledge Graph Construction │
│    (RoBERTa)    │   (BERTopic)    │    (NER + Relation Extraction)  │
└─────────────────┴─────────────────┴─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       KNOWLEDGE GRAPH (Neo4j)                       │
│                                                                     │
│    [Maduro]──OPPOSES──▶[US Gov]──IMPOSES──▶[Sanctions]             │
│        │                   │                    │                   │
│        ▼                   ▼                    ▼                   │
│   sentiment: -0.7    sentiment: -0.5     topic: economy            │
│   platform: Reddit   platform: GDELT     time: 2019-01             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INTERACTIVE WEB APP                            │
├────────────────┬────────────────┬────────────────┬──────────────────┤
│   Dashboard    │   KG Explorer  │    Timeline    │  Real-time Feed  │
│ (Trends/Stats) │  (Graph Viz)   │  (Event View)  │ (Weekly Update)  │
└────────────────┴────────────────┴────────────────┴──────────────────┘
```

---

## Slide 8: Knowledge Graph Design

### Entities
- **Actors**: Maduro, Guaidó, Trump, Biden, US Gov, VEN Gov, Opposition
- **Events**: Sanctions, Protests, Elections, Diplomatic breaks
- **Topics**: Economy, Human Rights, Oil, Migration, Democracy

### Relations
```
(Actor)─[SUPPORTS/OPPOSES]─▶(Actor)
(Actor)─[PARTICIPATES_IN]─▶(Event)
(Event)─[TRIGGERS]─▶(Event)
(Post)─[MENTIONS]─▶(Entity)
(Post)─[HAS_SENTIMENT]─▶(Score)
```

### Temporal Versioning
- Weekly/Monthly snapshots
- Track graph evolution over time
- Example: US-Guaidó relationship strength in Jan 2019 vs Dec 2019

### Tools
- NER: spaCy, Flair
- Relation Extraction: OpenIE, Custom models
- Graph DB: Neo4j / NetworkX

---

## Slide 9: Interactive Web Application

### Features

| Feature | Description | Tech |
|---------|-------------|------|
| **Dashboard** | Sentiment/Topic 시계열 차트 | Plotly, Recharts |
| **KG Explorer** | Interactive graph 탐색 | D3.js, Cytoscape.js |
| **Timeline View** | Event별 platform 비교 | Custom component |
| **Real-time Feed** | 주간/월간 자동 업데이트 | Cron + API polling |
| **Cross-platform Filter** | GDELT vs Reddit vs TikTok 선택 | Dropdown filters |

### Tech Stack
- **Frontend**: React / Streamlit
- **Backend**: FastAPI + Celery
- **Database**: PostgreSQL + Neo4j
- **Deploy**: Docker + AWS/GCP

---

## Slide 10: Progress & Next Steps

### Current Progress

| Phase | Status | Details |
|-------|--------|---------|
| Reddit Collection | ✅ Complete | 426,435 posts collected |
| Reddit Preprocessing | ✅ Complete | Cleaned, filtered |
| GDELT Collection | ✅ Complete | Downloaded |
| TikTok Collection | 🔄 In Progress | Data gathering |
| Event Alignment | 🔄 In Progress | GDELT-Reddit matching |
| Sentiment Analysis | ⬜ Planned | RoBERTa pipeline |
| Topic Modeling | ⬜ Planned | BERTopic |
| Knowledge Graph | ⬜ Planned | Entity extraction → Neo4j |
| Web App Development | ⬜ Planned | Dashboard + KG viewer |
| Real-time Integration | ⬜ Planned | Streaming pipeline |
| ICWSM Paper Draft | ⬜ Planned | Target deadline TBD |

### Next Steps
1. **GDELT-Reddit Event Alignment**: Match events by timestamp and keywords
2. **Sentiment Analysis**: Apply RoBERTa to all text data
3. **Topic Modeling**: Extract topics using BERTopic
4. **Knowledge Graph Construction**: Build entity-relation graph
5. **Web App MVP**: Dashboard with basic visualizations
6. **Real-time Pipeline**: Scheduled data collection and analysis
7. **Paper Writing**: Draft for ICWSM 2026

---

## Slide 11: Expected Outcomes

### Deliverables

| Output | Description |
|--------|-------------|
| **Academic Paper** | ICWSM 2026 submission (RQ1-4 분석 결과) |
| **Dataset** | Event-aligned multi-platform corpus |
| **Knowledge Graph** | Temporal KG with sentiment/topic annotations |
| **Web Application** | Live dashboard with real-time updates |
| **Code Repository** | Reproducible pipeline (GitHub) |

### ICWSM 2026 Timeline
- **Submission Deadline**: ~January 2026
- **Notification**: ~March 2026
- **Camera-ready**: ~April 2026
- **Conference**: ~June 2026

### Impact
- First comprehensive multi-platform narrative analysis for US-Venezuela relations
- Reusable framework for other geopolitical case studies
- Open-source tools for computational social science research

---

## References

1. Kwak, H., & An, J. (2016). A First Look at Global News Coverage of Disasters by Using the GDELT Dataset. *ICWSM*.
2. Olteanu, A., et al. (2015). Comparing Events Coverage in Online News and Social Media. *ICWSM*.
3. Zhao, Y., et al. (2024). Event-Centric Framing and Media Attitude Detection. *EMNLP*.
4. Kuila, A., et al. (2024). Aspect-Level Media Bias and Narrative Variance. *ACM*.
5. GDELT Project: https://www.gdeltproject.org/
6. Arctic Shift API for Reddit Collection

---

## Appendix: Key Events Timeline

| Date | Event | Expected Impact |
|------|-------|-----------------|
| 2013-04 | Maduro Inauguration | Baseline period |
| 2014-02 | Venezuelan Protests | First major spike |
| 2017-08 | Trump Sanctions | Increased US attention |
| 2019-01 | Guaidó Recognition | Highest activity peak |
| 2019-04 | Failed Uprising | Crisis escalation |
| 2024-07 | 2024 Presidential Election | Recent surge |
| 2026-01 | Maduro Captured | Latest peak |

---

*Generated: 2026-02-02*
