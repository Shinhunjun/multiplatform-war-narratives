# Venezuela-US GDELT Comprehensive Analysis Report

## Overview

| Metric | Value |
|--------|-------|
| **Data Period** | 2013-01-07 ~ 2026-03-25 |
| **Total Events** | 301,007 |
| **Avg Goldstein Scale** | 0.02 |
| **Avg Tone** | -3.08 |
| **Median Tone** | -3.24 |
| **Initiator Split (VEN / USA)** | 140,275 / 160,732 |
| **Unique URLs** | 108,066 |
| **Articles in Analysis Corpus** | 67,022 |

### Data Source
- **Dataset**: Analysis-ready GDELT parquet join (Venezuela-US filtered interactions)
- **Scope**: Event metadata from `analysis_events.parquet` + scraped article title/text content from `analysis_url_content.parquet`

---

## Content Analysis

### Title Word Cloud

![Title Word Cloud](08_title_wordcloud.png)

### Top Title Terms

| Word | Frequency | Share |
|------|-----------|-------|
| venezuela | 34,693 | 5.51% |
| us | 16,398 | 2.60% |
| trump | 13,744 | 2.18% |
| u.s. | 12,649 | 2.01% |
| venezuelan | 11,904 | 1.89% |
| maduro | 10,863 | 1.72% |
| oil | 5,609 | 0.89% |
| sanction | 4,034 | 0.64% |
| president | 3,750 | 0.60% |
| american | 3,346 | 0.53% |

### Text Word Cloud

![Text Word Cloud](09_text_wordcloud.png)

### Top Text Terms

| Word | Frequency | Share |
|------|-----------|-------|
| venezuela | 67,211 | 0.36% |
| state | 63,507 | 0.34% |
| president | 62,066 | 0.33% |
| venezuelan | 58,363 | 0.31% |
| country | 57,605 | 0.31% |
| government | 49,376 | 0.26% |
| maduro | 48,382 | 0.26% |
| united | 47,261 | 0.25% |
| american | 45,811 | 0.24% |
| take | 43,798 | 0.23% |

---

## Timeline Analysis

### Full Timeline (2013 - 2026)

![Timeline](01_gdelt_timeline.png)

### Key Insights
- **Volume**: Spikes in event volume correlate with major political milestones.
- **Stability**: Monthly Goldstein means moving below zero indicate more conflict-heavy periods.

### Top 10 Peak Activity Months

| Month | Events |
|-------|--------|
| 2026-01 | 27,925 |
| 2019-02 | 11,409 |
| 2019-03 | 8,700 |
| 2019-01 | 8,569 |
| 2025-12 | 7,191 |
| 2019-05 | 6,268 |
| 2018-05 | 5,045 |
| 2019-04 | 4,874 |
| 2017-08 | 4,721 |
| 2015-03 | 4,437 |

---

## Yearly Trends

![Yearly Stats](02_gdelt_yearly_stats.png)

### Summary
- **Activity**: Event volume by year captures macro-level intensity of interaction.
- **Tone Distribution**: Box plots show within-year dispersion and outliers in media tone.

### Smoothed Tone Trend

![Tone Trend](05_gdelt_tone_trend.png)

- **Rolling Mean**: Highlights medium-term direction shifts.
- **Volatility Band (±1 SD)**: Shows periods of high/low tone variability.

---

## Event Categories (QuadClass)

![Categories](03_gdelt_categories.png)

### Categories Defined
- **Verbal Cooperation**: Statements of support, negotiation, promises.
- **Material Cooperation**: Economic aid, agreements, visits.
- **Verbal Conflict**: Threats, demands, disapproval.
- **Material Conflict**: Sanctions, protests, military acts.

---

## Intensity & Sentiment

![Intensity](04_gdelt_intensity.png)

### Metric Distributions
- **Goldstein Scale**: Event impact on stability from conflict (-) to cooperation (+).
- **AvgTone**: Sentiment proxy of related coverage.

---

## Extreme Events

### Top Conflict Events (Lowest Goldstein)
| Date | Actor 1 | Actor 2 | Code | Goldstein | Title |
|------|---------|---------|------|-----------|-------|
| 2026-03-25 | CHICAGO | VENEZUELA | 190 | -10.0 | Report: Taxpayers Footed Housing Costs for Illegal Alien Accused of Murdering Sheridan Gorman |
| 2013-12-29 | SAN ANTONIO | VENEZUELA | 195 | -10.0 | The NSA Uses Powerful Toolbox in Effort to Spy on Global Networks |
| 2013-12-29 | SAN ANTONIO | VENEZUELA | 195 | -10.0 | The NSA Uses Powerful Toolbox in Effort to Spy on Global Networks |
| 2018-09-08 | ORLANDO | VENEZUELA | 190 | -10.0 | Wife of critically injured Ofc. Valencia shares story |
| 2026-03-25 | VENEZUELA | HOUSTON | 190 | -10.0 | Trump has destroyed Venezuela's socialist ideology: opposition leader |

### Top Cooperation Events (Highest Goldstein)
| Date | Actor 1 | Actor 2 | Code | Goldstein | Title |
|------|---------|---------|------|-----------|-------|
| 2018-02-02 | VENEZUELA | THE WHITE HOUSE | 874 | 10.0 | Top diplomat leaves State, opens door for Cuba, Venezuela hawks | Miami Herald |
| 2015-07-20 | VENEZUELA | PITTSBURGH | 874 | 10.0 | Obituary: Mary Loretta Harrison / Swissvale native made difference as volunteer |
| 2013-07-24 | VENEZUELAN | MIAMI | 874 | 10.0 | Venezuela Consulate Empty But Paying |
| 2023-12-04 | VENEZUELA | UNITED STATES | 874 | 10.0 | Guyana Says Oil Producers Are Moving Ahead Despite Venezuela’s Threats |
| 2018-06-20 | VENEZUELA | AMERICAN | 874 | 10.0 | US exits UN rights body: Principled, or another retreat? |

---

## Appendix: Data Quality & Methodology

### Scrape Quality

![Scrape Status](06_scraped_status.png)

#### Scrape Status Breakdown

| Status | Count |
|--------|-------|
| Success | 173,678 |
| Error | 61,473 |
| Success (Archived) | 59,628 |
| Empty_Content | 6,228 |

| **Successful Scrapes** | 233,306 |
| **Scrape Success Rate** | 77.51% |
| **Duplicate URL Rows** | 192,941 |

#### URL Uniqueness

![URL Uniqueness](07_scraped_url_uniqueness.png)

---

### Article Length Distribution

![Article Length](10_article_length.png)

#### Statistics (Included Articles)

| Metric | Value |
|--------|-------|
| **Count** | 67,022 |
| **Min** | 40 words |
| **25th Percentile** | 365 words |
| **Median** | 588 words |
| **Mean** | 755 words |
| **75th Percentile** | 925 words |
| **99th Percentile** | 3,447 words |
| **Max** | 18,243 words |

---

### Relevance Score Distribution

![Relevance Score](11_relevance_score.png)

#### Statistics by Inclusion Status (In-Scope Articles)

| Metric | Included | Excluded |
|--------|----------|----------|
| **Count** | 67,022 | 22,325 |
| **Min** | 25.0 | -22.8 |
| **Median** | 58.3 | 26.1 |
| **Mean** | 58.5 | 32.8 |
| **Max** | 121.1 | 117.6 |

---

### Filter Stage Breakdown

![Filter Stage Breakdown](12_filter_stage_breakdown.png)

#### Drop Count by Stage (In-Scope Articles, Stages Not Mutually Exclusive)

| Stage | Dropped | % of In-Scope |
|-------|---------|---------------|
| **Duplicate** | 12,688 | 14.2% |
| **Length** | 3,928 | 4.4% |
| **Score** | 10,844 | 12.1% |
| **Anchor** | 16,006 | 17.9% |

---

### Content Filter Funnel

![Content Filter Funnel](13_filter_funnel.png)

#### Funnel Steps

| Step | Articles | Removed |
|------|----------|---------|
| **Total Scraped** | 107,934 | — |
| **After Scope Filter** | 89,347 | 18,587 out of scope |
| **After Content Filters** | 67,022 | 22,325 failed filters |

---

*Generated: 2026-03-27*
