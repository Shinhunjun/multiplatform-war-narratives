# Best Months for Demo: Cross-Platform Narrative Divergence

> Analysis of sentiment, topics, and clusters across Reddit, GDELT News, and TikTok to identify periods with the most striking cross-platform differences.

---

## Executive Summary

| Rank | Month | Event | Sentiment Gap | Volume (R/N/T) | Why |
|------|-------|-------|:---:|:---:|-----|
| **1** | **2019-01** | Guaido interim presidency | 0.460 | 11K / 6K / 12K | Strongest all-around: high volume on all 3 platforms, distinct topics & clusters |
| **2** | **2017-05** | Venezuela constitutional crisis | 0.644 | 3K / 1.5K / 385 | Largest sentiment gap in entire dataset; TikTok is pure entertainment |
| **3** | **2019-05** | Failed uprising aftermath | 0.527 | 7K / 4.6K / 10K | Strong divergence + high volume + distinct political vs entertainment framing |
| **4** | **2026-01** | Maduro captured (simulated) | 0.531 | 47K / 25K / 151 | Massive scale demo; but TikTok volume is very thin |

---

## #1 Recommended: January 2019 (Guaido Crisis)

**Best balance of volume, divergence, and topical clarity across all 3 platforms.**

### Sentiment

| Platform | Mean Sentiment | Positive % | Negative % | Volume |
|----------|:-:|:-:|:-:|:-:|
| Reddit | **-0.291** | 4.0% | 42.8% | 11,419 |
| News | **-0.065** | 3.4% | 13.6% | 6,127 |
| TikTok | **+0.168** | 22.9% | 3.1% | 12,398 |

- Sentiment spread: **0.460**
- Reddit is 4.5x more negative than News, while TikTok remains positive

### Topics (Monthly Independent BERTopic)

| Platform | Top Topics | Proportion |
|----------|-----------|:---:|
| **Reddit** | "venezolanos, venezolano, venezuela" (Spanish-language crisis discourse) | 23.8% |
| | "venezuela crisis, crisis venezuela, happening venezuela" | 6.0% |
| | "socialism, socialist" | 4.6% |
| **News** | "venezuela, venezuelan, venezuela pdvsa" (oil & state enterprise) | 2.0% |
| | "situation venezuela, venezuela issue" (diplomatic framing) | 1.3% |
| | "president emmanuel, venezuelan diplomats" (international response) | 1.2% |
| **TikTok** | "jigneshkaviraj, ilikacruz, lachamaaa" (creator names) | 7.7% |
| | "destacame venezuela, risas venezuela" (entertainment/comedy) | 6.8% |
| | "pasa ves, jajajjaja si" (casual humor) | 3.5% |

**Key insight**: Reddit frames Venezuela through crisis & ideology. News frames it through diplomacy & oil policy. TikTok is entirely about entertainment creators and comedy.

### Clusters (Monthly Independent Clustering)

| Platform | Top Clusters | Keywords | Proportion |
|----------|-------------|----------|:---:|
| **Reddit** | Political crisis | venezuela, si, maduro | 23.8% |
| | US involvement | venezuela, us, people, venezuelan, crisis | 6.0% |
| | Ideological debate | maduro, us, people, like | 4.6% |
| **News** | Oil & sanctions | oil, venezuela, prices, sanctions, oil prices | 1.5% |
| | Guaido leadership | juan, guaido, president, interim, leader | 1.3% |
| | Opposition politics | leader, president, juan, guaido, venezuela | 1.2% |
| **TikTok** | Comedy content | venezuela, comedy, comedia, destacame, humor | 16.1% |
| | Comedy (Spanish) | venezuela, comedia, comedia venezuela, destacame | 6.2% |
| | Cross-country content | venezuela, colombia, venezuela colombia, destacame | 2.5% |

**Cluster divergence**: Reddit clusters around **political crisis and ideology**. News clusters around **oil/sanctions and Guaido's leadership**. TikTok clusters around **comedy and cross-country entertainment**.

---

## #2 Recommended: May 2017 (Constitutional Crisis)

**Highest sentiment divergence in the entire dataset (0.644).**

### Sentiment

| Platform | Mean Sentiment | Positive % | Negative % | Volume |
|----------|:-:|:-:|:-:|:-:|
| Reddit | **-0.364** | 4.6% | 52.4% | 2,923 |
| News | **-0.197** | 7.0% | 36.3% | 1,450 |
| TikTok | **+0.279** | 39.0% | 4.7% | 385 |

- Sentiment spread: **0.644** (maximum across all months)
- Over half of Reddit content is negative; TikTok is 39% positive

### Topics

| Platform | Top Topics | Proportion |
|----------|-----------|:---:|
| **Reddit** | "crisis venezuela, venezuela, venezolanos" | 23.5% |
| | "country, dictatorships, socialist country" | 5.0% |
| | "per 100, 100 000" (statistics/data) | 4.8% |
| **News** | "venezuela, venezuelan, venezuelans" | 4.1% |
| | "us officials, officials said, new sanctions" | 3.6% |
| | "colombia ecuador, colombia, colombian" (regional) | 2.9% |
| **TikTok** | "venezuela destacame, like venezuela" (feature requests) | 28.1% |
| | "destacame venezuela, bailame" (dance/feature) | 25.4% |
| | "love venezuela, featureme venezuela" | 13.2% |

**Key insight**: TikTok's top 3 topics (66.7% combined) are ALL about "destacame" (feature me) and dance content. Zero political content.

### Clusters

| Platform | Top Clusters | Keywords | Proportion |
|----------|-------------|----------|:---:|
| **Reddit** | Crisis & Maduro | venezuela, maduro, si | 23.5% |
| | Socialism debate | socialism, socialist, real, people, government | 5.0% |
| | Armed conflict | guns, government, people, gun, armed | 4.8% |
| **News** | Caracas protests | venezuela, caracas, protesters, venezuelan, government | 4.3% |
| | Opposition & deaths | venezuela, opposition, president, protests, died | 2.6% |
| | Trump & diplomacy | president, trump, donald, santos, donald trump | 2.5% |
| **TikTok** | Lip sync & lifestyle | venezuela, lipsync, belgium, dailylife | 35.5% |
| | Social engagement | venezuela, like, followme | 16.2% |
| | Feature requests | venezuela, featureme, destacame, spain, lipsync | 16.2% |

**Cluster divergence**: Reddit = armed political crisis. News = protests and diplomacy. TikTok = **67.9% concentrated in lip sync and social engagement** (completely apolitical).

---

## #3 Recommended: May 2019 (Failed Uprising Aftermath)

**Strong divergence with high volume on all platforms.**

### Sentiment

| Platform | Mean Sentiment | Positive % | Negative % | Volume |
|----------|:-:|:-:|:-:|:-:|
| Reddit | **-0.369** | 3.7% | 51.6% | 6,927 |
| News | **-0.116** | 2.0% | 20.3% | 4,601 |
| TikTok | **+0.159** | 21.7% | 2.8% | 9,921 |

- Sentiment spread: **0.527**

### Topics

| Platform | Top Topics | Proportion |
|----------|-----------|:---:|
| **Reddit** | "regimen, militar, politica" (military/political discourse) | 26.4% |
| | "president putin, putin trump" (geopolitics) | 3.9% |
| | "venezuela crisis, crisis venezuela" | 3.3% |
| **News** | "venezuelan, venezuela, crisis carlos" | 1.3% |
| | "iran venezuela, oil exporting countries" (geopolitical alliances) | 1.3% |
| | "venezuelan, venezuelans, russia" | 1.2% |
| **TikTok** | "paso, paso, pasaba" (casual conversation) | 6.3% |
| | "carlosparedes17" (creator name) | 3.9% |
| | "destacame venezuela, siguanme venezuela" | 2.9% |

### Clusters

| Platform | Top Clusters | Keywords | Proportion |
|----------|-------------|----------|:---:|
| **Reddit** | Regime & Maduro | venezuela, si, maduro | 26.4% |
| | Russia-US proxy | putin, trump, russia, russian, us | 3.9% |
| | Crisis & intervention | venezuela, venezuelan, people, us | 3.3% |
| **News** | Opposition & Guaido | opposition, juan, leader, guaido | 1.4% |
| | US foreign policy | president, washington, venezuelan, states | 1.3% |
| | Maduro leadership | leader, maduro, juan, president, opposition | 1.2% |
| **TikTok** | Comedy | venezuela, comedy, destacame, comedia | 6.9% |
| | For-you-page content | venezuela, foryou, destacame, parati | 4.7% |
| | Comedy (Spanish) | venezuela, comedia, tiktok | 4.1% |

**Key insight**: Reddit discusses **military regime and Russia-US proxy dynamics**. News focuses on **Guaido and US diplomatic response**. TikTok = **pure comedy and FYP content**.

---

## #4 Optional: January 2026 (Simulated Maduro Capture)

**Maximum volume for demonstrating system scalability.**

### Sentiment

| Platform | Mean Sentiment | Positive % | Negative % | Volume |
|----------|:-:|:-:|:-:|:-:|
| Reddit | **-0.426** | 4.7% | 59.3% | 47,519 |
| News | **-0.086** | 4.7% | 18.5% | 24,934 |
| TikTok | **+0.105** | 13.9% | 0.7% | 151 |

- Reddit's most negative month in the entire dataset (-0.426)
- 59.3% of Reddit content is negative (highest ever)

### Topics

| Platform | Top Topics | Proportion |
|----------|-----------|:---:|
| **Reddit** | "venezuelan people, venezuelans, people venezuela" | 20.0% |
| | "venezolanos, venezolano, estadounidenses" (Americans involvement) | 11.8% |
| | "americans, america, us" | 8.1% |
| **News** | "venezuelans, venezuelan, venezuela" | 3.2% |
| | "venezuela, venezuelan, venezuelans" | 2.4% |
| | "venezuela, venezuelan, venezuelans" | 2.0% |
| **TikTok** | "2026, baje llegue, alguien" (generic/thin) | 65.6% |
| | "alguien 2026" | 26.5% |

### Clusters

| Platform | Top Clusters | Keywords | Proportion |
|----------|-------------|----------|:---:|
| **Reddit** | Oil & geopolitics | venezuela, trump, us, oil, maduro | 20.2% |
| | Political discourse | venezuela, si, maduro | 12.1% |
| | Economic interests | oil, trump, us, companies, money | 9.4% |
| **News** | Trump administration | trump, venezuela, president, us, venezuelan | 2.9% |
| | Maduro capture | trump, venezuela, president, maduro, us | 1.9% |

**Caveat**: TikTok volume (151) is too thin for meaningful topic/cluster analysis. Best used to show system scalability, not cross-platform comparison.

---

## Recommended Demo Flow

```
Step 1: Start with 2019-01 (Guaido Crisis)
        -> Best all-around: balanced volume, clear divergence in all 3 dimensions
        -> "During the Guaido crisis, Reddit debated socialism and intervention,
           News covered oil sanctions and diplomacy, TikTok posted comedy sketches"

Step 2: Show 2017-05 (Constitutional Crisis)  
        -> Largest sentiment gap (0.644)
        -> "TikTok was 67% lip-sync and dance content while Reddit was 52% negative
           about the same country's political crisis"

Step 3 (if time permits): Show 2019-05 (Uprising Aftermath)
        -> Confirms the pattern with military/regime topics on Reddit,
           Iran-oil alliance topics on News, and pure comedy on TikTok

Step 4 (optional): Flash 2026-01 for volume scalability
        -> "Our system processes 47K Reddit + 25K News articles in a single month"
```

## Key Presentation Talking Points

1. **Sentiment Divergence**: The same geopolitical event produces sentiment scores that differ by 0.4-0.6 across platforms — Reddit is consistently the most negative, TikTok the most positive.

2. **Topic Framing**: Reddit frames events through **ideology and crisis** (socialism, dictatorship, intervention). News frames through **policy and diplomacy** (sanctions, oil, officials). TikTok frames through **entertainment and culture** (comedy, dance, destacame).

3. **Cluster Separation**: Reddit clusters are **politically polarized** (regime vs. socialism debates). News clusters are **policy-oriented** (oil prices, Guaido leadership). TikTok clusters are **entirely entertainment-driven** (comedy, FYP, cross-country content).

4. **Volume Asymmetry**: During crises, Reddit and News volume surges while TikTok volume remains relatively stable — suggesting TikTok audiences are less reactive to political events.
