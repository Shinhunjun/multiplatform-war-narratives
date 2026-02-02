# Presentation Script
## Cross-Platform Narrative Analysis of U.S.-Venezuela Relations

**Total Duration:** ~8 minutes

---

## Slide 1: Title (30 seconds)

> Hello everyone. Today, I'll be presenting our team's project: **"Cross-Platform Narrative Analysis of U.S.-Venezuela Relations."**
>
> Our goal is to analyze how narratives about U.S.-Venezuela relations from 2013 to 2025 are constructed differently across news media and social media platforms, and to build an Interactive Dashboard that visualizes these differences in real-time.
>
> We are targeting submission to ICWSM 2026 with our findings.

---

## Slide 2: Problem Statement (45 seconds)

> Let me start with the problem we're addressing.
>
> When the same geopolitical event occurs, U.S. news, Venezuelan news, and social media platforms like Reddit describe that event in very different ways.
>
> For example, when Guaidó declared himself interim president in 2019, U.S. media framed it as "defending democracy," while Venezuelan government-aligned media called it a "coup attempt." On Reddit, these two perspectives appeared differently depending on the subreddit community.
>
> Existing research has been mostly limited to single platforms and static analysis. We aim to fill this gap by building a multi-platform, real-time analysis system.

---

## Slide 3: Research Questions (40 seconds)

> We have established four Research Questions.
>
> **RQ1**: Do U.S. and Venezuelan news media construct different narratives when reporting on the same geopolitical events?
>
> **RQ2**: How do narratives in social media discourse, specifically Reddit, differ from those in traditional news media?
>
> **RQ3**: To what extent do social media narratives mirror, amplify, or diverge from national news media narratives?
>
> **RQ4**: How do narrative differences across media systems evolve around major geopolitical events?

---

## Slide 4: Team Objectives & Roles (40 seconds)

> Our team has four main objectives.
>
> First, to derive quantitative answers to our four Research Questions.
> Second, to analyze structural relationships through Knowledge Graph construction.
> Third, to deploy a Real-time Interactive Dashboard.
> Fourth, to submit our findings to ICWSM 2026.
>
> Our team consists of three members.
> I was responsible for Reddit data collection, Rich handled GDELT news data, and Ameir managed TikTok data collection.
> Analysis, Knowledge Graph construction, and Web App development are collaborative efforts among all three of us.

---

## Slide 5: Datasets (50 seconds)

> Here are the datasets we're using.
>
> **Reddit data** was collected via the Arctic Shift API. From 2013 to January 2026, we collected a total of 426,435 posts. This includes 11 subreddits: Venezuelan communities like r/vzla and r/venezuela, and U.S. communities like r/politics, r/worldnews, and r/Conservative.
>
> **GDELT** stands for Global Database of Events, Language, and Tone. It provides worldwide news events in an actor-action-actor structure. We filtered events related to the United States and Venezuela.
>
> We are also collecting **TikTok** data as an additional source.
>
> Key challenges include filtering deleted content, cross-platform event alignment, and handling multilingual text in English and Spanish.

---

## Slide 6: Related Work (45 seconds)

> Let me briefly introduce related work.
>
> Kwak and An's 2016 ICWSM paper analyzed GDELT dataset characteristics, but did not address narrative framing.
>
> Olteanu et al.'s 2015 study examined the event coverage gap between news and social media, but focused only on frequency without text content analysis.
>
> Zhao et al.'s 2024 EMNLP paper proposed event-centric framing methods, but was limited to a single platform.
>
> Our research fills these gaps by being the first to construct **event-aligned multi-platform narrative comparison** with a **real-time knowledge graph**.

---

## Slide 7: System Architecture (1 minute)

> This is our system architecture.
>
> At the top, the **Data Layer** collects data from GDELT, Reddit, and TikTok. This also includes scheduled jobs for real-time streaming.
>
> In the middle, the **Processing Layer** performs three types of analysis:
> - **Sentiment Analysis** using the RoBERTa model
> - **Topic Modeling** using BERTopic
> - **Knowledge Graph Construction** through Named Entity Recognition and Relation Extraction
>
> Analysis results are stored in the **Neo4j Knowledge Graph**. Here, we store relationships between entities like Maduro, US Government, and Sanctions, along with sentiment, topic, and temporal information.
>
> Finally, the **Interactive Web App** provides a Dashboard, Knowledge Graph Explorer, Timeline View, and Real-time Feed.

---

## Slide 8: Knowledge Graph Design (45 seconds)

> Here's our Knowledge Graph design.
>
> For **Entities**, we extract key figures like Maduro, Guaidó, Trump, and Biden; organizations like US Government and Venezuelan Opposition; and events like Sanctions, Protests, and Elections.
>
> For **Relations**, we define relationships such as SUPPORTS, OPPOSES, PARTICIPATES_IN, and TRIGGERS.
>
> A crucial aspect is **Temporal Versioning**. We store weekly or monthly snapshots to track how the graph evolves over time.
>
> For example, in January 2019, the US-Guaidó relationship appears strongly as SUPPORTS, while the Maduro-US relationship appears as OPPOSES.

---

## Slide 9: Interactive Web Application (45 seconds)

> Here are the main features of our Web Application.
>
> The **Dashboard** displays time-series charts of sentiment and topic changes, with filtering by platform and time period.
>
> The **Knowledge Graph Explorer** uses D3.js or Cytoscape for interactive graph exploration. Clicking on a specific entity shows related posts and sentiment scores.
>
> The **Timeline View** compares each platform's response around key events.
>
> The **Real-time Feed** automatically collects new data weekly or monthly and updates the analysis results.
>
> Our tech stack includes React or Streamlit for the frontend, FastAPI for the backend, PostgreSQL and Neo4j for databases, and Docker for deployment.

---

## Slide 10: Progress & Next Steps (45 seconds)

> Here's our current progress.
>
> **Completed work:**
> - Reddit data collection: 426,435 posts
> - Reddit preprocessing: bot removal, deleted content filtering
> - GDELT data download complete
>
> **In progress:**
> - TikTok data collection
> - GDELT-Reddit event alignment
>
> **Planned work:**
> - Sentiment Analysis with RoBERTa
> - Topic Modeling with BERTopic
> - Knowledge Graph construction
> - Web App development
> - Real-time integration
> - ICWSM paper writing

---

## Slide 11: Expected Outcomes (30 seconds)

> Finally, our expected outcomes.
>
> **Academic**: We plan to submit a paper to ICWSM 2026 presenting our analysis results for RQ1 through RQ4.
>
> **Dataset**: We will release an event-aligned multi-platform corpus to contribute to future research.
>
> **Product**: We will deploy an Interactive Dashboard with real-time updates that researchers and journalists can utilize.
>
> **Code**: All analysis pipelines will be open-sourced on GitHub for reproducibility.
>
> That concludes our presentation. I'm happy to take any questions.

---

## Timing Summary

| Slide | Duration | Key Message |
|-------|----------|-------------|
| 1. Title | 30 sec | Project introduction, ICWSM goal |
| 2. Problem | 45 sec | Why this research is needed |
| 3. RQ | 40 sec | Four Research Questions |
| 4. Team | 40 sec | Objectives and role distribution |
| 5. Datasets | 50 sec | Data scale and characteristics |
| 6. Related Work | 45 sec | Existing research and gaps |
| 7. Architecture | 1 min | Overall system structure |
| 8. KG Design | 45 sec | Knowledge Graph design |
| 9. Web App | 45 sec | Interactive features |
| 10. Progress | 45 sec | Current status and next steps |
| 11. Outcomes | 30 sec | Expected deliverables |
| **Total** | **~8 min** | |

---

## Q&A Preparation

### Anticipated Questions and Answers

**Q1: Why focus on US-Venezuela relations specifically?**
> US-Venezuela relations provide rich data and diverse narratives due to continuous political tensions and major events since 2013, including sanctions, the Guaidó crisis, and election controversies. Additionally, there is active online discourse in both English and Spanish, making it ideal for cross-national comparison.

**Q2: How do you handle multilingual data (English/Spanish)?**
> For Reddit data, r/vzla is primarily in Spanish while r/politics is in English. We plan to use multilingual transformer models like XLM-RoBERTa, or analyze each language separately and then compare the results.

**Q3: What makes your approach different from existing work?**
> Existing research has primarily focused on single platforms (either news or social media) and static analysis. Our work is the first to combine: (1) multi-platform integrated analysis, (2) event-aligned comparison, (3) temporal knowledge graph, and (4) real-time dashboard.

**Q4: How do you align events across platforms?**
> We use GDELT's event timestamps and event codes as anchors. Reddit posts are matched by finding related keyword posts within a specific time window (e.g., 7 days) after the event occurrence.

**Q5: What are the limitations of your approach?**
> - Reddit may be biased toward certain demographics
> - GDELT's event coding may contain errors
> - Technical constraints in TikTok data collection
> - Computational costs of real-time analysis

**Q6: Why did you choose BERTopic for topic modeling?**
> BERTopic leverages transformer-based embeddings, which capture semantic meaning better than traditional methods like LDA. It also provides interpretable topic representations and handles the dynamic nature of social media text well.

**Q7: How will you validate your Knowledge Graph?**
> We plan to use a combination of automated validation (checking relation consistency) and manual annotation on a sample subset. We'll also compare our extracted relations against known events documented in news sources.

**Q8: What's your timeline for ICWSM submission?**
> ICWSM 2026 submission deadline is expected around January 2026. We aim to complete data integration and core analysis by Fall 2025, allowing time for paper writing and revision.

---

## Notes for Presenter

### Key Points to Emphasize
1. **ICWSM submission** - Emphasize academic goals
2. **Real-time + Interactive** - Key differentiator from existing research
3. **Knowledge Graph** - Novelty in structural relationship analysis
4. **Multi-platform** - Integration of Reddit + GDELT + TikTok

### Visual Aids
- Slide 7 (Architecture): Most important slide, explain slowly
- Slide 8 (KG Design): Use concrete examples to illustrate
- Slide 10 (Progress): Be honest about current status

### Delivery Tips
- Maintain eye contact with audience
- Pause briefly after each RQ to let them sink in
- Point to specific parts of the architecture diagram
- Show enthusiasm when discussing novel contributions

### Backup Slides (if needed)
- Detailed EDA results from Reddit
- Sample Knowledge Graph visualization
- Tech stack comparison table
- ICWSM submission timeline details
- Reddit data distribution by subreddit
- Key events timeline with post counts

---

*Generated: 2026-02-02*
