from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.clustering import cluster, embedder, summarizer
from analysis.sentiment import roberta_analyzer as sentiment
from analysis.topic import bertopic_model as topic


def test_sentiment_batch_and_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pipe(batch: list[str]) -> list[dict[str, float | str]]:
        out: list[dict[str, float | str]] = []
        for text in batch:
            t = text.lower()
            if "good" in t:
                out.append({"label": "POSITIVE", "score": 0.9})
            elif "bad" in t:
                out.append({"label": "NEGATIVE", "score": 0.8})
            else:
                out.append({"label": "NEUTRAL", "score": 0.5})
        return out

    monkeypatch.setattr(sentiment, "_load_model", lambda model_name="x": fake_pipe)
    results = sentiment.analyze_sentiment_batch(["good news", "bad news", ""], batch_size=2)
    assert [r["label"] for r in results] == ["positive", "negative", "neutral"]
    assert results[0]["sentiment_score"] == pytest.approx(0.9)
    assert results[1]["sentiment_score"] == pytest.approx(-0.8)

    monkeypatch.setattr(
        sentiment,
        "_load_model",
        lambda model_name="x": (lambda batch: (_ for _ in ()).throw(RuntimeError("fail"))),
    )
    fallback = sentiment.analyze_sentiment_batch(["a", "b"], batch_size=2)
    assert all(r["label"] == "neutral" for r in fallback)

    monkeypatch.setattr(
        sentiment,
        "analyze_sentiment_batch",
        lambda texts, model_name, batch_size: [
            {"label": "positive", "confidence": 0.7, "sentiment_score": 0.7}
            for _ in texts
        ],
    )
    df = sentiment.analyze_dataframe(pd.DataFrame({"text": ["one", "two"]}))
    assert df["sentiment_label"].tolist() == ["positive", "positive"]


def test_sentiment_aggregation_and_summary() -> None:
    df = pd.DataFrame(
        {
            "source_domain": ["a", "a", "b"],
            "year_month": ["2020-01", "2020-01", "2020-02"],
            "sentiment_label": ["positive", "negative", "neutral"],
            "sentiment_score": [0.8, -0.4, 0.0],
        }
    )
    agg = sentiment.aggregate_sentiment(df, ["source_domain", "year_month"])
    assert set(agg.columns) >= {
        "source_domain",
        "year_month",
        "mean_sentiment",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
    }
    summary = sentiment.get_sentiment_summary(df)
    assert summary["total_records"] == 3
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1


def test_topic_helpers_and_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTopicModel:
        def __init__(self) -> None:
            self.saved_path = None

        def fit_transform(self, texts: list[str]) -> tuple[list[int], list[float]]:
            return [0 if i % 2 == 0 else 1 for i in range(len(texts))], [0.9] * len(texts)

        def _extract_embeddings(self, texts: list[str]) -> np.ndarray:
            return np.array([[float(i), float(i + 1)] for i in range(len(texts))], dtype=float)

        def get_topic_info(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Topic": [-1, 0, 1],
                    "Name": ["outliers", "topic zero", "topic one"],
                    "Count": [1, 2, 2],
                }
            )

        def get_topic(self, topic_id: int) -> list[tuple[str, float]]:
            return [("venezuela", 0.5), ("us", 0.4)]

        def topics_over_time(self, docs: list[str], timestamps: list[object], nr_bins: int = 20) -> pd.DataFrame:
            return pd.DataFrame({"Topic": [0], "Timestamp": ["2020-01"], "Frequency": [1]})

        def save(self, path: str) -> None:
            self.saved_path = path

    captured: dict[str, object] = {}

    def fake_create_model(
        embedding_model: str = "x",
        n_topics: int | None = None,
        min_topic_size: int = 50,
        n_gram_range: tuple[int, int] = (1, 2),
    ) -> FakeTopicModel:
        captured["min_topic_size"] = min_topic_size
        return FakeTopicModel()

    monkeypatch.setattr(topic, "create_bertopic_model", fake_create_model)
    df = pd.DataFrame({"text": [f"doc {i}" for i in range(40)]})
    out_df, model, embeddings = topic.fit_topics(df, min_topic_size=50)
    assert len(out_df) == 40
    assert "topic_id" in out_df.columns
    assert embeddings.shape == (40, 2)
    assert captured["min_topic_size"] == 10

    info = topic.get_topic_info(model)
    assert set(info["Topic"].tolist()) == {-1, 0, 1}
    keywords = topic.get_topic_keywords(model, 0, n_words=1)
    assert keywords == [("venezuela", 0.5)]
    over_time = topic.topics_over_time(model, ["a"], ["2020-01"], nr_bins=5)
    assert len(over_time) == 1

    docs = topic.get_representative_docs(model, 0, out_df.assign(topic_prob=0.5), n_docs=5)
    assert len(docs) <= 5

    agg = topic.aggregate_topics_by_group(
        pd.DataFrame(
            {
                "source_domain": ["a", "a", "b"],
                "year_month": ["2020-01", "2020-01", "2020-02"],
                "topic_id": [0, 1, 0],
            }
        ),
        group_by=["source_domain", "year_month"],
    )
    assert "proportion" in agg.columns

    topic.save_topic_model(model, "mock_path")
    assert model.saved_path == "mock_path"

    fake_bertopic = types.ModuleType("bertopic")

    class FakeBERTopic:
        @staticmethod
        def load(path: str) -> object:
            return {"loaded_from": path}

    fake_bertopic.BERTopic = FakeBERTopic
    monkeypatch.setitem(sys.modules, "bertopic", fake_bertopic)
    loaded = topic.load_topic_model("model_dir")
    assert loaded == {"loaded_from": "model_dir"}


def test_embedder_and_dimensionality_reduction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    emb = embedder.TextEmbedder(model_name="fake", batch_size=2)

    class FakeSentenceModel:
        def encode(
            self,
            texts: list[str],
            batch_size: int,
            show_progress_bar: bool,
            convert_to_numpy: bool,
        ) -> np.ndarray:
            return np.array([[float(i), float(i + 1), float(i + 2)] for i in range(len(texts))], dtype=float)

    monkeypatch.setattr(emb, "_load_model", lambda: FakeSentenceModel())
    vectors = emb.embed_texts(["a", "b"], ids=["id1", "id2"])
    assert vectors.shape == (2, 3)
    assert emb.get_embedding_by_id("id1") is not None
    assert emb.get_ids_by_indices([0, 1]) == ["id1", "id2"]

    df = pd.DataFrame(
        {
            "id": ["id1", "id2"],
            "text": ["alpha", "beta"],
            "type": ["event", "event"],
            "subreddit": ["na", "na"],
            "year_month": ["2020-01", "2020-01"],
            "created_utc": [1, 2],
        }
    )
    embs, index_df = emb.embed_dataframe(df)
    assert embs.shape[0] == len(index_df)

    parquet_store: dict[str, pd.DataFrame] = {}

    def fake_to_parquet(self: pd.DataFrame, path: str | Path, index: bool = False) -> None:
        parquet_store[str(path)] = self.copy()

    def fake_read_parquet(path: str | Path) -> pd.DataFrame:
        return parquet_store[str(path)].copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    emb.save(tmp_path)
    emb2 = embedder.TextEmbedder(model_name="fake")
    emb2.load(tmp_path)
    assert emb2.embeddings is not None
    assert emb2.get_embedding_by_id("id2") is not None

    fake_umap = types.ModuleType("umap")

    class FakeUMAP:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def fit_transform(self, arr: np.ndarray) -> np.ndarray:
            return arr[:, :2]

    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)
    reduced = embedder.reduce_dimensions(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), n_components=2)
    assert reduced.shape == (2, 2)


def test_temporal_clusterer_and_evolution(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_hdbscan = types.ModuleType("hdbscan")

    class FakeHDBSCAN:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.labels_ = None
            self.probabilities_ = None

        def fit(self, data: np.ndarray) -> "FakeHDBSCAN":
            self.labels_ = np.array([0, 0, 1, -1, 1, 1])
            self.probabilities_ = np.array([0.9, 0.8, 0.7, 0.1, 0.85, 0.88])
            return self

        def fit_predict(self, data: np.ndarray) -> np.ndarray:
            self.probabilities_ = np.full(len(data), 0.6, dtype=float)
            return np.array([0 if i % 2 == 0 else 1 for i in range(len(data))])

    fake_hdbscan.HDBSCAN = FakeHDBSCAN
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)

    fake_umap = types.ModuleType("umap")

    class FakeUMAP:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, arr: np.ndarray) -> np.ndarray:
            return arr[:, :2]

    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)

    tc = cluster.TemporalClusterer(min_cluster_size=2, min_samples=1, group_column="source_domain")
    embeddings = np.array(
        [
            [1.0, 0.0, 0.1],
            [1.1, 0.1, 0.2],
            [0.0, 1.0, 0.2],
            [0.2, 0.2, 1.0],
            [0.0, 1.1, 0.3],
            [0.1, 0.9, 0.2],
        ]
    )
    labels = tc.fit(embeddings, reduce_first=True, n_components=2)
    assert len(labels) == 6

    df = pd.DataFrame(
        {
            "id": [f"gdelt_{i}" for i in range(1, 7)],
            "source_domain": ["a", "a", "b", "c", "b", "b"],
            "created_datetime": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-15", "2020-03-01", "2020-03-05"]
            ),
            "year_month": ["2020-01", "2020-01", "2020-02", "2020-02", "2020-03", "2020-03"],
            "sentiment_score": [0.1, 0.2, -0.3, 0.0, -0.1, -0.2],
            "avg_tone": [0.2, 0.3, -0.1, 0.0, -0.2, -0.3],
            "goldstein_scale": [1, 2, -1, 0, -2, -3],
        }
    )
    dfc = tc.add_clusters_to_df(df, labels)
    assert "cluster_id" in dfc.columns
    assert tc.get_cluster_ids(0)

    summary = tc.get_cluster_summary(dfc)
    assert {"cluster_id", "count", "top_group"}.issubset(summary.columns)
    temporal = tc.get_temporal_clusters(dfc)
    assert "proportion" in temporal.columns

    by_period = tc.cluster_by_period(
        dfc,
        embeddings,
        periods=["2020-01", "2020-02", "2020-03"],
        time_column="year_month",
    )
    assert set(by_period.keys()) == {"2020-01", "2020-02", "2020-03"}

    evo_df = cluster.track_cluster_evolution(
        {
            "2020-01": dfc.iloc[:3].assign(cluster_id=[0, 0, 1]),
            "2020-02": dfc.iloc[3:].assign(cluster_id=[1, 1, 0]),
        },
        embeddings,
        dfc,
        similarity_threshold=0.0,
    )
    assert not evo_df.empty
    assert {"from_period", "to_period", "similarity"}.issubset(evo_df.columns)


def test_cluster_summarizer_helpers_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, -1],
            "cluster_prob": [0.9, 0.8, 0.7, 0.95, 0.1],
            "text": [
                "Venezuela sanctions update from US officials.",
                "Diplomatic talks continue amid tensions.",
                "Economic pressure and oil policy changes.",
                "Regional cooperation story.",
                "noise",
            ],
            "source_domain": ["a.com", "a.com", "b.com", "c.com", "noise.com"],
            "sentiment_score": [0.2, -0.1, -0.3, 0.5, 0.0],
            "avg_tone": [0.1, -0.2, -0.4, 0.6, 0.0],
            "goldstein_scale": [1, -2, -3, 2, 0],
            "created_datetime": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01", "2020-03-01"]
            ),
        }
    )

    samples = summarizer.sample_cluster_texts(df, 0, n_samples=2, text_column="text", random_state=7)
    assert len(samples) == 2
    assert samples[0].startswith("Venezuela")

    prompt_text = summarizer.format_samples_for_prompt(["x" * 20, "y" * 20], max_chars_per_text=10, max_total_chars=30)
    assert "1." in prompt_text
    prompt = summarizer.create_summary_prompt(samples, 0, metadata={"count": 3, "top_group": "a.com"})
    assert "Cluster #0" in prompt
    assert "Dominant source domain: a.com" in prompt

    fake_openai = types.ModuleType("openai")

    class FakeCompletions:
        @staticmethod
        def create(**kwargs: object) -> object:
            payload = (
                '{"theme":"policy","summary":"desc","key_topics":["a","b"],'
                '"perspective":"mixed","tone":"critical"}'
            )
            message = types.SimpleNamespace(content=payload)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAIClient:
        def __init__(self, api_key: str) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai.OpenAI = FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    openai_summary = summarizer.summarize_cluster_with_openai(samples, 0, api_key="x")
    assert openai_summary["theme"] == "policy"

    fake_anthropic = types.ModuleType("anthropic")

    class FakeAnthropicClient:
        def __init__(self, api_key: str) -> None:
            self.messages = types.SimpleNamespace(
                create=lambda **kwargs: types.SimpleNamespace(
                    content=[
                        types.SimpleNamespace(
                            text='{"theme":"t","summary":"s","key_topics":["k"],"perspective":"p","tone":"n"}'
                        )
                    ]
                )
            )

    fake_anthropic.Anthropic = FakeAnthropicClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    anthropic_summary = summarizer.summarize_cluster_with_anthropic(samples, 1, api_key="y")
    assert anthropic_summary["theme"] == "t"

    monkeypatch.setattr(
        summarizer,
        "summarize_cluster_with_openai",
        lambda texts, cluster_id, metadata, model, api_key: {"cluster_id": cluster_id, "theme": "openai"},
    )
    monkeypatch.setattr(
        summarizer,
        "summarize_cluster_with_anthropic",
        lambda texts, cluster_id, metadata, model, api_key: {"cluster_id": cluster_id, "theme": "anthropic"},
    )
    all_openai = summarizer.summarize_all_clusters(df, n_samples=2, llm_provider="openai")
    all_anthropic = summarizer.summarize_all_clusters(df, n_samples=2, llm_provider="anthropic")
    assert set(all_openai["theme"]) == {"openai"}
    assert set(all_anthropic["theme"]) == {"anthropic"}

    fake_sklearn = types.ModuleType("sklearn")
    fake_feature_extraction = types.ModuleType("sklearn.feature_extraction")
    fake_text_mod = types.ModuleType("sklearn.feature_extraction.text")

    class FakeTfidfMatrix:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def mean(self, axis: int = 0) -> object:
            return types.SimpleNamespace(A1=self.arr.mean(axis=axis))

    class FakeTfidfVectorizer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, texts: list[str]) -> FakeTfidfMatrix:
            return FakeTfidfMatrix(np.array([[0.1, 0.5, 0.2], [0.2, 0.4, 0.1]], dtype=float))

        def get_feature_names_out(self) -> np.ndarray:
            return np.array(["venezuela", "sanctions", "policy"])

    fake_text_mod.TfidfVectorizer = FakeTfidfVectorizer
    fake_feature_extraction.text = fake_text_mod
    fake_sklearn.feature_extraction = fake_feature_extraction
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction", fake_feature_extraction)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", fake_text_mod)

    keywords = summarizer.generate_keyword_summary(df, cluster_id=0, n_keywords=2)
    assert keywords == ["sanctions", "policy"]
    assert summarizer.generate_keyword_summary(df, cluster_id=999) == []
