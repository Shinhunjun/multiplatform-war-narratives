from __future__ import annotations

import sys
import types
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from analysis import main as analysis_main
from analysis.clustering import temporal_viz
from analysis.config import AnalysisConfig
from analysis import visualize_temporal


def build_visualization_data() -> dict[str, pd.DataFrame]:
    topic_info = pd.DataFrame(
        {
            "Topic": [0, 1, -1],
            "Name": ["Topic 0 long description", "Topic 1 description", "Outliers"],
            "Count": [30, 20, 5],
        }
    )
    topic_assignments = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "year_month": ["2020-01", "2020-01", "2020-02", "2020-03"],
            "topic_id": [0, 1, 0, 1],
            "source_domain": ["x.com", "x.com", "y.com", "z.com"],
            "event_category": ["Verbal Cooperation"] * 4,
            "text": ["doc a", "doc b", "doc c", "doc d"],
            "cluster_id": [0, 1, 0, 1],
        }
    )
    topics_over_time = pd.DataFrame({"Topic": [0], "Timestamp": ["2020-01"], "Frequency": [2]})
    cluster_summary = pd.DataFrame(
        {
            "cluster_id": [0, 1, -1],
            "count": [22, 11, 2],
            "top_group": ["x.com", "y.com", "noise"],
        }
    )
    temporal_clusters = pd.DataFrame(
        {
            "year_month": ["2020-01", "2020-02"],
            "cluster_id": [0, 1],
            "count": [4, 5],
            "proportion": [0.4, 0.5],
        }
    )
    cluster_assignments = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "year_month": ["2020-01", "2020-01", "2020-02", "2020-03"],
            "cluster_id": [0, 1, 0, 1],
            "source_domain": ["x.com", "x.com", "y.com", "z.com"],
            "event_category": ["Verbal Cooperation"] * 4,
            "text": ["doc a", "doc b", "doc c", "doc d"],
        }
    )
    return {
        "topic_info": topic_info,
        "topics_over_time": topics_over_time,
        "topic_assignments": topic_assignments,
        "cluster_summary": cluster_summary,
        "temporal_clusters": temporal_clusters,
        "cluster_assignments": cluster_assignments,
    }


def test_visualize_temporal_load_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = build_visualization_data()

    output_dir = tmp_path / "outputs"
    topics_dir = output_dir / "topics"
    clusters_dir = output_dir / "clusters"
    viz_dir = output_dir / "visualizations"
    topics_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    data["topic_info"].to_csv(topics_dir / "topic_info.csv", index=False)
    data["topics_over_time"].to_csv(topics_dir / "topics_over_time.csv", index=False)
    data["cluster_summary"].to_csv(clusters_dir / "cluster_summary.csv", index=False)
    data["temporal_clusters"].to_csv(clusters_dir / "temporal_clusters.csv", index=False)
    pd.DataFrame({"cluster_id": [0], "keywords": ["venezuela, sanctions"]}).to_csv(
        clusters_dir / "cluster_keywords.csv", index=False
    )

    parquet_map = {
        str(topics_dir / "topic_assignments.parquet"): data["topic_assignments"],
        str(clusters_dir / "cluster_assignments.parquet"): data["cluster_assignments"],
    }

    monkeypatch.setattr(visualize_temporal, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(visualize_temporal, "VIZ_DIR", viz_dir)
    monkeypatch.setattr(pd, "read_parquet", lambda path: parquet_map[str(path)].copy())

    loaded = visualize_temporal.load_data()
    assert set(loaded.keys()) == {
        "topic_info",
        "topics_over_time",
        "topic_assignments",
        "cluster_summary",
        "temporal_clusters",
        "cluster_assignments",
    }

    visualize_temporal.plot_top_topics_over_time(loaded, top_n=2)
    visualize_temporal.plot_topics_heatmap(loaded, top_n=2)
    visualize_temporal.plot_top_clusters_over_time(loaded, top_n=2)
    visualize_temporal.plot_clusters_heatmap(loaded, top_n=2)
    visualize_temporal.plot_topic_trends(loaded, top_n=2)
    visualize_temporal.plot_cluster_by_source_domain(loaded, top_n=2)

    topic_summary = visualize_temporal.create_topic_summary_table(loaded)
    cluster_summary = visualize_temporal.create_cluster_summary_table(loaded)
    assert not topic_summary.empty
    assert not cluster_summary.empty

    expected_files = [
        "topics_over_time_stacked.png",
        "topics_heatmap.png",
        "clusters_over_time_stacked.png",
        "clusters_heatmap.png",
        "topic_individual_trends.png",
        "clusters_by_source_domain.png",
        "topic_summary_table.csv",
        "cluster_summary_table.csv",
    ]
    for name in expected_files:
        assert (viz_dir / name).exists()


def test_temporal_viz_functions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "cluster_id": [0, 1, 0, -1],
            "year_month": ["2020-01", "2020-01", "2020-02", "2020-03"],
            "source_domain": ["x.com", "x.com", "y.com", "z.com"],
            "event_category": ["Verbal Cooperation"] * 4,
            "text": ["t1", "t2", "t3", "t4"],
        }
    )

    temporal_viz.create_umap_scatter(embeddings, df, output_path=tmp_path / "scatter.png")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Animation was deleted without rendering anything.*",
            category=UserWarning,
        )
        temporal_viz.create_animated_umap(embeddings, df, output_path=None, fps=1)
    temporal_viz.create_cluster_river_plot(df, output_path=tmp_path / "river.png", top_n_clusters=2)
    temporal_viz.create_cluster_heatmap(df, output_path=tmp_path / "heatmap.png")
    assert (tmp_path / "scatter.png").exists()
    assert (tmp_path / "river.png").exists()
    assert (tmp_path / "heatmap.png").exists()

    with pytest.raises(ValueError):
        temporal_viz.create_cluster_heatmap(df.drop(columns=["source_domain"]), group_column="source_domain")

    fake_plotly = types.ModuleType("plotly")
    fake_go = types.ModuleType("plotly.graph_objects")

    class FakeFigure:
        def __init__(self, data: object = None) -> None:
            self.data = data
            self.layout = {}
            self.saved = None

        def update_layout(self, **kwargs: object) -> None:
            self.layout.update(kwargs)

        def write_html(self, path: Path) -> None:
            self.saved = path
            Path(path).write_text("<html>ok</html>", encoding="utf-8")

    class FakeSankey:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_go.Figure = FakeFigure
    fake_go.Sankey = FakeSankey
    fake_plotly.graph_objects = fake_go
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)
    sankey_df = pd.DataFrame(
        {
            "from_period": ["2020-01"],
            "to_period": ["2020-02"],
            "from_cluster": [0],
            "to_cluster": [1],
            "from_count": [10],
            "similarity": [0.9],
        }
    )
    fig = temporal_viz.create_sankey_diagram(sankey_df, output_path=tmp_path / "sankey.html")
    assert fig is not None
    assert (tmp_path / "sankey.html").exists()

    fake_px = types.ModuleType("plotly.express")

    class FakeScatterFigure:
        def __init__(self) -> None:
            self.saved = None

        def update_traces(self, **kwargs: object) -> None:
            return None

        def update_layout(self, **kwargs: object) -> None:
            return None

        def write_html(self, path: Path) -> None:
            self.saved = path
            Path(path).write_text("<html>scatter</html>", encoding="utf-8")

    fake_px.scatter = lambda *args, **kwargs: FakeScatterFigure()
    fake_plotly.express = fake_px
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", fake_px)
    out_fig = temporal_viz.create_interactive_scatter(embeddings, df, output_path=tmp_path / "interactive.html")
    assert out_fig is not None
    assert (tmp_path / "interactive.html").exists()


def test_main_stage_runners_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AnalysisConfig(base_dir=tmp_path)
    cfg.ensure_directories()
    df = pd.DataFrame(
        {
            "id": ["gdelt_1", "gdelt_2"],
            "type": ["event", "event"],
            "text": ["doc one", "doc two"],
            "year_month": ["2020-01", "2020-02"],
            "source_domain": ["x.com", "y.com"],
            "event_category": ["Verbal Cooperation", "Material Conflict"],
            "actor_pair": ["VEN->USA", "USA->VEN"],
            "doc_relevance_score": [70, 60],
            "created_datetime": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        }
    )

    import analysis.sentiment as sentiment_pkg
    import analysis.topic as topic_pkg
    import analysis.clustering as clustering_pkg

    monkeypatch.setattr(
        sentiment_pkg,
        "analyze_dataframe",
        lambda in_df, text_column, model_name, batch_size: in_df.assign(
            sentiment_label=["positive", "negative"],
            sentiment_confidence=[0.9, 0.8],
            sentiment_score=[0.9, -0.8],
        ),
    )
    monkeypatch.setattr(
        sentiment_pkg,
        "aggregate_sentiment",
        lambda in_df, group_by: pd.DataFrame({"group": ["x"], "mean_sentiment": [0.1]}),
    )
    monkeypatch.setattr(
        sentiment_pkg,
        "get_sentiment_summary",
        lambda in_df: {
            "mean_sentiment": 0.05,
            "positive_ratio": 0.5,
            "negative_ratio": 0.5,
        },
    )

    parquet_store: dict[str, pd.DataFrame] = {}

    def fake_to_parquet(self: pd.DataFrame, path: str | Path, index: bool = False) -> None:
        parquet_store[str(path)] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    sent_df = analysis_main.run_sentiment_analysis(df, cfg, save=True)
    assert "sentiment_score" in sent_df.columns

    class FakeTopicModel:
        def __init__(self) -> None:
            self.saved_path = None

        def save(self, path: str) -> None:
            self.saved_path = path

    monkeypatch.setattr(
        topic_pkg,
        "fit_topics",
        lambda in_df, text_column, embedding_model, n_topics, min_topic_size: (
            in_df.assign(topic_id=[0, 1], topic_label=["A", "B"], topic_prob=[0.9, 0.8]),
            FakeTopicModel(),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ),
    )
    monkeypatch.setattr(
        topic_pkg,
        "get_topic_info",
        lambda model: pd.DataFrame({"Topic": [0, 1], "Name": ["A", "B"], "Count": [1, 1]}),
    )
    monkeypatch.setattr(
        topic_pkg,
        "topics_over_time",
        lambda model, texts, timestamps, nr_bins: pd.DataFrame({"Topic": [0], "Timestamp": ["2020-01"], "Frequency": [1]}),
    )
    monkeypatch.setattr(
        topic_pkg,
        "aggregate_topics_by_group",
        lambda in_df, group_by: pd.DataFrame({"group": ["x"], "topic_id": [0], "count": [1], "proportion": [1.0]}),
    )

    topic_df, _, embs = analysis_main.run_topic_modeling(sent_df, cfg, save=True)
    assert "topic_id" in topic_df.columns
    assert embs.shape == (2, 2)

    class FakeClusterer:
        def __init__(self, min_cluster_size: int, min_samples: int, group_column: str) -> None:
            self.probabilities = np.array([0.7, 0.8])

        def fit(self, embeddings: np.ndarray) -> np.ndarray:
            return np.array([0, 1])

        def add_clusters_to_df(self, in_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
            return in_df.assign(cluster_id=labels, cluster_prob=[0.7, 0.8])

        def get_cluster_summary(self, in_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"cluster_id": [0, 1], "count": [1, 1], "top_group": ["x.com", "y.com"]})

        def get_temporal_clusters(self, in_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"year_month": ["2020-01"], "cluster_id": [0], "count": [1], "proportion": [1.0]})

    monkeypatch.setattr(clustering_pkg, "TemporalClusterer", FakeClusterer)
    monkeypatch.setattr(clustering_pkg, "reduce_dimensions", lambda embeddings, n_components=2: np.array([[0.1, 0.2], [0.3, 0.4]]))
    monkeypatch.setattr(clustering_pkg, "TextEmbedder", lambda model_name: None)

    clustered_df, _, emb2d = analysis_main.run_clustering(topic_df, embs, cfg, save=True)
    assert "cluster_id" in clustered_df.columns
    assert emb2d.shape == (2, 2)

    calls: list[str] = []
    monkeypatch.setattr(clustering_pkg, "create_umap_scatter", lambda *args, **kwargs: calls.append("scatter"))
    monkeypatch.setattr(clustering_pkg, "create_animated_umap", lambda *args, **kwargs: calls.append("anim"))
    monkeypatch.setattr(clustering_pkg, "create_cluster_river_plot", lambda *args, **kwargs: calls.append("river"))
    monkeypatch.setattr(clustering_pkg, "create_cluster_heatmap", lambda *args, **kwargs: calls.append("heatmap"))
    monkeypatch.setattr(clustering_pkg, "create_interactive_scatter", lambda *args, **kwargs: calls.append("interactive"))
    analysis_main.run_visualizations(clustered_df.assign(umap_1=emb2d[:, 0], umap_2=emb2d[:, 1]), emb2d, cfg)
    assert set(calls) == {"scatter", "anim", "river", "heatmap", "interactive"}

    monkeypatch.setattr(clustering_pkg, "generate_keyword_summary", lambda in_df, cluster_id: ["venezuela", "sanctions"])
    monkeypatch.setattr(
        clustering_pkg,
        "summarize_all_clusters",
        lambda in_df, n_samples, llm_provider: pd.DataFrame(
            {"cluster_id": [0], "theme": ["policy"], "summary": ["desc"]}
        ),
    )
    summaries = analysis_main.run_cluster_summarization(
        clustered_df.assign(cluster_id=[0, 1]),
        cfg,
        llm_provider="openai",
    )
    assert not summaries.empty

    monkeypatch.setattr(analysis_main, "load_all_data", lambda config: df.copy())
    monkeypatch.setattr(analysis_main, "run_sentiment_analysis", lambda in_df, config: in_df.assign(sentiment_score=0.1))
    monkeypatch.setattr(
        analysis_main,
        "run_topic_modeling",
        lambda in_df, config: (in_df.assign(topic_id=0, topic_label="A", topic_prob=0.9), object(), np.array([[1.0, 2.0], [3.0, 4.0]])),
    )
    monkeypatch.setattr(
        analysis_main,
        "run_clustering",
        lambda in_df, embeddings, config: (
            in_df.assign(cluster_id=0, cluster_prob=0.9, umap_1=[0.1, 0.2], umap_2=[0.3, 0.4]),
            embeddings,
            np.array([[0.1, 0.3], [0.2, 0.4]]),
        ),
    )
    marker: dict[str, bool] = {"viz": False, "sum": False}
    monkeypatch.setattr(analysis_main, "run_visualizations", lambda in_df, embeddings_2d, config: marker.__setitem__("viz", True))
    monkeypatch.setattr(
        analysis_main,
        "run_cluster_summarization",
        lambda in_df, config, llm: marker.__setitem__("sum", True),
    )

    monkeypatch.setattr(sys, "argv", ["main.py"])
    analysis_main.main()
    assert marker["viz"] is True
    assert marker["sum"] is True
