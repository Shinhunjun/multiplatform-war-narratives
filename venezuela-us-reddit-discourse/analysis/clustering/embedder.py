"""
Sentence-BERT embedder with ID mapping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class TextEmbedder:
    """
    Text embedder with ID tracking.

    Maintains mapping between document IDs and embedding indices.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = None,  # Auto-detect
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

        # ID tracking
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            # Auto-detect best device
            if self.device is None:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"

            print(f"Embedding model using device: {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def embed_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings
            ids: Optional list of IDs (same length as texts)
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        model = self._load_model()

        print(f"Embedding {len(texts):,} texts with {self.model_name}...")

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Track IDs if provided
        if ids is not None:
            for idx, doc_id in enumerate(ids):
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id

        self.embeddings = embeddings
        return embeddings

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        id_column: str = "id",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Embed texts from DataFrame with full ID tracking.

        Returns:
        - embeddings: numpy array
        - index_df: DataFrame mapping indices to IDs and metadata
        """
        df = df.copy().reset_index(drop=True)

        texts = df[text_column].fillna("").tolist()
        ids = df[id_column].tolist()

        embeddings = self.embed_texts(texts, ids)

        # Create index DataFrame for tracking
        index_df = pd.DataFrame({
            "embedding_idx": range(len(df)),
            "id": ids,
            "type": df["type"] if "type" in df.columns else "unknown",
            "subreddit": df["subreddit"] if "subreddit" in df.columns else None,
            "year_month": df["year_month"] if "year_month" in df.columns else None,
            "created_utc": df["created_utc"] if "created_utc" in df.columns else None,
            "text_preview": [t[:200] + "..." if len(t) > 200 else t for t in texts],
        })

        self.metadata = index_df
        return embeddings, index_df

    def get_embedding_by_id(self, doc_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific document ID."""
        if doc_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[doc_id]
        return self.embeddings[idx]

    def get_ids_by_indices(self, indices: List[int]) -> List[str]:
        """Get document IDs for a list of indices."""
        return [self.idx_to_id.get(idx) for idx in indices]

    def save(self, output_dir: Path) -> None:
        """
        Save embeddings and mappings to disk.

        Saves:
        - embeddings.npy: numpy array of embeddings
        - embedding_index.parquet: ID mappings and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(output_dir / "embeddings.npy", self.embeddings)

        # Save metadata
        if self.metadata is not None:
            self.metadata.to_parquet(output_dir / "embedding_index.parquet", index=False)

        # Save ID mappings
        mapping_df = pd.DataFrame({
            "id": list(self.id_to_idx.keys()),
            "embedding_idx": list(self.id_to_idx.values()),
        })
        mapping_df.to_parquet(output_dir / "id_mapping.parquet", index=False)

        print(f"Saved embeddings to {output_dir}")

    def load(self, input_dir: Path) -> None:
        """Load embeddings and mappings from disk."""
        input_dir = Path(input_dir)

        # Load embeddings
        self.embeddings = np.load(input_dir / "embeddings.npy")

        # Load metadata
        if (input_dir / "embedding_index.parquet").exists():
            self.metadata = pd.read_parquet(input_dir / "embedding_index.parquet")

        # Load ID mappings
        mapping_df = pd.read_parquet(input_dir / "id_mapping.parquet")
        self.id_to_idx = dict(zip(mapping_df["id"], mapping_df["embedding_idx"]))
        self.idx_to_id = dict(zip(mapping_df["embedding_idx"], mapping_df["id"]))

        print(f"Loaded {len(self.embeddings):,} embeddings from {input_dir}")


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Reduce embedding dimensions using UMAP.

    Args:
        embeddings: High-dimensional embeddings
        n_components: Output dimensions (2 or 3 for visualization)
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        metric: Distance metric

    Returns:
        Reduced embeddings
    """
    from umap import UMAP

    print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}...")

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )

    reduced = reducer.fit_transform(embeddings)
    return reduced
