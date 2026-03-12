"""
Sentence-BERT embedder with ID mapping.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        device: Optional[str] = None,  # Auto-detect
    ) -> None:
        """Initialize the class instance.
        
        Args:
            model_name (str): Model identifier used by the NLP component. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            batch_size (int): Batch size used for model inference. Defaults to 64.
            device (Optional[str]): Value for `device`. Defaults to None.
        
        Returns:
            None: No return value.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

        # ID tracking
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None

    def _load_model(self) -> Any:
        """Lazy load the embedding model.
        
        Returns:
            Any: Object returned by the underlying library or runtime path.
        """
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
        """Embed a list of texts.
        
        Args:
            texts (List[str]): Input text strings.
            ids (Optional[List[str]]): Collection of document IDs. Defaults to None.
            show_progress (bool): Whether to display progress bars during inference. Defaults to True.
        
        Returns:
            np.ndarray: NumPy array result for downstream computation.
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
        """Embed texts from DataFrame with full ID tracking.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process.
            text_column (str): Column name containing article text. Defaults to 'text'.
            id_column (str): Column name containing unique document IDs. Defaults to 'id'.
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Processed pandas DataFrame.
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
        """Get embedding for a specific document ID.
        
        Args:
            doc_id (str): Single document ID.
        
        Returns:
            Optional[np.ndarray]: NumPy array result for downstream computation.
        """
        if doc_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[doc_id]
        return self.embeddings[idx]

    def get_ids_by_indices(self, indices: List[int]) -> List[str]:
        """Get document IDs for a list of indices.
        
        Args:
            indices (List[int]): Value for `indices`.
        
        Returns:
            List[str]: List result produced by this function.
        """
        return [self.idx_to_id.get(idx) for idx in indices]

    def save(self, output_dir: Path) -> None:
        """Save embeddings and mappings to disk.
        
        Args:
            output_dir (Path): Directory where outputs will be written.
        
        Returns:
            None: No return value.
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
        """Load embeddings and mappings from disk.
        
        Args:
            input_dir (Path): Directory containing input artifacts.
        
        Returns:
            None: No return value.
        """
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
    """Reduce embedding dimensions using UMAP.
    
    Args:
        embeddings (np.ndarray): High-dimensional embedding matrix.
        n_components (int): Target number of components/dimensions. Defaults to 2.
        n_neighbors (int): UMAP neighborhood size parameter. Defaults to 15.
        min_dist (float): UMAP minimum-distance parameter. Defaults to 0.1.
        metric (str): Distance metric name. Defaults to 'cosine'.
    
    Returns:
        np.ndarray: NumPy array result for downstream computation.
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
