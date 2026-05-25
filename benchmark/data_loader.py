# benchmark/data_loader.py

"""
Dataset loading utilities for benchmark.

Handles:
- Loading PubMedQA from HuggingFace
- Caching datasets locally
- Generating embeddings
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from benchmark.benchmark_config import config as benchmark_config


class DataLoader:
    """
    Handles dataset loading and embedding generation.
    """

    def __init__(self):
        """Initialize data loader."""
        self.dataset = None
        self.embedder = None

        print("\n[DATA] Initializing data loader...")
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load embedding model."""
        print(f"[DATA] Loading embedding model: {benchmark_config.EMBEDDING_MODEL}")

        self.embedder = SentenceTransformer(
            benchmark_config.EMBEDDING_MODEL,
            cache_folder=str(benchmark_config.MODEL_CACHE_DIR),
        )

        print(f"[DATA] OK Model loaded ({benchmark_config.EMBEDDING_DIM}D)")

    def load_dataset(self, use_cache: bool = True) -> "Dataset":
        """
        Load PubMedQA dataset.

        Args:
            use_cache: Use cached dataset if available

        Returns:
            HuggingFace Dataset object
        """
        if self.dataset is not None:
            return self.dataset

        print(f"\n[DATA] Loading {benchmark_config.DATASET_NAME} dataset...")

        self.dataset = load_dataset(
            benchmark_config.DATASET_NAME,
            benchmark_config.DATASET_SUBSET,
            split="train",
            cache_dir=os.getenv(
                "HF_DATASETS_CACHE", str(benchmark_config.DATA_DIR / "cache")
            ),
        )

        print(f"[DATA] OK Loaded {len(self.dataset):,} documents")

        return self.dataset

    def extract_texts(self, indices: List[int]) -> List[str]:
        """
        Extract text from dataset at given indices.

        Args:
            indices: List of document indices

        Returns:
            List of text strings
        """
        if self.dataset is None:
            self.load_dataset()

        texts = []
        for idx in indices:
            item = self.dataset[idx]
            question = item.get("question", "")
            context = " ".join(item.get("context", {}).get("contexts", []))
            texts.append(f"{question} {context}")

        return texts

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            prefix: E5-style prefix ("query: " or "passage: "). Auto-detected for E5 models.

        Returns:
            Numpy array of embeddings (N, D)
        """
        if not texts:
            return np.array([])

        # Per E5 convention: add prefix if model requires it
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    def get_sample_indices(
        self, count: int, skip_first: int = 0, seed: int = 42
    ) -> List[int]:
        """
        Get random sample indices from dataset.

        Args:
            count: Number of indices to sample
            skip_first: Skip first N documents (e.g., TIER 1)
            seed: Random seed

        Returns:
            List of indices
        """
        if self.dataset is None:
            self.load_dataset()

        np.random.seed(seed)

        available_indices = range(skip_first, len(self.dataset))
        sampled = np.random.choice(
            list(available_indices),
            size=min(count, len(available_indices)),
            replace=False,
        )

        return sampled.tolist()

    def get_documents_batch(
        self, start_idx: int, count: int
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Get a batch of documents with embeddings.

        Args:
            start_idx: Starting index
            count: Number of documents

        Returns:
            (doc_ids, doc_texts, doc_embeddings)
        """
        if self.dataset is None:
            self.load_dataset()

        # Extract texts
        indices = range(start_idx, min(start_idx + count, len(self.dataset)))
        texts = self.extract_texts(list(indices))

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Create IDs
        doc_ids = [f"pubmed_{i}" for i in indices]

        return doc_ids, texts, embeddings
