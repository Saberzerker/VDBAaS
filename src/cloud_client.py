# src/cloud_client.py
"""
Qdrant Cloud integration for hybrid VDB system.
Handles cloud storage, search, and vector retrieval.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import CLOUD_TIMEOUT_SECONDS
from src.config import *
from src.qdrant_ids import external_doc_id, to_qdrant_point_id

logger = logging.getLogger(__name__)


class QdrantCloudClient:
    """Real Qdrant Cloud client for canonical vector storage."""

    def __init__(
        self, url: str, api_key: str, collection_name: str, dimension: int = None
    ):
        self.collection_name = collection_name
        self.dimension = dimension or VECTOR_DIMENSION  # Use config default

        print(f"[QDRANT] Connecting to {url}...")
        self.client = QdrantClient(
            url=url, api_key=api_key, timeout=CLOUD_TIMEOUT_SECONDS
        )

        self._ensure_collection_exists()
        print(f"[QDRANT] OK Connected to collection '{collection_name}'")

    def _ensure_collection_exists(self):
        """Create collection if doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            print(f"[QDRANT] Collection '{self.collection_name}' exists")
        except Exception:
            print(f"[QDRANT] Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )
            print("[QDRANT] Collection created")

    def populate_with_documents(
        self, documents: List[str], embedder: SentenceTransformer
    ):
        """
        Populate cloud with document corpus.
        This is the canonical knowledge base.
        """
        print(f"\n[QDRANT] Populating cloud with {len(documents)} documents...")

        # Generate embeddings
        print("[QDRANT] Generating embeddings...")
        embeddings = embedder.encode(
            [f"passage: {d}" for d in documents],
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Create points
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"text": documents[i], "doc_id": f"doc_{i}"},
            )
            for i in range(len(documents))
        ]

        # Upload
        print("[QDRANT] Uploading to cloud...")
        self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"[QDRANT] OK Uploaded {len(documents)} vectors")

        # Verify
        collection_info = self.client.get_collection(self.collection_name)
        print(f"[QDRANT] Cloud contains {collection_info.points_count} vectors\n")

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> Tuple[List[str], List[float], float]:
        """
        Search cloud for similar vectors.

        Returns:
            (ids, scores, latency_ms)
        """
        start_time = time.time()

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector.tolist(),
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=k,
                    with_payload=True,
                )

            latency_ms = (time.time() - start_time) * 1000

            ids = [
                external_doc_id(result.id, getattr(result, "payload", None))
                for result in results
            ]
            scores = [result.score for result in results]

            return ids, scores, latency_ms

        except Exception as e:
            print(f"[QDRANT] Search error: {e}")
            return [], [], 0.0

    def get_vectors_by_ids(self, ids: List[Any]) -> List[np.ndarray]:
        """
        Get vectors by IDs (batch operation).

        OPTIMIZED: Single API call for multiple IDs.

        Per Fix 3 (B1): returns vectors IN INPUT ORDER. Qdrant retrieve()
        returns records in arbitrary order, so we rebuild by ID mapping.
        Missing IDs get None entries (caller must handle).

        Args:
            ids: List of vector IDs to fetch

        Returns:
            List of numpy arrays (vectors) in same order as input ids.
            Missing IDs produce None entries.
        """
        try:
            # BATCH RETRIEVE (single API call)
            normalized_ids = [to_qdrant_point_id(point_id) for point_id in ids]
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=normalized_ids,
                with_vectors=True,
            )

            # Per Fix 3: build ID→vector map, then reconstruct input order.
            # Qdrant retrieve() returns records in arbitrary order.
            # external_doc_id() converts internal Qdrant ID back to doc ID.
            id_to_vector = {}
            for record in records:
                if record.vector:
                    payload = getattr(record, "payload", None) or {}
                    ext_id = external_doc_id(record.id, payload)
                    id_to_vector[ext_id] = np.array(record.vector, dtype="float32")

            # Reconstruct in input order; None for missing
            vectors = [id_to_vector.get(str(doc_id)) for doc_id in ids]

            logger.debug(f"[CLOUD] Batch retrieved {len(vectors)} vectors ({len(id_to_vector)} found)")
            return vectors

        except Exception as e:
            logger.error(f"[CLOUD] Batch retrieve failed: {e}")
            return []

    def get_payloads_by_ids(self, ids: List[Any]) -> Dict[str, Dict]:
        """Fetch payloads (including text) by doc IDs (batch, no vectors).

        Per cross-encoder re-ranker integration: the router needs document
        text to score query-document relevance.  Returns: {external_doc_id: payload_dict, ...}
        """
        try:
            normalized_ids = [to_qdrant_point_id(point_id) for point_id in ids]
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=normalized_ids,
                with_vectors=False,
            )
            result = {}
            for record in records:
                payload = getattr(record, "payload", None) or {}
                ext_id = external_doc_id(record.id, payload)
                result[ext_id] = payload
            return result
        except Exception as e:
            logger.error(f"[CLOUD] Batch payload retrieve failed: {e}")
            return {}

    def get_collection_stats(self) -> dict:
        """Get cloud collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "status": info.status,
                "vectors_count": getattr(info, "vectors_count", None)
                or info.points_count,
            }
        except Exception as e:
            return {"error": str(e)}

    def recommend(
        self,
        positive_ids: List[Any],
        limit: int = 20,
    ) -> Tuple[List[str], List[float], float]:
        """Find vectors similar to the given positive examples.

        Per warm-seed neighborhood expansion (THEORY.md §Q2):
        Q1 cloud top-K results are used as positive examples. We fetch
        their vectors, average them as the query centroid, and search
        for neighbors — ~30 vectors seeded to Tier 2.

        Returns (ids, scores, latency_ms).
        """
        start_time = time.time()
        try:
            # Fetch vectors for positive examples
            normalized = [to_qdrant_point_id(pid) for pid in positive_ids]
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=normalized,
                with_vectors=True,
            )

            if not records:
                return [], [], (time.time() - start_time) * 1000

            # Average the positive vectors as the query centroid
            vectors = [np.array(r.vector) for r in records if r.vector]
            if not vectors:
                return [], [], (time.time() - start_time) * 1000

            centroid = np.mean(vectors, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Search for neighbors using the centroid
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=centroid.tolist(),
                    limit=limit,
                    with_payload=False,
                    with_vectors=False,
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=centroid.tolist(),
                    limit=limit,
                    with_payload=False,
                )

            latency_ms = (time.time() - start_time) * 1000

            ids = [
                external_doc_id(r.id, getattr(r, "payload", None))
                for r in results
            ]
            scores = [r.score for r in results]

            logger.info(
                f"[CLOUD] Recommend: {len(ids)} neighbors from {len(vectors)} seeds ({latency_ms:.0f}ms)"
            )
            return ids, scores, latency_ms

        except Exception as e:
            logger.warning(f"[CLOUD] Recommend failed: {e}")
            return [], [], (time.time() - start_time) * 1000

    def clear_collection(self):
        """Clear all vectors (for testing)."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            print("[QDRANT] Collection cleared")
        except Exception as e:
            print(f"[QDRANT] Clear error: {e}")
