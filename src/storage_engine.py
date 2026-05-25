# src/storage_engine.py

"""
Storage Engine with Fixed-Size Dynamic Layer and Weight Tracking

Key Features:
- Fixed capacity from config (doesn't grow)
- Tracks weights for each vector
- Neighborhood dedup check before admission
- Weight-based eviction when full
- HNSW index for fast searches
- Optional INT8 quantization for memory reduction
- Metadata filtering with inverted index
"""

import faiss
import numpy as np
import os
import glob
import json
import pickle
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from src.config import (
    BASE_LAYER_PATH,
    DYNAMIC_LAYER_PATH,
    VECTOR_DIMENSION,
    DYNAMIC_LAYER_CAPACITY,
    HOT_PARTITION_RAM_LIMIT,
    NEIGHBORHOOD_THRESHOLD_ADMISSION,
    NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP,
    EVICTION_MODE,
)
from src.anchor_system import CONFIDENCE_EVICT_FLOOR
from src.quantization import INT8Quantizer

logger = logging.getLogger(__name__)


class StorageEngine:
    """
    Two-tier storage with fixed-size dynamic layer.

    TIER 1 (Permanent):
    - Bounded (read-only, loaded from disk)

    TIER 2 (Dynamic):
    - Bounded capacity (read-write, FIXED SIZE)

    OPTIMIZATIONS:
    - HNSW index for fast approximate search
    - INT8 quantization for memory efficiency
    - Metadata inverted index for fast filtering
    """

    def __init__(self, config):
        """Initialize two-tier storage with optimizations."""
        self.config = config
        self.dimension = config.VECTOR_DIMENSION

        # Paths
        self.base_path = Path(config.BASE_LAYER_PATH)
        self.dynamic_path = Path(config.DYNAMIC_LAYER_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.dynamic_path.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self.lock = threading.RLock()

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TIER 1: Permanent Layer (Kitchen)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        self.permanent_partitions = []  # List of FAISS indexes
        self.permanent_metadata = {}  # {id: metadata}
        self.permanent_lookup = {}  # {(normalized_partition_path, local_idx): id}

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TIER 2: Dynamic Layer (Backpack)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        # âœ… OPTIMIZATION 1: HNSW index
        self.use_hnsw = getattr(config, "USE_HNSW", True)

        if self.use_hnsw:
            # HNSW parameters
            M = getattr(config, "HNSW_M", 16)
            ef_construction = getattr(config, "HNSW_EF_CONSTRUCTION", 200)
            ef_search = getattr(config, "HNSW_EF_SEARCH", 50)

            self.dynamic_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.dynamic_index.hnsw.efConstruction = ef_construction
            self.dynamic_index.hnsw.efSearch = ef_search

            logger.info(
                f"[STORAGE] âš¡ Using HNSW index (M={M}, efC={ef_construction}, efS={ef_search})"
            )
        else:
            self.dynamic_index = faiss.IndexFlatL2(self.dimension)
            logger.info("[STORAGE] Using flat index (exact search)")

        self.dynamic_ids = []  # List of vector IDs
        self.dynamic_metadata = {}  # {id: {weight, timestamp, ...}}
        self.dynamic_vectors_cache = {}  # {id: vector} for quick access
        self.deleted_ids = set()  # Tombstone mechanism

        # âœ… OPTIMIZATION 2: Inverted index for metadata filtering
        self.metadata_index = {
            "source": defaultdict(set),  # source â†’ {vector_ids}
            "cluster_id": defaultdict(set),  # cluster_id â†’ {vector_ids}
            "anchor_id": defaultdict(set),  # anchor_id â†’ {vector_ids}
            "weight_bucket": defaultdict(set),  # weight range â†’ {vector_ids}
        }

        # FIXED CAPACITY
        self.dynamic_capacity = config.DYNAMIC_LAYER_CAPACITY

        # Per co-evolutionary loop: anchor weight callback for T2 eviction
        self._anchor_weight_callback = None  # Set by HybridRouter

        # Eviction mode: "anchor" (confidence cascade) or "lru" (strict timestamp)
        self.eviction_mode = getattr(config, "EVICTION_MODE", "anchor")

        # âœ… OPTIMIZATION 3: INT8 Quantization
        self.use_quantization = getattr(config, "USE_QUANTIZATION", False)
        self.quantizer = None

        if self.use_quantization:
            self.quantizer = INT8Quantizer()
            logger.info("[STORAGE] INT8 quantization will be enabled after calibration")

        # Load existing data
        self._load_permanent_layer()
        self._load_dynamic_layer()

        # Calibrate quantizer on permanent layer (if enabled)
        if self.use_quantization and self._count_permanent() > 0:
            self._calibrate_quantizer()

        logger.info(f"[STORAGE] Initialized")
        logger.info(f"[STORAGE] TIER 1 (Permanent): {self._count_permanent()} vectors")
        logger.info(
            f"[STORAGE] TIER 2 (Dynamic): {self._count_dynamic()}/{self.dynamic_capacity} vectors"
        )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TIER 1: PERMANENT LAYER (Kitchen)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _load_permanent_layer(self):
        """Load permanent layer from disk."""
        if not self.base_path.exists():
            logger.warning(f"[STORAGE] Permanent path doesn't exist: {self.base_path}")
            return

        partition_files = sorted(glob.glob(str(self.base_path / "*.index")))

        for pfile in partition_files:
            try:
                index = faiss.read_index(pfile)

                # âœ… Convert flat indexes to HNSW for faster search
                if self.use_hnsw and isinstance(index, faiss.IndexFlatL2):
                    logger.info(f"[STORAGE] Converting {pfile} to HNSW...")

                    n_vectors = index.ntotal
                    vectors = np.zeros((n_vectors, self.dimension), dtype="float32")
                    for i in range(n_vectors):
                        vectors[i] = index.reconstruct(i)

                    hnsw_index = faiss.IndexHNSWFlat(self.dimension, 16)
                    hnsw_index.hnsw.efSearch = 50
                    hnsw_index.add(vectors)

                    index = hnsw_index
                    logger.info(
                        f"[STORAGE] âœ… Converted to HNSW ({n_vectors} vectors)"
                    )

                self.permanent_partitions.append(
                    {"index": index, "file": pfile, "nvectors": index.ntotal}
                )

                logger.info(
                    f"[STORAGE] Loaded permanent: {pfile} ({index.ntotal} vectors)"
                )
            except Exception as e:
                logger.error(f"[STORAGE] Failed to load {pfile}: {e}")

        # Load metadata
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.permanent_metadata = json.load(f)
            self._rebuild_permanent_lookup()

    def _count_permanent(self) -> int:
        """Count vectors in permanent layer."""
        return sum(p["nvectors"] for p in self.permanent_partitions)

    def get_permanent_vectors(self) -> Optional[np.ndarray]:
        """Reconstruct all Tier 1 vectors from FAISS partitions.

        Returns:
            (n, d) float32 array of all permanent vectors, or None if empty.
        """
        if not self.permanent_partitions:
            return None
        all_vectors = []
        for partition in self.permanent_partitions:
            index = partition["index"]
            n = index.ntotal
            if n == 0:
                continue
            vectors = np.zeros((n, self.dimension), dtype="float32")
            for i in range(n):
                vectors[i] = index.reconstruct(i)
            all_vectors.append(vectors)
        if not all_vectors:
            return None
        return np.vstack(all_vectors)

    def _normalize_partition_path(self, partition_path: str) -> str:
        """Normalize stored partition paths so relative and absolute forms match."""
        path = Path(partition_path)
        if path.is_absolute():
            return str(path.resolve())

        candidates = [
            path,
            self.base_path / path,
            self.base_path.parent / path,
            self.base_path.parent.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        return str((self.base_path.parent.parent / path).resolve())

    def _rebuild_permanent_lookup(self):
        """Build a fast ID lookup keyed by normalized partition path and local index."""
        self.permanent_lookup = {}
        for vec_id, meta in self.permanent_metadata.items():
            partition_file = meta.get("partition_file")
            local_idx = meta.get("local_idx")
            if partition_file is None or local_idx is None:
                continue
            normalized_path = self._normalize_partition_path(str(partition_file))
            self.permanent_lookup[(normalized_path, int(local_idx))] = vec_id

    def search_permanent(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[List[str], List[float]]:
        """Search only permanent layer."""
        with self.lock:
            all_ids = []
            all_distances = []

            query_vec = query_vector.reshape(1, -1).astype("float32")

            for partition in self.permanent_partitions:
                if partition["index"].ntotal == 0:
                    continue

                D, I = partition["index"].search(
                    query_vec, min(k, partition["index"].ntotal)
                )

                for local_idx, dist in zip(I[0], D[0]):
                    if local_idx != -1 and dist != float("inf"):
                        vec_id = self._get_permanent_id(partition, local_idx)
                        if vec_id:
                            all_ids.append(vec_id)
                            all_distances.append(dist)

            # Merge, sort, return top-k
            combined = list(zip(all_ids, all_distances))
            combined.sort(key=lambda x: x[1])
            top_k = combined[:k]

            return [vid for vid, _ in top_k], [d for _, d in top_k]

    def _get_permanent_id(self, partition, local_idx) -> Optional[str]:
        """Get vector ID from permanent metadata."""
        partition_file = self._normalize_partition_path(str(partition["file"]))
        return self.permanent_lookup.get((partition_file, int(local_idx)))

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TIER 2: DYNAMIC LAYER (Backpack)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _load_dynamic_layer(self):
        """Load dynamic layer from disk."""
        index_file = self.dynamic_path / "dynamic.index"
        ids_file = self.dynamic_path / "dynamic_ids.pkl"
        metadata_file = self.dynamic_path / "dynamic_metadata.pkl"

        if index_file.exists():
            try:
                self.dynamic_index = faiss.read_index(str(index_file))

                # Convert loaded index to HNSW if needed
                if self.use_hnsw and isinstance(self.dynamic_index, faiss.IndexFlatL2):
                    logger.info("[STORAGE] Converting loaded dynamic index to HNSW...")

                    n_vectors = self.dynamic_index.ntotal
                    if n_vectors > 0:
                        vectors = np.zeros((n_vectors, self.dimension), dtype="float32")
                        for i in range(n_vectors):
                            vectors[i] = self.dynamic_index.reconstruct(i)

                        hnsw_index = faiss.IndexHNSWFlat(self.dimension, 16)
                        hnsw_index.hnsw.efSearch = 50
                        hnsw_index.add(vectors)

                        self.dynamic_index = hnsw_index
                        logger.info(
                            f"[STORAGE] âœ… Converted dynamic to HNSW ({n_vectors} vectors)"
                        )

                with open(ids_file, "rb") as f:
                    self.dynamic_ids = pickle.load(f)
                with open(metadata_file, "rb") as f:
                    self.dynamic_metadata = pickle.load(f)

                # Rebuild metadata index
                logger.info("[STORAGE] Rebuilding metadata index...")
                for vid, meta in self.dynamic_metadata.items():
                    if vid not in self.deleted_ids:
                        self._index_metadata(vid, meta)

                logger.info(
                    f"[STORAGE] Loaded dynamic: {len(self.dynamic_ids)} vectors"
                )

                # Per T2 bounded admission fix: if loaded T2 exceeds capacity,
                # trim to capacity by evicting lowest-weight vectors.
                # This happens when a previous run left T2 over-capacity on disk
                # (e.g., CAsT 2020 had 455 vectors with capacity 151).
                current = self._count_dynamic()
                if current > self.dynamic_capacity:
                    excess = current - self.dynamic_capacity
                    logger.warning(
                        f"[STORAGE] T2 over-capacity on load: {current}/{self.dynamic_capacity}. "
                        f"Trimming {excess} lowest-weight vectors."
                    )
                    # Sort by weight ascending, keep top-capacity
                    active = [
                        (vid, meta.get("weight", 1.0))
                        for vid, meta in self.dynamic_metadata.items()
                        if vid not in self.deleted_ids
                    ]
                    active.sort(key=lambda x: x[1], reverse=True)  # highest weight first
                    keep_ids = set(vid for vid, _ in active[:self.dynamic_capacity])
                    
                    # Rebuild index with only kept vectors
                    kept_ids = []
                    kept_meta = {}
                    kept_cache = {}
                    new_index = faiss.IndexFlatL2(self.dimension)
                    
                    for vid in keep_ids:
                        if vid not in self.dynamic_metadata:
                            continue
                        # Try cache first, then FAISS reconstruct
                        vector = self.dynamic_vectors_cache.get(vid)
                        if vector is None:
                            meta = self.dynamic_metadata[vid]
                            local_idx = meta.get("local_idx")
                            if local_idx is not None and local_idx < self.dynamic_index.ntotal:
                                try:
                                    vector = self.dynamic_index.reconstruct(int(local_idx))
                                except Exception:
                                    continue
                        if vector is not None:
                            new_index.add(vector.reshape(1, -1).astype("float32"))
                            new_local_idx = len(kept_ids)
                            meta = dict(self.dynamic_metadata[vid])
                            meta["local_idx"] = new_local_idx
                            kept_ids.append(vid)
                            kept_meta[vid] = meta
                            if vid in self.dynamic_vectors_cache:
                                kept_cache[vid] = self.dynamic_vectors_cache[vid]
                    
                    self.dynamic_index = new_index
                    self.dynamic_ids = kept_ids
                    self.dynamic_metadata = kept_meta
                    self.dynamic_vectors_cache = kept_cache
                    self.deleted_ids.clear()
                    
                    # Rebuild metadata index
                    self.metadata_index = {
                        "source": __import__("collections").defaultdict(set),
                        "cluster_id": __import__("collections").defaultdict(set),
                        "anchor_id": __import__("collections").defaultdict(set),
                        "weight_bucket": __import__("collections").defaultdict(set),
                    }
                    for vid, meta in self.dynamic_metadata.items():
                        self._index_metadata(vid, meta)
                    
                    logger.info(
                        f"[STORAGE] Post-trim: {self._count_dynamic()}/{self.dynamic_capacity}"
                    )

            except Exception as e:
                logger.error(f"[STORAGE] Failed to load dynamic: {e}")

    def _calibrate_quantizer(self):
        """Calibrate quantizer on permanent layer vectors."""
        if not self.quantizer:
            return

        logger.info("[STORAGE] Calibrating INT8 quantizer...")

        samples = []
        target_samples = min(1000, self._count_permanent())

        # Sample vectors from permanent partitions
        for partition in self.permanent_partitions:
            n_vecs = partition["index"].ntotal
            if n_vecs == 0:
                continue

            sample_size = min(target_samples - len(samples), n_vecs)
            indices = np.linspace(0, n_vecs - 1, sample_size, dtype=int)

            for idx in indices:
                vec = partition["index"].reconstruct(int(idx))
                samples.append(vec)

                if len(samples) >= target_samples:
                    break

            if len(samples) >= target_samples:
                break

        if samples:
            sample_array = np.array(samples, dtype="float32")
            self.quantizer.calibrate(sample_array)

            # Test accuracy
            if len(samples) >= 100:
                test_set = sample_array[:100]
                accuracy = self.quantizer.estimate_accuracy_loss(test_set)
                logger.info(
                    f"[STORAGE] Quantization accuracy: {accuracy['accuracy_retained']:.2f}%"
                )
                logger.info(
                    f"[STORAGE] Accuracy loss: {accuracy['accuracy_loss']:.2f}%"
                )

            # Calculate savings
            savings = self.quantizer.get_memory_savings(
                self.dynamic_capacity, self.dimension
            )
            logger.info(
                f"[STORAGE] Memory savings: {savings['savings_mb']:.1f} MB "
                f"({savings['savings_percent']:.1f}%)"
            )
        else:
            logger.warning("[STORAGE] No samples for quantizer calibration")

    def _count_dynamic(self) -> int:
        """Count vectors in dynamic layer."""
        return len(self.dynamic_ids) - len(self.deleted_ids)

    def is_dynamic_full(self) -> bool:
        """Check if dynamic layer is at capacity."""
        return self._count_dynamic() >= self.dynamic_capacity

    def has_dynamic_space(self, n: int) -> bool:
        """Check if dynamic has space for n vectors."""
        return self._count_dynamic() + n <= self.dynamic_capacity

    def has_dynamic_id(self, vec_id: str) -> bool:
        """Return True when a dynamic resident with this ID is currently present."""
        return vec_id in self.dynamic_metadata and vec_id not in self.deleted_ids

    def has_permanent_id(self, vec_id: str) -> bool:
        """Return True when a permanent (Tier 1) resident with this ID is currently present."""
        return vec_id in self.permanent_metadata

    def insert_dynamic(
        self, vectors: np.ndarray, ids: List[str], metadata: Optional[Dict] = None
    ):
        """
        Insert vectors into dynamic layer.

        OPTIMIZED: Uses INT8 quantization for 4Ã— memory savings in cache
        """
        with self.lock:
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            n_vectors = vectors.shape[0]

            # Check capacity (CRITICAL!)
            if not self.has_dynamic_space(n_vectors):
                raise ValueError(
                    f"Dynamic layer full! "
                    f"({self._count_dynamic()}/{self.dynamic_capacity}). "
                    f"Must evict {n_vectors} vectors first."
                )

            # Add to FAISS index (always FP32)
            self.dynamic_index.add(vectors.astype("float32"))

            # âœ… OPTIMIZATION: Quantize vectors for cache storage
            if self.quantizer and self.quantizer.calibrated:
                # Quantize to INT8 (4Ã— memory reduction!)
                quantized_vectors = self.quantizer.quantize(vectors)

                for i, vid in enumerate(ids):
                    self.dynamic_ids.append(vid)

                    # Store QUANTIZED vector
                    self.dynamic_vectors_cache[vid] = quantized_vectors[i]

                    # Store metadata
                    meta = metadata.copy() if metadata else {}
                    meta.update(
                        {
                            "weight": meta.get("weight", 1.0),
                            "inserted_at": time.time(),
                            "last_accessed": time.time(),  # Per strict LRU: initial access time
                            "local_idx": len(self.dynamic_ids) - 1,
                            "quantized": True,
                        }
                    )
                    self.dynamic_metadata[vid] = meta

                    # âœ… Index metadata for fast filtering
                    self._index_metadata(vid, meta)

                logger.debug(
                    f"[STORAGE] Inserted {n_vectors} vectors (INT8 quantized) "
                    f"({self._count_dynamic()}/{self.dynamic_capacity})"
                )
            else:
                # No quantization - store FP32
                for i, vid in enumerate(ids):
                    self.dynamic_ids.append(vid)

                    # Store FP32 vector
                    self.dynamic_vectors_cache[vid] = vectors[i]

                    # Store metadata
                    meta = metadata.copy() if metadata else {}
                    meta.update(
                        {
                            "weight": meta.get("weight", 1.0),
                            "inserted_at": time.time(),
                            "last_accessed": time.time(),  # Per strict LRU: initial access time
                            "local_idx": len(self.dynamic_ids) - 1,
                            "quantized": False,
                        }
                    )
                    self.dynamic_metadata[vid] = meta

                    # âœ… Index metadata for fast filtering
                    self._index_metadata(vid, meta)

                logger.debug(
                    f"[STORAGE] Inserted {n_vectors} vectors (FP32) "
                    f"({self._count_dynamic()}/{self.dynamic_capacity})"
                )

    def admit_dynamic(
        self,
        vector: np.ndarray,
        vec_id: str,
        metadata: Optional[Dict] = None,
        *,
        neighborhood_threshold: Optional[float] = None,
    ) -> Dict[str, int]:
        """Atomically dedupe, evict if needed, and admit one vector into the dynamic layer.

        Per §22 Step 1.5 / HAKES insight: the neighborhood_threshold here controls
        ADMISSION — how similar a candidate must be to an existing Tier 2 resident
        to be skipped. This should be LOOSE (0.85) so diverse vectors get in.
        For SEARCH dedup (preventing duplicate results to the user), use
        search_dedup_check() which uses a TIGHT threshold (0.98).
        """
        diagnostics = {
            "cache_admitted_count": 0,
            "cache_evicted_count": 0,
            "cache_duplicate_skip_count": 0,
            "cache_insert_failed_count": 0,
        }

        # Per §22 Step 1.5: default to admission threshold (loose, 0.85)
        if neighborhood_threshold is None:
            neighborhood_threshold = NEIGHBORHOOD_THRESHOLD_ADMISSION

        with self.lock:
            vec_id = str(vec_id)
            candidate = vector.reshape(1, -1).astype("float32")

            if self.has_dynamic_id(vec_id):
                diagnostics["cache_duplicate_skip_count"] = 1
                return diagnostics

            if neighborhood_threshold is not None and self.dynamic_index.ntotal > 0:
                # Per anchor policy / Tier 2 bounded admission: use cosine conversion
                distances, indices = self.dynamic_index.search(
                    candidate, min(1, self.dynamic_index.ntotal)
                )
                if (
                    len(indices[0])
                    and indices[0][0] != -1
                    and distances[0][0] != float("inf")
                ):
                    # Per anchor policy / Tier 2 bounded admission / FAISS L2 inversion
                    # INVARIANT: vectors in FAISS are unit-normalized (see hybrid_router.py)
                    d = float(distances[0][0])
                    similarity = max(0.0, 1.0 - d * d / 2.0)  # cos_sim = 1 - L2²/2 for unit vectors
                    if similarity >= neighborhood_threshold:
                        diagnostics["cache_duplicate_skip_count"] = 1
                        return diagnostics

            # Per T2 bounded admission fix: evict in a LOOP until space is available.
            # Previous code evicted only 1 vector, but if T2 is far over capacity
            # (e.g., 455/151 after loading from disk), a single eviction doesn't
            # free enough space → insert_dynamic raises ValueError → admission fails.
            # This caused CAsT 2020 T2 to grow unboundedly (455 vectors, 0 evictions).
            MAX_EVICT_PER_ADMIT = 50  # Safety limit per single admission
            evict_count = 0
            while not self.has_dynamic_space(1) and evict_count < MAX_EVICT_PER_ADMIT:
                # Per eviction mode: choose anchor-weight or strict LRU
                if self.eviction_mode == "lru":
                    evict_id = self._evict_by_lru()
                else:
                    evict_id = self._evict_by_anchor_strength()
                    if not evict_id:
                        evict_id = self.get_weakest_dynamic_vector()
                if not evict_id:
                    break  # Nothing left to evict
                self.delete_dynamic(evict_id)
                evict_count += 1

            diagnostics["cache_evicted_count"] = evict_count

            if not self.has_dynamic_space(1):
                # Still no space after max evictions — give up
                diagnostics["cache_insert_failed_count"] = 1
                return diagnostics

            try:
                self.insert_dynamic(candidate, [vec_id], metadata)
                diagnostics["cache_admitted_count"] = 1
            except ValueError:
                diagnostics["cache_insert_failed_count"] = 1

        return diagnostics

    def search_dedup_check(
        self, query_vector: np.ndarray, threshold: Optional[float] = None
    ) -> bool:
        """Check if a near-duplicate of query_vector exists in Tier 2.

        Per §22 Step 1.5 / HAKES insight: this uses the TIGHT search dedup
        threshold (0.98) to prevent returning near-identical results to the
        user. This is SEPARATE from admission (which uses a loose 0.85 threshold).

        Args:
            query_vector: Vector to check for near-duplicates
            threshold: Override threshold (default: NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP)

        Returns:
            True if near-duplicate exists (should dedup from results)
        """
        if threshold is None:
            threshold = NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP
        return self.exists_in_dynamic_neighborhood(query_vector, threshold)

    def _index_metadata(self, vec_id: str, metadata: Dict):
        """Index metadata for fast filtering."""
        # Index by source
        if "source" in metadata:
            self.metadata_index["source"][metadata["source"]].add(vec_id)

        # Index by cluster_id
        if "cluster_id" in metadata:
            self.metadata_index["cluster_id"][metadata["cluster_id"]].add(vec_id)

        # Index by anchor_id
        if "anchor_id" in metadata:
            self.metadata_index["anchor_id"][metadata["anchor_id"]].add(vec_id)

        # Index by weight bucket
        if "weight" in metadata:
            weight = metadata["weight"]
            bucket = int(weight // 10) * 10
            self.metadata_index["weight_bucket"][bucket].add(vec_id)

    def delete_dynamic(self, vec_id: str):
        """Delete a vector from the dynamic layer (tombstone + lazy compaction)."""
        with self.lock:
            if vec_id not in self.dynamic_metadata:
                return

            self.deleted_ids.add(vec_id)
            self._remove_from_metadata_index(vec_id)

            # Lazy compaction: only rebuild HNSW when deleted_ids exceeds threshold
            # per anchor policy / Tier 2 bounded admission
            COMPACTION_THRESHOLD = 0.10
            if len(self.deleted_ids) >= max(
                1, int(COMPACTION_THRESHOLD * self.dynamic_capacity)
            ):
                self._compact_dynamic_storage()

            logger.debug(
                f"[STORAGE] Deleted {vec_id} from dynamic (pending_compact={len(self.deleted_ids)})"
            )

    def _compact_dynamic_storage(self):
        """Physically rebuild the small dynamic index after deletions."""
        active_ids = [
            vec_id
            for vec_id in self.dynamic_ids
            if vec_id not in self.deleted_ids and vec_id in self.dynamic_metadata
        ]

        if self.use_hnsw:
            M = getattr(self.config, "HNSW_M", 16)
            ef_construction = getattr(self.config, "HNSW_EF_CONSTRUCTION", 200)
            ef_search = getattr(self.config, "HNSW_EF_SEARCH", 50)
            new_index = faiss.IndexHNSWFlat(self.dimension, M)
            new_index.hnsw.efConstruction = ef_construction
            new_index.hnsw.efSearch = ef_search
        else:
            new_index = faiss.IndexFlatL2(self.dimension)

        new_ids = []
        new_metadata = {}
        new_vectors_cache = {}

        for new_local_idx, vec_id in enumerate(active_ids):
            # Per compaction fix: try cache first, then FAISS reconstruct.
            # get_vector_by_id only checks dynamic_vectors_cache and returns None
            # for vectors not cached, which causes compaction to drop them entirely.
            vector = self.dynamic_vectors_cache.get(vec_id)
            if vector is None:
                # Fall back to FAISS reconstruct using local_idx from metadata
                meta = self.dynamic_metadata.get(vec_id)
                if meta is not None:
                    local_idx = meta.get("local_idx")
                    if local_idx is not None and local_idx < self.dynamic_index.ntotal:
                        try:
                            vector = self.dynamic_index.reconstruct(int(local_idx))
                        except Exception:
                            continue
                if vector is None:
                    continue

            new_index.add(vector.reshape(1, -1).astype("float32"))
            new_ids.append(vec_id)

            meta = dict(self.dynamic_metadata[vec_id])
            meta["local_idx"] = new_local_idx
            new_metadata[vec_id] = meta

            cached_vector = self.dynamic_vectors_cache.get(vec_id)
            if cached_vector is not None:
                new_vectors_cache[vec_id] = cached_vector

        self.dynamic_index = new_index
        self.dynamic_ids = new_ids
        self.dynamic_metadata = new_metadata
        self.dynamic_vectors_cache = new_vectors_cache
        self.deleted_ids.clear()

        self.metadata_index = {
            "source": defaultdict(set),
            "cluster_id": defaultdict(set),
            "anchor_id": defaultdict(set),
            "weight_bucket": defaultdict(set),
        }
        for vec_id, meta in self.dynamic_metadata.items():
            self._index_metadata(vec_id, meta)

    def _remove_from_metadata_index(self, vec_id: str):
        """Remove vector from all metadata indexes."""
        for category, index in self.metadata_index.items():
            for value, id_set in list(index.items()):
                if vec_id in id_set:
                    id_set.discard(vec_id)
                    if not id_set:
                        del index[value]

    def search_dynamic(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[List[str], List[float]]:
        """Search only dynamic layer (HNSW optimized)."""
        with self.lock:
            if self.dynamic_index.ntotal == 0:
                return [], []

            query_vec = query_vector.reshape(1, -1).astype("float32")

            D, I = self.dynamic_index.search(
                query_vec, min(k, self.dynamic_index.ntotal)
            )

            ids = []
            distances = []

            for local_idx, dist in zip(I[0], D[0]):
                if (
                    local_idx != -1
                    and dist != float("inf")
                    and local_idx < len(self.dynamic_ids)
                ):
                    vec_id = self.dynamic_ids[local_idx]

                    if vec_id not in self.deleted_ids:
                        ids.append(vec_id)
                        distances.append(dist)
                        # Per strict LRU eviction: update last_accessed on every hit
                        if vec_id in self.dynamic_metadata:
                            self.dynamic_metadata[vec_id]["last_accessed"] = time.time()

            return ids[:k], distances[:k]

    def search_dynamic_filtered(
        self, query_vector: np.ndarray, k: int, filters: Optional[Dict] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search dynamic layer with metadata filtering.

        OPTIMIZED: Uses inverted index for 7Ã— speedup
        """
        with self.lock:
            if self.dynamic_index.ntotal == 0:
                return [], []

            # Fast path: no filters
            if not filters:
                return self.search_dynamic(query_vector, k)

            # Get candidate IDs using inverted index
            candidate_ids = self._apply_metadata_filters(filters)

            if not candidate_ids:
                logger.debug("[STORAGE] No vectors match filters")
                return [], []

            logger.debug(f"[STORAGE] Filtered to {len(candidate_ids)} candidates")

            # Search and filter results
            query_vec = query_vector.reshape(1, -1).astype("float32")
            search_k = min(k * 10, self.dynamic_index.ntotal)
            D, I = self.dynamic_index.search(query_vec, search_k)

            filtered_ids = []
            filtered_distances = []

            for local_idx, dist in zip(I[0], D[0]):
                if (
                    local_idx != -1
                    and dist != float("inf")
                    and local_idx < len(self.dynamic_ids)
                ):
                    vec_id = self.dynamic_ids[local_idx]

                    if vec_id in candidate_ids and vec_id not in self.deleted_ids:
                        filtered_ids.append(vec_id)
                        filtered_distances.append(dist)

                        if len(filtered_ids) >= k:
                            break

            return filtered_ids[:k], filtered_distances[:k]

    def _apply_metadata_filters(self, filters: Dict) -> set:
        """Apply metadata filters using inverted index."""
        candidate_ids = None

        # Filter by source
        if "source" in filters:
            source_ids = self.metadata_index["source"].get(filters["source"], set())
            candidate_ids = (
                source_ids if candidate_ids is None else candidate_ids & source_ids
            )

        # Filter by cluster_id
        if "cluster_id" in filters:
            cluster_ids = self.metadata_index["cluster_id"].get(
                filters["cluster_id"], set()
            )
            candidate_ids = (
                cluster_ids if candidate_ids is None else candidate_ids & cluster_ids
            )

        # Filter by anchor_id
        if "anchor_id" in filters:
            anchor_ids = self.metadata_index["anchor_id"].get(
                filters["anchor_id"], set()
            )
            candidate_ids = (
                anchor_ids if candidate_ids is None else candidate_ids & anchor_ids
            )

        # Filter by weight range
        if "weight_min" in filters or "weight_max" in filters:
            weight_min = filters.get("weight_min", 0)
            weight_max = filters.get("weight_max", float("inf"))

            weight_ids = set()
            for bucket, ids in self.metadata_index["weight_bucket"].items():
                bucket_min = bucket
                bucket_max = bucket + 10

                if bucket_max >= weight_min and bucket_min <= weight_max:
                    weight_ids |= ids

            # Exact weight filtering
            exact_weight_ids = set()
            for vid in weight_ids:
                if vid in self.dynamic_metadata:
                    weight = self.dynamic_metadata[vid].get("weight", 1.0)
                    if weight_min <= weight <= weight_max:
                        exact_weight_ids.add(vid)

            candidate_ids = (
                exact_weight_ids
                if candidate_ids is None
                else candidate_ids & exact_weight_ids
            )

        return candidate_ids if candidate_ids is not None else set()

    def exists_in_dynamic_neighborhood(
        self, query_vector: np.ndarray, threshold: float = 0.98  # Per §19.2 D1: raised from 0.90
    ) -> bool:
        """Check if similar vector exists in dynamic (SMART CHECK)."""
        ids, distances = self.search_dynamic(query_vector, k=1)

        if ids and distances:
            # Per anchor policy / Tier 2 bounded admission: use cosine conversion
            d = float(distances[0])
            similarity = max(0.0, 1.0 - d * d / 2.0)

            if similarity >= threshold:
                logger.debug(
                    f"[SMART CHECK] Found similar vector (sim={similarity:.3f} >= {threshold})"
                )
                return True

        logger.debug(f"[SMART CHECK] No similar vector, need to fetch")
        return False

    def get_weakest_dynamic_vector(self) -> Optional[str]:
        """Find vector with lowest weight (for eviction)."""
        with self.lock:
            if not self.dynamic_metadata:
                return None

            weakest_id = None
            min_weight = float("inf")

            for vid, meta in self.dynamic_metadata.items():
                if vid in self.deleted_ids:
                    continue

                weight = meta.get("weight", 1.0)
                if weight < min_weight:
                    min_weight = weight
                    weakest_id = vid

            if weakest_id:
                logger.debug(
                    f"[EVICTION] Weakest vector: {weakest_id} (weight={min_weight})"
                )

            return weakest_id

    def set_anchor_weight_callback(self, callback):
        """Register callback for anchor-weight-based eviction.

        Per co-evolutionary loop: T2 eviction uses anchor confidence
        to decide which documents to evict. The callback is the
        AnchorSystem instance, which provides get_confidence() and
        get_dead_anchor_id().
        """
        self._anchor_weight_callback = callback

    def _evict_by_anchor_strength(self) -> Optional[str]:
        """Evict document from weakest anchor. Uses anchor_id metadata index.

        Per co-evolutionary loop: T2 eviction prioritizes documents from
        low-confidence anchors. When an anchor dies (confidence < 0.02),
        ALL its documents are evicted (cascade).

        Priority:
        1. If any anchor has confidence < 0.02 (eviction floor), evict its docs
        2. Otherwise, evict the document whose anchor has lowest confidence
        """
        if not self._anchor_weight_callback:
            return None  # No callback registered, fall back to weight-based eviction

        # Check for dead anchors first (confidence < eviction floor)
        dead_anchor_id = self._anchor_weight_callback.get_dead_anchor_id()
        if dead_anchor_id is not None:
            # Evict documents from dead anchor (cascade)
            anchor_key = str(dead_anchor_id)
            dead_docs = list(self.metadata_index.get("anchor_id", {}).get(anchor_key, set()))
            if dead_docs:
                evict_id = dead_docs[0]
                logger.debug(
                    f"[EVICTION] Cascading: evicting doc {evict_id} from "
                    f"dead anchor #{dead_anchor_id} (confidence < {CONFIDENCE_EVICT_FLOOR})"
                )
                return evict_id

        # Find document from weakest anchor
        weakest_doc = None
        weakest_confidence = float('inf')

        for vid, meta in self.dynamic_metadata.items():
            if vid in self.deleted_ids:
                continue
            anchor_id = meta.get("anchor_id")
            if anchor_id is not None:
                confidence = self._anchor_weight_callback.get_confidence(anchor_id)
                if confidence is not None and confidence < weakest_confidence:
                    weakest_confidence = confidence
                    weakest_doc = vid

        if weakest_doc:
            logger.debug(
                f"[EVICTION] Anchor-weight: evicting doc {weakest_doc} "
                f"(anchor confidence={weakest_confidence:.3f})"
            )

        return weakest_doc

    def _evict_by_lru(self) -> Optional[str]:
        """Strict LRU eviction: evict the vector with oldest last_accessed timestamp.

        Per ablation baseline: pure LRU cache with no anchor involvement.
        last_accessed is updated on every search hit and at insert time.
        """
        with self.lock:
            if not self.dynamic_metadata:
                return None

            oldest_id = None
            oldest_time = float("inf")

            for vid, meta in self.dynamic_metadata.items():
                if vid in self.deleted_ids:
                    continue
                accessed = meta.get("last_accessed", meta.get("inserted_at", 0))
                if accessed < oldest_time:
                    oldest_time = accessed
                    oldest_id = vid

            if oldest_id:
                logger.debug(
                    f"[EVICTION] LRU: evicting doc {oldest_id} "
                    f"(last_accessed={oldest_time:.1f})"
                )

            return oldest_id

    def update_dynamic_weight(self, vec_id: str, delta: float):
        """Update vector weight (reinforcement learning)."""
        with self.lock:
            if vec_id in self.dynamic_metadata:
                old_weight = self.dynamic_metadata[vec_id].get("weight", 1.0)
                new_weight = max(0.1, old_weight + delta)

                # Update weight bucket index
                old_bucket = int(old_weight // 10) * 10
                new_bucket = int(new_weight // 10) * 10

                if old_bucket != new_bucket:
                    self.metadata_index["weight_bucket"][old_bucket].discard(vec_id)
                    self.metadata_index["weight_bucket"][new_bucket].add(vec_id)

                self.dynamic_metadata[vec_id]["weight"] = new_weight
                logger.debug(
                    f"[WEIGHT] {vec_id}: {old_weight:.1f} â†’ {new_weight:.1f}"
                )

    def get_vector_by_id(self, vec_id: str) -> Optional[np.ndarray]:
        """Get vector by ID (with automatic dequantization)."""
        with self.lock:
            if vec_id in self.dynamic_vectors_cache:
                cached = self.dynamic_vectors_cache[vec_id]

                # Dequantize if needed
                if self.quantizer and cached.dtype == np.uint8:
                    return self.quantizer.dequantize(cached.reshape(1, -1))[0]
                else:
                    return cached

            return None

    def get_dynamic_stats(self) -> Dict:
        """Get dynamic layer statistics."""
        with self.lock:
            if not self.dynamic_metadata:
                return {
                    "current_size": 0,
                    "capacity": self.dynamic_capacity,
                    "fill_rate": 0.0,
                    "avg_weight": 0.0,
                    "deleted_count": 0,
                }

            active_weights = [
                meta.get("weight", 1.0)
                for vid, meta in self.dynamic_metadata.items()
                if vid not in self.deleted_ids
            ]

            current_size = self._count_dynamic()

            return {
                "current_size": current_size,
                "capacity": self.dynamic_capacity,
                "fill_rate": (current_size / self.dynamic_capacity * 100),
                "avg_weight": np.mean(active_weights) if active_weights else 0.0,
                "deleted_count": len(self.deleted_ids),
            }

    def get_metadata_stats(self) -> Dict:
        """Get comprehensive statistics about dynamic layer metadata."""
        source_counts = defaultdict(int)
        phase_counts = defaultdict(int)
        for meta in self.dynamic_metadata.values():
            source_counts[meta.get("source", "unknown")] += 1
            phase = meta.get("phase")
            if phase:
                phase_counts[phase] += 1
        return {
            "source_counts": dict(source_counts),
            "phase_counts": dict(phase_counts),
        }

    def get_dynamic_metadata_batch(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """Return metadata dicts for a batch of dynamic-layer doc IDs.

        Per Tier 2 hit decomposition: the benchmark needs to know whether
        each Tier 2 search hit was admitted via reactive_cache or
        predictive_prefetch.  Missing IDs (not in Tier 2) return {}.
        """
        with self.lock:
            return {
                doc_id: dict(self.dynamic_metadata.get(doc_id, {}))
                for doc_id in doc_ids
            }

    def save_dynamic_state(self):
        """Save dynamic layer to disk."""
        with self.lock:
            try:
                index_file = self.dynamic_path / "dynamic.index"
                ids_file = self.dynamic_path / "dynamic_ids.pkl"
                metadata_file = self.dynamic_path / "dynamic_metadata.pkl"

                # Save FAISS index
                faiss.write_index(self.dynamic_index, str(index_file))

                # Save IDs and metadata
                with open(ids_file, "wb") as f:
                    pickle.dump(self.dynamic_ids, f)
                with open(metadata_file, "wb") as f:
                    pickle.dump(self.dynamic_metadata, f)

                logger.info(
                    f"[STORAGE] âœ… Saved dynamic state ({len(self.dynamic_ids)} vectors)"
                )
                return True

            except Exception as e:
                logger.error(f"[STORAGE] âŒ Failed to save dynamic state: {e}")
                return False
