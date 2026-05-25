# benchmark/benchmark.py

"""
Main Benchmark Script for Hybrid VDB System

Two modes:
- Quick: 100 queries for fast validation
- Full: 1000 queries with realistic distribution (65/25/10)

Usage:
    python benchmark/benchmark.py --mode quick
    python benchmark/benchmark.py --mode full
    python benchmark/benchmark.py --mode quick --queries-file queries/custom_queries.txt

Author: Saberzerker
Date: 2025-11-30
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import middleware
from src.hybrid_router import HybridRouter
from src.local_vdb import LocalVDB
from src.cloud_client import QdrantCloudClient
from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.metrics import MetricsTracker
from src.config import Config

# Import benchmark modules
from benchmark.benchmark_config import config as benchmark_config, validate_config
from benchmark.data_loader import DataLoader
from benchmark.query_generator import QueryGenerator
from benchmark.visualizer import BenchmarkVisualizer
from benchmark.eval_metrics import (
    compute_ranking_metrics,
    summarize_query_metrics,
    summarize_session_metrics,
)
from benchmark.workload_utils import (
    derive_collection_name,
    derive_local_data_dirs,
    resolve_workload_dir,
)


class HybridVDBBenchmark:
    """
    Main benchmark orchestrator.
    """

    def __init__(
        self,
        test_mode: str = "full",
        custom_queries_file: Optional[str] = None,
        manifest_file: Optional[str] = None,
        top_k: int = 5,
        variant: str = "full_hybrid",
        collection_name: Optional[str] = None,
        output_name: Optional[str] = None,
        cold_start: bool = False,
        dynamic_dir_override: Optional[str] = None,
    ):
        """
        Initialize benchmark.

        Args:
            test_mode: "quick" (100 queries) or "full" (1000 queries)
            custom_queries_file: Optional path to custom queries
            output_name: Override output directory name for sweep runs
            cold_start: Delete dynamic/ (Tier 2) before init to measure
                        prefetch/reactive contribution from empty cache.
            dynamic_dir_override: Use this path instead of default dynamic/.
                        Prevents FAISS corruption when running variants in parallel.
        """
        self.test_mode = test_mode
        self.custom_queries_file = custom_queries_file
        self.manifest_file = manifest_file
        self.top_k = top_k
        self.variant = variant
        self.output_name = output_name
        self.cold_start = cold_start
        self.dynamic_dir_override = dynamic_dir_override
        self.workload_dir = resolve_workload_dir(manifest_file=manifest_file)
        self.git_sha = self._get_git_sha()
        self.collection_name = collection_name or (
            derive_collection_name(self.workload_dir) if self.workload_dir else None
        )
        self.permanent_dir: Optional[Path] = None
        self.dynamic_dir: Optional[Path] = None

        # Validate configuration
        if not validate_config():
            raise ValueError("Invalid configuration. Check your .env file.")

        # Determine query count
        if manifest_file:
            with open(manifest_file, "r", encoding="utf-8") as handle:
                self.num_queries = sum(1 for line in handle if line.strip())
        elif test_mode == "quick":
            self.num_queries = benchmark_config.QUICK_TEST_SIZE
        else:
            self.num_queries = benchmark_config.FULL_TEST_SIZE

        # Print configuration
        self._print_header()

        # Initialize components
        self.data_loader = DataLoader()
        self.query_generator = QueryGenerator(self.data_loader)
        self.visualizer = BenchmarkVisualizer(test_mode)

        # VDB system (initialized later)
        self.router = None
        self.metrics = None
        self.anchor_system = None
        self.local_vdb = None

        # Results storage
        self.results = {
            "config": {
                "test_mode": test_mode,
                "num_queries": self.num_queries,
                "tier1_size": benchmark_config.TIER1_SIZE,
                "tier2_capacity": benchmark_config.TIER2_CAPACITY,
                "top_k": self.top_k,
                "variant": self.variant,
                "manifest_file": self.manifest_file,
                "collection_name": self.collection_name,
                "git_sha": self.git_sha,
                "embedding_model": benchmark_config.EMBEDDING_MODEL,
                "timestamp": time.time(),
                "per_anchor_threshold": benchmark_config.PER_ANCHOR_THRESHOLD,
                "adaptive_gate": False,  # Per P1: static threshold for ablation
                "gate_threshold_override": None,  # Set if --gate-threshold passed
            },
            "per_query": [],
            "anchor_snapshots": [],
            "cache_snapshots": [],
        }

    def _print_header(self):
        """Print benchmark header."""
        print("=" * 70)
        print("HYBRID VDB BENCHMARK")
        print("=" * 70)
        print(f"Test Mode:        {self.test_mode.upper()}")
        print(f"Variant:          {self.variant}")
        print(f"Total Queries:    {self.num_queries:,}")
        print("\nStorage Configuration:")
        print(f"  TIER 1 (Perm):  {benchmark_config.TIER1_SIZE:,} docs ({benchmark_config.TIER1_PERCENT*100:.0f}% of corpus)")
        print(f"  TIER 2 (Dyn):   {benchmark_config.TIER2_CAPACITY:,} capacity ({benchmark_config.TIER2_PERCENT*100:.0f}% of corpus)")
        print("  TIER 3 (Cloud): configured Qdrant collection")

        if self.test_mode == "full":
            in_count = int(self.num_queries * benchmark_config.IN_DATASET_RATIO)
            edge_count = int(self.num_queries * benchmark_config.EDGE_CASE_RATIO)
            ood_count = int(self.num_queries * benchmark_config.OOD_RATIO)

            print("\nQuery Distribution:")
            print(
                f"  In-dataset:     {in_count} ({benchmark_config.IN_DATASET_RATIO * 100:.0f}%)"
            )
            print(
                f"  Edge cases:     {edge_count} ({benchmark_config.EDGE_CASE_RATIO * 100:.0f}%)"
            )
            print(
                f"  Out-of-dist:    {ood_count} ({benchmark_config.OOD_RATIO * 100:.0f}%)"
            )

        est_time = self.num_queries * benchmark_config.QUERY_INTERVAL_MS / 1000 / 60
        print(f"\nEstimated Time:   ~{est_time:.1f} minutes")
        print("=" * 70)

    def _get_git_sha(self) -> Optional[str]:
        """Best-effort git SHA for reproducible benchmark artifacts."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        except Exception:
            return None

    def _resolve_output_dir(self) -> Path:
        """Resolve the benchmark output directory for the active workload/variant."""
        # Per P2: allow sweep runner to override output name directly
        if self.output_name:
            return benchmark_config.RESULTS_DIR / self.output_name

        output_dir = benchmark_config.RESULTS_DIR / self.test_mode / self.variant
        if self.manifest_file:
            workload_name = Path(self.manifest_file).resolve().parent.name
            output_dir = (
                benchmark_config.RESULTS_DIR
                / workload_name
                / self.test_mode
                / self.variant
            )
        return output_dir

    def initialize_vdb_system(self):
        """Initialize VDB components."""
        print("\n[VDB] Initializing Hybrid VDB system...")

        # Create middleware config
        config = Config()

        # Per §22.6: Compute tier capacities from corpus size using percentage-based sizing.
        # Tier 1 = 1% of corpus, Tier 2 = 5% of corpus.
        # This ensures bounded hybrid retrieval across datasets of any size.
        corpus_size = 0
        if self.workload_dir:
            metadata_path = self.workload_dir / "metadata.json"
            if metadata_path.exists():
                import json as _json
                with metadata_path.open("r", encoding="utf-8") as _f:
                    workload_meta = _json.load(_f)
                corpus_size = int(workload_meta.get("corpus_count", 0))

        if corpus_size > 0:
            tier1_size = max(10, int(corpus_size * benchmark_config.TIER1_PERCENT))
            tier2_capacity = max(50, int(corpus_size * benchmark_config.TIER2_PERCENT))
            print(f"[VDB] Corpus size: {corpus_size:,} docs")
            print(f"[VDB] Tier 1 ({benchmark_config.TIER1_PERCENT*100:.0f}%): {tier1_size:,} vectors")
            print(f"[VDB] Tier 2 ({benchmark_config.TIER2_PERCENT*100:.0f}%): {tier2_capacity:,} vectors")
        else:
            # Fallback to fixed sizes when corpus size unknown
            tier1_size = benchmark_config.TIER1_SIZE
            tier2_capacity = benchmark_config.TIER2_CAPACITY
            print("[VDB] Corpus size unknown, using fixed capacities")
            print(f"[VDB] Tier 1: {tier1_size:,} vectors (fixed)")
            print(f"[VDB] Tier 2: {tier2_capacity:,} vectors (fixed)")

        config.PERMANENT_LAYER_CAPACITY = tier1_size
        config.DYNAMIC_LAYER_CAPACITY = tier2_capacity
        config.BASE_LAYER_PATH = str(benchmark_config.PERMANENT_DIR)
        config.DYNAMIC_LAYER_PATH = str(benchmark_config.DYNAMIC_DIR)
        config.PREFETCH_ENABLED = benchmark_config.PREFETCH_ENABLED
        config.USE_HNSW = benchmark_config.USE_HNSW
        config.USE_QUANTIZATION = benchmark_config.USE_QUANTIZATION
        if self.workload_dir:
            permanent_dir, dynamic_dir = derive_local_data_dirs(self.workload_dir)
            config.BASE_LAYER_PATH = str(permanent_dir)
            config.DYNAMIC_LAYER_PATH = str(dynamic_dir)
        if self.collection_name:
            config.CLOUD_COLLECTION_NAME = self.collection_name

        self.permanent_dir = Path(config.BASE_LAYER_PATH).resolve()
        self.dynamic_dir = Path(config.DYNAMIC_LAYER_PATH).resolve()
        self.collection_name = config.CLOUD_COLLECTION_NAME

        # Per parallel-safety fix: override dynamic dir to isolate per-variant.
        # Without this, concurrent variants corrupt each other's FAISS indices.
        if self.dynamic_dir_override:
            self.dynamic_dir = Path(self.dynamic_dir_override).resolve()
            config.DYNAMIC_LAYER_PATH = str(self.dynamic_dir)

        # Per cold-start fix: clear dynamic/ so Tier 2 starts empty.
        # Without this, pre-seeded FAISS index masks the contribution of
        # prefetch vs reactive_cache (both read same disk state).
        if self.cold_start and self.dynamic_dir.exists():
            import shutil
            print(f"[COLD-START] Clearing Tier 2 dynamic dir: {self.dynamic_dir}")
            shutil.rmtree(self.dynamic_dir)
            self.dynamic_dir.mkdir(parents=True, exist_ok=True)
        self.results["config"].update(
            {
                "collection_name": self.collection_name,
                "permanent_dir": str(self.permanent_dir),
                "dynamic_dir": str(self.dynamic_dir),
                "gate_threshold_override": config.GATE_INITIAL_THRESHOLD if hasattr(config, 'GATE_INITIAL_THRESHOLD') else None,
                "cold_start": self.cold_start,
                "dynamic_dir_override": self.dynamic_dir_override,
            }
        )

        # Initialize components
        self.local_vdb = LocalVDB(config)
        self.cloud_vdb = QdrantCloudClient(
            url=config.CLOUD_URL,
            api_key=config.CLOUD_API_KEY,
            collection_name=config.CLOUD_COLLECTION_NAME,
            dimension=config.VECTOR_DIMENSION,
        )
        self.semantic_cache = SemanticClusterCache()
        self.anchor_system = AnchorSystem()
        self.metrics = MetricsTracker()

        # Create router
        # Release-04: true_lru and freq_cache disable prefetch (no anchor-driven prediction)
        prefetch_enabled = benchmark_config.PREFETCH_ENABLED and self.variant not in {
            "no_prefetch",
            "local_only",
            "cloud_only",
            "true_lru",
            "freq_cache",
        }
        self.router = HybridRouter(
            local_vdb=self.local_vdb,
            cloud_vdb=self.cloud_vdb,
            semantic_cache=self.semantic_cache,
            anchor_system=self.anchor_system,
            metrics=self.metrics,
            prefetch_enabled=prefetch_enabled,
            routing_mode=self.variant,
        )

        # Per cross-encoder re-ranker: pre-load corpus texts into router
        # so _get_doc_texts() never needs cloud roundtrips for re-ranking.
        # Bug 6 fix: also load when workload_dir is None (non-manifest mode).
        #
        # For pure-local baselines (local_full, random_3pct), we skip the router
        # and instead build a FAISS index from the full corpus (or random subset).
        if self.variant in ("local_full", "random_3pct"):
            print("[VDB] Pure-local baseline variant — skipping router initialization")
            self._build_pure_local_index()
            print("[VDB] Baseline index ready")
        else:
            # Validate T1 loaded correctly for session/routing variants.
            # Catches the double-nested unzip bug where data ends up at
            # data/workloads/qrecc/qrecc/permanent/ instead of
            # data/workloads/qrecc/permanent/ — T1 would be silently empty.
            t1_count = self.local_vdb.get_permanent_count()
            if t1_count == 0:
                print(f"[FATAL] T1 index is EMPTY at {self.permanent_dir}")
                print("        Expected k-means centroids from seed_permanent_kmeans.py.")
                print(f"        Run: python scripts/seed_permanent_kmeans.py"
                      f" --workload-dir {self.workload_dir}")
                raise SystemExit(1)
            print(f"[VDB] T1 loaded: {t1_count} vectors from {self.permanent_dir}")
            corpus_path = None
            if self.workload_dir:
                corpus_path = self.workload_dir / "corpus.jsonl"
            else:
                # Fallback: infer workload dir from collection name
                # e.g. "hybrid_vdb_beir_nfcorpus" -> "beir/nfcorpus"
                _coll = self.collection_name or ""
                for _dataset in ["nfcorpus", "scifact", "trec-covid"]:
                    if _dataset in _coll:
                        corpus_path = (
                            benchmark_config.BENCHMARK_DIR
                            / "workloads"
                            / "beir"
                            / _dataset
                            / "corpus.jsonl"
                        )
                        break

            if corpus_path and corpus_path.exists():
                import json as _json

                doc_texts = {}
                with corpus_path.open("r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        _rec = _json.loads(_line)
                        _doc_id = str(_rec.get("doc_id") or _rec.get("_id", ""))
                        _text = _rec.get("text", "")
                        if _doc_id and _text:
                            doc_texts[_doc_id] = _text
                self.router.set_doc_texts(doc_texts)
                print(f"[VDB] Pre-loaded {len(doc_texts)} corpus texts for re-ranker")
            else:
                print(f"[VDB] WARNING: No corpus texts loaded for re-ranker. "
                      f"Local serving will be disabled (corpus_path={corpus_path}).")

        print("[VDB] System initialized")
        print(f"[VDB] Permanent dir:   {self.permanent_dir}")
        print(f"[VDB] Dynamic dir:     {self.dynamic_dir}")

    # ── Pure-local baseline variants: local_full, random_3pct ──────────
    #
    # These bypass the three-tier router entirely. They build a FAISS index
    # from the full corpus (or a random 3% subset) and search it directly.
    # Purpose: prove that k-means seeding > random seeding, and establish
    # the 100%-local quality ceiling.
    #
    # Based on: ablation methodology per BEIR evaluation framework
    # [Thakur et al., 2021]; random baseline follows ANN-Benchmarks
    # control methodology [Aumüller et al., 2020].

    def _build_pure_local_index(self) -> None:
        """Build a pure-local FAISS index for local_full / random_3pct variants.

        Fetches ALL vectors from the cloud collection, optionally subsamples
        them, then builds a flat L2 index.

        After this method:
          self._baseline_index  — faiss.IndexFlatL2
          self._baseline_ids    — list[str] aligned to index rows
          self._baseline_dim    — int, vector dimensionality

        Compute: O(N*d) for index construction; N = corpus size.
        Memory: O(N*d) float32 (e.g. NFCorpus 3633×768 ≈ 11 MB).
        """
        import faiss
        from qdrant_client import QdrantClient
        from src.qdrant_ids import external_doc_id

        # Variant-specific sampling
        sample_fraction = 0.03 if self.variant == "random_3pct" else 1.0
        rng_seed = 42  # Deterministic for reproducibility
        print(f"\n[BASELINE] Building pure-local index for variant '{self.variant}'")
        print(f"[BASELINE] Sample fraction: {sample_fraction*100:.0f}%")

        # Connect to cloud (same pattern as seed_permanent_kmeans.py).
        # .env already loaded by Config() in initialize_vdb_system().
        import os as _os

        client = QdrantClient(
            url=_os.getenv("QDRANT_URL", ""),
            api_key=_os.getenv("QDRANT_API_KEY", ""),
        )
        _coll = self.collection_name
        info = client.get_collection(_coll)
        points_count = getattr(info, "vectors_count", None) or getattr(info, "points_count", 0)
        print(f"[BASELINE] Cloud collection '{_coll}': {points_count:,} vectors")

        # Fetch all vectors via scroll (same pattern as seed_permanent_kmeans.py)
        all_vectors: list[np.ndarray] = []
        all_ids: list[str] = []
        offset = None
        batch_size = 256

        with tqdm(total=points_count, desc="Fetching vectors") as pbar:
            while len(all_vectors) < points_count:
                results, offset = client.scroll(
                    collection_name=_coll,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                if not results:
                    break
                for point in results:
                    all_vectors.append(np.array(point.vector, dtype="float32"))
                    payload = point.payload or {}
                    doc_id = external_doc_id(point.id, payload)
                    all_ids.append(doc_id)
                    pbar.update(1)
                if offset is None:
                    break

        n_fetched = len(all_vectors)
        print(f"[BASELINE] Fetched {n_fetched:,} vectors")

        # Subsample for random_3pct
        if sample_fraction < 1.0:
            target_n = max(10, int(n_fetched * sample_fraction))
            rng = np.random.default_rng(rng_seed)
            indices = rng.choice(n_fetched, size=target_n, replace=False)
            indices.sort()  # Keep deterministic order
            all_vectors = [all_vectors[i] for i in indices]
            all_ids = [all_ids[i] for i in indices]
            print(f"[BASELINE] Randomly sampled {len(all_vectors):,} vectors ({sample_fraction*100:.0f}%)")

        # Build FAISS flat L2 index
        vectors_array = np.vstack(all_vectors).astype("float32")
        dim = vectors_array.shape[1]
        self._baseline_index = faiss.IndexFlatL2(dim)
        self._baseline_index.add(vectors_array)
        self._baseline_ids = all_ids
        self._baseline_dim = dim
        self._baseline_n = len(all_ids)

        print(f"[BASELINE] FAISS IndexFlatL2 built: {self._baseline_n:,} vectors, dim={dim}")

    def _search_pure_local(self, query_vector: np.ndarray, k: int) -> dict:
        """Search the pure-local baseline index (local_full / random_3pct).

        Returns a result dict compatible with the benchmark result format.

        Compute: O(N*d) per query (brute-force L2), ~0.1ms for 5K vectors.
        """
        start = time.time()
        qvec = query_vector.reshape(1, -1).astype("float32")
        distances, indices = self._baseline_index.search(qvec, k)
        elapsed_ms = (time.time() - start) * 1000

        ids = [self._baseline_ids[i] for i in indices[0] if i >= 0]
        scores = [float(d) for d, i in zip(distances[0], indices[0]) if i >= 0]

        # Convert L2 distance to cosine similarity for confidence reporting
        # Per FAISS L2 inversion: cos_sim = 1 - d²/2 for unit-normalized vectors
        from src.hybrid_router import HybridRouter
        confidences = [HybridRouter._distance_to_similarity(s) for s in scores]

        return {
            "ids": ids,
            "scores": scores,
            "source": f"baseline_{self.variant}",
            "latency_ms": elapsed_ms,
            "confidence": confidences[0] if confidences else 0.0,
            "variant": self.variant,
            "parallel_search_latency_ms": elapsed_ms,
            "tier3_latency_ms": 0,
            "best_local_tier": f"baseline_{self.variant}",
            "best_local_distance": scores[0] if scores else None,
            "best_local_similarity": confidences[0] if confidences else 0.0,
            "tier1_results": self._baseline_n,
            "tier2_results": 0,
            "dynamic_size_before": 0,
            "dynamic_size_after": 0,
            "dynamic_capacity": 0,
            "cache_admitted_count": 0,
            "cache_evicted_count": 0,
            "cache_duplicate_skip_count": 0,
            "cache_id_skip_count": 0,
            "cache_tier1_skip_count": 0,
            "cache_neighborhood_skip_count": 0,
            "cache_insert_failed_count": 0,
            "local_confident_hit": True,
            "prediction_hit": False,
            "served_locally": True,
            "reranker_best_relevance": None,
            "reranker_best_redundancy": None,
            "reranker_gate_decision": None,
            "reranker_gate_alpha": None,
            "reranker_gate_threshold": None,
        }

    def configure_anchor_system(self, query_embeddings: np.ndarray, session_ids: Optional[List[str]] = None):
        """Configure anchor system from corpus shape after T1 seeding.

        Per Bug #1/#2 fix: epsilon and max_radius must be derived from
        actual corpus geometry. Previous defaults (epsilon=0.05, max_radius=0.12)
        caused anchors to be inert on NFCorpus and TREC-COVID.

        Per mega-anchor fix: for session-ordered workloads, epsilon is computed
        from within-session query distances (topic coherence) instead of Tier 1
        inter-distances (corpus diversity). This prevents a single anchor from
        absorbing all queries.

        Computes:
          epsilon = P75 of within-session distances (if session_ids provided)
                   or P75 of T1 k-NN distances (fallback for static workloads)
          max_radius = P95 of query-T1 distances (anchors must cover typical query range)

        Args:
            query_embeddings: (n_queries, d) array of query vectors.
            session_ids: Optional list of session IDs per query. When provided,
                epsilon is computed from within-session distances.
        """

        # Get Tier 1 vectors from local VDB
        t1_vectors = self.local_vdb.get_permanent_vectors()
        if t1_vectors is None or len(t1_vectors) < 2:
            print("[ANCHOR-CFG] Not enough T1 vectors for anchor config, using defaults")
            return

        print(f"[ANCHOR-CFG] Configuring anchors from {len(t1_vectors)} T1 vectors "
              f"and {len(query_embeddings)} queries...")

        # Compute query-T1 distances for max_radius
        norms_t1 = np.linalg.norm(t1_vectors, axis=1, keepdims=True)
        norms_t1 = np.maximum(norms_t1, 1e-10)
        t1_norm = t1_vectors / norms_t1

        norms_q = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms_q = np.maximum(norms_q, 1e-10)
        q_norm = query_embeddings / norms_q

        sim_q_t1 = q_norm @ t1_norm.T
        max_sim = np.max(sim_q_t1, axis=1)
        query_t1_distances = 1.0 - max_sim

        self.anchor_system.configure_from_corpus_shape(
            tier1_vectors=t1_vectors,
            query_vectors=query_embeddings,
            query_corpus_distances=query_t1_distances,
            session_ids=session_ids,
        )

        print(f"[ANCHOR-CFG] epsilon={self.anchor_system._epsilon:.4f}, "
              f"max_radius={self.anchor_system._max_radius:.4f}")

    def get_queries(
        self,
    ) -> Tuple[
        List[str],
        List[str],
        np.ndarray,
        Optional[List[str]],
        Optional[List[List[str]]],
        Optional[List[dict]],
    ]:
        """
        Get queries based on test mode.

        Returns:
            (query_ids, query_texts, query_embeddings, query_types)
        """
        if self.manifest_file:
            print(f"\n[QUERIES] Loading labeled workload from {self.manifest_file}...")
            (
                query_ids,
                query_texts,
                query_embeddings,
                query_types,
                expected_ids,
                query_metadata,
            ) = self.query_generator.load_manifest_queries(
                self.manifest_file, limit=self.num_queries
            )

        elif self.test_mode == "quick" and self.custom_queries_file:
            # Load from file
            print(f"\n[QUERIES] Loading from {self.custom_queries_file}...")
            query_ids, query_texts, query_embeddings = (
                self.query_generator.load_custom_queries(
                    self.custom_queries_file, limit=self.num_queries
                )
            )
            query_types = None
            expected_ids = None
            query_metadata = None

        elif self.test_mode == "quick":
            # Sample from dataset
            print(f"\n[QUERIES] Sampling {self.num_queries} quick test queries...")
            query_ids, query_texts, query_embeddings = (
                self.query_generator.generate_quick_queries(self.num_queries)
            )
            query_types = None
            expected_ids = None
            query_metadata = None

        else:
            # Full test with realistic distribution
            print(
                f"\n[QUERIES] Generating {self.num_queries} queries with realistic mix..."
            )
            query_ids, query_texts, query_embeddings, query_types = (
                self.query_generator.generate_full_test_queries(
                    total=self.num_queries,
                    in_dataset_ratio=benchmark_config.IN_DATASET_RATIO,
                    edge_case_ratio=benchmark_config.EDGE_CASE_RATIO,
                    ood_ratio=benchmark_config.OOD_RATIO,
                )
            )
            expected_ids = None
            query_metadata = None

        return (
            query_ids,
            query_texts,
            query_embeddings,
            query_types,
            expected_ids,
            query_metadata,
        )

    def validate_labeled_workload_alignment(
        self, expected_ids: Optional[List[List[str]]]
    ):
        """
        Fail fast if labeled expected_ids do not exist in the active cloud collection.

        This catches a common offline-eval error: running qrels from one corpus against
        a different collection or against a collection seeded with row indices instead
        of canonical corpus IDs.
        """
        if not expected_ids:
            return

        sample_expected_ids = []
        for row in expected_ids:
            for item_id in row:
                value = str(item_id)
                if value not in sample_expected_ids:
                    sample_expected_ids.append(value)
                if len(sample_expected_ids) >= 20:
                    break
            if len(sample_expected_ids) >= 20:
                break

        if not sample_expected_ids:
            return

        retrieved_vectors = self.cloud_vdb.get_vectors_by_ids(sample_expected_ids)
        if retrieved_vectors:
            print(
                f"[VALIDATION] Labeled workload aligned: found {len(retrieved_vectors)} "
                f"of {len(sample_expected_ids)} sampled expected_ids in cloud collection"
            )
            return

        raise ValueError(
            "Labeled workload does not align with the active Qdrant collection. "
            "Sampled expected_ids from the manifest were not found in cloud storage. "
            "This usually means you are evaluating SciFact/BEIR qrels against the wrong "
            "corpus or the collection was seeded with row indices instead of canonical doc IDs. "
            f"Seed the workload corpus into collection '{self.collection_name}' before benchmarking."
        )

    def run_benchmark(
        self,
        query_ids: List[str],
        query_texts: List[str],
        query_embeddings: np.ndarray,
        query_types: Optional[List[str]] = None,
        expected_ids: Optional[List[List[str]]] = None,
        query_metadata: Optional[List[dict]] = None,
    ):
        """Execute benchmark queries."""
        print("\n" + "=" * 70)
        print("STARTING BENCHMARK")
        print("=" * 70)

        start_time = time.time()
        snapshot_interval = 10 if self.test_mode == "quick" else 50

        for i in tqdm(range(len(query_ids)), desc="Running queries"):
            qid = query_ids[i]
            qtext = query_texts[i]
            qvec = query_embeddings[i]
            qtype = query_types[i] if query_types else "unknown"
            metadata = query_metadata[i] if query_metadata else {}

            # Execute query — route to pure-local baseline or three-tier router
            if self.variant in ("local_full", "random_3pct"):
                result = self._search_pure_local(qvec, self.top_k)
            else:
                result = self.router.search(
                    query_vector=qvec, query_id=qid, query_text=qtext, k=self.top_k
                )

            # Record result
            row = {
                "query_num": i + 1,
                "query_id": qid,
                "query_type": qtype,
                "query_text": qtext[:80],
                "latency_ms": result["latency_ms"],
                "source": result["source"],
                "confidence": result.get("confidence", 0),
                "timestamp": time.time() - start_time,
                "variant": result.get("variant", self.variant),
                "parallel_search_latency_ms": result.get("parallel_search_latency_ms"),
                "tier3_latency_ms": result.get("tier3_latency_ms"),
                "best_local_tier": result.get("best_local_tier"),
                "best_local_distance": result.get("best_local_distance"),
                "best_local_similarity": result.get("best_local_similarity"),
                "tier1_results": result.get("tier1_results"),
                "tier2_results": result.get("tier2_results"),
                "dynamic_size_before": result.get("dynamic_size_before"),
                "dynamic_size_after": result.get("dynamic_size_after"),
                "dynamic_capacity": result.get("dynamic_capacity"),
                "cache_admitted_count": result.get("cache_admitted_count", 0),
                "cache_evicted_count": result.get("cache_evicted_count", 0),
                "cache_duplicate_skip_count": result.get(
                    "cache_duplicate_skip_count", 0
                ),
                # Per §22 Step 1: split skip diagnostics
                "cache_id_skip_count": result.get("cache_id_skip_count", 0),
                "cache_tier1_skip_count": result.get("cache_tier1_skip_count", 0),
                "cache_neighborhood_skip_count": result.get("cache_neighborhood_skip_count", 0),
                "cache_insert_failed_count": result.get("cache_insert_failed_count", 0),
                "local_confident_hit": result.get("local_confident_hit"),
                "session_id": metadata.get("session_id"),
                "turn_id": metadata.get("turn_id"),
                "trace_id": metadata.get("trace_id"),
                "order_index": metadata.get("order_index"),
                # Per §19.1: Three prediction metrics tracked per-query
                "prediction_hit": result.get("prediction_hit", False),
                "served_locally": result.get("served_locally", False),
                # Per §19.1 diagnostic: reranker gate values (Phase 5: now hybrid gate)
                "reranker_best_relevance": result.get("reranker_best_relevance"),
                "reranker_passed_count": result.get("reranker_passed_count", 0),
                "reranker_total_count": result.get("reranker_total_count", 0),
                # Per Phase 5: hybrid gate diagnostics
                "gate_signal": result.get("gate_signal"),
                "gate_threshold": result.get("gate_threshold"),
                "gate_best_cosine": result.get("gate_best_cosine"),
                "gate_best_bm25_raw": result.get("gate_best_bm25_raw"),
                "gate_best_bm25_norm": result.get("gate_best_bm25_norm"),
                "gate_alpha": result.get("gate_alpha"),
                # Per Phase 5.7: shadow probe calibration diagnostics
                "calibration_phase": result.get("calibration_phase"),
                "shadow_probe": result.get("shadow_probe"),
                "calibration_served_cloud": result.get("calibration_served_cloud"),
                "calibration_fallback": result.get("calibration_fallback"),
            }

            # Per Tier 2 hit decomposition: count predictive vs reactive hits
            tier2_result_count = result.get("tier2_results", 0)
            if tier2_result_count and tier2_result_count > 0:
                returned_ids = result.get("ids", [])
                if returned_ids:
                    t2_meta = self.local_vdb.get_dynamic_metadata_batch(
                        [str(rid) for rid in returned_ids]
                    )
                    row["tier2_predictive_hits"] = sum(
                        1
                        for m in t2_meta.values()
                        if m.get("source") == "predictive_prefetch"
                    )
                    row["tier2_reactive_hits"] = sum(
                        1
                        for m in t2_meta.values()
                        if m.get("source") == "reactive_cache"
                    )
                else:
                    row["tier2_predictive_hits"] = 0
                    row["tier2_reactive_hits"] = 0
            if expected_ids is not None:
                row["expected_ids_count"] = len(expected_ids[i])
                ranking = compute_ranking_metrics(
                    result.get("ids", []), expected_ids[i], k=self.top_k
                )
                if ranking:
                    row.update(ranking)
            self.results["per_query"].append(row)

            # Take snapshots (skip for pure-local baselines — no router state)
            if (i + 1) % snapshot_interval == 0 and self.variant not in ("local_full", "random_3pct"):
                # Anchor snapshot
                anchor_stats = self.anchor_system.get_anchor_stats()
                self.results["anchor_snapshots"].append(
                    {"query_num": i + 1, **anchor_stats}
                )

                # Cache snapshot
                cache_stats = self.local_vdb.get_dynamic_stats()
                meta_stats = self.local_vdb.storage.get_metadata_stats()
                cache_stats["predictive_count"] = meta_stats["source_counts"].get(
                    "predictive_prefetch", 0
                )
                cache_stats["reactive_count"] = meta_stats["source_counts"].get(
                    "reactive_cache", 0
                )
                self.results["cache_snapshots"].append(
                    {"query_num": i + 1, **cache_stats}
                )

            # Realistic delay
            time.sleep(benchmark_config.QUERY_INTERVAL_MS / 1000.0)

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print("=" * 70)

    def analyze_results(self):
        """Analyze and display results."""
        print("\n" + "=" * 70)
        print("RESULTS ANALYSIS")
        print("=" * 70)

        # For pure-local baselines, metrics/router are not initialized
        if self.variant in ("local_full", "random_3pct"):
            metrics = {}
        else:
            metrics = self.metrics.get_summary()
        metrics.update(summarize_query_metrics(self.results["per_query"]))
        metrics.update(summarize_session_metrics(self.results["per_query"]))

        # Compute basic latency stats from per_query rows (works for all variants)
        latencies = [
            float(row["latency_ms"])
            for row in self.results["per_query"]
            if row.get("latency_ms") is not None
        ]
        if latencies:
            sorted_lat = sorted(latencies)
            metrics.setdefault("total_queries", len(latencies))
            metrics.setdefault("avg_latency", sum(latencies) / len(latencies))
            metrics.setdefault("p50_latency", sorted_lat[len(sorted_lat) // 2])
            metrics.setdefault("p95_latency", sorted_lat[int(len(sorted_lat) * 0.95)])
            metrics.setdefault("min_latency", sorted_lat[0])
            metrics.setdefault("max_latency", sorted_lat[-1])
            # Tier hit rates from per_query source field
            t1_count = sum(1 for r in self.results["per_query"] if r.get("source") == "tier1_permanent")
            t2_count = sum(1 for r in self.results["per_query"] if r.get("source") == "tier2_dynamic")
            t3_count = sum(1 for r in self.results["per_query"] if r.get("source") == "tier3_cloud")
            baseline_count = sum(1 for r in self.results["per_query"] if "baseline_" in str(r.get("source", "")))
            n = len(latencies)
            metrics.setdefault("tier1_hit_rate", t1_count / n * 100)
            metrics.setdefault("tier2_hit_rate", t2_count / n * 100)
            metrics.setdefault("tier3_hit_rate", t3_count / n * 100)
            metrics.setdefault("local_hit_rate", (t1_count + t2_count + baseline_count) / n * 100)
        metrics.update(
            {
                "dynamic_admissions_total": int(
                    sum(
                        row.get("cache_admitted_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                "dynamic_evictions_total": int(
                    sum(
                        row.get("cache_evicted_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                "dynamic_duplicate_skips_total": int(
                    sum(
                        row.get("cache_duplicate_skip_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                # Per §22 Step 1: split skip diagnostics
                "dynamic_id_skips_total": int(
                    sum(
                        row.get("cache_id_skip_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                "dynamic_tier1_skips_total": int(
                    sum(
                        row.get("cache_tier1_skip_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                "dynamic_neighborhood_skips_total": int(
                    sum(
                        row.get("cache_neighborhood_skip_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                "dynamic_insert_failures_total": int(
                    sum(
                        row.get("cache_insert_failed_count", 0) or 0
                        for row in self.results["per_query"]
                    )
                ),
                # Per §19.1: Three prediction metrics
                "prediction_accuracy_pct": (
                    sum(1 for r in self.results["per_query"] if r.get("prediction_hit"))
                    / len(self.results["per_query"]) * 100
                    if self.results["per_query"] else 0.0
                ),
                "tier2_hit_rate_pct": (
                    sum(1 for r in self.results["per_query"] if r.get("source") == "tier2_dynamic")
                    / len(self.results["per_query"]) * 100
                    if self.results["per_query"] else 0.0
                ),
                "prediction_yield_pct": (
                    sum(
                        1 for r in self.results["per_query"]
                        if r.get("prediction_hit") and r.get("source") == "tier2_dynamic"
                    )
                    / len(self.results["per_query"]) * 100
                    if self.results["per_query"] else 0.0
                ),
                # Per Phase 5.7: calibration diagnostics
                "calibration_cold_queries": int(
                    sum(1 for r in self.results["per_query"]
                        if r.get("calibration_phase") == "cold")
                ),
                "calibration_warmup_queries": int(
                    sum(1 for r in self.results["per_query"]
                        if r.get("calibration_phase") == "warmup")
                ),
                "calibration_steady_queries": int(
                    sum(1 for r in self.results["per_query"]
                        if r.get("calibration_phase") == "steady")
                ),
                "calibration_shadow_probes": int(
                    sum(1 for r in self.results["per_query"]
                        if r.get("shadow_probe") is True)
                ),
                "calibration_cloud_served": int(
                    sum(1 for r in self.results["per_query"]
                        if r.get("calibration_served_cloud") is True)
                ),
            }
        )

        # Aggregate final anchor stats into summary
        if self.results.get("anchor_snapshots"):
            final_snapshot = self.results["anchor_snapshots"][-1]
            metrics["anchor_system"] = {
                "total_anchors": final_snapshot.get("total_anchors", 0),
                "anchor_types": final_snapshot.get("anchor_types", {}),
                "avg_strength": final_snapshot.get("avg_strength", 0.0),
                "max_strength": final_snapshot.get("max_strength", 0.0),
                "avg_radius": final_snapshot.get("avg_radius", 0.0),
                "max_radius": final_snapshot.get("max_radius", 0.0),
                "total_hits": final_snapshot.get("total_hits", 0),
                "total_misses": final_snapshot.get("total_misses", 0),
                "prediction_accuracy": final_snapshot.get("prediction_accuracy", 0.0),
                "markov": final_snapshot.get("markov", {}),
            }

        self.results["summary"] = metrics

        # Overall performance
        print("\nðŸ“Š OVERALL PERFORMANCE:")
        print(f"Total queries:        {metrics['total_queries']:,}")
        print(f"Avg latency:          {metrics['avg_latency']:.1f}ms")
        print(f"P50 latency:          {metrics['p50_latency']:.1f}ms")
        print(f"P95 latency:          {metrics['p95_latency']:.1f}ms")
        print(f"Min latency:          {metrics['min_latency']:.1f}ms")
        print(f"Max latency:          {metrics['max_latency']:.1f}ms")

        # Hit rates
        print("\nðŸŽ¯ HIT RATES:")
        print(f"TIER 1 (Permanent):   {metrics['tier1_hit_rate']:.1f}%")
        print(f"TIER 2 (Dynamic):     {metrics['tier2_hit_rate']:.1f}%")
        print(f"TIER 3 (Cloud):       {metrics['tier3_hit_rate']:.1f}%")
        print(f"Local Total (T1+T2):  {metrics['local_hit_rate']:.1f}%")

        if metrics.get("labeled_queries", 0):
            print("\nRETRIEVAL QUALITY:")
            print(f"Labeled queries:      {int(metrics['labeled_queries'])}")
            print(
                f"Recall@{self.top_k}:           {metrics.get('mean_recall_at_k', 0.0):.3f}"
            )
            print(
                f"MRR@{self.top_k}:              {metrics.get('mean_mrr_at_k', 0.0):.3f}"
            )
            print(
                f"nDCG@{self.top_k}:             {metrics.get('mean_ndcg_at_k', 0.0):.3f}"
            )
            print(f"Success@1:            {metrics.get('mean_success_at_1', 0.0):.3f}")

        if metrics.get("session_count", 0):
            print("\nSESSION WORKLOAD:")
            print(f"Sessions:             {int(metrics['session_count'])}")
            print(
                f"Mean turns/session:   {metrics.get('mean_turns_per_session', 0.0):.2f}"
            )
            print(
                f"Turn-1 local hit:     {metrics.get('turn1_local_hit_rate', 0.0) * 100:.1f}%"
            )
            print(
                f"Later-turn local hit: {metrics.get('later_turn_local_hit_rate', 0.0) * 100:.1f}%"
            )
            print(
                f"Mean session local:   {metrics.get('mean_session_local_hit_rate', 0.0) * 100:.1f}%"
            )
            if metrics.get("mean_session_recall_at_k") is not None:
                print(
                    f"Mean session Recall@{self.top_k}: {metrics.get('mean_session_recall_at_k', 0.0):.3f}"
                )
            if metrics.get("mean_session_mrr_at_k") is not None:
                print(
                    f"Mean session MRR@{self.top_k}:    {metrics.get('mean_session_mrr_at_k', 0.0):.3f}"
                )
            if metrics.get("mean_session_ndcg_at_k") is not None:
                print(
                    f"Mean session nDCG@{self.top_k}:   {metrics.get('mean_session_ndcg_at_k', 0.0):.3f}"
                )

            # Per-turn breakdown — validates anchor-and-momentum quality improvement
            per_turn_keys = sorted(
                int(k.split('_')[0].replace('turn', ''))
                for k in metrics
                if k.startswith('turn') and k.endswith('_n')
            )
            if per_turn_keys:
                print("\nPER-TURN BREAKDOWN (anchor claim: quality improves over turns):")
                header = f"{'Turn':>4} {'N':>5} {'Local%':>7} {'Pred%':>7}"
                has_ndcg = any(metrics.get(f'turn{t}_ndcg') is not None for t in per_turn_keys)
                has_mrr = any(metrics.get(f'turn{t}_mrr') is not None for t in per_turn_keys)
                has_recall = any(metrics.get(f'turn{t}_recall') is not None for t in per_turn_keys)
                if has_ndcg:
                    header += f" {'nDCG':>6}"
                if has_mrr:
                    header += f" {'MRR':>6}"
                if has_recall:
                    header += f" {'Recall':>7}"
                print(header)
                print("-" * len(header))
                for t in per_turn_keys:
                    prefix = f"turn{t}"
                    line = f"{t:>4} {int(metrics.get(f'{prefix}_n', 0)):>5} {metrics.get(f'{prefix}_local_hit_rate', 0)*100:>6.1f}% {metrics.get(f'{prefix}_prediction_hit_rate', 0)*100:>6.1f}%"
                    if has_ndcg and metrics.get(f'{prefix}_ndcg') is not None:
                        line += f" {metrics[f'{prefix}_ndcg']:>6.3f}"
                    if has_mrr and metrics.get(f'{prefix}_mrr') is not None:
                        line += f" {metrics[f'{prefix}_mrr']:>6.3f}"
                    if has_recall and metrics.get(f'{prefix}_recall') is not None:
                        line += f" {metrics[f'{prefix}_recall']:>7.3f}"
                    print(line)

        # Learning progression
        if len(self.results["anchor_snapshots"]) > 0:
            final_anchors = self.results["anchor_snapshots"][-1]
            anchor_types = final_anchors.get("anchor_types", {})

            print("\nâš“ ANCHOR SYSTEM:")
            print(f"Total anchors:        {final_anchors.get('total_anchors', 0)}")
            print(
                f"  Weak:               {anchor_types.get('weak', final_anchors.get('weak_anchors', 0))}"
            )
            print(
                f"  Medium:             {anchor_types.get('medium', final_anchors.get('medium_anchors', 0))}"
            )
            print(
                f"  Strong:             {anchor_types.get('strong', final_anchors.get('strong_anchors', 0))}"
            )
            print(
                f"  Permanent:          {anchor_types.get('permanent', final_anchors.get('permanent_anchors', 0))}"
            )
            print(f"Active predictions:   {final_anchors.get('active_predictions', 0)}")
            print(
                f"Prediction accuracy:  {final_anchors.get('prediction_accuracy', 0.0):.1f}%"
            )

        # Per §19.1: Three prediction metrics
        if metrics.get("prediction_accuracy_pct") is not None:
            print("\nPREDICTION METRICS (per §19.1):")
            print(
                f"  Accuracy (geometric): {metrics['prediction_accuracy_pct']:.1f}%"
                f"  — did query land in predicted basin?"
            )
            print(
                f"  Tier 2 hit rate:      {metrics['tier2_hit_rate_pct']:.1f}%"
                f"  — fraction served from Tier 2"
            )
            print(
                f"  Prediction yield:     {metrics['prediction_yield_pct']:.1f}%"
                f"  — predicted AND served from Tier 2"
            )

        # Per Phase 5.7: calibration diagnostics
        if metrics.get("calibration_cold_queries", 0) > 0 or metrics.get("calibration_shadow_probes", 0) > 0:
            print("\nCALIBRATION (Phase 5.7):")
            print(f"  Cold queries (cloud-served): {metrics.get('calibration_cold_queries', 0)}")
            print(f"  Warmup queries:              {metrics.get('calibration_warmup_queries', 0)}")
            print(f"  Steady queries:             {metrics.get('calibration_steady_queries', 0)}")
            print(f"  Shadow probes:               {metrics.get('calibration_shadow_probes', 0)}")
            print(f"  Cloud-served (cold start):   {metrics.get('calibration_cloud_served', 0)}")

        if len(self.results["cache_snapshots"]) > 0:
            final_cache = self.results["cache_snapshots"][-1]

            print("\nðŸ’¾ DYNAMIC CACHE:")
            print(
                f"Size:                 {final_cache.get('current_size', 0):,} / "
                f"{final_cache.get('capacity', 0):,}"
            )
            print(f"Fill rate:            {final_cache.get('fill_rate', 0):.1f}%")
            print(f"Avg weight:           {final_cache.get('avg_weight', 0):.2f}")

        print("\nCACHE ADMISSION:")
        print(f"Admitted:             {metrics.get('dynamic_admissions_total', 0)}")
        print(f"Evicted:              {metrics.get('dynamic_evictions_total', 0)}")
        print(
            f"Duplicate skips:      {metrics.get('dynamic_duplicate_skips_total', 0)}"
        )
        print(
            f"Insert failures:      {metrics.get('dynamic_insert_failures_total', 0)}"
        )

        # Speedup — use measured cloud latency if available
        cloud_latency = metrics.get("cloud_avg_latency", metrics.get("tier3_latency", 0))
        if cloud_latency == 0 or cloud_latency is None:
            cloud_latency = 200  # Fallback if no cloud queries occurred
        speedup = (
            (cloud_latency / metrics["avg_latency"]) if metrics["avg_latency"] else 0.0
        )

        print("\nðŸš€ SPEEDUP:")
        print(f"Cloud-only baseline:  {cloud_latency}ms")
        print(f"Hybrid system:        {metrics['avg_latency']:.1f}ms")
        print(f"Speedup:              {speedup:.1f}x faster")

        # Proactive prefetch diagnostics
        if self.results.get("queries"):
            pf_triggers = sum(1 for q in self.results["queries"] if q.get("prefetch_trigger") == "proactive")
            pf_preds = sum(q.get("prefetch_predictions_generated", 0) for q in self.results["queries"])
            pf_admitted = sum(q.get("cache_admitted_count", 0) for q in self.results["queries"])
            pf_evicted = sum(q.get("cache_evicted_count", 0) for q in self.results["queries"])
            print(f"\nPROACTIVE PREFETCH:")
            print(f"Prefetch triggers:    {pf_triggers}/{len(self.results['queries'])} queries")
            print(f"Predictions generated:{pf_preds}")
            print(f"Docs admitted to T2:  {pf_admitted}")
            print(f"Docs evicted from T2: {pf_evicted}")

        print("\n" + "=" * 70)

    def save_results(self):
        """Save results to files."""
        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] Saving results to {output_dir}...")

        # Save JSON
        import json

        json_file = output_dir / "results.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[SAVE] âœ… JSON: {json_file.name}")

        run_manifest = {
            "git_sha": self.git_sha,
            "variant": self.variant,
            "mode": self.test_mode,
            "manifest_file": str(Path(self.manifest_file).resolve())
            if self.manifest_file
            else None,
            "collection_name": self.collection_name,
            "permanent_dir": str(self.permanent_dir) if self.permanent_dir else None,
            "dynamic_dir": str(self.dynamic_dir) if self.dynamic_dir else None,
            "embedding_model": benchmark_config.EMBEDDING_MODEL,
            "tier1_size": benchmark_config.TIER1_SIZE,
            "tier2_capacity": benchmark_config.TIER2_CAPACITY,
            "tier1_percent": benchmark_config.TIER1_PERCENT,
            "tier2_percent": benchmark_config.TIER2_PERCENT,
            "top_k": self.top_k,
            "timestamp": time.time(),
        }
        manifest_file = output_dir / "run_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(run_manifest, f, indent=2)
        print(f"[SAVE] âœ… MANIFEST: {manifest_file.name}")

        # Save CSV
        import pandas as pd

        df = pd.DataFrame(self.results["per_query"])
        csv_file = output_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        print(f"[SAVE] âœ… CSV: {csv_file.name}")

    def run(self):
        """Run complete benchmark pipeline."""
        try:
            # Initialize VDB
            self.initialize_vdb_system()

            # Get queries
            (
                query_ids,
                query_texts,
                query_embeddings,
                query_types,
                expected_ids,
                query_metadata,
            ) = self.get_queries()

            # Validate labeled workload against active collection before spending minutes benchmarking.
            # Skip for pure-local baselines — they don't use the router or cloud at query time.
            if self.variant not in ("local_full", "random_3pct"):
                self.validate_labeled_workload_alignment(expected_ids)

            # Per Bug #1/#2 fix: configure anchor system from corpus shape
            # Per mega-anchor fix: pass session_ids for session-aware epsilon
            # Skip for pure-local baselines — no anchor system.
            if self.variant not in ("local_full", "random_3pct"):
                session_ids = None
                if query_metadata:
                    session_ids = [m.get("session_id") if m else None for m in query_metadata]
                self.configure_anchor_system(query_embeddings, session_ids=session_ids)

            # Run benchmark
            self.run_benchmark(
                query_ids,
                query_texts,
                query_embeddings,
                query_types,
                expected_ids,
                query_metadata,
            )

            # Make the final cache state inspectable and reproducible.
            # Skip for pure-local baselines — no router or dynamic cache.
            if self.variant not in ("local_full", "random_3pct"):
                self.router.wait_for_background_prefetch()
                self.local_vdb.save_state()

            # Analyze
            self.analyze_results()

            # Per Step 4: Save gate status (including per-anchor thresholds)
            # after all queries so we can inspect anchor-specific adaptation.
            if self.variant not in ("local_full", "random_3pct"):
                try:
                    gate_status = self.router._hybrid_gate.status()
                    self.results["gate_status"] = gate_status
                except Exception as e:
                    print(f"[WARN] Could not capture gate status: {e}")

            # Visualize
            self.visualizer.generate_plots(self.results)

            # Save
            self.save_results()

            print("\n" + "=" * 70)
            print("âœ… BENCHMARK COMPLETE!")
            print("=" * 70)
            final_dir = self._resolve_output_dir()
            print(f"Results: {final_dir}")

        except KeyboardInterrupt:
            print("\n\nâš ï¸  Benchmark interrupted by user")
            print("Partial results saved.")
            if self.router and self.local_vdb:
                self.router.wait_for_background_prefetch(timeout_s=5.0)
                self.local_vdb.save_state()
            self.save_results()

        except Exception as e:
            print(f"\nâŒ BENCHMARK FAILED: {e}")
            import traceback

            traceback.print_exc()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid VDB Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 queries)
  python benchmark.py --mode quick

  # Quick test with custom queries
  python benchmark.py --mode quick --queries-file queries/custom_queries.txt

  # Full test (1000 queries, 65/25/10 distribution)
  python benchmark.py --mode full
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Test mode: quick (100) or full (1000 queries)",
    )

    parser.add_argument(
        "--queries-file", type=str, help="Path to custom queries file (for quick mode)"
    )

    parser.add_argument(
        "--manifest-file",
        type=str,
        help="Path to a JSONL manifest with payload and expected_ids",
    )

    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-k results to request and evaluate"
    )

    parser.add_argument(
        "--variant",
        choices=[
            "full_hybrid",
            "parallel_hybrid",
            "reactive_cache",
            "no_prefetch",
            "local_only",
            "cloud_only",
            "t1_plus_cloud",
            "local_full",
            "random_3pct",
            "true_lru",         # Release-04: pure LRU eviction, no anchors
            "freq_cache",       # Release-04: frequency-based caching
        ],
        default="full_hybrid",
        help="Benchmark variant to run",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        help="Optional Qdrant collection override. For workload manifests, defaults to a derived workload-specific collection name.",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=None,
        help="Override GATE_INITIAL_THRESHOLD for sweep runs. Only affects gate-dependent variants.",
    )
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Override output directory name for sweep runs.",
    )
    parser.add_argument(
        "--cold-start", action="store_true", default=False,
        help="Delete dynamic/ (Tier 2) directory before init. "
             "Ensures empty Tier 2 so prefetch/reactive contribution is measured, not pre-seeded cache.",
    )
    parser.add_argument(
        "--dynamic-dir-override", type=str, default=None,
        help="Override dynamic/ (Tier 2) directory path. "
             "REQUIRED when running multiple variants in parallel — each variant must have "
             "its own isolated dynamic/ dir to prevent FAISS index corruption from concurrent writes.",
    )
    parser.add_argument(
        "--tier2-percent", type=float, default=None,
        help="Override Tier 2 size as fraction of corpus (e.g. 0.05 for 5%%, 0.30 for 30%%). "
             "Default is config.TIER2_PERCENT (0.15).",
    )

    args = parser.parse_args()

    # Per pre-flight P1: if gate-threshold override given, patch config before init
    # Also patch the HybridGate singleton's adaptive threshold (created at import time).
    if args.gate_threshold is not None:
        import src.config as cfg_module
        cfg_module.config.GATE_INITIAL_THRESHOLD = args.gate_threshold
        cfg_module.GATE_INITIAL_THRESHOLD = args.gate_threshold
        # Patch the already-created HybridGate singleton
        from src.reranker import get_hybrid_gate
        gate = get_hybrid_gate()
        gate.adaptive_threshold.threshold = args.gate_threshold
        gate.adaptive_threshold.min_threshold = args.gate_threshold  # Lock threshold
        gate.adaptive_threshold.max_threshold = args.gate_threshold  # Lock threshold
        # Also patch any anchor-specific thresholds
        gate.anchor_aware_threshold._global.threshold = args.gate_threshold

    # Per tier sizing sweep: override TIER2_PERCENT if given.
    if args.tier2_percent is not None:
        import src.config as cfg_module
        cfg_module.config.TIER2_PERCENT = args.tier2_percent
        cfg_module.TIER2_PERCENT = args.tier2_percent
        benchmark_config.TIER2_PERCENT = args.tier2_percent  # per tier sizing sweep

    # Run benchmark
    benchmark = HybridVDBBenchmark(
        test_mode=args.mode,
        custom_queries_file=args.queries_file,
        manifest_file=args.manifest_file,
        top_k=args.top_k,
        variant=args.variant,
        collection_name=args.collection_name,
        output_name=args.output_name,
        cold_start=args.cold_start,
        dynamic_dir_override=args.dynamic_dir_override,
    )
    benchmark.run()


if __name__ == "__main__":
    main()
