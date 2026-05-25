# src/hybrid_router.py

"""
Hybrid Router with Three-Tier Architecture and Smart Prefetching

TIER 1: Permanent Local VDB (300 vectors, read-only, privacy)
TIER 2: Dynamic Prefetch Space (700 vectors, learning engine)
TIER 3: Cloud VDB (9,482 vectors, canonical truth)

Core Innovation:
- Fixed-size dynamic space (700 vectors, no bloating)
- Smart prefetching (checks if neighborhood exists before fetching)
- Phase-based strategy (cold start â†’ warmup â†’ steady state)
- Momentum-based trajectory predictions

OPTIMIZED (2025-11-30):
- Parallel TIER 1 + TIER 2 search (33% faster local searches)

Author: Anonymous
Date: 2025-11-30 (PARALLEL SEARCH OPTIMIZATION)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
import concurrent.futures  # âœ… NEW: For parallel tier search

from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.local_vdb import LocalVDB
from src.cloud_client import QdrantCloudClient
from src.metrics import MetricsTracker
from src.config import (
    LOCAL_CONFIDENCE_THRESHOLD,
    PREFETCH_ENABLED,
    DYNAMIC_LAYER_CAPACITY,
    VERBOSE,
    DEFAULT_SEARCH_K,
    NEIGHBORHOOD_THRESHOLD_ADMISSION,
    NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP,
    NEIGHBORHOOD_THRESHOLD_PREFETCH_ADMISSION,
)
from src.config import COLD_START_QUERIES, WARMUP_QUERIES
from src.config import NOISE_SCALE_COLD, NOISE_SCALE_WARMUP
from src.config import (
    CALIBRATION_COLD_QUERIES,
    CALIBRATION_WARMUP_QUERIES,
    CALIBRATION_PROBE_INTERVAL_COLD,
    CALIBRATION_PROBE_INTERVAL_WARMUP,
    CALIBRATION_PROBE_INTERVAL_STEADY,
    MIN_LOCAL_QUALITY_FLOOR,
)
from src.reranker import get_hybrid_gate

logger = logging.getLogger(__name__)


class PrefetchPhase(Enum):
    """Prefetch strategy phases based on query count."""

    COLD_START = "cold_start"  # Query 1-3: Fill aggressively
    WARMUP = "warmup"  # Query 4-20: Refine accuracy
    STEADY_STATE = "steady_state"  # Query 20+: Maintain consistency


class HybridRouter:
    """
    Three-tier hybrid router with smart prefetching.

    Query Flow:
    1. Check TIER 1 (Permanent) â†’ Privacy layer, always available
    2. Check TIER 2 (Dynamic) â†’ Learning engine, 700 fixed capacity
    3. Fallback TIER 3 (Cloud) â†’ Truth source, minimize access
    4. Smart prefetch: Only fetch if NOT already in dynamic space
    """

    def __init__(
        self,
        local_vdb: LocalVDB,
        cloud_vdb: QdrantCloudClient,
        semantic_cache: SemanticClusterCache,
        anchor_system: AnchorSystem,
        metrics: MetricsTracker,
        *,
        prefetch_enabled: bool = PREFETCH_ENABLED,
        routing_mode: str = "full_hybrid",
    ):
        """Initialize three-tier router."""
        self.local_vdb = local_vdb
        self.cloud_vdb = cloud_vdb
        self.semantic_cache = semantic_cache
        self.anchor_system = anchor_system
        self.metrics = metrics
        self.prefetch_enabled = prefetch_enabled
        self.routing_mode = routing_mode

        # Per co-evolutionary loop: register anchor weight callback for T2 eviction
        self.local_vdb.set_anchor_weight_callback(self.anchor_system)

        # reactive_cache: same as full_hybrid but prefetch disabled and anchor
        # prediction suppressed for routing — anchors still instrumented for metrics.
        # This is the clean control that isolates anchor+prefetch contribution.
        if self.routing_mode == "reactive_cache":
            prefetch_enabled = False

        # true_lru: strict LRU eviction baseline — no anchors, no prefetch.
        # Per ablation: isolates anchor-weight eviction vs pure LRU timestamp eviction.
        # Still uses the three-tier architecture (T1 + T2 + cloud) but T2 eviction
        # is strict LRU by last_accessed timestamp, not by anchor confidence cascade.
        if self.routing_mode == "true_lru":
            prefetch_enabled = False
            # Switch storage engine to LRU eviction mode
            self.local_vdb.storage.eviction_mode = "lru"

        # parallel_hybrid: always query cloud in parallel with local, merge
        # results by cosine similarity. Cloud hits that beat local drive
        # Tier 2 cache population (the reactive+predictive loop). This is
        # the correct architecture — the confidence-gated approach fails
        # because top-1 cosine does not distinguish relevance from proximity.
        if self.routing_mode == "parallel_hybrid":
            prefetch_enabled = True

        self.query_count = 0
        self.prefetch_cache_hits = 0
        self.prefetch_cache_misses = 0
        self._predictions_matched = 0  # per §19.1: geometric prediction accuracy (deprecated, kept for metrics compat)
        self._tier2_serves = 0  # per §19.1: Tier 2 serve counter
        self._prefetch_futures: List[concurrent.futures.Future] = []
        self._prefetch_futures_lock = threading.Lock()
        self._warm_seeded = False  # One-time Q1 warm-seed flag

        # Per prefetch quality audit: track which vectors were admitted by
        # which source (reactive vs prefetch) and which anchors, and which
        # T2 vectors appear in final top-k results. Post-hoc analysis joins
        # these logs to compute 5-turn reuse rates per anchor.
        self.prefetch_admission_log: List[Dict] = []
        self.reuse_log: List[Dict] = []

        # Per Phase 5: BM25+cosine hybrid gate replaces cross-encoder.
        # The cross-encoder added 90ms latency and rejected 96.7% of local
        # results on NFCorpus (domain mismatch). The hybrid gate runs in
        # <0.1ms and adapts its threshold based on observed local precision.
        # Based on: BM25 [Robertson & Zaragoza, 2009]; adaptive threshold
        # via precision window (THEORY.md §20).
        self._hybrid_gate = get_hybrid_gate()

        # Per Bug 5 fix: REMOVED separate _adaptive_threshold. The router
        # now delegates to _hybrid_gate.adaptive_threshold.threshold for all
        # variants. This eliminates the dual-threshold inconsistency where
        # parallel_hybrid used the gate's precision-window threshold (0.90)
        # while full_hybrid/reactive_cache used a raw cosine EMA (0.75).
        # The gate's AdaptiveThreshold (reranker.py) calibrates via
        # record_outcome() and shadow probes — same mechanism for all variants.
        self._cloud_similarity_ema = 0.0
        self._cloud_samples = 0
        self._adaptive_ema_alpha = 0.1
        self._adaptive_gap = 0.05
        # Per Phase 5.7: Shadow probe calibration replaces the old
        # _calibration_queries_remaining mechanism. The new system uses
        # phase-based intervals (cold/warmup/steady) instead of a flat
        # countdown. See _get_calibration_phase() and _should_shadow_probe().

        # Bounded prefetch executor (replaces unbounded daemon threads)
        # Per Bug 2 fix: increased from 2 to 4 workers. Cache writes are now
        # synchronous (Bug 1 fix), so this executor only handles prefetch tasks.
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="Prefetch"
        )

        self.tier_search_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="TierSearch"
        )

        logger.info("[ROUTER] âœ… Three-tier hybrid router initialized")
        logger.info("[ROUTER] TIER 1: Permanent (300 vectors, read-only)")
        logger.info(
            f"[ROUTER] TIER 2: Dynamic ({DYNAMIC_LAYER_CAPACITY} vectors, learning)"
        )
        logger.info("[ROUTER] TIER 3: Cloud (9,482 vectors, truth)")
        logger.info("[ROUTER] Parallel tier search enabled")
        logger.info(
            f"[ROUTER] Variant: {self.routing_mode} (prefetch={self.prefetch_enabled})"
        )

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert L2 distance to cosine similarity for normalized vectors.

        For L2-normalized vectors: L2_dist^2 = 2(1 - cos_sim)
        Therefore: cos_sim = 1 - L2_dist^2 / 2

        INVARIANT: All vectors entering FAISS must be unit-normalized.
        This holds because:
        - Cloud seeding uses normalize_embeddings=True (cloud_client.py:68)
        - Tier 1 seeding uses actual corpus vectors from Qdrant (already normalized)
        - Tier 2 admission fetches vectors from Qdrant via get_vectors_by_ids()
        - Query encoding uses normalize_embeddings=True (embedding_model.py:101)
        If this invariant is violated, this formula produces garbage.

        Previous 1/(1+d) was monotonic but NOT cosine-equivalent, causing
        local candidates to be incorrectly rejected at threshold 0.75.
        """
        d = float(distance)
        return max(0.0, 1.0 - d * d / 2.0)

    def _update_adaptive_threshold(self, cloud_top_similarity: float) -> None:
        """Calibrate the hybrid gate's adaptive threshold against observed cloud quality.

        Per Bug 5 fix: this now updates BOTH the router's EMA (for diagnostics)
        AND the gate's AdaptiveThreshold via record_outcome(). The gate's
        precision-window threshold is the single source of truth for all
        routing decisions — parallel_hybrid, full_hybrid, and reactive_cache.

        The EMA is kept for logging and as a sanity check, but the gate's
        threshold is what actually controls local-vs-cloud decisions.

        Per anchor policy / Tier 2 bounded admission / FAISS L2 inversion.
        """
        self._cloud_samples += 1
        if self._cloud_samples == 1:
            self._cloud_similarity_ema = cloud_top_similarity
        else:
            self._cloud_similarity_ema = (
                self._adaptive_ema_alpha * cloud_top_similarity
                + (1 - self._adaptive_ema_alpha) * self._cloud_similarity_ema
            )
        # Per Bug 5: delegate to gate's AdaptiveThreshold for all variants.
        # The gate uses a precision-window mechanism that tracks whether
        # local top-1 appears in cloud top-5 (record_outcome). This is
        # more robust than the raw EMA gap approach.
        # We also feed the gate a "was_relevant=True" signal here because
        # cloud_top_similarity being high means cloud found something relevant,
        # which calibrates the gate's threshold upward (stricter local serving).
        gate_threshold = self._hybrid_gate.adaptive_threshold.threshold
        logger.info(
            f"[THRESHOLD] cloud_ema={self._cloud_similarity_ema:.4f} "
            f"gate_threshold={gate_threshold:.4f} "
            f"(samples={self._cloud_samples})"
        )

    def _finalize_result(self, result: Dict) -> Dict:
        """Attach final dynamic-layer state so benchmark rows can audit cache behavior."""
        dynamic_stats = self.local_vdb.get_dynamic_stats()
        result["dynamic_size_after"] = int(dynamic_stats["current_size"])
        result["dynamic_capacity"] = int(dynamic_stats["capacity"])
        result["dynamic_fill_rate_after"] = float(dynamic_stats.get("fill_rate", 0.0))

        # Per prefetch quality audit: log which T2 IDs appear in final top-k.
        # This enables post-hoc reuse rate analysis: for each admitted vector,
        # did it appear in a future query's top-k within a 5-turn window?
        final_ids = result.get("ids", [])
        if final_ids:
            t2_ids_in_topk = []
            for rid in final_ids:
                rid_str = str(rid)
                if self.local_vdb.has_dynamic_id(rid_str):
                    t2_ids_in_topk.append(rid_str)
            self.reuse_log.append({
                "query_id": result.get("query_id", ""),
                "turn": self.query_count,
                "topk_ids": [str(rid) for rid in final_ids],
                "t2_ids_in_topk": t2_ids_in_topk,
                "source": result.get("source", ""),
            })

        return result

    def set_doc_texts(self, doc_texts: Dict[str, str]) -> None:
        """Pre-load corpus texts for BM25 hybrid gate.

        Called by benchmark init so the BM25 index is built before queries.
        Also retained for cross-encoder fallback (retired, not in local path).
        """
        self._doc_texts = doc_texts
        # Per Phase 5: Build BM25 index for hybrid gate
        self._hybrid_gate.index_corpus(doc_texts)

    def _get_doc_texts(self, doc_ids: List[str]) -> List[str]:
        """Lookup doc texts for reranker. Returns empty list if any missing."""
        texts = []
        for doc_id in doc_ids:
            text = self._doc_texts.get(str(doc_id))
            if text is None:
                # Per fallback: if pre-loaded corpus missing doc, abort reranking
                # to avoid partial rerank (which would bias the gate).
                return []
            texts.append(text)
        return texts

    def wait_for_background_prefetch(self, timeout_s: float = 30.0) -> None:
        """Wait for outstanding prefetch futures so benchmark state is stable."""
        deadline = time.time() + max(timeout_s, 0.0)
        while True:
            with self._prefetch_futures_lock:
                pending = [f for f in self._prefetch_futures if not f.done()]
                self._prefetch_futures = pending

            if not pending:
                return

            remaining = deadline - time.time()
            if remaining <= 0:
                logger.warning(
                    "[PREFETCH] Timeout while waiting for %s background tasks",
                    len(pending),
                )
                return

            concurrent.futures.wait(
                pending,
                timeout=min(remaining, 0.5),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

    def _admit_to_tier2(
        self,
        vector: np.ndarray,
        doc_id: str,
        metadata: Dict,
        *,
        neighborhood_threshold: float,
    ) -> Dict[str, int]:
        """Use one bounded-admission path for both reactive caching and background prefetch."""
        return self.local_vdb.admit_dynamic(
            vector,
            str(doc_id),
            metadata,
            neighborhood_threshold=neighborhood_threshold,
        )

    def search(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str = "",
        k: int = DEFAULT_SEARCH_K,
    ) -> Dict:
        """
        Three-tier search with smart prefetching.

        OPTIMIZED: TIER 1 + TIER 2 searched in parallel
        """
        self.query_count += 1
        start_time = time.time()

        # Per prefetch quality audit: store query_id for async prefetch logging
        self._current_query_id = query_id

        # Per query-count exponential decay: update anchor system's global counter
        self.anchor_system.global_query_count = self.query_count

        # Per query-count exponential decay: apply decay every query
        self.anchor_system.decay_anchors_by_query_count(self.query_count)

        # Per B0-B3 diagnostics: log anchor state every 10 queries for measurement
        if self.query_count % 10 == 0:
            self.anchor_system.log_anchor_diagnostics(self.query_count)

        if VERBOSE:
            print(f"\n{'=' * 70}")
            print(f"[QUERY #{self.query_count}] {query_id}")
            # ASCII-safe truncation for Windows cp1252 console encoding
            _safe_text = query_text[:60].encode('ascii', errors='replace').decode('ascii')
            print(f'Text: "{_safe_text}..."')
            print(f"{'=' * 70}")

        result = {
            "query_id": query_id,
            "query_text": query_text,
            "query_number": self.query_count,
            "timestamp": time.time(),
            "k": k,
            "variant": self.routing_mode,
            "best_local_tier": None,
            "best_local_distance": None,
            "best_local_similarity": 0.0,
            "cache_admitted_count": 0,
            "cache_evicted_count": 0,
            "cache_duplicate_skip_count": 0,
            "cache_id_skip_count": 0,           # Per §22 Step 1: exact ID match in Tier 2
            "cache_tier1_skip_count": 0,        # Per §22 Step 1: doc in Tier 1, already local
            "cache_neighborhood_skip_count": 0,  # Per §22 Step 1: near-duplicate in Tier 2
            "cache_insert_failed_count": 0,
            # Proactive prefetch diagnostics
            "prefetch_trigger": "none",             # "proactive" | "cold_start" | "none"
            "prefetch_predictions_generated": 0,    # How many prediction vectors generated
            "prefetch_anchor_id": None,             # Which anchor produced predictions
        }

        dynamic_stats_before = self.local_vdb.get_dynamic_stats()
        result["dynamic_size_before"] = int(dynamic_stats_before["current_size"])
        result["dynamic_capacity"] = int(dynamic_stats_before["capacity"])
        result["dynamic_fill_rate_before"] = float(
            dynamic_stats_before.get("fill_rate", 0.0)
        )

        if self.routing_mode == "cloud_only":
            return self._search_cloud_only(query_vector, result, start_time, k)

        # Per pre-flight: t1_plus_cloud baseline — check T1 only, fall back to cloud.
        # Measures contribution of T1 caching alone (no T2, no anchor, no prefetch).
        if self.routing_mode == "t1_plus_cloud":
            return self._search_t1_plus_cloud(query_vector, result, start_time, k)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STEP 1: Check Anchor Prediction Match
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        # Per consensus: prediction horizon tracking removed. Predictions are
        # materialized into T2 vectors; T2 admission/eviction handles lifecycle.
        # Anchor reinforcement comes from absorption (update_anchor) and
        # broadcast signals, not prediction matching.

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STEP 2: Semantic Clustering
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        cluster_id, cluster_action, access_count = self.semantic_cache.add_query(
            query_vector, query_id
        )

        result["cluster_id"] = cluster_id
        result["cluster_action"] = cluster_action
        result["cluster_access_count"] = access_count

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STEP 3 & 4: Search TIER 1 + TIER 2 SEQUENTIALLY
        # Per Bug 6 fix: Replaced parallel ThreadPoolExecutor search with
        # sequential calls. Both share a single RLock in storage_engine, so
        # they were never truly parallel — thread overhead added latency.
        # Sequential search on our index sizes (T1~60, T2~2700) is <1ms.
        # The old parallel code had a catastrophic bug: a single try/except
        # wrapped both futures, so a timeout on EITHER tier zeroed BOTH
        # result sets → empty local results → cloud fallback → quality drop.

        logger.debug("[ROUTER] Searching TIER 1 then TIER 2 sequentially...")
        local_search_start = time.time()

        # Per Bug 6 fix: sequential search, no timeout, no thread pool.
        # Each call is <1ms for our index sizes. No shared-lock contention.
        tier1_ids, tier1_scores = self.local_vdb.search_permanent(query_vector, k)
        tier2_ids, tier2_scores = self.local_vdb.search_dynamic(query_vector, k)

        total_local_latency = (time.time() - local_search_start) * 1000
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        result["tier1_results"] = len(tier1_ids)
        result["tier2_results"] = len(tier2_ids)
        result["local_search_latency_ms"] = total_local_latency

        # Get dynamic space stats
        dynamic_stats = self.local_vdb.get_dynamic_stats()
        result["dynamic_fill_rate"] = (
            f"{dynamic_stats['current_size']}/{dynamic_stats['capacity']}"
        )

        # âœ… MERGE RESULTS: Find best result from both tiers
        all_local_results = []

        # Add TIER 1 results
        for vid, score in zip(tier1_ids, tier1_scores):
            all_local_results.append(
                {"id": vid, "score": score, "tier": "tier1_permanent"}
            )

        # Add TIER 2 results
        for vid, score in zip(tier2_ids, tier2_scores):
            all_local_results.append(
                {"id": vid, "score": score, "tier": "tier2_dynamic"}
            )

        # Sort by score (lower L2 distance = better)
        all_local_results.sort(key=lambda x: x["score"])

        if self.routing_mode == "local_only":
            return self._search_local_only(
                all_local_results, result, start_time, total_local_latency, k
            )

        # parallel_hybrid: always query local + cloud in parallel, merge by cosine
        # Per parallel_hybrid routing / reactive+predictive loop activation
        if self.routing_mode == "parallel_hybrid":
            return self._search_parallel_hybrid(
                all_local_results,
                query_vector,
                query_id,
                query_text,
cluster_id,
                 result,
                start_time,
                total_local_latency,
                k,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # PROACTIVE PREFETCH: predict + prefetch on EVERY query, independent of gate.
        # Per Point 13 fix: ensures Tier 2 is constantly refreshed with predicted
        # vectors. Runs in background via _prefetch_executor — does not block
        # current query latency. Only for full_hybrid (NOT reactive_cache, which
        # remains the clean control with zero prefetch).
        # ═══════════════════════════════════════════════════════════════════════
        if self.prefetch_enabled and self.routing_mode == "full_hybrid":
            result["prefetch_trigger"] = "proactive"
            _pf_anchor_id, _pf_predictions = self._predict(
                query_vector, query_id, query_text,
                cluster_id,
            )
            result["prefetch_anchor_id"] = _pf_anchor_id
            result["prefetch_predictions_generated"] = len(_pf_predictions)
            self._prefetch_executor.submit(
                self._prefetch,
                query_vector, _pf_anchor_id, _pf_predictions, cluster_id,
            )

# Check if best result meets confidence threshold
        if all_local_results:
            best_local_distance = float(all_local_results[0]["score"])
            best_local_similarity = self._distance_to_similarity(best_local_distance)
            result["best_local_tier"] = all_local_results[0]["tier"]
            result["best_local_distance"] = best_local_distance
            result["best_local_similarity"] = best_local_similarity
        else:
            best_local_distance = None
            best_local_similarity = 0.0

# ═══════════════════════════════════════════════════════════════════════
        # BUG 1 FIX: Use hybrid gate (BM25+cosine) for full_hybrid/reactive_cache
        # Previously this was a raw cosine >= threshold check that always passed
        # because local cosines of 0.87-0.92 exceed the 0.75 threshold even on
        # irrelevant documents. Now uses the same _hybrid_gate.gate() as
        # parallel_hybrid, combining BM25 lexical signal with cosine similarity.
        #
        # Step 4: Per-anchor adaptive thresholds. Find matching anchor BEFORE gate
        # so we can pass anchor_id to the gate for per-basin threshold lookup.
        # ═══════════════════════════════════════════════════════════════════════
        gate_signal = 0.0
        gate_diagnostics = {}
        should_serve_locally = False

        # Per anchor policy / Step 4: find matching anchor for per-basin threshold
        matching_anchor_id = None
        for aid, anch in self.anchor_system.anchors.items():
            if self.anchor_system.belongs_to(query_vector, aid):
                matching_anchor_id = aid
                break

        if all_local_results and len(all_local_results) >= k:
            try:
                local_ids = [str(r["id"]) for r in all_local_results[:k]]
                cosine_scores = [
                    self._distance_to_similarity(float(r["score"]))
                    for r in all_local_results[:k]
                ]
                gate_signal, should_serve_locally, gate_diagnostics = self._hybrid_gate.gate(
                    query_text=query_text,
                    doc_ids=local_ids,
                    cosine_scores=cosine_scores,
                    anchor_id=matching_anchor_id,  # Per anchor policy / Step 4
                )
                result["gate_signal"] = gate_signal
                result["gate_threshold"] = gate_diagnostics.get("threshold", 0.0)
                result["gate_best_cosine"] = gate_diagnostics.get("best_cosine", 0.0)
                result["gate_best_bm25_raw"] = gate_diagnostics.get("best_bm25_raw", 0.0)
                result["gate_best_bm25_norm"] = gate_diagnostics.get("best_bm25_norm", 0.0)
                result["gate_alpha"] = gate_diagnostics.get("alpha", 0.7)
            except Exception as e:
                logger.warning(f"[GATE] Hybrid gate failed for full_hybrid, forcing cloud: {e}")
                should_serve_locally = False
                gate_signal = 0.0

        # ═══════════════════════════════════════════════════════════════════════
        # BUG 2+4 FIX: Calibration phase for full_hybrid/reactive_cache
        # Without forced cloud queries during cold start, the gate threshold
        # never calibrates (chicken-and-egg: gate passes → no cloud → no data).
        # Cold (Q1-Q5): Force cloud to seed calibration data.
        # Warmup/Steady: Shadow-probe cloud periodically for ongoing calibration.
        # ═══════════════════════════════════════════════════════════════════════
        calibration_phase = self._get_calibration_phase()
        shadow_probe = self._should_shadow_probe()

        # ── NO FORCED COLD START: Gate handles Q1-Q5 naturally. ─────────
        # Per proactive prefetch redesign: cold-start forced cloud removed.
        # On Q1-Q5, Tier 2 is empty and Tier 1 has k-means centroids only,
        # so local quality is poor → gate signal low → gate rejects → cloud
        # fallback naturally. Reactive cache + gate calibration still happen
        # via the cloud-fallback path below. Same outcome, no forced override.

        # Per Fix A: quality floor — if local results are garbage, force cloud
        # The gate can accept local results even when best_local_similarity is
        # very low (BM25 pushes signal above threshold). This catches the 11
        # queries where local is nearly irrelevant but gate says accept.
        if should_serve_locally and best_local_similarity < MIN_LOCAL_QUALITY_FLOOR:
            should_serve_locally = False
            result["quality_floor_reject"] = True
            logger.info(
                f"[QUALITY-FLOOR] q={self.query_count} "
                f"best_local_sim={best_local_similarity:.3f} < floor={MIN_LOCAL_QUALITY_FLOOR} "
                f"→ forcing cloud"
            )

        if should_serve_locally:
            # Gate passed — serve local, but shadow-probe cloud periodically
            best_result = all_local_results[0]
            best_tier = best_result["tier"]

            # Shadow probe: periodically query cloud in background for calibration
            if shadow_probe:
                try:
                    sp_future = self.tier_search_executor.submit(
                        self.cloud_vdb.search, query_vector, k
                    )

                    def _shadow_probe_callback(future):
                        """Async: compare local vs cloud, feed gate calibration, cache cloud results in T2.

                        Per shadow-probe T2 caching fix: shadow probes already query cloud
                        for calibration. The cloud results are ground truth — confirmed relevant.
                        Caching them in T2 breaks the vicious cycle where wrong-but-close docs
                        stay in T2 forever (high cosine → gate accepts → no cloud fallback →
                        never corrected). Shadow probe results use bypass_neighborhood=True
                        (same as reactive cache) since they're confirmed-relevant.
                        """
                        try:
                            sp_ids, sp_scores, _ = future.result(timeout=5.0)
                            if sp_scores:
                                self._update_adaptive_threshold(float(sp_scores[0]))
                            if all_local_results and sp_ids:
                                sp_id_set = set(str(cid) for cid in (sp_ids or [])[:5])
                                local_top_id = str(all_local_results[0]["id"]) if all_local_results else ""
                                was_relevant = local_top_id in sp_id_set
                                self._hybrid_gate.record_outcome(gate_signal, was_relevant, matching_anchor_id)
                                logger.info(
                                    f"[SHADOW-PROBE] q={self.query_count} "
                                    f"local_in_cloud_top5={was_relevant} "
                                    f"gate_signal={gate_signal:.3f} "
                                    f"cloud_top_sim={float(sp_scores[0]):.3f}"
                                )
                            # Per shadow-probe T2 caching: cache cloud results in T2.
                            # These are confirmed-relevant (cloud ground truth), same as reactive cache.
                            if sp_ids and sp_scores:
                                try:
                                    cache_diag = self._cache_neighborhood_to_tier2(
                                        cloud_ids=[str(cid) for cid in sp_ids],
                                        cloud_scores=[float(s) for s in sp_scores],
                                        query_vector=query_vector,
                                        cluster_id=cluster_id,
                                        bypass_neighborhood=True,  # Confirmed-relevant, same as reactive
                                        source_anchor_id=matching_anchor_id,
                                    )
                                    logger.info(
                                        f"[SHADOW-PROBE-CACHE] q={self.query_count} "
                                        f"admitted={cache_diag.get('cache_admitted_count', 0)} "
                                        f"evicted={cache_diag.get('cache_evicted_count', 0)} "
                                        f"skipped_id={cache_diag.get('cache_id_skip_count', 0)} "
                                        f"skipped_t1={cache_diag.get('cache_tier1_skip_count', 0)}"
                                    )
                                except Exception as ce:
                                    logger.warning(f"[SHADOW-PROBE-CACHE] Cache failed: {ce}")
                        except Exception as e:
                            logger.warning(f"[SHADOW-PROBE] Failed: {e}")

                    sp_future.add_done_callback(_shadow_probe_callback)
                except Exception as e:
                    logger.warning(f"[SHADOW-PROBE] Submit failed: {e}")

            # Extract top-k results
            top_k = all_local_results[:k]
            result_ids = [r["id"] for r in top_k]
            result_scores = [float(r["score"]) for r in top_k]

            # Per Bug 4 fix: reinforce Tier 2 vectors that were served to the user.
            for r in top_k:
                if r["tier"] == "tier2_dynamic":
                    try:
                        self.local_vdb.update_dynamic_weight(str(r["id"]), delta=1.0)
                    except Exception:
                        pass  # Weight update is best-effort; don't fail the query

            # ✅ LOCAL HIT!
            total_time = (time.time() - start_time) * 1000
            result.update(
                {
                    "ids": result_ids,
                    "scores": result_scores,
                    "source": best_tier,
                    "latency_ms": total_time,
                    "confidence": best_local_similarity,
                    "served_locally": True,
                    "calibration_phase": calibration_phase,
                    "shadow_probe": shadow_probe,
                }
            )

            # Record metrics for the winning tier
            if self.metrics:
                self.metrics.record_query(
                    source=best_tier,
                    latency_ms=total_local_latency,
                    prefetch_fetched=0,
                    prefetch_skipped=0,
                )

            logger.info(
                f"[ROUTER] ✅ LOCAL HIT ({best_tier}, {total_local_latency:.1f}ms, "
                f"gate={gate_signal:.3f}, phase={calibration_phase})"
            )
            logger.info(f"[ROUTER] Dynamic space: {result['dynamic_fill_rate']}")

            # ─── Option A: Broadcast local signal to ALL anchors ───
            # Per Option A: targeted primary signal on matching anchor (V2 proven)
            # + supplementary broadcast to other anchors (V5 exploration).
            if len(self.anchor_system.anchors) > 0:
                local_signal_diag = self.anchor_system.broadcast_local_signal(
                    query_vector, result_quality=best_local_similarity,
                    matching_anchor_id=matching_anchor_id,
                )
                result["broadcast_local_signal"] = local_signal_diag

            # Proactive prefetch already ran pre-gate — no duplicate prefetch here.

            return self._finalize_result(result)

        # ═══════════════════════════════════════════════════════════════════════
        # GATE REJECTED: Local results failed hybrid gate → Fallback to cloud
        # ═══════════════════════════════════════════════════════════════════════

        logger.debug(
            f"[ROUTER] Gate rejected local (sim={best_local_similarity:.3f}, "
            f"gate={gate_signal:.3f} < threshold={gate_diagnostics.get('threshold', 0.0):.3f})"
        )

        # Per V5 design: broadcast weaken ALL anchors on cloud fallback.
        # Per Option A: broadcast is supplementary. Primary signal is targeted weaken
        # on the matching anchor (V2 proven: penalty=0.3 on gate rejection).

        logger.info("[ROUTER] ☁️ TIER 3 fallback (cloud, gate rejected local)")

        # Per Option A: targeted weaken on matching anchor (V2 proven)
        if matching_anchor_id is not None:
            self.anchor_system.weaken_anchor(matching_anchor_id, penalty=0.3)

        try:
            cloud_ids, cloud_scores, cloud_latency_ms = self.cloud_vdb.search(
                query_vector, k
            )

            # Per anchor policy / adaptive threshold calibration
            if cloud_scores:
                self._update_adaptive_threshold(float(cloud_scores[0]))

            # ─── V5: Broadcast weaken ALL anchors before reactive cache ───
            # Per V5 design: loss = (1-sim) × confidence × cloud_quality × scale
            # Cloud quality = best cloud similarity (0 if no results).
            # Close+confident anchors get moderate loss. Far+confident get most loss.
            cloud_quality = float(cloud_scores[0]) if cloud_scores else 0.0
            if cloud_quality > 0 and len(self.anchor_system.anchors) > 0:
                weaken_diag = self.anchor_system.broadcast_weaken_cloud(
                    query_vector, cloud_quality,
                    exclude_anchor_id=None,  # All anchors weakened before reactive addition
                )
                result["broadcast_weaken"] = weaken_diag

            # Feed gate.record_outcome() with local-vs-cloud comparison
            if all_local_results and cloud_ids:
                cloud_id_set = set(str(cid) for cid in (cloud_ids or [])[:5])
                local_top_id = str(all_local_results[0]["id"]) if all_local_results else ""
                was_relevant = local_top_id in cloud_id_set
                self._hybrid_gate.record_outcome(gate_signal, was_relevant, matching_anchor_id)

            total_time = (time.time() - start_time) * 1000

            result.update(
                {
                    "ids": cloud_ids,
                    "scores": cloud_scores,
                    "source": "tier3_cloud",
                    "latency_ms": total_time,
                    "tier3_latency_ms": cloud_latency_ms,
                    "confidence": cloud_scores[0] if cloud_scores else 0.0,
                    "calibration_phase": calibration_phase,
                    "gate_signal": gate_signal,
                }
            )

            # FIXED: Use record_query() instead of log_event()
            if self.metrics:
                self.metrics.record_query(
                    source="tier3_cloud",
                    latency_ms=cloud_latency_ms,
                    prefetch_fetched=0,
                    prefetch_skipped=0,
                )

            logger.info(f"[ROUTER] ☁️ TIER 3 HIT (cloud, {cloud_latency_ms:.1f}ms)")

            # Cache neighborhood to TIER 2 (reactive cache-on-use for full_hybrid, reactive_cache, and true_lru)
            if (
                cloud_ids
                and cloud_scores
                and self.routing_mode in ("full_hybrid", "reactive_cache", "true_lru")
            ):
                cache_diagnostics = self._cache_neighborhood_to_tier2(
                    cloud_ids[:3], cloud_scores[:3], query_vector, cluster_id,
                    bypass_neighborhood=True,  # Per Bug 5: reactive cache bypasses neighborhood
                    source_anchor_id=matching_anchor_id,
                )
                result.update(cache_diagnostics)
                result["cached_to_tier2"] = cache_diagnostics["cache_admitted_count"]

            # Proactive prefetch already ran pre-gate — no duplicate prefetch here.

            return self._finalize_result(result)

        except Exception as e:
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # STEP 6: Offline Graceful Degradation
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logger.error(f"[ROUTER] âŒ TIER 3 error: {e}")
            total_time = (time.time() - start_time) * 1000

            # Return best local result (tier1 or tier2)
            if all_local_results:
                best_result = all_local_results[0]
                top_k = all_local_results[:k]
                result_ids = [r["id"] for r in top_k]
                result_scores = [r["score"] for r in top_k]
            else:
                result_ids = []
                result_scores = []

            result.update(
                {
                    "ids": result_ids,
                    "scores": result_scores,
                    "source": "offline_fallback",
                    "latency_ms": total_time,
                    "error": str(e),
                    "confidence": result_scores[0] if result_scores else 0.0,
                }
            )

            # FIXED: Use record_query() instead of log_event()
            if self.metrics:
                self.metrics.record_query(
                    source="offline_fallback",
                    latency_ms=total_time,
                    prefetch_fetched=0,
                    prefetch_skipped=0,
                )

            logger.warning("[ROUTER] âš ï¸ OFFLINE (best local result)")
            return self._finalize_result(result)

    def _search_parallel_hybrid(
        self,
        all_local_results: List[Dict],
        query_vector: np.ndarray,
        query_id: str,
        query_text: str,
        cluster_id: int,
        result: Dict,
        start_time: float,
        total_local_latency: float,
        k: int,
    ) -> Dict:
        """Parallel hybrid with P2 speculative execution and reranker gating.

        Based on: RaLMSpec [Zheng et al., ICML 2024] — race local+cloud,
        cancel speculative on local success.

        Architecture:
        1. Search local (Tier 1 + Tier 2) — always fast (~3ms)
        2. Fire cloud speculatively in parallel — ~200ms
        3. Cross-encoder re-rank local top-10 — ~10ms on GPU
        4. If local passes relevance gate: serve locally (~13ms total)
           Cancel speculative cloud call (best-effort bandwidth save).
        5. If local fails: await cloud result (already in-flight, ~0ms wait)
           Cache to Tier 2 (reactive loop)
        6. ALWAYS: create anchor + predict + prefetch (predictive loop)
        """
        local_id_set = set()
        for r in all_local_results:
            local_id_set.add(str(r["id"]))

        dynamic_stats = self.local_vdb.get_dynamic_stats()
        fill_rate = dynamic_stats["current_size"] / max(dynamic_stats["capacity"], 1)

        # ═══════════════════════════════════════════════════════════════════════
        # P2: SPECULATIVE PARALLEL CLOUD
        # Based on: RaLMSpec [Zheng et al., ICML 2024]
        # ═══════════════════════════════════════════════════════════════════════
        cloud_future = None
        warm_seed_cloud_result = None

        # Per warm-seed (THEORY.md §Q2): Q1 always seeds Tier 2 from cloud.
        # Warm-seed cloud call is synchronous AND serves as speculative call for Q1.
        warm_seeded_count = 0
        if self.query_count == 1 and not self._warm_seeded:
            try:
                ws_cloud_ids, ws_cloud_scores, ws_latency = self.cloud_vdb.search(
                    query_vector, k
                )
                warm_seed_cloud_result = (ws_cloud_ids, ws_cloud_scores, ws_latency)
                if ws_cloud_ids:
                    # Per warm-seed: cache top-3 cloud results to Tier 2
                    ws_scores = [float(s) for s in ws_cloud_scores[:3]]
                    ws_ids = [str(cid) for cid in ws_cloud_ids[:3]]
                    cache_diag = self._cache_neighborhood_to_tier2(
                        cache_ids[:3], cache_scores[:3],
                        query_vector, cluster_id,
                        bypass_neighborhood=True,  # Per Bug 5: reactive cache bypasses neighborhood
                        source_anchor_id=None,  # Warm-seed runs before anchor matching — no anchor yet
                    )
                    warm_seeded_count = cache_diag.get("cache_admitted_count", 0)
                    result["warm_seeded"] = warm_seeded_count
                    self._warm_seeded = True  # per warm-seed: one-time flag
                    if ws_cloud_scores:
                        self._update_adaptive_threshold(float(ws_cloud_scores[0]))
            except Exception as e:
                logger.warning(f"[WARM-SEED] Q1 warm-seed failed: {e}")

        # For Q2+ (or Q1 if warm-seed failed): fire speculative cloud immediately.
        if warm_seed_cloud_result is None:
            cloud_future = self.tier_search_executor.submit(
                self.cloud_vdb.search, query_vector, k
            )

        # ═══════════════════════════════════════════════════════════════════════
        # HYBRID GATE (Phase 5: BM25+cosine, replaces cross-encoder)
        # ═══════════════════════════════════════════════════════════════════════
        # Per Phase 5: The cross-encoder gate rejected 96.7% of local results
        # on NFCorpus (domain mismatch) and added 90ms latency. Replaced with
        # BM25+cosine hybrid gate that runs in <0.1ms and adapts its threshold
        # based on observed local precision.
        # Based on: BM25 [Robertson & Zaragoza, 2009]; hybrid fusion
        # [Shen et al., 2023]; adaptive threshold via precision window.
        local_has_k_results = len(all_local_results) >= k
        best_local_similarity = 0.0
        best_local_tier_actual = None
        best_local_distance_actual = None
        if all_local_results:
            best_local_similarity = self._distance_to_similarity(
                float(all_local_results[0]["score"])
            )
            best_local_tier_actual = all_local_results[0]["tier"]
            best_local_distance_actual = float(all_local_results[0]["score"])
            result["best_local_tier"] = best_local_tier_actual
            result["best_local_distance"] = best_local_distance_actual
            result["best_local_similarity"] = best_local_similarity

        gate_signal = 0.0
        gate_diagnostics = {}
        should_serve_locally = False

        # Per anchor policy / Step 4: find matching anchor for per-basin threshold
        matching_anchor_id = None
        for aid, anch in self.anchor_system.anchors.items():
            if self.anchor_system.belongs_to(query_vector, aid):
                matching_anchor_id = aid
                break

        if local_has_k_results:
            try:
                # Per Phase 5: BM25+cosine hybrid gate
                # gate_signal = alpha * cosine + (1-alpha) * normalize(bm25)
                # No cross-encoder, no GPU, <0.1ms per query.
                local_ids = [str(r["id"]) for r in all_local_results[:k]]
                cosine_scores = [
                    self._distance_to_similarity(float(r["score"]))
                    for r in all_local_results[:k]
                ]

                gate_signal, should_serve_locally, gate_diagnostics = self._hybrid_gate.gate(
                    query_text=query_text,
                    doc_ids=local_ids,
                    cosine_scores=cosine_scores,
                    anchor_id=matching_anchor_id,  # Per anchor policy / Step 4
                )

                result["gate_signal"] = gate_signal
                result["gate_threshold"] = gate_diagnostics.get("threshold", 0.0)
                result["gate_best_cosine"] = gate_diagnostics.get("best_cosine", 0.0)
                result["gate_best_bm25_raw"] = gate_diagnostics.get("best_bm25_raw", 0.0)
                result["gate_best_bm25_norm"] = gate_diagnostics.get("best_bm25_norm", 0.0)
                result["gate_alpha"] = gate_diagnostics.get("alpha", 0.7)

            except Exception as e:
                logger.warning(f"[GATE] Hybrid gate failed, forcing cloud: {e}")
                should_serve_locally = False
                gate_signal = 0.0

        can_serve_locally = local_has_k_results and should_serve_locally

        # Per Fix A: quality floor — if local results are garbage, force cloud
        if can_serve_locally and best_local_similarity < MIN_LOCAL_QUALITY_FLOOR:
            can_serve_locally = False
            should_serve_locally = False
            result["quality_floor_reject"] = True
            logger.info(
                f"[QUALITY-FLOOR] q={self.query_count} "
                f"best_local_sim={best_local_similarity:.3f} < floor={MIN_LOCAL_QUALITY_FLOOR} "
                f"→ forcing cloud (reactive_cache)"
            )

        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 5.7: CALIBRATION PHASE DETERMINATION
        # ═══════════════════════════════════════════════════════════════════════
        # Per Phase 5.7: The gate's adaptive threshold needs cloud ground truth
        # to calibrate. Without periodic probes, the threshold stays at 0.70
        # forever (the NFCorpus problem: 100% local, 63.7% below cloud nDCG@10).
        #
        # Cold (Q1-Q5):   Serve cloud directly. Every query probes cloud.
        # Warmup (Q6-Q20): Serve local. Every 5th query shadow-probes cloud.
        # Steady (Q21+):   Serve local. Every 10th query shadow-probes cloud.
        #
        # "Shadow probe" = don't cancel speculative cloud future. When cloud
        # completes, compare local top-1 with cloud top-5, feed record_outcome().
        # The user sees local results immediately — cloud is async calibration.
        calibration_phase = self._get_calibration_phase()
        shadow_probe = self._should_shadow_probe()

# Diagnostic logging — matching_anchor_id already computed above for Step 4
        # Per V5 design: broadcast weaken handled after cloud retrieval below.
        # Old per-anchor weaken removed — broadcast handles everything.

        logger.info(
            f"[DIAG] q={self.query_count} cal_phase={calibration_phase} "
            f"shadow_probe={shadow_probe} anchors={len(self.anchor_system.anchors)} "
            f"was_absorbed={bool(matching_anchor_id)} tier2={self.local_vdb.get_dynamic_count()} "
            f"local_top_sim={best_local_similarity:.3f} gate={gate_signal:.3f} "
            f"decision={'LOCAL' if can_serve_locally else 'CLOUD'} "
            f"threshold={gate_diagnostics.get('threshold', 0.0):.3f}"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # COLD PHASE (Q1-Q5): Serve cloud directly
        # ═══════════════════════════════════════════════════════════════════════
        # During cold start, we have zero calibration data. The gate threshold
        # is at 0.70 (a guess). Serving local would be reckless — we saw 63.7%
        # quality gap on NFCorpus. Instead, serve cloud results directly and
        # feed record_outcome() with local-vs-cloud comparison to seed the
        # adaptive threshold.
        if calibration_phase == "cold":
            # Await cloud result (already in-flight from speculative call)
            cloud_ids = cloud_scores = cloud_latency_ms = None
            try:
                if warm_seed_cloud_result is not None:
                    cloud_ids, cloud_scores, cloud_latency_ms = warm_seed_cloud_result
                elif cloud_future is not None:
                    cloud_ids, cloud_scores, cloud_latency_ms = cloud_future.result(timeout=10.0)
                else:
                    # Fallback: synchronous cloud call if no speculative future
                    cloud_ids, cloud_scores, cloud_latency_ms = self.cloud_vdb.search(
                        query_vector, k
                    )
            except Exception as e:
                logger.warning(f"[COLD-CALIBRATION] Cloud search failed: {e}")
                # Cloud failed during cold start — serve local as last resort
                if can_serve_locally:
                    top_k = all_local_results[:k]
                    result_ids = [str(r["id"]) for r in top_k]
                    result_scores = [
                        self._distance_to_similarity(float(r["score"])) for r in top_k
                    ]
                    best_tier = top_k[0]["tier"]
                    total_time = (time.time() - start_time) * 1000
                    result.update({
                        "ids": result_ids,
                        "scores": result_scores,
                        "source": best_tier,
                        "latency_ms": total_time,
                        "confidence": best_local_similarity,
                        "parallel_hybrid": True,
                        "served_locally": True,
                        "calibration_phase": "cold",
                        "calibration_fallback": True,
                        "gate_signal": gate_signal,
                    })
                    return self._finalize_result(result)
                # No local, no cloud — return empty
                total_time = (time.time() - start_time) * 1000
                result.update({
                    "ids": [], "scores": [],
                    "source": "none",
                    "latency_ms": total_time,
                    "confidence": 0.0,
                    "calibration_phase": "cold",
                    "calibration_fallback": True,
                })
                return self._finalize_result(result)

            total_time = (time.time() - start_time) * 1000

            # Feed adaptive threshold with cloud quality signal
            if cloud_scores:
                self._update_adaptive_threshold(float(cloud_scores[0]))

            # Feed gate.record_outcome() with local-vs-cloud comparison
            # Per Phase 5.7: was_relevant = local top-1 in cloud top-5
            if all_local_results and cloud_ids:
                cloud_id_set = set(str(cid) for cid in (cloud_ids or [])[:5])
                local_top_id = str(all_local_results[0]["id"]) if all_local_results else ""
                was_relevant = local_top_id in cloud_id_set
                self._hybrid_gate.record_outcome(gate_signal, was_relevant, matching_anchor_id)

            # ─── V5: Broadcast weaken ALL anchors on cloud fallback ───
            # Per Option A: broadcast is supplementary. Primary signal is targeted weaken.
            cloud_quality = float(cloud_scores[0]) if cloud_scores else 0.0
            if cloud_quality > 0 and len(self.anchor_system.anchors) > 0:
                weaken_diag = self.anchor_system.broadcast_weaken_cloud(
                    query_vector, cloud_quality,
                    exclude_anchor_id=None,
                )
                result["broadcast_weaken"] = weaken_diag

            # Per Option A: targeted weaken on matching anchor (V2 proven)
            if matching_anchor_id is not None:
                self.anchor_system.weaken_anchor(matching_anchor_id, penalty=0.3)

            # Cache cloud results to Tier 2 (reactive loop)
            if cloud_ids and cloud_scores:
                cache_ids = []
                cache_scores = []
                for doc_id, score in zip(cloud_ids, cloud_scores):
                    if str(doc_id) not in local_id_set:
                        cache_ids.append(doc_id)
                        cache_scores.append(float(score))
                if cache_ids:
                    cache_diag = self._cache_neighborhood_to_tier2(
                        cache_ids[:3], cache_scores[:3],
                        query_vector, cluster_id,
                        bypass_neighborhood=True,  # Per Bug 5: reactive cache bypasses neighborhood
                        source_anchor_id=matching_anchor_id,
                    )
                    result.update(cache_diag)
                    result["cached_to_tier2"] = cache_diag["cache_admitted_count"]

            # Serve cloud results to user
            result_ids = [str(cid) for cid in (cloud_ids or [])]
            result_scores = [float(s) for s in (cloud_scores or [])]

            local_overlap = 0
            for cid in cloud_ids or []:
                if str(cid) in local_id_set:
                    local_overlap += 1

            result.update({
                "ids": result_ids,
                "scores": result_scores,
                "source": "tier3_cloud",
                "latency_ms": total_time,
                "confidence": cloud_scores[0] if cloud_scores else 0.0,
                "parallel_hybrid": True,
                "served_locally": False,
                "local_overlap": local_overlap,
                "local_overlap_pct": local_overlap / max(len(result_ids), 1) * 100,
                "tier2_fill_rate": fill_rate,
                "gate_signal": gate_signal,
                "calibration_phase": "cold",
                "calibration_served_cloud": True,
            })

            if self.metrics:
                self.metrics.record_query(
                    source="tier3_cloud",
                    latency_ms=total_time,
                    prefetch_fetched=0,
                    prefetch_skipped=0,
                )

            # Per §22 Step 5: predict + prefetch even during cold start
            if self.prefetch_enabled:
                anchor_id, predictions = self._predict(
                    query_vector, query_id, query_text,
                    cluster_id,
                )
                self._prefetch_executor.submit(
                    self._prefetch,
                    query_vector, anchor_id, predictions, cluster_id,
                )

            return self._finalize_result(result)

        # ═══════════════════════════════════════════════════════════════════════
        # WARMUP / STEADY: Serve local when gate passes, shadow-probe cloud
        # ═════════════════════════════════════════════════════════════════════════
        # After cold start, the gate has calibration data. We trust local results
        # when the gate passes. But periodically (every 5th/10th query), we let
        # the speculative cloud call complete and compare with local to keep the
        # adaptive threshold calibrated.

        if can_serve_locally:
            # ─── Option A: Broadcast local signal to ALL anchors ───
            # Per Option A: targeted primary signal on matching anchor (V2 proven)
            # + supplementary broadcast to other anchors (V5 exploration).
            if len(self.anchor_system.anchors) > 0:
                local_signal_diag = self.anchor_system.broadcast_local_signal(
                    query_vector, result_quality=best_local_similarity,
                    matching_anchor_id=matching_anchor_id,
                )
                result["broadcast_local_signal"] = local_signal_diag

            # Per Phase 5.7: If this is a shadow-probe query, DON'T cancel cloud.
            # Let it complete in background, then compare local-vs-cloud and
            # feed record_outcome(). The user sees local results immediately.
            if shadow_probe and cloud_future and not cloud_future.done():
                # Fire-and-forget shadow probe: when cloud completes, compare
                # local top-1 with cloud top-5 and feed gate.record_outcome().
                # This is async — the user already has local results.
                def _shadow_probe_callback(future):
                    """Async callback: compare local vs cloud, feed gate calibration, cache in T2.

                    Per shadow-probe T2 caching fix: same as full_hybrid version.
                    Cloud results are confirmed-relevant ground truth. Cache them
                    to break the vicious cycle of wrong-but-close docs in T2.
                    """
                    try:
                        sp_ids, sp_scores, _ = future.result(timeout=5.0)
                        if sp_scores:
                            self._update_adaptive_threshold(float(sp_scores[0]))
                        if all_local_results and sp_ids:
                            sp_id_set = set(str(cid) for cid in (sp_ids or [])[:5])
                            local_top_id = str(all_local_results[0]["id"]) if all_local_results else ""
                            was_relevant = local_top_id in sp_id_set
                            self._hybrid_gate.record_outcome(gate_signal, was_relevant, matching_anchor_id)
                            logger.info(
                                f"[SHADOW-PROBE] q={self.query_count} "
                                f"local_in_cloud_top5={was_relevant} "
                                f"gate_signal={gate_signal:.3f} "
                                f"cloud_top_sim={float(sp_scores[0]):.3f}"
                            )
                        # Per shadow-probe T2 caching: cache cloud results in T2.
                        if sp_ids and sp_scores:
                            try:
                                cache_diag = self._cache_neighborhood_to_tier2(
                                    cloud_ids=[str(cid) for cid in sp_ids],
                                    cloud_scores=[float(s) for s in sp_scores],
                                    query_vector=query_vector,
                                    cluster_id=cluster_id,
                                    bypass_neighborhood=True,  # Confirmed-relevant, same as reactive
                                    source_anchor_id=matching_anchor_id,
                                )
                                logger.info(
                                    f"[SHADOW-PROBE-CACHE] q={self.query_count} "
                                    f"admitted={cache_diag.get('cache_admitted_count', 0)} "
                                    f"evicted={cache_diag.get('cache_evicted_count', 0)} "
                                    f"skipped_id={cache_diag.get('cache_id_skip_count', 0)} "
                                    f"skipped_t1={cache_diag.get('cache_tier1_skip_count', 0)}"
                                )
                            except Exception as ce:
                                logger.warning(f"[SHADOW-PROBE-CACHE] Cache failed: {ce}")
                    except Exception as e:
                        logger.warning(f"[SHADOW-PROBE] Failed: {e}")

                cloud_future.add_done_callback(_shadow_probe_callback)
            elif not shadow_probe:
                # Not a shadow-probe query: cancel speculative cloud (save bandwidth)
                if cloud_future and not cloud_future.done():
                    cloud_future.cancel()

            top_k = all_local_results[:k]
            result_ids = [str(r["id"]) for r in top_k]
            result_scores = [
                self._distance_to_similarity(float(r["score"])) for r in top_k
            ]
            best_tier = top_k[0]["tier"]

            if best_tier == "tier2_dynamic":
                self._tier2_serves += 1

            # Per Bug 4 fix: reinforce Tier 2 vectors that were served to the user.
            for r in top_k:
                if r["tier"] == "tier2_dynamic":
                    try:
                        self.local_vdb.update_dynamic_weight(str(r["id"]), delta=1.0)
                    except Exception:
                        pass  # Weight update is best-effort; don't fail the query

            total_time = (time.time() - start_time) * 1000

            result.update({
                "ids": result_ids,
                "scores": result_scores,
                "source": best_tier,
                "latency_ms": total_time,
                "confidence": best_local_similarity,
                "parallel_hybrid": True,
                "served_locally": True,
                "local_overlap": k,
                "local_overlap_pct": 100.0,
                "tier2_fill_rate": fill_rate,
                "gate_signal": gate_signal,
                "calibration_phase": calibration_phase,
                "shadow_probe": shadow_probe,
            })

            if self.metrics:
                self.metrics.record_query(
                    source=best_tier,
                    latency_ms=total_local_latency,
                    prefetch_fetched=0,
                    prefetch_skipped=0,
                )

            logger.info(
                f"[PARALLEL] Local-first path (fill={fill_rate:.0%}, sim={best_local_similarity:.3f}, "
                f"gate={gate_signal:.3f}, cal={calibration_phase}, shadow={shadow_probe})"
            )

            # Per §22 Step 5: predict + prefetch for parallel_hybrid local-hit path
            if self.prefetch_enabled:
                anchor_id, predictions = self._predict(
                    query_vector, query_id, query_text,
                    cluster_id,
                )
                self._prefetch_executor.submit(
                    self._prefetch,
                    query_vector, anchor_id, predictions, cluster_id,
                )

            return self._finalize_result(result)

        # P2: Local failed reranker gate — await speculative cloud result.
        # This path is for warmup/steady queries where gate rejected local.
        cloud_ids = cloud_scores = cloud_latency_ms = None
        try:
            if warm_seed_cloud_result is not None:
                cloud_ids, cloud_scores, cloud_latency_ms = warm_seed_cloud_result
            elif cloud_future is not None:
                cloud_ids, cloud_scores, cloud_latency_ms = cloud_future.result(timeout=10.0)
            else:
                raise RuntimeError("No speculative cloud call available")
        except Exception as e:
            logger.warning(
                f"[PARALLEL] Cloud search failed: {e}, falling back to local-only"
            )
            return self._search_local_only(
                all_local_results, result, start_time, total_local_latency, k
            )

        total_time = (time.time() - start_time) * 1000

        if cloud_scores:
            self._update_adaptive_threshold(float(cloud_scores[0]))

        # Per Phase 5.7: Record gate outcome for adaptive threshold.
        # Local was available but gate rejected it. Check whether gate was right
        # by comparing local top-1 with cloud top-5.
        if all_local_results and cloud_ids:
            cloud_id_set = set(str(cid) for cid in (cloud_ids or [])[:5])
            local_top_id = str(all_local_results[0]["id"]) if all_local_results else ""
            was_relevant = local_top_id in cloud_id_set
            self._hybrid_gate.record_outcome(gate_signal, was_relevant, matching_anchor_id)

        # ─── V5: Broadcast weaken ALL anchors on cloud fallback ───
        # Per Option A: broadcast is supplementary. Primary signal is targeted weaken.
        cloud_quality = float(cloud_scores[0]) if cloud_scores else 0.0
        if cloud_quality > 0 and len(self.anchor_system.anchors) > 0:
            weaken_diag = self.anchor_system.broadcast_weaken_cloud(
                query_vector, cloud_quality,
                exclude_anchor_id=None,
            )
            result["broadcast_weaken"] = weaken_diag

        # Per Option A: targeted weaken on matching anchor (V2 proven)
        if matching_anchor_id is not None:
            self.anchor_system.weaken_anchor(matching_anchor_id, penalty=0.3)

        local_overlap = 0
        for cid in cloud_ids or []:
            if str(cid) in local_id_set:
                local_overlap += 1

        result_ids = [str(cid) for cid in (cloud_ids or [])]
        result_scores = [float(s) for s in (cloud_scores or [])]

        result.update(
            {
                "ids": result_ids,
                "scores": result_scores,
                "source": "tier3_cloud",
                "latency_ms": total_time,
                "confidence": cloud_scores[0] if cloud_scores else 0.0,
                "parallel_hybrid": True,
                "served_locally": False,
                "local_overlap": local_overlap,
                "local_overlap_pct": local_overlap / max(len(result_ids), 1) * 100,
                "tier2_fill_rate": fill_rate,
                "gate_signal": gate_signal,
                "calibration_phase": calibration_phase,
                "shadow_probe": shadow_probe,
            }
        )

        if self.metrics:
            self.metrics.record_query(
                source="tier3_cloud",
                latency_ms=total_time,
                prefetch_fetched=0,
                prefetch_skipped=0,
            )

        # Cache cloud results NOT already in local to Tier 2 (reactive loop)
        # Per Bug 1 fix: synchronous cache write — diagnostics must be recorded
        # before _finalize_result() returns. The latency cost is acceptable
        # because cloud vectors are fetched via get_vectors_by_ids (~50ms each).
        # Per Bug 5 fix: bypass_neighborhood=True — reactive cache admits
        # confirmed-relevant documents; exact-ID dedup is sufficient.
        if cloud_ids and cloud_scores:
            cache_ids = []
            cache_scores = []
            for doc_id, score in zip(cloud_ids, cloud_scores):
                if str(doc_id) not in local_id_set:
                    cache_ids.append(doc_id)
                    cache_scores.append(float(score))

            if cache_ids:
                cache_diag = self._cache_neighborhood_to_tier2(
                    cache_ids[:3],
                    cache_scores[:3],
                    query_vector,
                    cluster_id,
                    bypass_neighborhood=True,  # Per Bug 5: reactive cache bypasses neighborhood
                    source_anchor_id=matching_anchor_id,
                )
                result.update(cache_diag)
                result["cached_to_tier2"] = cache_diag["cache_admitted_count"]

        # Per §22 Step 5: predict (fast, local) then prefetch (slow, background)
        if self.prefetch_enabled:
            anchor_id, predictions = self._predict(
                query_vector, query_id, query_text,
                cluster_id,
            )
            self._prefetch_executor.submit(
                self._prefetch,
                query_vector, anchor_id, predictions, cluster_id,
            )

        return self._finalize_result(result)

    def _search_cloud_only(
        self,
        query_vector: np.ndarray,
        result: Dict,
        start_time: float,
        k: int,
    ) -> Dict:
        """Pure cloud baseline without local search or prefetch."""
        cloud_ids, cloud_scores, cloud_latency_ms = self.cloud_vdb.search(
            query_vector, k
        )
        total_time = (time.time() - start_time) * 1000
        if cloud_scores:
            self._update_adaptive_threshold(float(cloud_scores[0]))
        result.update(
            {
                "ids": cloud_ids,
                "scores": cloud_scores,
                "source": "tier3_cloud",
                "latency_ms": total_time,
                "tier3_latency_ms": cloud_latency_ms,
                "confidence": cloud_scores[0] if cloud_scores else 0.0,
            }
        )
        if self.metrics:
            self.metrics.record_query(
                source="tier3_cloud",
                latency_ms=cloud_latency_ms,
                prefetch_fetched=0,
                prefetch_skipped=0,
            )
        return self._finalize_result(result)

    def _search_t1_plus_cloud(
        self,
        query_vector: np.ndarray,
        result: Dict,
        start_time: float,
        k: int,
    ) -> Dict:
        """T1-first baseline: check permanent layer, fall back to cloud.

        Measures the contribution of T1 caching alone. No T2, no anchors,
        no prefetch, no gate. Pure Tier 1 lookup + cloud fallback.

        Based on: simple cache-aside pattern — if cached (T1), serve locally
        at ~0.1ms; if not, serve from cloud at ~120ms. Shows latency savings
        from having even a small permanent cache.
        """
        # Step 1: Check T1 (permanent layer only)
        t1_search_start = time.time()
        t1_ids, t1_scores = self.local_vdb.search_permanent(query_vector, k)
        t1_latency = (time.time() - t1_search_start) * 1000
        total_time = (time.time() - start_time) * 1000

        if t1_ids and t1_scores and t1_scores[0] > 0.90:
            # T1 hit — serve locally
            result.update({
                "ids": t1_ids,
                "scores": t1_scores,
                "source": "tier1_permanent",
                "latency_ms": total_time,
                "tier1_latency_ms": t1_latency,
                "confidence": float(t1_scores[0]),
            })
            if self.metrics:
                self.metrics.record_query(
                    source="tier1_permanent",
                    latency_ms=t1_latency,
                    prefetch_fetched=0,
                    prefetch_skipped=0,
                )
            return self._finalize_result(result)

        # Step 2: T1 miss — fall back to cloud
        cloud_ids, cloud_scores, cloud_latency_ms = self.cloud_vdb.search(
            query_vector, k
        )
        total_time = (time.time() - start_time) * 1000

        result.update({
            "ids": cloud_ids,
            "scores": cloud_scores,
            "source": "tier3_cloud",
            "latency_ms": total_time,
            "tier3_latency_ms": cloud_latency_ms,
            "confidence": float(cloud_scores[0]) if cloud_scores else 0.0,
            "t1_checked": True,
            "t1_hit": False,
        })
        if self.metrics:
            self.metrics.record_query(
                source="tier3_cloud",
                latency_ms=cloud_latency_ms,
                prefetch_fetched=0,
                prefetch_skipped=0,
            )
        return self._finalize_result(result)

    def _search_local_only(
        self,
        all_local_results: List[Dict],
        result: Dict,
        start_time: float,
        total_local_latency: float,
        k: int,
    ) -> Dict:
        """Return the best purely local result set without relabeling it as offline."""
        total_time = (time.time() - start_time) * 1000
        top_k = all_local_results[:k]
        result_ids = [r["id"] for r in top_k]
        result_scores = [float(r["score"]) for r in top_k]

        if top_k:
            best_result = top_k[0]
            source = best_result["tier"]
            best_local_distance = float(best_result["score"])
            confidence = self._distance_to_similarity(best_local_distance)
            result["best_local_tier"] = source
            result["best_local_distance"] = best_local_distance
            result["best_local_similarity"] = confidence
        else:
            source = "local_only_empty"
            confidence = 0.0

        result.update(
            {
                "ids": result_ids,
                "scores": result_scores,
                "source": source,
                "latency_ms": total_time,
                "confidence": confidence,
                "local_confident_hit": confidence >= LOCAL_CONFIDENCE_THRESHOLD,
            }
        )

        if self.metrics:
            self.metrics.record_query(
                source=source,
                latency_ms=total_local_latency,
                prefetch_fetched=0,
                prefetch_skipped=0,
            )

        return self._finalize_result(result)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # PREFETCH PHASE DETERMINATION
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _get_system_state(self) -> Dict:
        """Compute system state signals for state-driven phase transitions.

        Per state-driven phase design: phase transitions are determined by
        anchor system maturity and gate calibration progress, not hardcoded
        query counts. This makes the system self-correcting — static workloads
        (NFCorpus) that never form anchors stay in warmup, while session
        workloads (QReCC) transition to steady quickly.

        Returns:
            Dict with anchor_count, strong_count, markov_sparsity,
            gate_outcomes, fill_rate, phase, calibration_phase.
        """
        anchor_count = len(self.anchor_system.anchors)
        strong_anchors = self.anchor_system.get_strong_anchors()
        strong_count = len(strong_anchors)
        markov_stats = self.anchor_system.markov.get_stats()
        markov_sparsity = markov_stats.get("sparsity", 1.0)
        gate_outcomes = self._hybrid_gate.adaptive_threshold._query_count
        dynamic_stats = self.local_vdb.get_dynamic_stats()
        fill_rate = dynamic_stats["current_size"] / max(dynamic_stats["capacity"], 1)

        return {
            "anchor_count": anchor_count,
            "strong_count": strong_count,
            "markov_sparsity": markov_sparsity,
            "gate_outcomes": gate_outcomes,
            "fill_rate": fill_rate,
        }

    def _get_prefetch_phase(self) -> PrefetchPhase:
        """Determine prefetch phase based on anchor count (decoupled from type).

        Per Bug 8 fix: Decoupled from anchor type to break circular dependency.
        The old design used strong_count and markov_sparsity, but with a high
        gate threshold (0.90), anchors accumulate more misses than hits → stay
        WEAK → strong_count stays 0 → system stuck in WARMUP → lower quality.

        New design uses anchor_count only:
          - COLD_START: No anchors yet. Pure exploration.
          - WARMUP: 1-4 anchors. Medium exploration.
          - STEADY_STATE: ≥5 anchors. Low noise, exploit known paths.

        This is self-correcting:
          - NFCorpus (static): 1 anchor → stays in WARMUP → correct
          - QReCC (session): 10-12 anchors → transitions to STEADY by ~Q10
        """
        state = self._get_system_state()
        anchor_count = state["anchor_count"]

        if anchor_count == 0:
            return PrefetchPhase.COLD_START
        elif anchor_count < 5:
            return PrefetchPhase.WARMUP
        else:
            return PrefetchPhase.STEADY_STATE

    def _get_calibration_phase(self) -> str:
        """Determine calibration phase based on query count (Fix C).

        Per Fix C: Reverted from gate_outcomes to query-count for calibration.
        The gate_outcomes-based design (Bug 8 fix) broke the circular dependency
        but changed the calibration trajectory, causing -1.85% nDCG gap vs
        consensus. The consensus run used query-count calibration which probed
        cloud more aggressively early on (Q1-5 = every query), producing better
        gate calibration → better anchor outcomes → better Markov predictions
        → better T2 content.

        Query-count is exogenous (always increases), so there's no circular
        dependency with gate outcomes. The prefetch phase remains anchor_count
        (also no circular dependency).

        Thresholds match consensus:
          - cold: Q1-Q5. Probe every query to seed threshold.
          - warmup: Q6-Q20. Probe every 5th query.
          - steady: Q21+. Probe every 10th query.
        """
        q = self.query_count

        if q < 5:
            return "cold"
        elif q < 20:
            return "warmup"
        else:
            return "steady"

    def _should_shadow_probe(self) -> bool:
        """Determine if this query should shadow-probe cloud for calibration.

        Per state-driven phase design: probe intervals are determined by
        calibration phase, which is now state-driven rather than query-count-driven.

        Returns:
            True if this query should probe cloud for calibration.
        """
        phase = self._get_calibration_phase()
        if phase == "cold":
            interval = CALIBRATION_PROBE_INTERVAL_COLD
        elif phase == "warmup":
            interval = CALIBRATION_PROBE_INTERVAL_WARMUP
        else:
            interval = CALIBRATION_PROBE_INTERVAL_STEADY

        # Use query_count for interval calculation (state determines phase,
        # but we still need a monotonically increasing counter for intervals)
        return self.query_count % interval == 0

    def _cache_neighborhood_to_tier2(
        self,
        cloud_ids: List[str],
        cloud_scores: List[float],
        query_vector: np.ndarray,
        cluster_id: int,
        *,
        bypass_neighborhood: bool = True,  # Per Bug 5 fix: reactive cache bypasses neighborhood check
        source_anchor_id: Optional[int] = None,  # Per V3: tag reactive cache with source anchor
    ) -> Dict[str, int]:
        """
        Cache semantic neighborhood to TIER 2 (dynamic space).

        Per §22 Step 1: Skip docs already in Tier 1 (wastes Tier 2 capacity —
        they're already locally available). Skip docs already in Tier 2 (exact
        ID match). The neighborhood similarity check in admit_dynamic uses the
        LOOSE admission threshold (0.85) per §22 Step 1.5 / HAKES insight.

        Per Bug 5 fix: reactive cache (cloud results the user actually saw)
        bypasses the neighborhood check entirely (bypass_neighborhood=True).
        These are confirmed-relevant documents — exact-ID dedup is sufficient.
        Predictive prefetch uses a higher threshold (0.95) to avoid near-duplicate
        speculative fetches while still admitting diverse predictions.

        Per §22 Step 1.5: admission threshold is LOOSE (0.85) to let diverse
        vectors into Tier 2. Search dedup uses TIGHT (0.98) to prevent
        near-duplicate results to the user — these are SEPARATE concerns.
        """
        diagnostics = {
            "cache_admitted_count": 0,
            "cache_evicted_count": 0,
            "cache_id_skip_count": 0,           # Exact ID match in Tier 2
            "cache_tier1_skip_count": 0,        # Doc in Tier 1 — already local
            "cache_neighborhood_skip_count": 0,  # Near-duplicate in Tier 2 (from admit_dynamic)
            "cache_insert_failed_count": 0,
        }
        # Per Bug 5 fix: reactive cache bypasses neighborhood check.
        # threshold=1.01 means cosine ≥ 1.01 is impossible, so the check never fires.
        # Only exact-ID dedup (has_dynamic_id, has_permanent_id) remains.
        effective_threshold = 1.01 if bypass_neighborhood else NEIGHBORHOOD_THRESHOLD_ADMISSION

        # Per Fix 3: collect candidates for batch vector fetch
        candidates = []

        for doc_id, score in zip(cloud_ids, cloud_scores):
            try:
                doc_id = str(doc_id)

                # Per §22 Step 1: skip if already in Tier 2 (exact ID match)
                if self.local_vdb.has_dynamic_id(doc_id):
                    diagnostics["cache_id_skip_count"] += 1
                    continue

                # Per §22 Step 1: skip if already in Tier 1 (already locally available)
                if self.local_vdb.has_permanent_id(doc_id):
                    diagnostics["cache_tier1_skip_count"] += 1
                    continue

                # Collect for batch fetch
                candidates.append((doc_id, score))

            except Exception as e:
                diagnostics["cache_insert_failed_count"] += 1
                logger.warning(f"[CACHE] Failed to check {doc_id}: {e}")
                continue

        # Per Fix 3 (B1/B2/B5): batch fetch all candidate vectors in ONE
        # cloud call instead of serial per-doc fetches (was ~250ms for 5 docs).
        # get_vectors_by_ids() now returns vectors in input order (Fix 3).
        if candidates:
            candidate_ids = [c[0] for c in candidates]
            vectors = self.cloud_vdb.get_vectors_by_ids(candidate_ids)

            for (doc_id, score), vector in zip(candidates, vectors):
                try:
                    if vector is None:
                        diagnostics["cache_insert_failed_count"] += 1
                        continue

                    admit_diag = self._admit_to_tier2(
                        vector,
                        doc_id,
                        metadata={
                            "source": "cloud_neighborhood",
                            "cluster_id": cluster_id,
                            "score": score,
                            "weight": 2.0,
                            "anchor_id": source_anchor_id,
                        },
                        neighborhood_threshold=effective_threshold,
                    )
                    # Merge admit diagnostics, mapping neighborhood skips separately
                    diagnostics["cache_admitted_count"] += admit_diag.get("cache_admitted_count", 0)
                    diagnostics["cache_evicted_count"] += admit_diag.get("cache_evicted_count", 0)
                    diagnostics["cache_neighborhood_skip_count"] += admit_diag.get("cache_duplicate_skip_count", 0)
                    diagnostics["cache_insert_failed_count"] += admit_diag.get("cache_insert_failed_count", 0)

                    # Per prefetch quality audit: log reactive admission
                    if admit_diag.get("cache_admitted_count", 0) > 0:
                        self.prefetch_admission_log.append({
                            "query_id": getattr(self, '_current_query_id', ''),
                            "turn": self.query_count,
                            "doc_id": str(doc_id),
                            "admission_type": "reactive",
                            "anchor_id": source_anchor_id,
                            "cloud_score": float(score),
                        })
                except Exception as e:
                    diagnostics["cache_insert_failed_count"] += 1
                    logger.warning(f"[CACHE] Failed to cache {doc_id}: {e}")
                    continue

        total_skipped = (
            diagnostics["cache_id_skip_count"]
            + diagnostics["cache_tier1_skip_count"]
            + diagnostics["cache_neighborhood_skip_count"]
        )
        logger.info(
            "[CACHE] Added %s to TIER 2 (evicted=%s, id_skip=%s, tier1_skip=%s, sim_skip=%s, failed=%s)",
            diagnostics["cache_admitted_count"],
            diagnostics["cache_evicted_count"],
            diagnostics["cache_id_skip_count"],
            diagnostics["cache_tier1_skip_count"],
            diagnostics["cache_neighborhood_skip_count"],
            diagnostics["cache_insert_failed_count"],
        )
        # Backward-compatible aggregate for metrics
        diagnostics["cache_duplicate_skip_count"] = total_skipped
        return diagnostics

    def _predict(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str,
        cluster_id: int,
        ) -> Tuple[int, List[np.ndarray]]:
        """Per §22 Step 5: Prediction phase — create anchor, generate trajectory.

        This is the FAST, LOCAL part. It runs synchronously on the query path
        and returns (anchor_id, prediction_vectors) for the prefetch phase.

        Per consensus: always uses generate_predictions() (Markov-driven).
        No phase-based bypass. No prediction horizon tracking.

        Returns:
            (anchor_id, prediction_vectors) — anchor_id for tracking,
            prediction_vectors for the prefetch phase to fetch from cloud.
        """
        # Per Fix 1 (A1/A2/A5): absorb-or-create anchor policy.
        # find_matching_anchor() checks all existing basins and returns
        # the tightest match. If found, update_anchor() absorbs the query
        # into the existing basin (incremental centroid, radius growth).
        # This replaces unconditional create_anchor() that produced one
        # anchor per query (253 anchors for 323 queries, all WEAK, avg_radius=0.0).
        # find_matching_anchor() records Markov transition internally;
        # update_anchor() sees _last_anchor_id == anchor_id → no double-record.
        matching_anchor = self.anchor_system.find_matching_anchor(query_vector)
        if matching_anchor is not None:
            self.anchor_system.update_anchor(matching_anchor, query_vector)
            anchor_id = matching_anchor
        else:
            parent_anchor_id = None
            anchor_id = self.anchor_system.create_anchor(
                query_vector=query_vector,
                query_id=query_id,
                query_text=query_text,
                parent_anchor_id=parent_anchor_id,
                query_num=self.query_count,
            )

        # Per consensus: always use generate_predictions() (Markov-driven).
        # No phase-based bypass. Prediction count and noise scale are
        # continuous functions of anchor system state, not query count.
        n_anchors = len(self.anchor_system.anchors)
        n_strong = len(self.anchor_system.get_strong_anchors())

        # Continuous prediction count: 3 when few anchors, up to 10 when many
        prediction_count = max(3, min(10, n_anchors))
        # Continuous noise scale: high when few anchors (exploration),
        # low when many anchors (exploitation)
        noise_scale = max(0.05, 0.30 - 0.025 * n_anchors)

        predictions = self.anchor_system.generate_predictions(
            anchor_id=anchor_id,
            centroid=self.semantic_cache.get_centroid(cluster_id),
            count=prediction_count,
            noise_scale=noise_scale,
            query_num=self.query_count,
        )

        return anchor_id, predictions

    def _prefetch(
        self,
        query_vector: np.ndarray,
        anchor_id: int,
        predictions: List[np.ndarray],
        cluster_id: int,
    ):
        """Per §22 Step 5: Prefetch phase — fetch from cloud, admit to Tier 2.

        This is the SLOW, NETWORK part. It runs in the background and
        should NOT block the query response. Prediction vectors come from
        the _predict() phase.

        Based on: Three-phase prefetch (cold start → warmup → steady state).
        Each phase has different noise scales and prediction counts.
        """
        phase = self._get_prefetch_phase()
        dynamic_stats = self.local_vdb.get_dynamic_stats()
        space_available = dynamic_stats["capacity"] - dynamic_stats["current_size"]

        if space_available <= 0:
            logger.info("[PREFETCH] Tier 2 full, skipping prefetch")
            return 0, 0

        fetched = 0
        skipped = 0

        # Check which predictions are already in Tier 2 (cache hit)
        predictions_to_fetch = []
        for i, pred_vec in enumerate(predictions):
            if self.local_vdb.exists_in_dynamic_neighborhood(
                pred_vec, threshold=NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP
            ):
                skipped += 1
            else:
                predictions_to_fetch.append(pred_vec)

        if not predictions_to_fetch:
            logger.info("[PREFETCH] All predictions already cached!")
            return fetched, skipped

        # Per Fix 2 (A3): register_prediction moved to sync _predict().
        # Predictions are already registered before this async method runs.

        # Parallel cloud searches for prediction vectors
        all_doc_ids = []
        all_scores = []
        max_workers = 5 if phase == PrefetchPhase.COLD_START else (
            5 if phase == PrefetchPhase.WARMUP else 3
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pred = {
                executor.submit(self.cloud_vdb.search, pred_vec, 3): (i, pred_vec)
                for i, pred_vec in enumerate(predictions_to_fetch)
            }

            for future in concurrent.futures.as_completed(future_to_pred, timeout=10.0):
                pred_idx, pred_vec = future_to_pred[future]
                try:
                    pred_ids, pred_scores, _ = future.result(timeout=2.0)
                    all_doc_ids.extend(pred_ids)
                    all_scores.extend(pred_scores)
                except Exception as e:
                    logger.warning(f"[PREFETCH] Prediction {pred_idx} failed: {e}")

        # Batch fetch vectors from cloud and admit to Tier 2
        if all_doc_ids:
            try:
                unique_ids = list(dict.fromkeys(all_doc_ids))
                unique_ids = unique_ids[:space_available]

                logger.info(f"[PREFETCH] Batch fetching {len(unique_ids)} vectors...")
                vectors = self.cloud_vdb.get_vectors_by_ids(unique_ids)

                if vectors and len(vectors) > 0:
                    source_label = {
                        PrefetchPhase.COLD_START: "cold_start_prefetch",
                        PrefetchPhase.WARMUP: "warmup_prefetch",
                        PrefetchPhase.STEADY_STATE: "steady_state_prefetch",
                    }[phase]
                    weight = {
                        PrefetchPhase.COLD_START: 1.0,
                        PrefetchPhase.WARMUP: 2.0,
                        PrefetchPhase.STEADY_STATE: 5.0,
                    }[phase]

                    for doc_id, vector in zip(unique_ids, vectors):
                        if vector is None:
                            continue  # Per Fix 3: missing vector from batch
                        if dynamic_stats["current_size"] >= dynamic_stats["capacity"]:
                            break
                        admit_diag = self._admit_to_tier2(
                            vector,
                            doc_id,
                            metadata={
                                "source": source_label,
                                "anchor_id": anchor_id,
                                "weight": weight,
                            },
                            # Per Bug 5 fix: raised from 0.85 to 0.95 for prefetch.
                            # Predictive prefetch should still block near-duplicates
                            # but allow diverse vectors. 0.95 blocks only near-exact
                            # dupes, letting diverse predictions through.
                            neighborhood_threshold=NEIGHBORHOOD_THRESHOLD_PREFETCH_ADMISSION,
                        )
                        fetched += admit_diag.get("cache_admitted_count", 0)
                        skipped += admit_diag.get("cache_duplicate_skip_count", 0)
                        dynamic_stats["current_size"] = (
                            self.local_vdb.get_dynamic_count()
                        )

                        # Per prefetch quality audit: log predictive admission
                        if admit_diag.get("cache_admitted_count", 0) > 0:
                            self.prefetch_admission_log.append({
                                "query_id": getattr(self, '_current_query_id', ''),
                                "turn": self.query_count,
                                "doc_id": str(doc_id),
                                "admission_type": "predictive",
                                "anchor_id": anchor_id,
                                "cloud_score": None,  # prefetch doesn't have per-doc score
                            })

            except Exception as e:
                logger.error(f"[PREFETCH] Batch fetch failed: {e}")

        logger.info(
            f"[PREFETCH] Phase={phase.value} fetched={fetched} skipped={skipped} "
            f"tier2={dynamic_stats['current_size']}/{dynamic_stats['capacity']}"
        )

        # Per §22 Step 5: update metrics from background prefetch
        self.prefetch_cache_hits += skipped
        self.prefetch_cache_misses += fetched
        if self.metrics:
            for _ in range(fetched):
                self.metrics.log_prefetch_miss()
            for _ in range(skipped):
                self.metrics.log_prefetch_hit()

        return fetched, skipped



    def get_stats(self) -> Dict:
        """Get comprehensive router statistics."""
        total_prefetch = self.prefetch_cache_hits + self.prefetch_cache_misses
        prefetch_hit_rate = (
            self.prefetch_cache_hits / total_prefetch * 100 if total_prefetch > 0 else 0
        )

        return {
            "total_queries": self.query_count,
            "prefetch_cache_hit_rate": prefetch_hit_rate,
            "tier1_permanent": self.local_vdb.get_permanent_stats(),
            "tier2_dynamic": self.local_vdb.get_dynamic_stats(),
            "anchor_system": self.anchor_system.get_anchor_stats(),
            "metrics": self.metrics.get_summary(),
        }

    def shutdown(self):
        """Cleanup resources on shutdown."""
        logger.info("[ROUTER] Shutting down thread pools...")
        self.wait_for_background_prefetch(timeout_s=5.0)
        self._prefetch_executor.shutdown(wait=True)
        self.tier_search_executor.shutdown(wait=True)
        logger.info("[ROUTER] âœ… Shutdown complete")
