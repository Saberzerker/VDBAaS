# src/anchor_system.py
"""
Anchor System — Centroid+Radius Semantic Basins with Query-Count Decay

v3: Centroid + radius anchors replacing point-vector + Kalman velocity.

Architecture (per THEORY.md §Q1):
  Anchor = {centroid: vector, radius: float, n: int, strength: float}
  
  update_anchor(anchor, query_vector):
    anchor.n += 1
    anchor.centroid = ((anchor.n-1)/anchor.n) * anchor.centroid + (1/anchor.n) * query_vector
    anchor.radius = max(anchor.radius, cosine_dist(query_vector, anchor.centroid))
  
  belongs_to(anchor, query_vector, epsilon=0.05):
    return cosine_dist(query_vector, anchor.centroid) <= anchor.radius + epsilon

An anchor is a "semantic basin" — not a point, but a region of embedding
space that has been visited. The centroid tracks the mean of all queries
assigned to this anchor, and the radius tracks the basin's extent.

Epsilon is calibrated from corpus geometry at initialization and held fixed.
For session-ordered workloads: P25 of within-session query distances.
For static workloads: P50 of Tier 1 k-NN distances.

The velocity-based prediction (v2) is removed. Prediction will be replaced
by Markov transitions in Phase 3. For now, generate_predictions() uses
centroid-guided noise exploration as a placeholder.

Based on: Cognitive memory models — "gists" as precomputed molds
  (Collins & Loftus 1975, Cowan 1999).
  Query-count decay: session-proportional, not clock-proportional.

Author: Saberzerker (v3: centroid+radius anchors)
Date: 2026-04-19
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    ANCHOR_SIGNAL_SCALE,
    PREFETCH_K,
)
from src.markov_transitions import MarkovTransitionMatrix

logger = logging.getLogger(__name__)
# Per consensus: prediction horizon removed. Predictions are materialized
# into T2 vectors; T2 admission/eviction handles the lifecycle. No need for
# separate prediction tracking. Anchor reinforcement comes from absorption
# (update_anchor) and broadcast signals, not prediction matching.
MIN_ANCHOR_ATTEMPTS = 20
MEDIUM_ANCHOR_HIT_RATE = 0.10
STRONG_ANCHOR_HIT_RATE = 0.25
# PERMANENT tier removed — all anchors decay. Former PERMANENT anchors
# now classify as STRONG (decay at λ=0.995).

# Default epsilon for belongs_to() — will be overridden by configure_from_corpus_shape()
DEFAULT_EPSILON = 0.05
# Default max_radius — will be overridden by configure_from_corpus_shape()
DEFAULT_MAX_RADIUS = 0.12

# ─── Option A: Additive strength, no PERMANENT tier (all anchors decay) ───
# Strength is additive (unbounded, grows with hits). Confidence mirrors strength
# for backward compat but is NOT the primary signal.
# No PERMANENT tier — even high-hit-rate anchors decay at STRONG rate (λ=0.995).

# Per query-count exponential decay (V2 proven)
DECAY_LAMBDA_WEAK = 0.95       # 5% decay per query gap
DECAY_LAMBDA_MEDIUM = 0.98     # 2% decay per query gap
DECAY_LAMBDA_STRONG = 0.995    # 0.5% decay per query gap (was PERMANENT λ=1.0, now all decay)
DECAY_STRENGTH_FLOOR = 0.05    # Prune threshold (Bug #4)

# V2 reinforcement constants (proven)
ANCHOR_HIT_REWARD = 1.0        # Additive reward on prediction hit
ANCHOR_MISS_PENALTY = 0.5      # Linear penalty on gate rejection
ANCHOR_PARENT_CREDIT = 0.5    # Parent gets half the reward

# Broadcast signal scale (supplementary, not primary)
# Per Option A: broadcast is supplementary. Primary signal is targeted strengthen/weaken.
BROADCAST_SIGNAL_SCALE = ANCHOR_SIGNAL_SCALE  # From config, typically 0.05

# Centroid update: simple incremental mean (V2 proven, no loss modulation)
# No centroid shift penalty — V2 had none and it worked.
# No absorption reward — strength is additive, not EMA.

CONFIDENCE_FLOOR = 0.01       # Absolute minimum confidence (never fully dead)
CONFIDENCE_EVICT_FLOOR = 0.02 # Evict anchor below this confidence
CONFIDENCE_INITIAL = 0.5      # Neutral starting confidence


class AnchorType(Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    PERMANENT = "permanent"  # Legacy — no longer assigned, maps to STRONG at runtime


@dataclass
class Anchor:
    """
    An anchor is a semantic basin — centroid + radius in embedding space.

    Tracks:
    - centroid: running mean of all query vectors assigned to this basin
    - radius: max cosine distance from centroid (basin extent)
    - n: number of query vectors absorbed
    - strength: reinforcement signal (prediction hit/miss)
    - parent/children: anchor graph structure
    - Query-count-based decay: strength *= lambda^Delta_q
    """

    id: int
    # Per centroid+radius anchor (THEORY.md §Q1)
    centroid: np.ndarray      # Running mean of query vectors
    radius: float             # Max cosine distance from centroid
    n: int                    # Number of query vectors absorbed

    query_id: str
    query_text: str
    strength: float           # Additive reinforcement signal (unbounded, grows with hits)
    confidence: float          # Mirrors strength for backward compat; primary signal is strength
    anchor_type: AnchorType
    created_at: float
    last_accessed: float
    hits: int
    misses: int
    parent_id: Optional[int]
    children_ids: List[int]
    predictions: List[np.ndarray]
    metadata: Dict

    # Per query-count exponential decay
    last_access_query_num: int = 0

    # Keep vector field for backward compat with router code
    # that accesses anchor.vector — delegates to centroid
    @property
    def vector(self) -> np.ndarray:
        return self.centroid


class AnchorSystem:
    """
    Manages anchor graph with centroid+radius basins and query-count decay.

    Key operations:
    1. create_anchor() - Create new semantic basin, update parent
    2. update_anchor() - Absorb query vector into existing anchor
    3. belongs_to() - Check if query falls within anchor's basin
    4. generate_predictions() - Markov-driven prediction vectors
    5. strengthen_anchor() - Reward correct predictions
    6. decay_anchors_by_query_count() - Query-count exponential decay

    Per consensus: prediction horizon tracking removed. Predictions are
    materialized into T2 vectors; T2 admission/eviction handles lifecycle.
    Anchor reinforcement comes from absorption (update_anchor) and broadcast
    signals, not prediction matching.
    """

    def __init__(self):
        self.anchors: Dict[int, Anchor] = {}
        self.anchor_counter = 0
        self.global_query_count = 0
        self.lock = threading.Lock()

        # Per Tier 1 density epsilon: set externally via configure_from_corpus_shape()
        self._epsilon = DEFAULT_EPSILON

        # Per radius cap: maximum basin radius (prevents single mega-anchor)
        # Overridden by configure_from_corpus_shape() from corpus inter-query distances
        self._max_radius = DEFAULT_MAX_RADIUS

        # Per Phase 3: Markov transition matrix for anchor prediction
        self.markov = MarkovTransitionMatrix()

        # Track last anchor for transition recording
        self._last_anchor_id: Optional[int] = None

        logger.info(
            "[ANCHORS] Initialized anchor system (v3: centroid+radius basins + Markov)"
        )

    def set_epsilon(self, epsilon: float):
        """Set epsilon from corpus geometry calibration.

        Per anchor policy: epsilon is calibrated once from corpus geometry
        (P25 of within-session distances for session workloads, P50 of
        Tier 1 k-NN distances for static workloads) and held fixed during
        inference.
        """
        self._epsilon = epsilon
        logger.info(f"[ANCHORS] Epsilon set to {epsilon:.4f} (from Tier 1 density)")

    def set_max_radius(self, max_radius: float):
        """Set maximum anchor basin radius.

        Per corpus shape profile: max_radius should cover the typical
        inter-query distance in the corpus embedding space.
        NFCorpus mean dist=0.178 → max_radius >= 0.20
        TREC-COVID mean dist=0.128 → max_radius >= 0.15
        SciFact mean dist=0.059 → max_radius >= 0.10
        """
        self._max_radius = max_radius
        logger.info(f"[ANCHORS] Max radius set to {max_radius:.4f}")

    def configure_from_corpus_shape(
        self,
        tier1_vectors: Optional[np.ndarray] = None,
        query_vectors: Optional[np.ndarray] = None,
        query_corpus_distances: Optional[np.ndarray] = None,
        session_ids: Optional[List[str]] = None,
    ):
        """Auto-configure epsilon and max_radius from corpus geometry.

        Per anchor policy: epsilon and max_radius must be derived from the
        actual embedding space, not hardcoded. The previous DEFAULT_EPSILON=0.05
        and MAX_RADIUS=0.12 caused anchors to be inert on NFCorpus (mean dist=0.178)
        and TREC-COVID (mean dist=0.128).

        Derivation:
          - If session_ids provided (session-ordered workload):
            epsilon = P75 of within-session query distances (topic coherence)
            This creates topic-specific anchors that don't merge into mega-anchors.
          - Otherwise (static workload):
            epsilon = P75 of Tier 1 intra-distances (corpus diversity)
          max_radius = max(
              p95 of query-corpus distances,  # anchors must cover typical query range
              p75 of Tier 1 intra-distances * 1.5,  # scale up for safety
              0.10  # absolute floor
          )

        Based on: QReCC analysis showing P75 within-session dist=0.1954 vs
        P25 across-session dist=0.2716. Epsilon must be below the across-session
        gap to create topic-specific anchors.

        Args:
            tier1_vectors: (n, d) array of Tier 1 medoid vectors. Used for epsilon.
            query_vectors: (m, d) array of query embeddings. Used for max_radius.
            query_corpus_distances: (m,) array of mean query-corpus cosine distances.
                If provided, overrides query_vectors for max_radius computation.
            session_ids: Optional list of session IDs per query. When provided,
                epsilon is computed from within-session distances instead of
                Tier 1 inter-distances. This is critical for session-ordered
                workloads where topic coherence differs from corpus diversity.
        """
        # --- Epsilon from within-session distances (session-ordered) ---
        # Per mega-anchor fix: Tier 1 inter-distances measure corpus diversity,
        # not topic coherence. For session workloads, within-session distances
        # are the right signal for anchor basin size.
        if session_ids is not None and query_vectors is not None and len(session_ids) == len(query_vectors):
            within_dists = []
            session_groups = {}
            for idx, sid in enumerate(session_ids):
                session_groups.setdefault(sid, []).append(idx)

            for sid, indices in session_groups.items():
                if len(indices) < 2:
                    continue
                session_vecs = query_vectors[indices]
                # Normalize
                norms = np.linalg.norm(session_vecs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                session_norm = session_vecs / norms
                # Pairwise cosine distances within session
                sim = session_norm @ session_norm.T
                dist = 1.0 - sim
                # Upper triangle only (avoid self-match and double-counting)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        within_dists.append(dist[i, j])

            if within_dists:
                within_dists = np.array(within_dists)
                # Per consensus: P25 creates ~30-40 specific anchors per session.
                # P50 created ~20-30 anchors that were too broad (generic centroids).
                # P25 creates tighter basins that capture subtopic structure,
                # giving Markov transitions more states to work with.
                # Based on: Bruhn & Zaragoza 2004; Han 2004 query reformulation types.
                epsilon = float(np.percentile(within_dists, 25))
                epsilon = max(epsilon, 0.02)  # floor at 0.02
                self.set_epsilon(epsilon)
                # Per mega-anchor fix: max_radius from within-session P95.
                # For session workloads, max_radius should bound anchor basins
                # to within-topic range, not across-topic range.
                max_radius_session = float(np.percentile(within_dists, 95))
                max_radius_session = max(max_radius_session, 0.10)  # absolute floor
                self.set_max_radius(max_radius_session)
                logger.info(
                    f"[ANCHORS] Session-aware epsilon: {epsilon:.4f} "
                    f"(P25 of {len(within_dists)} within-session distances, "
                    f"{len(session_groups)} sessions, "
                    f"mean={within_dists.mean():.4f}, P75={np.percentile(within_dists, 75):.4f})"
                )
                logger.info(
                    f"[ANCHORS] Session-aware max_radius: {max_radius_session:.4f} "
                    f"(P95 of within-session distances)"
                )
            else:
                logger.warning(
                    "[ANCHORS] No within-session pairs for epsilon, falling back to T1"
                )

        # --- Epsilon from Tier 1 k-NN distances (fallback / static workload) ---
        if self._epsilon == 0.05 and tier1_vectors is not None and len(tier1_vectors) >= 2:
            # Only compute if session-aware epsilon wasn't set above
            n = len(tier1_vectors)
            k = min(10, max(2, int(np.sqrt(n))))
            norms = np.linalg.norm(tier1_vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normalized = tier1_vectors / norms
            sim_matrix = normalized @ normalized.T
            dist_matrix = 1.0 - sim_matrix
            np.fill_diagonal(dist_matrix, np.inf)
            knn_dists = np.sort(dist_matrix, axis=1)[:, :k]
            # Per anchor policy: P50 for finer-grained anchors (same rationale as session-aware)
            epsilon = float(np.percentile(knn_dists, 50))
            epsilon = max(epsilon, 0.02)
            self.set_epsilon(epsilon)
        elif self._epsilon == 0.05:
            logger.warning(
                "[ANCHORS] No Tier 1 vectors for epsilon computation, using default"
            )

        # --- Max radius from query-corpus distances ---
        # Per mega-anchor fix: skip if session-aware max_radius was already set
        # (within the session_ids block above). Session-aware max_radius uses
        # P95 of within-session distances, which is the correct bound for
        # topic-specific anchors. Check if max_radius was already overridden
        # by comparing against the default value.
        _DEFAULT_MAX_RADIUS = 0.12  # Must match DEFAULT_MAX_RADIUS at module level
        if self._max_radius == _DEFAULT_MAX_RADIUS and query_corpus_distances is not None and len(query_corpus_distances) > 0:
            # Default value still set — session-aware didn't override
            max_radius = float(np.percentile(query_corpus_distances, 95))
            max_radius = max(max_radius, 0.10)  # absolute floor
            self.set_max_radius(max_radius)
        elif self._max_radius == _DEFAULT_MAX_RADIUS and query_vectors is not None and tier1_vectors is not None and len(tier1_vectors) >= 2:
            norms_t1 = np.linalg.norm(tier1_vectors, axis=1, keepdims=True)
            norms_t1 = np.maximum(norms_t1, 1e-10)
            t1_norm = tier1_vectors / norms_t1
            norms_q = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            norms_q = np.maximum(norms_q, 1e-10)
            q_norm = query_vectors / norms_q
            sim_q_t1 = q_norm @ t1_norm.T
            max_sim = np.max(sim_q_t1, axis=1)
            min_dist = 1.0 - max_sim
            max_radius = float(np.percentile(min_dist, 95))
            max_radius = max(max_radius, 0.10)
            self.set_max_radius(max_radius)
        elif self._max_radius == _DEFAULT_MAX_RADIUS:
            logger.warning(
                "[ANCHORS] No query-corpus distances for max_radius, using default"
            )

    # ═══════════════════════════════════════════════════════════
    # ANCHOR CREATION & UPDATE
    # ═══════════════════════════════════════════════════════════

    def create_anchor(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str = "",
        parent_anchor_id: Optional[int] = None,
        query_num: int = 0,
    ) -> int:
        """
        Create new anchor as a semantic basin.

        Per THEORY.md §Q1: centroid initialized to query_vector, radius=0, n=1.
        The radius will grow as more queries are absorbed into this basin.
        """
        anchor_id = self.anchor_counter
        self.anchor_counter += 1

        anchor = Anchor(
            id=anchor_id,
            centroid=query_vector.copy(),
            radius=0.0,  # Single point — radius will grow with absorption
            n=1,
            query_id=query_id,
            query_text=query_text,
            strength=CONFIDENCE_INITIAL,  # mirrors confidence for backward compat
            confidence=CONFIDENCE_INITIAL,  # EMA confidence [0.01, 1.0]
            anchor_type=AnchorType.WEAK,
            created_at=time.time(),
            last_accessed=time.time(),
            hits=0,
            misses=0,
            parent_id=parent_anchor_id,
            children_ids=[],
            predictions=[],
            metadata={},
            last_access_query_num=query_num,
        )

        self.anchors[anchor_id] = anchor

        # Per Phase 3: register anchor in Markov transition matrix
        self.markov.register_anchor(anchor_id, query_vector)

        # Record transition: last_anchor -> new_anchor
        # Per co-evolutionary loop: weight by source anchor confidence
        if self._last_anchor_id is not None and self._last_anchor_id in self.anchors:
            src_confidence = self.anchors[self._last_anchor_id].confidence
            self.markov.record_transition(self._last_anchor_id, anchor_id, weight=src_confidence)
        self._last_anchor_id = anchor_id
        self.markov.set_last_anchor(anchor_id)

        # Update parent's children list
        if parent_anchor_id is not None and parent_anchor_id in self.anchors:
            parent = self.anchors[parent_anchor_id]
            parent.children_ids.append(anchor_id)
            logger.debug(
                f"[ANCHORS] Created Anchor #{anchor_id} (child of #{parent_anchor_id})"
            )
        else:
            logger.debug(f"[ANCHORS] Created Anchor #{anchor_id} (root)")

        return anchor_id

    def update_anchor(self, anchor_id: int, query_vector: np.ndarray) -> None:
        """Absorb a query vector into an anchor's semantic basin.

        Per Option A: Reverted to V2's simple incremental mean (no loss modulation,
        no centroid shift penalty, no absorption reward). V2's simple approach
        produced nDCG@5=0.627 on QReCC vs V5's 0.538.

        Algorithm (V2 proven):
          anchor.n += 1
          anchor.centroid = ((n-1)/n) * old_centroid + (1/n) * query_vector
          anchor.radius = max(anchor.radius, cosine_dist(query_vector, anchor.centroid))

        Based on: Incremental mean (numerically stable, no death spiral).
        """
        if anchor_id not in self.anchors:
            return

        anchor = self.anchors[anchor_id]

        old_n = anchor.n
        anchor.n = old_n + 1

        # ─── V2: Simple incremental mean (no loss modulation) ───
        # c_new = ((n-1)/n) * c_old + (1/n) * v_new
        anchor.centroid = (
            (old_n / anchor.n) * anchor.centroid + (1.0 / anchor.n) * query_vector
        )

        # Re-normalize centroid to unit sphere (E5 vectors are normalized)
        norm = np.linalg.norm(anchor.centroid)
        if norm > 1e-10:
            anchor.centroid = anchor.centroid / norm

        # Update radius: max cosine distance from updated centroid
        dist = self._cosine_distance(query_vector, anchor.centroid)
        anchor.radius = max(anchor.radius, dist)

        # Per V5 fix: update last_access_query_num so decay uses correct delta_q
        anchor.last_access_query_num = self.global_query_count
        anchor.last_accessed = time.time()

        # Record transition if this is a different anchor from last query
        if self._last_anchor_id is not None and self._last_anchor_id != anchor_id:
            if self._last_anchor_id in self.anchors:
                src_strength = self.anchors[self._last_anchor_id].strength
                self.markov.record_transition(self._last_anchor_id, anchor_id, weight=src_strength)
        self._last_anchor_id = anchor_id
        self.markov.set_last_anchor(anchor_id)

        # ─── Diagnostics ───
        logger.debug(
            f"[ABSORB] Anchor #{anchor_id}: n={anchor.n}, "
            f"strength={anchor.strength:.1f}, type={anchor.anchor_type.value}"
        )

    def belongs_to(
        self,
        query_vector: np.ndarray,
        anchor_id: int,
        epsilon: Optional[float] = None,
    ) -> bool:
        """Check if query falls within anchor's semantic basin.

        Per mega-anchor fix: uses epsilon-only threshold (not radius + epsilon).
        A query belongs to an anchor if it's within epsilon of the centroid.
        Radius is tracked for diagnostics but doesn't expand the basin.
        """
        if anchor_id not in self.anchors:
            return False

        anchor = self.anchors[anchor_id]
        eps = epsilon if epsilon is not None else self._epsilon
        dist = self._cosine_distance(query_vector, anchor.centroid)
        return dist <= eps

    def find_matching_anchor(
        self, query_vector: np.ndarray
    ) -> Optional[int]:
        """Find the anchor whose basin contains this query vector.

        Per mega-anchor fix: competitive assignment with minimum distance check.
        Returns anchor_id of best-matching anchor ONLY if the query is within
        epsilon of the anchor's centroid. If the closest anchor is farther than
        epsilon, returns None (triggering new anchor creation).

        This prevents a single anchor from absorbing all queries — only queries
        genuinely within the anchor's topic neighborhood are absorbed.

        Returns anchor_id of best-matching anchor, or None.
        """
        best_id = None
        best_dist = float('inf')

        for anchor_id, anchor in self.anchors.items():
            # Per DORMANT removal: all active anchors participate in matching.
            # Evicted anchors (confidence < DEATH_FLOOR) are removed from self.anchors.
            dist = self._cosine_distance(query_vector, anchor.centroid)
            if dist < best_dist:
                best_dist = dist
                best_id = anchor_id

        # Per mega-anchor fix: competitive assignment.
        # Only absorb if query is within epsilon of the closest anchor centroid.
        # If the closest anchor is farther than epsilon, create a new anchor.
        if best_id is not None and best_dist <= self._epsilon:
            # Per Phase 3: record transition for matched anchor
            # Per co-evolutionary loop: weight by source anchor confidence
            # Guard: _last_anchor_id may have been evicted by decay
            if self._last_anchor_id is not None and self._last_anchor_id != best_id and self._last_anchor_id in self.anchors:
                src_confidence = self.anchors[self._last_anchor_id].confidence
                self.markov.record_transition(self._last_anchor_id, best_id, weight=src_confidence)
            self._last_anchor_id = best_id
            self.markov.set_last_anchor(best_id)
            return best_id

        # Query is outside all anchor basins — will trigger new anchor creation
        return None

    # ═══════════════════════════════════════════════════════════
    # PREDICTION GENERATION (PLACEHOLDER — Markov replaces in Phase 3)
    # ═══════════════════════════════════════════════════════════

    def generate_predictions(
        self,
        anchor_id: int,
        centroid: Optional[np.ndarray] = None,
        count: int = PREFETCH_K,
        noise_scale: float = 0.15,
        query_num: int = 0,
    ) -> List[np.ndarray]:
        """Generate predictions using first-order Markov from current anchor.

        P[current → ?] with Laplace smoothing + semantic prior.
        Weighted by confidence × recency — stronger, more active anchors
        contribute more to predictions.

        Per B0: direct Markov P[current → ?] (not collaborative P[other → ?]).
        Per V3 data: 47% yield, 56.9% accuracy with raw probs — weighting
        should improve targeting of useful predictions.
        Per DORMANT removal: no DORMANT filter — all active anchors participate.

        Based on: First-order Markov prefetch; Laplace-smoothed transitions
        with semantic prior (THEORY.md §Q2).
        """
        if anchor_id not in self.anchors:
            logger.warning(f"[ANCHORS] Anchor #{anchor_id} not found")
            return []

        anchor = self.anchors[anchor_id]
        anchor_vec = anchor.centroid

        # ─── Primary: first-order Markov from current anchor ───
        # P[current → ?] with Laplace smoothing + semantic prior.
        probs = self.markov.get_transition_probs(anchor_id)

        if probs:
            # Filter evicted anchors — only active anchors participate
            active_targets = {
                tid: prob for tid, prob in probs.items()
                if tid in self.anchors
            }

            if active_targets:
                # ─── Confidence × recency weighting ───
                # V2 had this inside collaborative block — restore it.
                # Strong (high confidence) and recent (low query gap) anchors
                # get more prediction budget. Prevents weak/stale anchors from
                # polluting predictions.
                weighted = {}
                for tid, prob in active_targets.items():
                    target = self.anchors[tid]
                    recency = 1.0 / (1 + self.global_query_count - target.last_access_query_num)
                    weight = target.confidence * recency
                    weighted[tid] = prob * weight

                # Normalize
                total = sum(weighted.values())
                if total > 1e-10:
                    for tid in weighted:
                        weighted[tid] /= total

                # Sort by weighted probability (highest first)
                sorted_targets = sorted(weighted.items(), key=lambda x: -x[1])

                # ─── Trajectory prediction toward target anchors ───
                # V2 steady-state used this with momentum=0.9 — extrapolates
                # from current query toward target centroid. More diverse than
                # raw centroid. Per V3 data: trajectory gave 0.627 nDCG vs
                # centroid-only 0.572.
                predictions = []
                for target_id, prob in sorted_targets[:count]:
                    if target_id in self.anchors:
                        target_centroid = self.anchors[target_id].centroid
                        pred_vec = self.generate_trajectory_prediction(
                            anchor_vec, target_centroid, momentum=0.9,
                        )
                        predictions.append(pred_vec)

                if predictions:
                    logger.info(
                        f"[PREDICTIONS] Markov+trajectory: {len(predictions)} predictions "
                        f"from Anchor #{anchor_id}, "
                        f"top_prob={sorted_targets[0][1]:.3f}, "
                        f"top_target=#{sorted_targets[0][0]}"
                    )
                    anchor.predictions = predictions
                    return predictions

        # ─── Fallback: single-anchor Markov ───
        # Used when no other anchors exist (query 1) or no transitions yet
        markov_preds = self.markov.predict_next_vectors(anchor_id, top_k=count)
        if markov_preds:
            predictions = [centroid for centroid, prob in markov_preds]
            # Pad with centroid-interpolated if fewer than count
            while len(predictions) < count:
                top_target = markov_preds[0][0]
                alpha = np.random.uniform(0.3, 0.7)
                interp = alpha * top_target + (1 - alpha) * anchor_vec
                norm = np.linalg.norm(interp)
                if norm > 0:
                    interp = interp / norm
                predictions.append(interp)

            anchor.predictions = predictions[:count]
            return predictions[:count]

        # ─── Last resort: centroid-guided noise ───
        # Only reached when query 1 has no Markov transitions at all
        adaptive_noise = max(noise_scale, self._epsilon * 0.5)

        if centroid is not None:
            total_predictions = anchor.hits + anchor.misses
            success_rate = anchor.hits / max(total_predictions, 1)
            momentum = 0.5 + (0.4 * success_rate)

            predictions = []
            for i in range(count):
                pred = momentum * centroid + (1 - momentum) * anchor_vec
                noise = np.random.normal(0, adaptive_noise, pred.shape)
                pred = pred + noise
                norm = np.linalg.norm(pred)
                if norm > 0:
                    pred = pred / norm
                predictions.append(pred)

            logger.debug(
                f"[PREDICTIONS] Generated {count} centroid predictions (momentum={momentum:.2f}, "
                f"noise={adaptive_noise:.3f})"
            )
        else:
            predictions = []
            for i in range(count):
                noise = np.random.normal(0, adaptive_noise * 2, anchor_vec.shape)
                pred = anchor_vec + noise
                norm = np.linalg.norm(pred)
                if norm > 0:
                    pred = pred / norm
                predictions.append(pred)

            logger.debug(f"[PREDICTIONS] Generated {count} random walk predictions")

        anchor.predictions = predictions
        return predictions

    def generate_trajectory_prediction(
        self, query_vector: np.ndarray, target_vector: np.ndarray, momentum: float = 0.9
    ) -> np.ndarray:
        """Generate single prediction along trajectory (legacy interface)."""
        pred = momentum * target_vector + (1 - momentum) * query_vector
        norm = np.linalg.norm(pred)
        if norm > 0:
            pred = pred / norm
        return pred

    # ═══════════════════════════════════════════════════════════
    # REINFORCEMENT (STRENGTHEN/WEAKEN)
    # ═══════════════════════════════════════════════════════════

    def strengthen_anchor(self, anchor_id: int, reward: float = 1.0):
        """Strengthen anchor after correct prediction.

        Per Option A: Reverted to V2's additive strength model.
        strength += reward (unbounded, grows with hits).
        This produces the dynamic range V2 had (0→80+) that V5's EMA
        compressed to 0.29-0.45, causing the regression.

        Parent propagation: parent gets reward * 0.5 (V2 proven).
        """
        if anchor_id not in self.anchors:
            return

        anchor = self.anchors[anchor_id]

        old_strength = anchor.strength
        anchor.strength += reward
        anchor.hits += 1
        anchor.last_accessed = time.time()
        anchor.confidence = min(1.0, anchor.strength / 10.0)  # Map strength→confidence for compat

        old_type = anchor.anchor_type
        anchor.anchor_type = self._determine_anchor_type(anchor)

        # Propagate to parent (V2 proven: parent gets half credit)
        if anchor.parent_id is not None and anchor.parent_id in self.anchors:
            parent = self.anchors[anchor.parent_id]
            parent.strength += reward * ANCHOR_PARENT_CREDIT
            parent.confidence = min(1.0, parent.strength / 10.0)
            parent.anchor_type = self._determine_anchor_type(parent)

        logger.info(
            f"[REINFORCE] Anchor #{anchor_id}: "
            f"{old_strength:.1f} -> {anchor.strength:.1f} "
            f"({old_type.value} -> {anchor.anchor_type.value})"
        )

    def weaken_anchor(self, anchor_id: int, penalty: float = 0.5):
        """Weaken anchor after incorrect prediction or gate rejection.

        Per Option A: Reverted to V2's linear subtraction.
        strength -= penalty (floor at 0).
        V2 used penalty=0.3 on gate rejection, penalty=0.5 default.
        """
        if anchor_id not in self.anchors:
            return

        anchor = self.anchors[anchor_id]

        old_strength = anchor.strength
        anchor.strength = max(0.0, anchor.strength - penalty)
        anchor.misses += 1
        anchor.confidence = min(1.0, anchor.strength / 10.0)  # Map strength→confidence for compat

        old_type = anchor.anchor_type
        anchor.anchor_type = self._determine_anchor_type(anchor)

        logger.debug(
            f"[WEAKEN] Anchor #{anchor_id}: "
            f"{old_strength:.1f} -> {anchor.strength:.1f} "
            f"({old_type.value} -> {anchor.anchor_type.value})"
        )

    def broadcast_weaken_cloud(
        self,
        query_vector: np.ndarray,
        cloud_quality: float,
        exclude_anchor_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """Broadcast weaken ALL anchors on cloud fallback (supplementary signal).

        Per Option A: This is now supplementary, not primary. The primary
        weaken signal comes from targeted weaken_anchor() on the matching anchor
        in the router. Broadcast weaken provides gentle exploration signal.

        Loss = (1 - sim) × strength_scale × cloud_quality × BROADCAST_SIGNAL_SCALE / sqrt(n)
        where strength_scale uses additive strength (not EMA confidence).

        Call BEFORE reactive cache addition so new anchor starts fresh.

        Returns dict with {anchor_id: loss_applied} for diagnostics.
        """
        if not self.anchors:
            return {}

        diagnostics = {}
        n_anchors = len(self.anchors)
        sqrt_n = max(1.0, np.sqrt(n_anchors))

        for anchor_id, anchor in list(self.anchors.items()):
            if anchor_id == exclude_anchor_id:
                continue

            sim = self._cosine_similarity(query_vector, anchor.centroid)
            # Diminishing returns: mature anchors change slowly
            maturity_scale = 1.0 / (1.0 + np.log(1.0 + anchor.hits))

            # Per Option A: use BROADCAST_SIGNAL_SCALE (supplementary, not primary)
            loss = (
                (1.0 - sim)
                * anchor.confidence  # Use confidence for broadcast scaling
                * cloud_quality
                * BROADCAST_SIGNAL_SCALE
                / sqrt_n
                * maturity_scale
            )

            if loss > 1e-6:
                self.weaken_anchor(anchor_id, penalty=min(loss, 0.5))
                diagnostics[str(anchor_id)] = round(loss, 5)

        if diagnostics:
            logger.info(
                f"[BROADCAST-WEAKEN] Cloud fallback: weakened {len(diagnostics)} anchors "
                f"(cloud_quality={cloud_quality:.3f}, top_loss={max(diagnostics.values()):.4f})"
            )

        return diagnostics

    def broadcast_local_signal(
        self,
        query_vector: np.ndarray,
        result_quality: float,
        matching_anchor_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """V2+V5 hybrid: targeted primary signal + broadcast supplementary.

        Per Option A: The matching anchor gets a strong additive reward (V2 proven).
        All other anchors get a weak distribution-relative signal (V5 supplementary).
        This preserves V2's high-signal targeted feedback while keeping V5's
        broadcast for exploration.

        Returns dict with {anchor_id: signal_applied} for diagnostics.
        Positive = strengthened, negative = weakened.
        """
        if len(self.anchors) < 1:
            return {}

        diagnostics = {}

        # ─── Primary: targeted strengthen on matching anchor (V2 proven) ───
        if matching_anchor_id is not None and matching_anchor_id in self.anchors:
            self.strengthen_anchor(matching_anchor_id, reward=ANCHOR_HIT_REWARD)
            diagnostics[str(matching_anchor_id)] = ANCHOR_HIT_REWARD

        # ─── Supplementary: broadcast to all anchors (V5 exploration) ───
        if len(self.anchors) < 2:
            return diagnostics  # Need at least 2 for distribution-relative signal

        # Compute similarities to all anchors
        sims = {}
        for anchor_id, anchor in self.anchors.items():
            if anchor_id == matching_anchor_id:
                continue  # Already strengthened above
            sims[anchor_id] = self._cosine_similarity(query_vector, anchor.centroid)

        if not sims:
            return diagnostics

        sim_values = list(sims.values())
        sim_min = min(sim_values)
        sim_max = max(sim_values)

        # All equidistant → no information → no signal
        if sim_max - sim_min < 1e-6:
            return diagnostics

        # Normalize to [0, 1]
        normalized = {
            aid: (s - sim_min) / (sim_max - sim_min)
            for aid, s in sims.items()
        }

        # Median as neutral point
        sorted_norms = sorted(normalized.values())
        median_norm = sorted_norms[len(sorted_norms) // 2]

        n_anchors = len(sims)
        sqrt_n = max(1.0, np.sqrt(n_anchors))

        for anchor_id, norm in list(normalized.items()):
            anchor = self.anchors[anchor_id]
            proximity_signal = norm - median_norm  # positive=above median, negative=below

            # Diminishing returns: mature anchors change slowly
            maturity_scale = 1.0 / (1.0 + np.log(1.0 + anchor.hits))

            magnitude = (
                result_quality
                * BROADCAST_SIGNAL_SCALE
                / sqrt_n
                * maturity_scale
            )
            signal = proximity_signal * magnitude

            if abs(signal) < 1e-7:
                continue

            if signal > 0:
                self.strengthen_anchor(anchor_id, reward=min(signal, 0.2))
                diagnostics[str(anchor_id)] = round(signal, 5)
            else:
                self.weaken_anchor(anchor_id, penalty=min(abs(signal), 0.1))
                diagnostics[str(anchor_id)] = round(signal, 5)

        if diagnostics:
            pos_count = sum(1 for v in diagnostics.values() if v > 0)
            neg_count = sum(1 for v in diagnostics.values() if v < 0)
            logger.info(
                f"[BROADCAST-LOCAL] Local hit: {pos_count} strengthened, "
                f"{neg_count} weakened, {len(self.anchors) - pos_count - neg_count} neutral "
                f"(quality={result_quality:.3f}, matching=#{matching_anchor_id})"
            )

        return diagnostics

    def _determine_anchor_type(self, anchor: Anchor) -> AnchorType:
        """Determine anchor type from attempts and hit-rate (V2 proven).

        No PERMANENT tier — all anchors decay. Anchors that would have been
        PERMANENT (attempts≥50, hit_rate≥0.40) now classify as STRONG and
        decay at λ=0.995 per query gap.
        """
        attempts = anchor.hits + anchor.misses
        hit_rate = (anchor.hits / attempts) if attempts > 0 else 0.0

        # Former PERMANENT threshold → STRONG (they still decay)
        if (
            attempts >= 50
            and hit_rate >= 0.40
        ):
            return AnchorType.STRONG
        if attempts >= MIN_ANCHOR_ATTEMPTS and hit_rate > STRONG_ANCHOR_HIT_RATE:
            return AnchorType.STRONG
        elif attempts >= MIN_ANCHOR_ATTEMPTS and hit_rate > MEDIUM_ANCHOR_HIT_RATE:
            return AnchorType.MEDIUM
        else:
            return AnchorType.WEAK

    # ═══════════════════════════════════════════════════════════
    # DECAY & EVICTION (QUERY-COUNT EXPONENTIAL DECAY)
    # ═══════════════════════════════════════════════════════════

    def decay_anchors_by_query_count(self, current_query_num: int) -> int:
        """Decay anchors using query-count exponential decay (V2 proven).

        No PERMANENT tier — all anchors decay:
        - STRONG anchors decay slowly (lambda=0.995)
        - MEDIUM anchors decay moderately (lambda=0.98)
        - WEAK anchors decay fast (lambda=0.95)
        - Only decay anchors with recent misses (Bug #3 fix)
        - Prune at DECAY_STRENGTH_FLOOR (0.05)

        Thread-safe.
        """
        to_remove = []

        with self.lock:
            for anchor_id, anchor in self.anchors.items():
                # Per consensus: decay ALL anchors every query.
                # Decay is constant downward pressure; hits/absorptions fight back.
                # Removing the misses==0 skip that prevented idle anchors from decaying.
                delta_q = current_query_num - anchor.last_access_query_num
                if delta_q <= 0:
                    continue

                # No PERMANENT skip — all anchors decay
                if anchor.anchor_type == AnchorType.STRONG:
                    lam = DECAY_LAMBDA_STRONG
                elif anchor.anchor_type == AnchorType.MEDIUM:
                    lam = DECAY_LAMBDA_MEDIUM
                else:
                    lam = DECAY_LAMBDA_WEAK

                decay_factor = lam ** delta_q
                anchor.strength *= decay_factor
                anchor.confidence = min(1.0, anchor.strength / 10.0)  # Keep in sync
                # NOTE: Do NOT reset last_access_query_num here.
                # Only create_anchor() and update_anchor() set it.
                # Resetting here defeats exponential decay (delta_q always 1).

                # Per Bug #4: prune at 0.05 with Markov cleanup
                if anchor.strength < DECAY_STRENGTH_FLOOR:
                    to_remove.append(anchor_id)

            # Per Bug #4: remove pruned anchors + clean Markov
            for aid in to_remove:
                del self.anchors[aid]
                self.markov.remove_anchor(aid)

        if to_remove:
            logger.info(
                f"[DECAY] Removed {len(to_remove)} decayed anchors "
                f"(query-count exponential decay)"
            )

        return len(to_remove)

    def decay_weak_anchors(self, decay_rate: float = 0.1):
        """Legacy wall-clock decay. Delegates to query-count decay."""
        self.decay_anchors_by_query_count(self.global_query_count)

    # ═══════════════════════════════════════════════════════════
    # QUERIES
    # ═══════════════════════════════════════════════════════════

    def get_strong_anchors(self) -> List[Anchor]:
        """Get all STRONG anchors (no PERMANENT tier — all decay)."""
        return [
            anchor
            for anchor in self.anchors.values()
            if anchor.anchor_type == AnchorType.STRONG
        ]

    def get_dead_anchor_id(self) -> Optional[int]:
        """Return anchor_id with confidence < eviction floor, for T2 cascade cleanup."""
        for aid, anchor in self.anchors.items():
            if anchor.confidence < CONFIDENCE_EVICT_FLOOR:
                return aid
        return None

    def get_confidence(self, anchor_id: int) -> Optional[float]:
        """Return anchor confidence for T2 eviction priority."""
        if anchor_id in self.anchors:
            return self.anchors[anchor_id].confidence
        return None

    def get_anchor_by_id(self, anchor_id: int) -> Optional[Anchor]:
        """Get specific anchor by ID."""
        return self.anchors.get(anchor_id)

    def _evict_anchor(self, anchor_id: int) -> None:
        """Remove an anchor entirely. Per Step 7: death floor eviction.
        
        Handles parent/child cleanup and Markov cleanup.
        Thread-safe: must be called within lock or from locked context.
        """
        if anchor_id not in self.anchors:
            return
        
        anchor = self.anchors[anchor_id]
        
        # Unlink from parent
        if anchor.parent_id is not None and anchor.parent_id in self.anchors:
            parent = self.anchors[anchor.parent_id]
            if anchor_id in parent.children_ids:
                parent.children_ids.remove(anchor_id)
        
        # Orphan children (they become roots)
        for child_id in anchor.children_ids:
            if child_id in self.anchors:
                self.anchors[child_id].parent_id = None
        
        # Note: Markov transitions are preserved (knowledge retained)
        # The transition matrix keeps stale entries but they'll have
        # zero weight since the source anchor no longer exists.
        
        del self.anchors[anchor_id]
        logger.debug(f"[STEP7] Evicted anchor #{anchor_id}")

    def get_anchor_stats(self) -> Dict:
        """Get comprehensive anchor statistics."""
        if not self.anchors:
            return {
                "total_anchors": 0,
                "active_predictions": 0,  # Per consensus: prediction tracking removed
                "anchor_types": {},
                "avg_strength": 0.0,
                "prediction_accuracy": 0.0,
                "avg_radius": 0.0,
            }

        type_counts = {"weak": 0, "medium": 0, "strong": 0, "permanent": 0}
        total_hits = 0
        total_predictions = 0
        total_strength = 0.0
        total_radius = 0.0

        for anchor in self.anchors.values():
            type_counts[anchor.anchor_type.value] = type_counts.get(anchor.anchor_type.value, 0) + 1
            total_hits += anchor.hits
            total_predictions += anchor.hits + anchor.misses
            total_strength += anchor.strength
            total_radius += anchor.radius

        n = len(self.anchors)
        prediction_accuracy = (
            total_hits / total_predictions * 100 if total_predictions > 0 else 0.0
        )

        return {
            "total_anchors": n,
            "active_predictions": 0,  # Per consensus: prediction tracking removed
            "anchor_types": type_counts,
            "avg_strength": total_strength / n,
            "max_strength": max(a.strength for a in self.anchors.values()),
            "prediction_accuracy": prediction_accuracy,
            "prediction_attempts": total_predictions,
            "total_hits": total_hits,
            "total_misses": total_predictions - total_hits,
            "avg_radius": total_radius / n,
            "epsilon": self._epsilon,
            "max_radius": self._max_radius,
            "prediction_threshold": "removed",  # Per consensus: prediction tracking removed
            "markov": self.markov.get_stats(),
            "option_a_hit_reward": ANCHOR_HIT_REWARD,
            "option_a_miss_penalty": ANCHOR_MISS_PENALTY,
            "option_a_parent_credit": ANCHOR_PARENT_CREDIT,
        }

    def get_anchor_graph(self) -> Dict:
        """Get anchor graph structure for visualization."""
        nodes = []
        edges = []

        for anchor in self.anchors.values():
            nodes.append(
                {
                    "id": anchor.id,
                    "label": anchor.query_text[:30],
                    "type": anchor.anchor_type.value,
                    "strength": anchor.strength,
                    "hits": anchor.hits,
                    "misses": anchor.misses,
                    "radius": anchor.radius,
                    "n": anchor.n,
                }
            )

            for child_id in anchor.children_ids:
                edges.append(
                    {"from": anchor.id, "to": child_id, "weight": anchor.strength}
                )

        return {"nodes": nodes, "edges": edges}

    # ═══════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine distance = 1 - cosine_similarity. In [0, 2] for unit vectors."""
        return 1.0 - self._cosine_similarity(v1, v2)

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTICS LOGGING (for QReCC measurement)
    # ═══════════════════════════════════════════════════════════

    def log_anchor_diagnostics(self, query_num: int, output_dir: str = "benchmark/results/v3"):
        """Log per-anchor diagnostics for measurement and ablation.

        Writes one JSONL line per call with:
        - Per-anchor: id, confidence, type, n (absorptions), radius, hits, misses
        - Per-anchor: centroid shift since last log (if available)
        - System-level: total anchors, Markov transition count, prediction count
        """
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "anchor_diagnostics.jsonl")

        anchor_data = []
        for aid, anchor in self.anchors.items():
            anchor_data.append({
                "id": aid,
                "confidence": float(anchor.confidence),
                "type": anchor.anchor_type.value,
                "n": anchor.n,
                "radius": float(anchor.radius),
                "hits": anchor.hits,
                "misses": anchor.misses,
                "last_access_query_num": anchor.last_access_query_num,
                "parent_id": anchor.parent_id,
            })

        markov_stats = self.markov.get_stats() if hasattr(self.markov, 'get_stats') else {}

        diagnostics = {
            "query_num": query_num,
            "timestamp": time.time(),
            "total_anchors": len(self.anchors),
            "global_query_count": self.global_query_count,
            "epsilon": float(self._epsilon),
            "max_radius": float(self._max_radius),
            "anchors": anchor_data,
            "markov": markov_stats,
            "active_predictions": 0,  # Per consensus: prediction tracking removed
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(diagnostics) + "\n")
