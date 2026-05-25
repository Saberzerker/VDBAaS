# src/markov_transitions.py
"""
Markov Transition Matrix with Semantic Prior — Phase 3

Per THEORY.md §Q2:
  P[A][B] = softmax(cosine_sim(centroid_A, centroid_B))   # semantic prior
  alpha_A = 1 / (1 + count_transitions_from_A)
  P[A][B] = (count[A][B] + alpha_A * P_prior[A][B]) / (count[A][*] + alpha_A)

The semantic prior bootstraps prediction before any transitions observed.
The alpha schedule starts at 1.0 (pure prior) and decays toward 0 as
transitions accumulate — the system starts ignorant and learns through
operation.

Based on:
  - Markov prefetch: Cortex (arXiv 2025) — first-order Markov for cache prefetch
  - Laplace smoothing: standard Bayesian interpolation
  - Semantic prior from anchor centroids: our contribution
  - 2nd-order prediction accuracy ~40%: Jansen et al. (JASIST 2009)

Compute: O(A^2) for prior init (A = number of anchors). Sparse for updates.
  Prediction: O(A) per query — scan row of current anchor.
  Update: O(1) per observed transition.

Failure modes:
  - < 3 anchors: prior dominates, predictions are centroid-nearest
  - Anchors all WEAK with n=1: centroids noisy, prior unreliable
  - Task switch: current anchor is wrong for 1-2 queries, self-corrects
    as new anchor forms and transitions accumulate

Author: Anonymous (Phase 3: Markov transitions)
Date: 2026-04-19
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Temperature for softmax in semantic prior
SOFTMAX_TEMPERATURE = 0.1

# Minimum alpha — never fully discard the prior
ALPHA_FLOOR = 0.05

# Number of top transitions to return for prefetch
TOP_K_PREDICTIONS = 5


class MarkovTransitionMatrix:
    """
    First-order Markov transition matrix over anchor IDs.

    Transition probabilities P[anchor_A -> anchor_B] are initialized from
    cosine similarity of anchor centroids (semantic prior) and updated
    with Laplace smoothing as transitions are observed.

    Thread-safe: all mutations go through self.lock.
    """

    def __init__(self):
        # Sparse count storage: _counts[src][dst] = observed transitions
        self._counts: Dict[int, Dict[int, int]] = {}

        # Semantic prior cache: _prior[src][dst] = P_prior(src -> dst)
        # Recomputed when anchors change
        self._prior: Dict[int, Dict[int, float]] = {}

        # Anchor centroids for prior computation
        self._centroids: Dict[int, np.ndarray] = {}

        # All known anchor IDs (for iteration)
        self._anchor_ids: set = set()

        # Last anchor assigned to a query (for transition recording)
        self._last_anchor_id: Optional[int] = None

        self.lock = threading.Lock()

        logger.info("[MARKOV] Initialized transition matrix (semantic prior + Laplace)")

    # ═══════════════════════════════════════════════════════════
    # ANCHOR REGISTRATION
    # ═══════════════════════════════════════════════════════════

    def register_anchor(self, anchor_id: int, centroid: np.ndarray):
        """Register a new anchor and recompute semantic prior.

        Called when an anchor is created. Centroid stored for prior computation.
        Prior is recomputed lazily on next prediction if multiple anchors
        are registered in batch.
        """
        with self.lock:
            self._anchor_ids.add(anchor_id)
            self._centroids[anchor_id] = centroid.copy()
            # Initialize count row
            if anchor_id not in self._counts:
                self._counts[anchor_id] = {}

        logger.debug(f"[MARKOV] Registered anchor #{anchor_id}")

    def update_anchor_centroid(self, anchor_id: int, centroid: np.ndarray):
        """Update centroid after anchor absorption. Invalidates all affected priors.

        Per Bug #7 fix: when centroid changes, ALL prior rows that include
        this anchor as a destination need recomputation. Since prior[src][dst]
        depends on ALL centroids, changing one centroid invalidates all rows.
        Simplest correct approach: clear entire prior cache.
        """
        with self.lock:
            if anchor_id in self._anchor_ids:
                self._centroids[anchor_id] = centroid.copy()
                # Invalidate all priors — any row referencing this centroid is stale
                self._prior.clear()

    def remove_anchor(self, anchor_id: int):
        """Remove anchor and all its transition data."""
        with self.lock:
            self._anchor_ids.discard(anchor_id)
            self._centroids.pop(anchor_id, None)
            self._counts.pop(anchor_id, None)
            self._prior.pop(anchor_id, None)
            # Remove from all count destinations
            for src in self._counts:
                self._counts[src].pop(anchor_id, None)
            if self._last_anchor_id == anchor_id:
                self._last_anchor_id = None

    # ═══════════════════════════════════════════════════════════
    # TRANSITION RECORDING
    # ═══════════════════════════════════════════════════════════

    def record_transition(self, from_anchor: int, to_anchor: int, weight: float = 1.0):
        """Record observed transition from_anchor -> to_anchor.

        Per co-evolutionary loop: transitions are weighted by source anchor
        confidence. High-confidence anchors contribute more to transition
        probabilities than low-confidence anchors. This prevents noisy
        transitions from weak anchors from polluting the prediction model.

        Args:
            from_anchor: Source anchor ID
            to_anchor: Destination anchor ID
            weight: Confidence weight (default 1.0 for backward compat).
                Typically set to source anchor's confidence [0.01, 1.0].
        """
        if from_anchor == to_anchor:
            # Self-transitions are not informative for prediction
            return

        with self.lock:
            if from_anchor not in self._counts:
                self._counts[from_anchor] = {}
            self._counts[from_anchor][to_anchor] = (
                self._counts[from_anchor].get(to_anchor, 0.0) + weight
            )

        logger.debug(
            f"[MARKOV] Recorded transition: #{from_anchor} -> #{to_anchor} "
            f"(count={self._counts[from_anchor][to_anchor]})"
        )

    def set_last_anchor(self, anchor_id: Optional[int]):
        """Track which anchor was assigned to the last query.

        Called by anchor_system on anchor assignment. Used to detect
        transitions (old_anchor -> new_anchor).
        """
        with self.lock:
            self._last_anchor_id = anchor_id

    def get_last_anchor(self) -> Optional[int]:
        """Return the anchor assigned to the previous query."""
        with self.lock:
            return self._last_anchor_id

    # ═══════════════════════════════════════════════════════════
    # SEMANTIC PRIOR COMPUTATION
    # ═══════════════════════════════════════════════════════════

    def _compute_prior_row(self, src_id: int) -> Dict[int, float]:
        """Compute semantic prior P[src -> *] from centroid similarities.

        Per THEORY.md §Q2:
          P_prior[src][dst] = softmax(cosine_sim(centroid_src, centroid_dst) / temperature)

        Returns dict mapping dst_id -> prior probability.
        """
        if src_id not in self._centroids:
            return {}

        src_centroid = self._centroids[src_id]
        similarities = {}

        for dst_id in self._anchor_ids:
            if dst_id == src_id:
                continue
            if dst_id not in self._centroids:
                continue
            # cosine similarity (centroids are unit-normalized)
            sim = float(np.dot(src_centroid, self._centroids[dst_id]))
            similarities[dst_id] = sim / SOFTMAX_TEMPERATURE

        if not similarities:
            return {}

        # Softmax: exp(x_i) / sum(exp(x_j))
        max_sim = max(similarities.values())
        exp_sims = {k: np.exp(v - max_sim) for k, v in similarities.items()}
        total = sum(exp_sims.values())

        if total < 1e-12:
            # Uniform fallback
            n = len(exp_sims)
            return {k: 1.0 / n for k in exp_sims}

        return {k: v / total for k, v in exp_sims.items()}

    def _ensure_prior(self, src_id: int):
        """Ensure prior row exists for src_id, computing if stale."""
        if src_id not in self._prior:
            self._prior[src_id] = self._compute_prior_row(src_id)

    # ═══════════════════════════════════════════════════════════
    # PROBABILITY COMPUTATION
    # ═══════════════════════════════════════════════════════════

    def get_transition_probs(self, from_anchor: int) -> Dict[int, float]:
        """Get P[from_anchor -> *] with Laplace smoothing.

        Per THEORY.md §Q2:
          alpha = 1 / (1 + total_count_from_src)
          P[A][B] = (count[A][B] + alpha * P_prior[A][B]) / (total_count + alpha)

        Returns dict mapping anchor_id -> probability, sorted descending.
        """
        with self.lock:
            self._ensure_prior(from_anchor)

            count_row = self._counts.get(from_anchor, {})
            prior_row = self._prior.get(from_anchor, {})

            if not prior_row:
                # No other anchors known — no predictions possible
                return {}

            total_count = sum(count_row.values())
            alpha = max(1.0 / (1.0 + total_count), ALPHA_FLOOR)

            probs = {}
            for dst_id in prior_row:
                observed = count_row.get(dst_id, 0)
                p_prior = prior_row[dst_id]
                # Laplace-smoothed probability
                probs[dst_id] = (observed + alpha * p_prior) / (total_count + alpha)

            # Normalize to sum to 1.0
            total_p = sum(probs.values())
            if total_p > 0:
                probs = {k: v / total_p for k, v in probs.items()}

            # Sort descending by probability
            return dict(sorted(probs.items(), key=lambda x: -x[1]))

    # ═══════════════════════════════════════════════════════════
    # PREDICTION INTERFACE
    # ═══════════════════════════════════════════════════════════

    def predict_next_anchors(
        self, current_anchor: int, top_k: int = TOP_K_PREDICTIONS
    ) -> List[Tuple[int, float]]:
        """Predict top-k most likely next anchors from current anchor.

        Returns list of (anchor_id, probability) tuples, sorted by probability.
        Uses Laplace-smoothed transition matrix.

        Per Jansen et al. (2009): 2nd-order context gives ~40% prediction.
        We use 1st-order Markov (last anchor only) as the base case.
        """
        probs = self.get_transition_probs(current_anchor)

        if not probs:
            logger.debug(
                f"[MARKOV] No predictions available from anchor #{current_anchor}"
            )
            return []

        predictions = list(probs.items())[:top_k]

        logger.info(
            f"[MARKOV] Top-{len(predictions)} predictions from #{current_anchor}: "
            + ", ".join(f"#{a}({p:.3f})" for a, p in predictions)
        )

        return predictions

    def predict_next_vectors(
        self, current_anchor: int, top_k: int = TOP_K_PREDICTIONS
    ) -> List[Tuple[np.ndarray, float]]:
        """Predict top-k most likely next query vectors.

        Returns list of (centroid_vector, probability) tuples.
        The centroid of the predicted anchor is the expected next query region.
        This is the prefetch target — fetch vectors near these centroids.
        """
        anchor_probs = self.predict_next_anchors(current_anchor, top_k)

        results = []
        with self.lock:
            for anchor_id, prob in anchor_probs:
                if anchor_id in self._centroids:
                    results.append((self._centroids[anchor_id].copy(), prob))

        return results

    # ═══════════════════════════════════════════════════════════
    # BATCH PRIOR RECOMPUTATION
    # ═══════════════════════════════════════════════════════════

    def recompute_all_priors(self):
        """Recompute all prior rows from current centroids.

        Called after multiple anchors are registered/updated in batch.
        Ensures prior reflects latest centroid positions.
        """
        with self.lock:
            self._prior.clear()
            for anchor_id in self._anchor_ids:
                self._prior[anchor_id] = self._compute_prior_row(anchor_id)

        logger.info(
            f"[MARKOV] Recomputed priors for {len(self._anchor_ids)} anchors"
        )

    # ═══════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return transition matrix statistics."""
        with self.lock:
            total_transitions = sum(
                sum(dst.values()) for dst in self._counts.values()
            )
            n_sources = len(self._counts)
            n_pairs = sum(len(dst) for dst in self._counts.values())

            # Sparsity
            max_pairs = len(self._anchor_ids) * max(len(self._anchor_ids) - 1, 0)
            sparsity = 1.0 - (n_pairs / max(max_pairs, 1))

            return {
                "n_anchors": len(self._anchor_ids),
                "total_transitions": total_transitions,
                "n_source_anchors": n_sources,
                "n_transition_pairs": n_pairs,
                "sparsity": sparsity,
                "has_prior": bool(self._prior),
            }
