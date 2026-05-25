# src/reranker.py
"""
Hybrid Re-Ranker Gate — BM25+cosine hybrid with adaptive threshold.

Phase 5: Replaces cross-encoder gate with a lightweight BM25+cosine hybrid.
The cross-encoder (MS-MARCO-MiniLM-L6-v2) rejected 96.7% of local results
on NFCorpus due to domain mismatch (medical literature vs web passages).
It also added 90ms latency to the local path, destroying the latency advantage.

New gate architecture:
  gate_signal = alpha * cosine_similarity + (1 - alpha) * normalize(bm25_score)
  If gate_signal >= adaptive_threshold: serve locally (< 5ms total)
  Else: fall through to cloud

The adaptive threshold tracks precision over a sliding window of 50 queries,
self-correcting: if local precision drops below target (0.70), threshold rises;
if local precision exceeds target, threshold relaxes.

Based on: BM25 [Robertson & Zaragoza, 2009]; hybrid fusion
[Shen et al., 2023]; adaptive threshold via precision window.

Retired components (kept for reference, not in local path):
  - OnlinePlattCalibrator: was for cross-encoder calibration. Retired because
    the cross-encoder is no longer in the local path. Platt has no input signal
    without the cross-encoder.
  - rerank(), rerank_with_threshold(): cross-encoder functions. Kept for
    potential cloud-side re-ranking but NOT called in the local path.

Author: Saberzerker
Date: 2026-04-23 (Phase 5: BM25+cosine gate replacement)
"""

import math
import logging
import os
import re
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# RETIRED: Cross-encoder infrastructure (kept for reference, not in local path)
# ═══════════════════════════════════════════════════════════════════════════════

_CROSS_ENCODER = None
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
_cross_encoder_lock = threading.Lock()


class OnlinePlattCalibrator:
    """RETIRED: Self-calibrating sigmoid for cross-encoder logits.

    Kept for reference. No longer called in the local path because the
    cross-encoder is removed from the local serving path (Phase 5).
    Platt's input signal was cross-encoder raw scores; without the
    cross-encoder, Platt has no input. The adaptive threshold with
    precision window is the new self-correcting mechanism.

    Based on: Platt (1999). "Probabilistic Outputs for SVMs."
    """

    def __init__(
        self,
        a: float = 0.5,
        b: float = -3.0,
        window_size: int = 50,
        refit_every: int = 5,
        min_updates_before_active: int = 20,
    ):
        self.a = a
        self.b = b
        self.window_size = window_size
        self.refit_every = refit_every
        self.min_updates_before_active = min_updates_before_active
        self.n_updates = 0
        self._query_count = 0
        self._window: deque = deque(maxlen=window_size)

    def calibrate(self, raw_score: float) -> float:
        if self.n_updates < self.min_updates_before_active:
            return raw_score
        return 1.0 / (1.0 + math.exp(-(self.a * raw_score + self.b)))

    def update(self, raw_score: float, label: float) -> None:
        self._window.append((float(raw_score), float(label)))
        self.n_updates += 1

    def end_query(self) -> None:
        self._query_count += 1

    def is_calibrated(self) -> bool:
        return self.n_updates >= self.min_updates_before_active

    def status(self) -> Dict:
        return {
            "a": self.a,
            "b": self.b,
            "n_updates": self.n_updates,
            "query_count": self._query_count,
            "window_size": len(self._window),
            "calibrated": self.is_calibrated(),
            "retired": True,
        }


_calibrator = OnlinePlattCalibrator()


def get_calibrator() -> OnlinePlattCalibrator:
    """Access the global calibrator (retired, kept for diagnostics)."""
    return _calibrator


def _get_cross_encoder():
    """RETIRED: Load cross-encoder model. Not called in local path."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        with _cross_encoder_lock:
            if _CROSS_ENCODER is None:
                from sentence_transformers import CrossEncoder

                logger.info(f"[RERANKER] Loading cross-encoder: {_RERANKER_MODEL}")
                _CROSS_ENCODER = CrossEncoder(_RERANKER_MODEL, max_length=512)
                logger.info("[RERANKER] Cross-encoder loaded")
    return _CROSS_ENCODER


def rerank(
    query_text: str,
    doc_texts: List[str],
    doc_ids: List[str],
    top_k: int = 10,
) -> List[Dict]:
    """RETIRED: Cross-encoder reranking. Not called in local path."""
    if not doc_texts:
        return []

    model = _get_cross_encoder()
    pairs = [(query_text, doc_text) for doc_text in doc_texts]
    raw_scores = model.predict(pairs, show_progress_bar=False)

    ranked = []
    for i, raw_score in enumerate(raw_scores):
        calibrated = _calibrator.calibrate(float(raw_score))
        ranked.append(
            {
                "id": doc_ids[i],
                "score": calibrated,
                "raw_score": float(raw_score),
                "text": doc_texts[i],
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def rerank_with_threshold(
    query_text: str,
    doc_texts: List[str],
    doc_ids: str,
    relevance_threshold: float = 0.5,
    top_k: int = 10,
) -> Tuple[List[Dict], float, List[Dict]]:
    """RETIRED: Cross-encoder reranking with threshold. Not called in local path."""
    ranked = rerank(query_text, doc_texts, doc_ids, top_k=top_k)

    if not ranked:
        return [], -10.0, []

    best_raw = ranked[0]["raw_score"]
    best_calibrated = ranked[0]["score"]

    GAP_THRESHOLD = 0.15
    RAW_CONFIDENCE_FLOOR = 3.0

    if len(ranked) >= 2 and best_raw < RAW_CONFIDENCE_FLOOR:
        gap = ranked[0]["raw_score"] - ranked[1]["raw_score"]
        if gap < GAP_THRESHOLD:
            return [], best_raw, ranked

    relevant = [r for r in ranked if r["raw_score"] >= relevance_threshold]
    return relevant, best_calibrated, ranked


def feed_calibration_signal(
    local_raw_scores: List[float],
    local_ids: List[str],
    cloud_ids: Optional[List[str]] = None,
    is_reformulation: bool = False,
) -> None:
    """RETIRED: Feed calibration signals. Not called in local path."""
    cal = _calibrator

    if cloud_ids is not None:
        cloud_set = set(str(cid) for cid in cloud_ids[:5])
        for raw_score, lid in zip(local_raw_scores, local_ids):
            label = 1.0 if str(lid) in cloud_set else 0.0
            cal.update(raw_score, label)

    elif is_reformulation and local_raw_scores:
        for raw_score in local_raw_scores[:3]:
            cal.update(raw_score, label=0.0)

    cal.end_query()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: BM25+cosine hybrid gate with adaptive threshold
# ═══════════════════════════════════════════════════════════════════════════════

# BM25 parameters — standard Robertson-Zaragoza defaults
BM25_K1 = 1.5   # Term frequency saturation
BM25_B = 0.75    # Length normalization


class BM25Scorer:
    """Okapi BM25 scorer for local document texts.

    Based on: Robertson & Zaragoza (2009). "The Probabilistic Relevance
    Framework: BM25 and Beyond." Foundations and Trends in IR.

    Computes BM25 scores for query-document pairs using pre-built
    inverted index. Designed for small local collections (Tier 1 + Tier 2
    typically < 2000 docs), so index construction is fast.

    Compute: O(|Q| * df_i * log(N/df_i)) per query, negligible for
    local collections. No GPU required.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> document frequency
        self.doc_lens: Dict[str, int] = {}   # doc_id -> document length
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = {}  # term -> [(doc_id, tf)]
        self._doc_texts: Dict[str, str] = {}  # doc_id -> raw text

    def index_corpus(self, doc_texts: Dict[str, str]) -> None:
        """Build BM25 index from corpus texts.

        Args:
            doc_texts: mapping from doc_id to document text.
        """
        self._doc_texts = doc_texts
        self.corpus_size = len(doc_texts)
        self.doc_freqs = {}
        self.doc_lens = {}
        self.inverted_index = {}

        total_len = 0
        for doc_id, text in doc_texts.items():
            tokens = self._tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)

            # Term frequency per document
            tf_map: Dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for term, tf in tf_map.items():
                # Document frequency
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                # Inverted index: term -> [(doc_id, tf)]
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_id, tf))

        self.avgdl = total_len / max(self.corpus_size, 1)
        logger.info(
            f"[BM25] Indexed {self.corpus_size} docs, "
            f"avgdl={self.avgdl:.1f}, vocab={len(self.doc_freqs)}"
        )

    def score(self, query_text: str, doc_ids: List[str]) -> Dict[str, float]:
        """Compute BM25 scores for query against specified documents.

        Args:
            query_text: query string.
            doc_ids: list of document IDs to score.

        Returns:
            Dict mapping doc_id -> BM25 score. Missing doc_ids get score 0.0.
        """
        if not doc_ids or self.corpus_size == 0:
            return {did: 0.0 for did in doc_ids}

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return {did: 0.0 for did in doc_ids}

        scores: Dict[str, float] = {did: 0.0 for did in doc_ids}
        doc_id_set = set(doc_ids)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            df = self.doc_freqs.get(term, 0)
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in self.inverted_index[term]:
                if doc_id not in doc_id_set:
                    continue

                dl = self.doc_lens.get(doc_id, self.avgdl)
                # BM25 TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                )
                scores[doc_id] += idf * tf_norm

        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenizer.

        For medical/scientific text, this is sufficient because BM25
        handles term frequency saturation. No stemming needed for the
        gate — we only need rough relevance, not perfect ranking.
        """
        # Lowercase, strip punctuation, split on whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens (< 2 chars) and very long tokens (> 30 chars)
        return [t for t in tokens if 2 <= len(t) <= 30]


class AdaptiveThreshold:
    """Self-correcting gate threshold based on precision window.

    Tracks whether locally-served results were actually relevant (by
    comparing with cloud top-k). If local precision drops below target,
    threshold rises to be more selective. If precision is high, threshold
    relaxes to serve more locally.

    Based on: adaptive threshold via precision window (THEORY.md §20).
    The window size (50 queries) balances responsiveness and stability.

    Failure modes:
    - If cloud_ids are never provided, precision is unknown and threshold
      stays at initial value. This is correct — without ground truth,
      we should not adapt.
    - If all local results are always relevant, threshold relaxes to
      minimum (GATE_MIN_THRESHOLD). This is correct — no need to be
      selective when everything is relevant.
    """

    def __init__(
        self,
        initial_threshold: float = 0.70,
        target_precision: float = 0.70,
        window_size: int = 50,
        step_up: float = 0.02,
        step_down: float = 0.01,
        min_threshold: float = 0.50,
        max_threshold: float = 0.95,
    ):
        self.threshold = initial_threshold
        self.target_precision = target_precision
        self.window_size = window_size
        self.step_up = step_up      # Precision below target → raise threshold
        self.step_down = step_down   # Precision above target → lower threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        # Sliding window of (gate_signal, was_relevant) for precision tracking
        self._window: deque = deque(maxlen=window_size)
        self._query_count = 0

    def record_outcome(self, gate_signal: float, was_relevant: bool) -> None:
        """Record a local serving outcome for precision tracking.

        Args:
            gate_signal: the hybrid gate signal that was used for the decision.
            was_relevant: True if the locally-served result was in cloud top-k.
        """
        self._window.append((gate_signal, was_relevant))
        self._query_count += 1

        # Adapt threshold every 10 queries for stability
        if self._query_count % 10 == 0 and len(self._window) >= 10:
            self._adapt()

    def _adapt(self) -> None:
        """Adjust threshold based on recent precision.

        Per pre-flight P1: When ADAPTIVE_GATE=False (config), this is a no-op.
        Threshold stays at GATE_INITIAL_THRESHOLD for clean ablation.
        record_outcome() still runs (precision logged for diagnostics).
        """
        from src.config import ADAPTIVE_GATE
        if not ADAPTIVE_GATE:
            return  # Static threshold mode for benchmark ablation

        relevant_count = sum(1 for _, was_rel in self._window if was_rel)
        precision = relevant_count / len(self._window) if self._window else 0.0

        if precision < self.target_precision:
            # Too many false positives — raise threshold
            self.threshold = min(self.threshold + self.step_up, self.max_threshold)
            logger.info(
                f"[GATE] Precision {precision:.3f} < target {self.target_precision:.3f}, "
                f"raising threshold to {self.threshold:.3f}"
            )
        elif precision > self.target_precision + 0.05:
            # Good precision — relax threshold to serve more locally
            self.threshold = max(self.threshold - self.step_down, self.min_threshold)
            logger.info(
                f"[GATE] Precision {precision:.3f} > target+0.05, "
                f"relaxing threshold to {self.threshold:.3f}"
            )

    def status(self) -> Dict:
        relevant_count = sum(1 for _, was_rel in self._window if was_rel)
        precision = relevant_count / len(self._window) if self._window else 0.0
        return {
            "threshold": self.threshold,
            "precision_window": len(self._window),
            "recent_precision": precision,
            "target_precision": self.target_precision,
            "query_count": self._query_count,
        }


class AnchorAwareThreshold:
    """Per-anchor adaptive thresholds with global fallback.

    Each anchor (semantic basin) gets its own AdaptiveThreshold that adapts
    based on local outcomes for queries belonging to that anchor. This allows
    well-established anchors (high confidence, many hits) to lower their
    thresholds (serve more locally from that basin), while uncertain anchors
    keep higher thresholds (route to cloud until proven).

    When a query doesn't belong to any anchor, the global threshold is used.

    Based on: Per-topic adaptive thresholds proven effective in session
    retrieval [Voskarides et al., 2020]; anchor basins as topic proxies
    per THEORY.md §Q1.

    Failure modes:
    - New anchors start at global threshold (safe default).
    - Anchor thresholds are clamped to [global_min, global_max] to prevent
      drift beyond reasonable bounds.
    - If an anchor has < MIN_ANCHOR_OUTCOMES outcomes, its threshold
      reverts to global (insufficient data).
    """

    # Minimum outcomes before an anchor gets its own threshold
    MIN_ANCHOR_OUTCOMES = 10

    def __init__(
        self,
        global_threshold: AdaptiveThreshold,
    ):
        self._global = global_threshold
        self._anchor_thresholds: Dict[int, AdaptiveThreshold] = {}
        self._lock = threading.Lock()

    def get_threshold(self, anchor_id: Optional[int] = None) -> float:
        """Get the effective threshold for a query.

        If anchor_id is provided and the anchor has enough outcomes,
        return the anchor-specific threshold. Otherwise, return global.

        Per anchor policy / Tier 2 bounded admission: anchors with high
        precision (well-established basins) get lower thresholds, allowing
        more local serving. Anchors with low precision get higher thresholds,
        routing uncertain queries to cloud.
        """
        if anchor_id is None:
            return self._global.threshold

        with self._lock:
            anchor_thr = self._anchor_thresholds.get(anchor_id)
            if anchor_thr is None:
                return self._global.threshold
            # Anchor must have enough outcomes to trust its own threshold
            if anchor_thr._query_count < self.MIN_ANCHOR_OUTCOMES:
                return self._global.threshold
            return anchor_thr.threshold

    def record_outcome(
        self,
        gate_signal: float,
        was_relevant: bool,
        anchor_id: Optional[int] = None,
    ) -> None:
        """Record outcome to both global and anchor-specific thresholds.

        Global always gets the outcome (for baseline calibration).
        Anchor-specific also gets it (for per-basin adaptation).

        Per anchor policy / per-anchor adaptive thresholds: each basin
        learns its own precision characteristics independently.
        """
        # Always record to global threshold
        self._global.record_outcome(gate_signal, was_relevant)

        # Also record to anchor-specific threshold if anchor is known
        if anchor_id is not None:
            with self._lock:
                if anchor_id not in self._anchor_thresholds:
                    # New anchor: start from current global threshold
                    # Per anchor policy: inherit global as starting point
                    self._anchor_thresholds[anchor_id] = AdaptiveThreshold(
                        initial_threshold=self._global.threshold,
                        target_precision=self._global.target_precision,
                        window_size=self._global.window_size,
                        step_up=self._global.step_up,
                        step_down=self._global.step_down,
                        min_threshold=self._global.min_threshold,
                        max_threshold=self._global.max_threshold,
                    )
                self._anchor_thresholds[anchor_id].record_outcome(gate_signal, was_relevant)

    def status(self) -> Dict:
        """Return status of global and per-anchor thresholds."""
        anchor_status = {}
        with self._lock:
            for aid, thr in self._anchor_thresholds.items():
                anchor_status[str(aid)] = thr.status()
        return {
            "global": self._global.status(),
            "anchor_thresholds": anchor_status,
            "n_anchors_with_thresholds": len(self._anchor_thresholds),
        }


class HybridGate:
    """BM25+cosine hybrid gate for local serving decisions.

    Replaces the cross-encoder gate (Phase 5). The hybrid gate combines:
    - Cosine similarity from the bi-encoder (E5-base-v2) — already computed
      during FAISS search, zero additional cost.
    - BM25 lexical score — computed from pre-built inverted index, ~0.1ms
      for 10 documents.

    gate_signal = alpha * cosine_similarity + (1 - alpha) * normalize(bm25_score)

    If gate_signal >= adaptive_threshold: serve locally.
    Else: fall through to cloud.

    Step 4: Per-anchor adaptive thresholds. Each anchor (semantic basin) gets
    its own AdaptiveThreshold. Well-established anchors lower their thresholds
    (serve more locally), uncertain anchors keep higher thresholds (route to
    cloud). Falls back to global threshold for queries without an anchor.

    Based on: BM25 [Robertson & Zaragoza, 2009]; hybrid fusion proven
    effective in BEIR [Thakur et al., 2021] where sparse+dense > either alone.
    Per-topic adaptive thresholds per Voskarides et al. (2020).

    Compute: ~0.1ms per query (BM25 scoring for 10 docs). No GPU required.
    Expected output shape: scalar gate_signal in [0, 1].
    Known failure modes:
    - BM25 returns 0 for queries with no token overlap (e.g., paraphrases).
      alpha=0.7 means cosine still dominates, so this is safe.
    - Very short documents may get inflated BM25 scores. Length normalization
      (b=0.75) mitigates this.
    """

    def __init__(
        self,
        alpha: float = None,
        initial_threshold: float = None,
        target_precision: float = None,
        precision_window: int = None,
    ):
        # Import here to avoid circular imports at module level
        from src.config import (
            GATE_ALPHA, GATE_INITIAL_THRESHOLD, GATE_TARGET_PRECISION,
            GATE_PRECISION_WINDOW, GATE_STEP_UP, GATE_STEP_DOWN,
            GATE_MIN_THRESHOLD, GATE_MAX_THRESHOLD,
        )
        self.alpha = alpha if alpha is not None else GATE_ALPHA
        self.bm25 = BM25Scorer()
        self.adaptive_threshold = AdaptiveThreshold(
            initial_threshold=initial_threshold if initial_threshold is not None else GATE_INITIAL_THRESHOLD,
            target_precision=target_precision if target_precision is not None else GATE_TARGET_PRECISION,
            window_size=precision_window if precision_window is not None else GATE_PRECISION_WINDOW,
            step_up=GATE_STEP_UP,
            step_down=GATE_STEP_DOWN,
            min_threshold=GATE_MIN_THRESHOLD,
            max_threshold=GATE_MAX_THRESHOLD,
        )
        # Per anchor policy / Step 4: anchor-aware thresholds
        self.anchor_aware_threshold = AnchorAwareThreshold(self.adaptive_threshold)
        self._indexed = False

    def index_corpus(self, doc_texts: Dict[str, str]) -> None:
        """Build BM25 index from corpus texts."""
        self.bm25.index_corpus(doc_texts)
        self._indexed = True

    def gate(
        self,
        query_text: str,
        doc_ids: List[str],
        cosine_scores: List[float],
        anchor_id: Optional[int] = None,
    ) -> Tuple[float, bool, Dict]:
        """Evaluate the hybrid gate for a local serving decision.

        Args:
            query_text: query string for BM25 scoring.
            doc_ids: document IDs (must be in BM25 index).
            cosine_scores: cosine similarity scores from FAISS (same order as doc_ids).
            anchor_id: optional anchor ID for per-anchor threshold lookup.
                If provided and the anchor has enough outcomes, uses the
                anchor-specific threshold instead of the global one.

        Returns:
            (gate_signal, should_serve_locally, diagnostics_dict)

        The gate_signal is the weighted combination:
            gate_signal = alpha * max(cosine_scores) + (1 - alpha) * normalize(max_bm25)

        should_serve_locally is True if gate_signal >= threshold.
        Threshold is anchor-specific if anchor_id is provided and has enough
        outcomes; otherwise falls back to global threshold.

        Diagnostics dict contains: gate_signal, best_cosine, best_bm25,
        alpha, threshold, gate_type, anchor_id.
        """
        if not doc_ids or not cosine_scores:
            return 0.0, False, {"gate_type": "empty", "reason": "no_local_results"}

        # Best cosine similarity (already computed during FAISS search)
        best_cosine = max(cosine_scores)

        # BM25 scoring for the same documents
        bm25_scores = self.bm25.score(query_text, doc_ids)
        best_bm25_raw = max(bm25_scores.values()) if bm25_scores else 0.0

        # Normalize BM25 to [0, 1] range
        # BM25 scores are unbounded; we use sigmoid normalization which is
        # monotonic and maps any real number to (0, 1).
        # Per BM25 normalization: sigmoid(0) = 0.5, sigmoid(5) ≈ 0.99.
        # This ensures BM25 contributes meaningfully even for low-scoring docs.
        best_bm25_norm = 1.0 / (1.0 + math.exp(-best_bm25_raw / 5.0))

        # Hybrid gate signal: alpha * cosine + (1 - alpha) * bm25_norm
        gate_signal = self.alpha * best_cosine + (1 - self.alpha) * best_bm25_norm

        # Per anchor policy / Step 4: use anchor-specific threshold if available
        threshold = self.anchor_aware_threshold.get_threshold(anchor_id)

        # Adaptive threshold decision
        should_serve = gate_signal >= threshold

        diagnostics = {
            "gate_signal": gate_signal,
            "best_cosine": best_cosine,
            "best_bm25_raw": best_bm25_raw,
            "best_bm25_norm": best_bm25_norm,
            "alpha": self.alpha,
            "threshold": threshold,
            "gate_type": "bm25_cosine_hybrid",
            "anchor_id": anchor_id,
        }

        return gate_signal, should_serve, diagnostics

    def record_outcome(
        self,
        gate_signal: float,
        was_relevant: bool,
        anchor_id: Optional[int] = None,
    ) -> None:
        """Record a local serving outcome for adaptive threshold tracking.

        Records to both global and anchor-specific thresholds.

        Args:
            gate_signal: the gate signal that was used for the decision.
            was_relevant: True if the locally-served result was in cloud top-k.
            anchor_id: optional anchor ID for per-anchor threshold adaptation.
        """
        # Per anchor policy / Step 4: record to both global and anchor-specific
        self.anchor_aware_threshold.record_outcome(gate_signal, was_relevant, anchor_id)

    def status(self) -> Dict:
        return {
            "alpha": self.alpha,
            "bm25_indexed": self._indexed,
            "corpus_size": self.bm25.corpus_size,
            "adaptive_threshold": self.adaptive_threshold.status(),
            "anchor_aware_threshold": self.anchor_aware_threshold.status(),
        }


# Global hybrid gate instance
_hybrid_gate = HybridGate()


def get_hybrid_gate() -> HybridGate:
    """Access the global hybrid gate for routing decisions."""
    return _hybrid_gate


def status() -> Dict:
    """Get status of all reranker components (for diagnostics)."""
    return {
        "hybrid_gate": _hybrid_gate.status(),
        "platt_calibrator": _calibrator.status(),
        "cross_encoder_loaded": _CROSS_ENCODER is not None,
    }