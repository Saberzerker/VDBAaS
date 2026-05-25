"""Small ranking-metric helpers for labeled benchmark runs.

Computes metrics at multiple k values (k=1,3,5,10) for SOTA comparison.
Most published results report nDCG@k and Recall@k at various k — we need
all of them to compare fairly with BEIR leaderboard, QReCC shared task, CAsT 2020.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


# Standard k values for SOTA comparison
MULTI_K = [1, 3, 5, 10]


def _compute_single_k(
    returned_ids: Sequence[str],
    expected_ids: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Compute retrieval metrics for one labeled query at a single k."""
    returned = list(returned_ids)[:k]
    expected = list(expected_ids)
    if not expected:
        return {}

    expected_set = set(expected)
    hits = [item_id in expected_set for item_id in returned]
    hit_count = sum(hits)

    recall = hit_count / len(expected_set)
    precision = hit_count / max(len(returned), 1)

    reciprocal_rank = 0.0
    for rank, item_id in enumerate(returned, start=1):
        if item_id in expected_set:
            reciprocal_rank = 1.0 / rank
            break

    dcg = 0.0
    for rank, is_relevant in enumerate(hits, start=1):
        if is_relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        f"ndcg_at_{k}": ndcg,
        f"mrr_at_{k}": reciprocal_rank,
        f"recall_at_{k}": recall,
        f"precision_at_{k}": precision,
    }


def compute_ranking_metrics(
    returned_ids: Iterable[str],
    expected_ids: Iterable[str],
    *,
    k: int = 5,
) -> dict[str, float] | None:
    """Compute retrieval metrics for one labeled query at multiple k values.

    Returns metrics at k=1,3,5,10 plus legacy keys at the given k for
    backward compatibility with existing benchmark code.
    """
    returned = [str(item_id) for item_id in returned_ids]
    expected = [str(item_id) for item_id in expected_ids]
    if not expected:
        return None

    # Compute at all standard k values
    result: dict[str, float] = {}
    for ki in MULTI_K:
        result.update(_compute_single_k(returned, expected, ki))

    # Legacy keys at the given k for backward compatibility
    single = _compute_single_k(returned, expected, k)
    result["recall_at_k"] = single.get(f"recall_at_{k}", 0.0)
    result["precision_at_k"] = single.get(f"precision_at_{k}", 0.0)
    result["mrr_at_k"] = single.get(f"mrr_at_{k}", 0.0)
    result["ndcg_at_k"] = single.get(f"ndcg_at_{k}", 0.0)
    result["success_at_1"] = result.get("ndcg_at_1", 0.0)

    return result


def summarize_query_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average retrieval metrics over labeled rows.

    Reports mean metrics at k=1,3,5,10 plus legacy k-specific keys.
    """
    labeled_rows = [row for row in rows if row.get("recall_at_k") is not None]
    summary: dict[str, float] = {
        "labeled_queries": float(len(labeled_rows)),
        "unlabeled_queries": float(max(len(rows) - len(labeled_rows), 0)),
        "labeled_query_fraction": (len(labeled_rows) / len(rows)) if rows else 0.0,
    }

    # Legacy metric names (backward compat)
    legacy_metric_names = [
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "success_at_1",
    ]

    # Multi-k metric names
    multi_k_metrics = []
    for ki in MULTI_K:
        multi_k_metrics.extend([
            f"ndcg_at_{ki}",
            f"mrr_at_{ki}",
            f"recall_at_{ki}",
            f"precision_at_{ki}",
        ])

    all_metrics = legacy_metric_names + multi_k_metrics

    for metric_name in all_metrics:
        values = [
            float(row[metric_name])
            for row in labeled_rows
            if row.get(metric_name) is not None
        ]
        if values:
            summary[f"mean_{metric_name}"] = sum(values) / len(values)

    return summary


def summarize_session_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate session-aware metrics when ordered session metadata is present."""
    session_rows = [row for row in rows if row.get("session_id") not in (None, "", "None")]
    if not session_rows:
        return {}

    local_sources = {"tier1_permanent", "tier2_dynamic"}
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in session_rows:
        trace_id = row.get("trace_id")
        session_id = row.get("session_id")
        key = str(trace_id if trace_id not in (None, "", "None") else session_id)
        sessions.setdefault(key, []).append(row)

    for session_key in sessions:
        sessions[session_key].sort(
            key=lambda row: (
                int(row.get("turn_id") or 0),
                int(row.get("order_index") or 0),
                int(row.get("query_num") or 0),
            )
        )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    session_local_hit_rates: list[float] = []
    session_cloud_hit_rates: list[float] = []
    session_lengths: list[float] = []
    session_recall: list[float] = []
    session_mrr: list[float] = []
    session_ndcg: list[float] = []

    turn1_rows: list[dict[str, Any]] = []
    later_turn_rows: list[dict[str, Any]] = []

    for session_rows_for_key in sessions.values():
        session_lengths.append(float(len(session_rows_for_key)))
        local_hits = sum(
            1 for row in session_rows_for_key if row.get("source") in local_sources
        )
        cloud_hits = sum(
            1 for row in session_rows_for_key if row.get("source") == "tier3_cloud"
        )
        session_local_hit_rates.append(local_hits / len(session_rows_for_key))
        session_cloud_hit_rates.append(cloud_hits / len(session_rows_for_key))

        labeled = [
            row for row in session_rows_for_key if row.get("recall_at_k") is not None
        ]
        if labeled:
            session_recall.append(
                _mean([float(row["recall_at_k"]) for row in labeled])
            )
            session_mrr.append(
                _mean([float(row["mrr_at_k"]) for row in labeled])
            )
            session_ndcg.append(
                _mean([float(row["ndcg_at_k"]) for row in labeled])
            )

        for row in session_rows_for_key:
            turn_id = int(row.get("turn_id") or 0)
            if turn_id <= 1:
                turn1_rows.append(row)
            else:
                later_turn_rows.append(row)

    def _local_hit_rate(rows_subset: list[dict[str, Any]]) -> float:
        if not rows_subset:
            return 0.0
        local_hits = sum(1 for row in rows_subset if row.get("source") in local_sources)
        return local_hits / len(rows_subset)

    summary: dict[str, float] = {
        "session_count": float(len(sessions)),
        "ordered_session_queries": float(len(session_rows)),
        "mean_turns_per_session": _mean(session_lengths),
        "mean_session_local_hit_rate": _mean(session_local_hit_rates),
        "mean_session_cloud_hit_rate": _mean(session_cloud_hit_rates),
        "turn1_local_hit_rate": _local_hit_rate(turn1_rows),
        "later_turn_local_hit_rate": _local_hit_rate(later_turn_rows),
    }

    if session_recall:
        summary["mean_session_recall_at_k"] = _mean(session_recall)
    if session_mrr:
        summary["mean_session_mrr_at_k"] = _mean(session_mrr)
    if session_ndcg:
        summary["mean_session_ndcg_at_k"] = _mean(session_ndcg)

    # Per-turn breakdown: metrics by turn number within sessions.
    # Critical for validating anchor-and-momentum claim: quality improves over turns.
    per_turn: dict[int, dict[str, list[float]]] = {}
    for session_rows_for_key in sessions.values():
        for row in session_rows_for_key:
            turn = int(row.get("turn_id") or row.get("order_index") or 0)
            if turn <= 0:
                continue
            per_turn.setdefault(turn, {
                "ndcg": [], "mrr": [], "recall": [],
                "local_hit": [], "prediction_hit": [],
            })
            if row.get("ndcg_at_k") is not None:
                per_turn[turn]["ndcg"].append(float(row["ndcg_at_k"]))
            if row.get("mrr_at_k") is not None:
                per_turn[turn]["mrr"].append(float(row["mrr_at_k"]))
            if row.get("recall_at_k") is not None:
                per_turn[turn]["recall"].append(float(row["recall_at_k"]))
            per_turn[turn]["local_hit"].append(
                1.0 if row.get("source") in local_sources else 0.0
            )
            per_turn[turn]["prediction_hit"].append(
                1.0 if row.get("prediction_hit") else 0.0
            )

    # Flatten per-turn into summary dict with turn number in key
    for turn in sorted(per_turn.keys()):
        prefix = f"turn{turn}"
        d = per_turn[turn]
        summary[f"{prefix}_n"] = float(len(d["local_hit"]))
        summary[f"{prefix}_local_hit_rate"] = _mean(d["local_hit"]) if d["local_hit"] else 0.0
        summary[f"{prefix}_prediction_hit_rate"] = _mean(d["prediction_hit"]) if d["prediction_hit"] else 0.0
        if d["ndcg"]:
            summary[f"{prefix}_ndcg"] = _mean(d["ndcg"])
        if d["mrr"]:
            summary[f"{prefix}_mrr"] = _mean(d["mrr"])
        if d["recall"]:
            summary[f"{prefix}_recall"] = _mean(d["recall"])

    return summary
