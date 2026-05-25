"""
Multi-pass benchmark: runs the same query manifest N times within a single
process so anchor state and Tier 2 cache persist across passes.

This proves the anchor-learning claim: quality should improve pass-over-pass
as anchors learn to predict relevant basins.

Usage:
    python scripts/multipass_benchmark.py \
        --manifest benchmark/workloads/qrecc/session_manifest_smoke.jsonl \
        --passes 3 --gate-threshold 0.88 --variant reactive_cache \
        --output-name qrecc_reactive_multipass

Based on: anchor-driven prefetch learning, per §19 CONTEXT.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch config BEFORE importing src modules
import src.config as cfg_module


def patch_gate(threshold: float):
    """Patch gate threshold before any src imports cascade."""
    cfg_module.config.GATE_INITIAL_THRESHOLD = threshold
    cfg_module.GATE_INITIAL_THRESHOLD = threshold


def compute_retrieval_metrics(results_with_labels):
    """Compute nDCG@5, MRR@5, Recall@5 from (predicted_ids, expected_ids) pairs."""
    ndcgs, mrrs, recalls = [], [], []
    for predicted, expected in results_with_labels:
        if not expected:
            continue
        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(predicted[:5]):
            if doc_id in expected:
                dcg += 1.0 / np.log2(rank + 2)
        # IDCG
        n_rel = min(len(expected), 5)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(n_rel))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(predicted[:5]):
            if doc_id in expected:
                mrr = 1.0 / (rank + 1)
                break
        mrrs.append(mrr)
        # Recall
        hits = sum(1 for d in predicted[:5] if d in expected)
        recalls.append(hits / len(expected) if expected else 0.0)

    return {
        "nDCG@5": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "MRR@5": float(np.mean(mrrs)) if mrrs else 0.0,
        "Recall@5": float(np.mean(recalls)) if recalls else 0.0,
        "n_labeled": len(ndcgs),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-pass benchmark")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL file")
    parser.add_argument("--passes", type=int, default=3, help="Number of passes")
    parser.add_argument("--gate-threshold", type=float, default=0.88)
    parser.add_argument("--variant", default="reactive_cache",
                        choices=["reactive_cache", "parallel_hybrid", "full_hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--output-dir", default="benchmark/results")
    args = parser.parse_args()

    # Patch gate BEFORE any other src imports
    patch_gate(args.gate_threshold)

    # Now import the rest
    from benchmark.benchmark import HybridVDBBenchmark
    from src.config import config

    print("=" * 70)
    print("MULTI-PASS BENCHMARK")
    print("=" * 70)
    print(f"Manifest:      {args.manifest}")
    print(f"Passes:        {args.passes}")
    print(f"Gate:          {args.gate_threshold}")
    print(f"Variant:       {args.variant}")
    print(f"Top-K:         {args.top_k}")
    print("=" * 70)

    # ---- Initialize system ONCE ----
    bench = HybridVDBBenchmark(
        test_mode="full",
        manifest_file=args.manifest,
        top_k=args.top_k,
        variant=args.variant,
        output_name=args.output_name or "multipass_temp",
        cold_start=True,  # Pass 1 must start with empty T2 for reproducibility
    )
    bench.initialize_vdb_system()

    # ---- Load queries ONCE ----
    (query_ids, query_texts, query_embeddings,
     query_types, expected_ids, query_metadata) = bench.get_queries()

    bench.validate_labeled_workload_alignment(expected_ids)

    session_ids = None
    if query_metadata:
        session_ids = [m.get("session_id") if m else None for m in query_metadata]
    bench.configure_anchor_system(query_embeddings, session_ids=session_ids)

    print(f"\n[READY] {len(query_ids)} queries, system initialized")
    print(f"[READY] Anchor system: {bench.anchor_system.get_anchor_stats()}")

    # ---- Multi-pass loop ----
    all_pass_results = []

    for pass_num in range(1, args.passes + 1):
        print(f"\n{'=' * 70}")
        print(f"PASS {pass_num}/{args.passes}")
        print(f"{'=' * 70}")

        pass_start = time.time()
        pass_results = []
        source_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "other": 0}
        latencies = []

        for i in range(len(query_ids)):
            qid = query_ids[i]
            qtext = query_texts[i]
            qvec = query_embeddings[i]
            metadata = query_metadata[i] if query_metadata else {}

            # Route through the system
            result = bench.router.search(
                query_vector=qvec, query_id=qid, query_text=qtext, k=args.top_k
            )

            # Track source
            source = result.get("source", "unknown")
            if "tier1" in source.lower() or source == "permanent":
                source_counts["tier1"] += 1
            elif "tier2" in source.lower() or source == "dynamic":
                source_counts["tier2"] += 1
            elif "tier3" in source.lower() or "cloud" in source.lower():
                source_counts["tier3"] += 1
            else:
                source_counts["other"] += 1

            latencies.append(result["latency_ms"])

            # Collect predicted IDs for retrieval metrics
            # Result structure: result["ids"] = [...], result["scores"] = [...]
            predicted_ids = [str(pid) for pid in result.get("ids", [])]

            # Expected IDs for this query
            exp = expected_ids[i] if expected_ids and i < len(expected_ids) else []
            exp_strs = [str(e) for e in exp] if exp else []

            if exp_strs:
                pass_results.append((predicted_ids[:args.top_k], exp_strs))

        pass_time = time.time() - pass_start
        total_queries = len(query_ids)

        # Compute metrics
        retrieval = compute_retrieval_metrics(pass_results)

        local_total = source_counts["tier1"] + source_counts["tier2"]
        local_pct = local_total / total_queries * 100

        # Anchor stats
        anchor_stats = bench.anchor_system.get_anchor_stats()

        pass_summary = {
            "pass": pass_num,
            "time_s": round(pass_time, 1),
            "queries": total_queries,
            **retrieval,
            "local_hit_pct": round(local_pct, 1),
            "tier1_pct": round(source_counts["tier1"] / total_queries * 100, 1),
            "tier2_pct": round(source_counts["tier2"] / total_queries * 100, 1),
            "tier3_pct": round(source_counts["tier3"] / total_queries * 100, 1),
            "avg_latency_ms": round(np.mean(latencies), 1),
            "p50_latency_ms": round(np.median(latencies), 1),
            "anchors_total": anchor_stats.get("total_anchors", 0),
            # Per Bug 7 fix: get_anchor_stats() returns anchor_types dict,
            # not flat strong_count/medium_count keys.
            "anchors_strong": anchor_stats.get("anchor_types", {}).get("strong", 0),
            "anchors_medium": anchor_stats.get("anchor_types", {}).get("medium", 0),
            "anchors_weak": anchor_stats.get("anchor_types", {}).get("weak", 0),
            "anchors_dormant": anchor_stats.get("anchor_types", {}).get("permanent", 0),
            "cache_size": result.get("dynamic_size_after", 0),
        }

        all_pass_results.append(pass_summary)

        # Per prefetch quality audit: dump admission and reuse logs after each pass
        router = bench.router
        audit_dir = Path(args.output_dir) / (args.output_name or f"multipass_{args.variant}_gate{args.gate_threshold}")
        audit_dir.mkdir(parents=True, exist_ok=True)
        admission_log_path = audit_dir / f"pass{pass_num}_admission_log.json"
        reuse_log_path = audit_dir / f"pass{pass_num}_reuse_log.json"
        try:
            with open(admission_log_path, "w") as f:
                json.dump(router.prefetch_admission_log, f, indent=2)
            with open(reuse_log_path, "w") as f:
                json.dump(router.reuse_log, f, indent=2)
            print(f"  Admission log: {len(router.prefetch_admission_log)} entries -> {admission_log_path}")
            print(f"  Reuse log:     {len(router.reuse_log)} entries -> {reuse_log_path}")
        except Exception as e:
            print(f"  [WARN] Failed to dump audit logs: {e}")

        print(f"\n--- PASS {pass_num} RESULTS ---")
        print(f"  nDCG@5:      {retrieval['nDCG@5']:.4f}")
        print(f"  MRR@5:       {retrieval['MRR@5']:.4f}")
        print(f"  Recall@5:    {retrieval['Recall@5']:.4f}")
        print(f"  Local hit:   {local_pct:.1f}%  (T1: {source_counts['tier1']}, T2: {source_counts['tier2']}, T3: {source_counts['tier3']})")
        print(f"  Avg latency: {np.mean(latencies):.1f}ms")
        print(f"  Anchors:     {anchor_stats.get('total_anchors', 0)} total, "
              f"{anchor_stats.get('strong_count', 0)} strong, "
              f"{anchor_stats.get('dormant_count', 0)} dormant")
        print(f"  Cache:       {result.get('dynamic_size_after', '?')} entries")
        print(f"  Time:        {pass_time:.1f}s")

    # ---- Save results ----
    output_name = args.output_name or f"multipass_{args.variant}_gate{args.gate_threshold}"
    output_dir = Path(args.output_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "manifest": args.manifest,
            "passes": args.passes,
            "gate_threshold": args.gate_threshold,
            "variant": args.variant,
            "top_k": args.top_k,
            "embedding_model": getattr(config, 'EMBEDDING_MODEL_NAME',
                                       getattr(config, 'MODEL_NAME', 'intfloat/e5-base-v2')),
        },
        "passes": all_pass_results,
        # Delta analysis: improvement from pass 1 to last pass
        "delta": {
            "ndcg_improvement": round(
                all_pass_results[-1]["nDCG@5"] - all_pass_results[0]["nDCG@5"], 4),
            "mrr_improvement": round(
                all_pass_results[-1]["MRR@5"] - all_pass_results[0]["MRR@5"], 4),
            "recall_improvement": round(
                all_pass_results[-1]["Recall@5"] - all_pass_results[0]["Recall@5"], 4),
            "local_improvement_pct": round(
                all_pass_results[-1]["local_hit_pct"] - all_pass_results[0]["local_hit_pct"], 1),
        },
    }

    with open(output_dir / "multipass_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("MULTI-PASS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Pass':>5} {'nDCG@5':>8} {'MRR@5':>8} {'Recall':>8} {'Local%':>8} {'Lat':>8} {'Anchors':>8}")
    print("-" * 55)
    for p in all_pass_results:
        print(f"{p['pass']:>5} {p['nDCG@5']:>8.4f} {p['MRR@5']:>8.4f} "
              f"{p['Recall@5']:>8.4f} {p['local_hit_pct']:>7.1f}% "
              f"{p['avg_latency_ms']:>7.1f}ms {p['anchors_total']:>8}")

    d = output["delta"]
    print(f"\nDelta (P1->P{args.passes}): nDCG {d['ndcg_improvement']:+.4f}  "
          f"MRR {d['mrr_improvement']:+.4f}  Recall {d['recall_improvement']:+.4f}  "
          f"Local {d['local_improvement_pct']:+.1f}%")

    print(f"\nResults saved to: {output_dir}")
    return output


if __name__ == "__main__":
    main()
