#!/usr/bin/env python3
"""
Multipass benchmark runner — P1 (cold-start) → P2 (learned) → P3 (converged).

Proves the anchor-and-momentum learning claim:
  - P1: cold-start, T2 empty, anchors fresh
  - P2: re-runs same workload with P1's learned T2 + anchors
  - P3: re-runs again with P2's learned T2 + anchors

Usage:
    # Print commands for 3 terminals (reactive_cache):
    python scripts/multipass_launch.py print-commands --variant reactive_cache

    # Run all 3 passes sequentially for one variant:
    python scripts/multipass_launch.py run --variant reactive_cache --dataset qrecc

    # Run all 3 passes for all 3 hybrid variants (9 runs total, sequential):
    python scripts/multipass_launch.py run-all --dataset qrecc

    # Run with custom gate threshold:
    python scripts/multipass_launch.py run --variant full_hybrid --gate-threshold 0.86
"""
import subprocess, os, sys, time, shutil, json
from pathlib import Path

PROJECT = Path(r"./vdb-benchmark")
PYTHON = r"python"

CLUSTERS = [
    {"name": "c1", "url": "YOUR_QDRANT_CLUSTER_URL",
     "key": "YOUR_QDRANT_API_KEY"},
    {"name": "c2", "url": "YOUR_QDRANT_CLUSTER_URL",
     "key": "YOUR_QDRANT_API_KEY"},
    {"name": "c3", "url": "YOUR_QDRANT_CLUSTER_URL",
     "key": "YOUR_QDRANT_API_KEY"},
    {"name": "c4", "url": "YOUR_QDRANT_CLUSTER_URL",
     "key": "YOUR_QDRANT_API_KEY"},
    {"name": "c5", "url": "YOUR_QDRANT_CLUSTER_URL",
     "key": "YOUR_QDRANT_API_KEY"},
    {"name": "c6", "url": "YOUR_QDRANT_CLUSTER_URL",
     "key": "YOUR_QDRANT_API_KEY"},
]

# Hybrid variants that use T2 and anchors — these are the ones that learn
HYBRID_VARIANTS = ["reactive_cache", "full_hybrid", "parallel_hybrid"]

# Baselines that don't learn — only need P1
BASELINE_VARIANTS = ["cloud_only", "true_lru"]

DATASETS = {
    "qrecc": ("hybrid_vdb_qrecc", "benchmark/workloads/qrecc/session_manifest.jsonl"),
    "qrecc-quick": ("hybrid_vdb_qrecc", "benchmark/workloads/qrecc/session_manifest_quick.jsonl"),
    "nfcorpus": ("hybrid_vdb_beir_nfcorpus", "benchmark/workloads/beir/nfcorpus/queries_manifest.jsonl"),
    "scifact": ("hybrid_vdb_beir_scifact", "benchmark/workloads/beir/scifact/queries_manifest.jsonl"),
}

NUM_PASSES = 3


def make_env(cluster, collection):
    return {
        **os.environ,
        "QDRANT_URL": cluster["url"],
        "QDRANT_API_KEY": cluster["key"],
        "QDRANT_COLLECTION": collection,
    }


def get_dynamic_dir(workload_dir: Path, variant: str, pass_num: int,
                    suffix: str = "") -> Path:
    """Dynamic dir for a specific pass. P1 uses dynamic_{variant}_p1, etc.
    If suffix is provided, appends _{suffix} to the dir name."""
    name = f"dynamic_{variant}_p{pass_num}"
    if suffix:
        name += f"_{suffix}"
    return workload_dir / name


def build_cmd(variant: str, dataset: str, pass_num: int, cold_start: bool,
              dynamic_dir_override: str, output_name: str,
              gate_threshold: float = None) -> list:
    """Build benchmark command for a single pass."""
    collection, manifest = DATASETS[dataset]
    cmd = [
        PYTHON, "-m", "benchmark.benchmark",
        "--variant", variant,
        "--manifest-file", str(PROJECT / manifest),
        "--top-k", "5",
        "--collection-name", collection,
        "--output-name", output_name,
        "--dynamic-dir-override", dynamic_dir_override,
    ]
    if cold_start:
        cmd.append("--cold-start")
    if gate_threshold is not None:
        cmd.extend(["--gate-threshold", str(gate_threshold)])
    return cmd


def run_pass(variant: str, dataset: str, pass_num: int, cluster_idx: int,
             cold_start: bool, gate_threshold: float = None, suffix: str = ""):
    """Run a single pass of the benchmark.

    For P1: cold_start=True, fresh dynamic dir.
    For P2/P3: cold_start=False, dynamic dir seeded from previous pass.
    """
    collection, manifest_path = DATASETS[dataset]
    workload_dir = Path(PROJECT / manifest_path).parent.parent
    cluster = CLUSTERS[cluster_idx]

    if pass_num == 1:
        dynamic_dir = get_dynamic_dir(workload_dir, variant, 1, suffix)
        # P1: cold start — clear the dir
        if dynamic_dir.exists():
            shutil.rmtree(dynamic_dir)
        dynamic_dir.mkdir(parents=True, exist_ok=True)
    else:
        # P2/P3: copy previous pass's dynamic dir as seed
        prev_dir = get_dynamic_dir(workload_dir, variant, pass_num - 1, suffix)
        dynamic_dir = get_dynamic_dir(workload_dir, variant, pass_num, suffix)

        if not prev_dir.exists():
            print(f"[FATAL] Previous pass dynamic dir not found: {prev_dir}")
            print(f"        Run P{pass_num - 1} first.")
            sys.exit(1)

        # Copy previous pass state to new dir
        if dynamic_dir.exists():
            shutil.rmtree(dynamic_dir)
        shutil.copytree(prev_dir, dynamic_dir)
        print(f"[P{pass_num}] Seeded dynamic dir from P{pass_num - 1}: {prev_dir} -> {dynamic_dir}")

    output_name = f"multipass/{dataset}_{variant}_p{pass_num}"
    if suffix:
        output_name += f"_{suffix}"
    cmd = build_cmd(
        variant=variant,
        dataset=dataset,
        pass_num=pass_num,
        cold_start=cold_start,
        dynamic_dir_override=str(dynamic_dir),
        output_name=output_name,
        gate_threshold=gate_threshold,
    )

    env = make_env(cluster, collection)
    label = f"{dataset}/{variant}/P{pass_num}@{cluster['name']}"

    print(f"\n{'='*70}")
    print(f"[START] {label}")
    print(f"  cold_start={cold_start}, dynamic_dir={dynamic_dir}")
    print(f"  output={output_name}")
    print(f"  cmd={' '.join(cmd[:6])}... (truncated)")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT), env=env)
    elapsed = time.time() - t0

    status = "PASS" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"\n[{status}] {label} — {elapsed:.0f}s ({elapsed/60:.1f}min)")

    if result.returncode != 0:
        print(f"[WARN] P{pass_num} failed for {variant}. Skipping subsequent passes.")
        return False

    # Verify output exists
    results_path = PROJECT / "benchmark" / "results" / output_name / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        n_queries = len(data.get("per_query", []))
        summary = data.get("summary", {})
        ndcg = summary.get("mean_ndcg_at_5", summary.get("mean_ndcg_at_k", "N/A"))
        lat = summary.get("avg_latency", "N/A")
        local = summary.get("local_hit_rate", "N/A")
        ndcg_s = f"{ndcg:.4f}" if isinstance(ndcg, (int, float)) else str(ndcg)
        lat_s = f"{lat:.1f}ms" if isinstance(lat, (int, float)) else str(lat)
        local_s = f"{local:.1f}%" if isinstance(local, (int, float)) else str(local)
        print(f"  → nDCG@5={ndcg_s}, latency={lat_s}, local={local_s}, queries={n_queries}")
    else:
        print(f"[WARN] No results file at {results_path}")

    return True


def run_multipass(variant: str, dataset: str, cluster_idx: int,
                  gate_threshold: float = None, suffix: str = ""):
    """Run P1→P2→P3 for a single variant on a single cluster."""
    for pass_num in range(1, NUM_PASSES + 1):
        cold_start = (pass_num == 1)
        success = run_pass(
            variant=variant,
            dataset=dataset,
            pass_num=pass_num,
            cluster_idx=cluster_idx,
            cold_start=cold_start,
            gate_threshold=gate_threshold,
            suffix=suffix,
        )
        if not success:
            print(f"[ABORT] Stopping multipass for {variant} after P{pass_num} failure.")
            return False
    return True


def run_all_multipass(dataset: str, gate_threshold: float = None, suffix: str = ""):
    """Run P1→P2→P3 for all 3 hybrid variants, each on a separate cluster.

    Uses clusters 1-3 for reactive_cache, full_hybrid, parallel_hybrid.
    Runs sequentially within each variant (P1 must finish before P2 starts).
    """
    results = {}
    for i, variant in enumerate(HYBRID_VARIANTS):
        print(f"\n{'#'*70}")
        print(f"# MULTIPASS: {variant}")
        print(f"{'#'*70}")
        success = run_multipass(
            variant=variant,
            dataset=dataset,
            cluster_idx=i,
            gate_threshold=gate_threshold,
            suffix=suffix,
        )
        results[variant] = "PASS" if success else "FAIL"

    print(f"\n{'='*70}")
    print("MULTIPASS SUMMARY")
    print(f"{'='*70}")
    for variant, status in results.items():
        print(f"  {variant:25s} : {status}")

    # Print convergence table
    print(f"\n{'='*70}")
    print("CONVERGENCE TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':25s} | {'P1 nDCG@5':>10s} | {'P2 nDCG@5':>10s} | {'P3 nDCG@5':>10s} | {'P1→P3 Δ':>10s}")
    print("-" * 80)

    for variant in HYBRID_VARIANTS:
        ndcgs = []
        for p in range(1, NUM_PASSES + 1):
            output_name = f"multipass/{dataset}_{variant}_p{p}"
            if suffix:
                output_name += f"_{suffix}"
            results_path = PROJECT / "benchmark" / "results" / output_name / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                ndcg = data.get("summary", {}).get("mean_ndcg_at_5",
                       data.get("summary", {}).get("mean_ndcg_at_k", None))
                ndcgs.append(ndcg)
            else:
                ndcgs.append(None)

        p1_s = f"{ndcgs[0]:.4f}" if ndcgs[0] is not None else "N/A"
        p2_s = f"{ndcgs[1]:.4f}" if ndcgs[1] is not None else "N/A"
        p3_s = f"{ndcgs[2]:.4f}" if ndcgs[2] is not None else "N/A"
        delta = f"{ndcgs[2] - ndcgs[0]:+.4f}" if ndcgs[0] is not None and ndcgs[2] is not None else "N/A"

        print(f"{variant:25s} | {p1_s:>10s} | {p2_s:>10s} | {p3_s:>10s} | {delta:>10s}")

    # Also print cloud_only baseline for reference
    cloud_path = PROJECT / "benchmark" / "results" / "validation" / f"{dataset}_cloud_only_quick" / "results.json"
    if cloud_path.exists():
        with open(cloud_path) as f:
            data = json.load(f)
        cloud_ndcg = data.get("summary", {}).get("mean_ndcg_at_5",
                     data.get("summary", {}).get("mean_ndcg_at_k", None))
        if cloud_ndcg is not None:
            print(f"{'cloud_only (baseline)':25s} | {cloud_ndcg:.4f}     | {'—':>10s} | {'—':>10s} | {'—':>10s}")


def print_commands(variant: str, dataset: str, cluster_idx: int,
                   gate_threshold: float = None, suffix: str = ""):
    """Print copy-paste commands for running multipass in separate terminals."""
    collection, manifest_path = DATASETS[dataset]
    workload_dir = Path(PROJECT / manifest_path).parent.parent
    cluster = CLUSTERS[cluster_idx]

    env_parts = [
        f'$env:QDRANT_URL="{cluster["url"]}"',
        f'$env:QDRANT_API_KEY="{cluster["key"]}"',
        f'$env:QDRANT_COLLECTION="{collection}"',
    ]
    env_line = " ; ".join(env_parts)

    suffix_tag = f" ({suffix})" if suffix else ""
    print(f"# === MULTIPASS COMMANDS for {variant} on {dataset}{suffix_tag} ===")
    print(f"# Run each pass sequentially in the SAME terminal.\n")

    for pass_num in range(1, NUM_PASSES + 1):
        cold_start = (pass_num == 1)
        dynamic_dir = get_dynamic_dir(workload_dir, variant, pass_num, suffix)

        if pass_num > 1:
            prev_dir = get_dynamic_dir(workload_dir, variant, pass_num - 1, suffix)
            print(f"# P{pass_num}: Copy P{pass_num-1} state")
            print(f'Copy-Item -Recurse -Force "{prev_dir}" "{dynamic_dir}"')
            print()

        output_name = f"multipass/{dataset}_{variant}_p{pass_num}"
        if suffix:
            output_name += f"_{suffix}"
        cmd = build_cmd(
            variant=variant,
            dataset=dataset,
            pass_num=pass_num,
            cold_start=cold_start,
            dynamic_dir_override=str(dynamic_dir),
            output_name=output_name,
            gate_threshold=gate_threshold,
        )

        print(f"# Terminal {cluster_idx + 1}: P{pass_num} ({'cold-start' if cold_start else 'learned'})")
        print(env_line)
        print(" ".join(cmd))
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  multipass_launch.py print-commands --variant reactive_cache [--dataset qrecc] [--gate-threshold 0.86] [--suffix P50]")
        print("  multipass_launch.py run --variant reactive_cache [--dataset qrecc] [--cluster 0] [--gate-threshold 0.86] [--suffix P50]")
        print("  multipass_launch.py run-all [--dataset qrecc] [--gate-threshold 0.86] [--suffix P50]")
        print(f"\nDatasets: {', '.join(DATASETS.keys())}")
        print(f"Hybrid variants: {', '.join(HYBRID_VARIANTS)}")
        print(f"Baseline variants: {', '.join(BASELINE_VARIANTS)}")
        print(f"Clusters: c1-c{len(CLUSTERS)} (indices 0-{len(CLUSTERS)-1})")
        print(f"Passes: {NUM_PASSES}")
        sys.exit(1)

    action = sys.argv[1]

    # Parse args
    variant = "reactive_cache"
    dataset = "qrecc"
    cluster_idx = 0
    gate_threshold = None
    suffix = ""

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--variant" and i + 1 < len(sys.argv):
            variant = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--cluster" and i + 1 < len(sys.argv):
            cluster_idx = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--gate-threshold" and i + 1 < len(sys.argv):
            gate_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--suffix" and i + 1 < len(sys.argv):
            suffix = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if dataset not in DATASETS:
        print(f"Unknown dataset: {dataset}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    if action == "print-commands":
        print_commands(variant, dataset, cluster_idx, gate_threshold, suffix)

    elif action == "run":
        if variant in BASELINE_VARIANTS:
            print(f"[INFO] {variant} is a baseline — no learning, running P1 only.")
            run_pass(variant, dataset, 1, cluster_idx, cold_start=True,
                     gate_threshold=gate_threshold, suffix=suffix)
        else:
            run_multipass(variant, dataset, cluster_idx, gate_threshold, suffix)

    elif action == "run-all":
        run_all_multipass(dataset, gate_threshold, suffix)

    else:
        print(f"Unknown action: {action}")
        print("Actions: print-commands, run, run-all")
        sys.exit(1)


if __name__ == "__main__":
    main()