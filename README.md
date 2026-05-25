# Bounded Hybrid Retrieval with Adaptive Working Memory

Anonymous submission for double-blind review.

## Problem Statement

Conversational and session-structured retrieval systems face a fundamental tension: cloud-based dense retrieval provides high quality but introduces 100-150ms latency per query, while local-only indices sacrifice quality for speed. Existing hybrid approaches either route queries statically (missing session dynamics) or require unbounded local memory.

We present a **three-tier adaptive retrieval framework** that maintains bounded local memory (5-15% of corpus) while preserving near-cloud retrieval quality. The system combines:

1. **Tier 1 (Permanent):** A static local index seeded from corpus-level embedding geometry via k-means clustering
2. **Tier 2 (Dynamic):** A bounded working-memory cache with anchor-weighted admission and eviction
3. **Tier 3 (Cloud):** A remote dense retrieval API (Qdrant) serving as the canonical corpus

The key architectural contribution is a **calibrated hybrid gate** that controls quality-latency tradeoff via a single threshold parameter tau, and an **anchor-and-momentum system** that models query trajectories within embedding space to drive Tier 2 prefetching.

## What We Have Done

### Architecture
- Three-tier vector retrieval: permanent local FAISS index + adaptive bounded Tier 2 + cloud fallback
- Hybrid trust gate combining cosine similarity and normalized BM25 signals
- Anchor system: semantic basins in embedding space that track query trajectories
- Markov transition model for anchor-to-anchor prediction
- Bounded admission: capacity enforcement, deduplication, anchor-weighted eviction
- INT8 quantization for memory-efficient Tier 2 storage

### Evaluation Datasets
| Dataset | Type | Queries | Corpus |
|---------|------|---------|--------|
| NFCorpus | Static (BEIR) | 323 | 3.6K |
| SciFact | Static (BEIR) | 300 | 5.2K |
| QReCC | Session (multi-turn) | 3,481 | 116K |
| TREC CAsT 2020 | Session (conversational) | 208 | 38K |
| TREC-COVID | Static (BEIR) | 50 | 171K |

### Key Results
- Gate threshold tau provides smooth quality-latency tradeoff across all datasets
- At tau=0.95: 84-97% of cloud quality with 6-78% local traffic
- Multipass anchor learning: up to +0.182 nDCG improvement with 40-64% latency reduction
- Tier 2 at 5% of corpus achieves near-identical quality to 30% (quality varies <0.006 nDCG)
- Local queries served in 1-3ms vs 145-150ms for cloud

## Repository Structure

```
src/                          # Core system implementation
  config.py                   # Configuration constants and tier sizing
  hybrid_router.py            # Query routing logic (gate, T1/T2/T3 dispatch)
  local_vdb.py                # Local vector database (FAISS + metadata)
  storage_engine.py           # Tier 2 dynamic storage with eviction
  anchor_system.py            # Anchor basins, epsilon calibration, Markov
  semantic_cache.py           # Semantic caching layer
  cloud_client.py             # Qdrant cloud API client
  reranker.py                 # BM25 + cosine reranking
  quantization.py             # INT8 quantization for memory savings
  markov_transitions.py       # Markov transition model for prediction
  metrics.py                  # Evaluation metrics (nDCG, MRR, Recall)
  qdrant_ids.py               # Qdrant ID management
  utils/                      # Embedding model loader, logging, timing

benchmark/                    # Benchmarking framework
  benchmark.py                # Main benchmark runner (CLI)
  benchmark_config.py         # Benchmark configuration
  data_loader.py              # Dataset loading utilities
  eval_metrics.py             # Metric computation
  query_generator.py          # Query generation
  visualizer.py               # Result visualization
  workload_utils.py           # Workload management
  workloads/                  # Datasets and manifests (see below)

scripts/                      # Run scripts
  multipass_launch.py         # Multi-pass benchmark (P1 cold -> P2 learned -> P3)
  multipass_benchmark.py      # Single multipass run
  seed_permanent_kmeans.py    # Seed Tier 1 permanent index via k-means

requirements.txt              # Python dependencies
```

## Requirements

- **Python 3.10+**
- **FAISS** (`faiss-cpu >= 1.9.0`)
- **sentence-transformers** (for embedding generation)
- **Qdrant Cloud** instance (for Tier 3 remote retrieval)
- **PyTorch** (for sentence-transformers backend)

See `requirements.txt` for full dependency list.

## How to Run

### 1. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Cloud Endpoint

Set environment variables for your Qdrant cloud instance:

```bash
export QDRANT_URL="YOUR_QDRANT_CLUSTER_URL"
export QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
export QDRANT_COLLECTION="YOUR_COLLECTION_NAME"
```

### 3. Seed Cloud Corpus

Upload dataset documents to your Qdrant collection. Each workload has a `corpus.jsonl` containing documents with pre-computed e5-base-v2 embeddings.

**Note:** TREC-COVID corpus (211MB) and QReCC pre-computed embeddings (53MB) exceed GitHub's file size limit and are excluded from this repository. To obtain them:
- **TREC-COVID corpus:** Download from [BEIR](https://github.com/beir-cellar/beir) (`beir-v1.0.0/trec-covid.zip`) and place `corpus.jsonl` in `benchmark/workloads/beir/trec-covid/`
- **QReCC embeddings:** Generate by running the embedding model on the QReCC corpus, or download from the authors' data repository

### 4. Seed Tier 1 Permanent Index

```bash
python scripts/seed_permanent_kmeans.py \
  --dataset nfcorpus \
  --num-clusters 3000
```

This creates the permanent local FAISS index via k-means clustering of corpus embeddings.

### 5. Run Single Benchmark

```bash
python -m benchmark.benchmark \
  --variant full_hybrid \
  --manifest-file benchmark/workloads/beir/nfcorpus/queries_manifest.jsonl \
  --top-k 5 \
  --collection-name YOUR_COLLECTION \
  --gate-threshold 0.90 \
  --cold-start \
  --output-name nfcorpus_fh_tau090
```

Variants:
- `cloud_only`: All queries to cloud (baseline)
- `reactive_cache`: T1 + reactive Tier 2 admission, no anchors
- `full_hybrid`: T1 + anchors + momentum-driven Tier 2 prefetch
- `true_lru`: T1 + strict LRU Tier 2 (ablation baseline)

### 6. Run Gate Sweep

```bash
for tau in 0.84 0.88 0.90 0.92 0.95 0.97; do
  python -m benchmark.benchmark \
    --variant full_hybrid \
    --manifest-file benchmark/workloads/beir/nfcorpus/queries_manifest.jsonl \
    --top-k 5 \
    --collection-name YOUR_COLLECTION \
    --gate-threshold $tau \
    --cold-start \
    --dynamic-dir-override data/workloads/beir/nfcorpus/dynamic_gate${tau} \
    --output-name nfcorpus_fh_gate${tau}
done
```

### 7. Run Multipass Benchmark

```bash
python scripts/multipass_launch.py run \
  --variant full_hybrid \
  --dataset qrecc-quick \
  --cluster 0 \
  --gate-threshold 0.95
```

This runs 3 passes: P1 (cold-start) -> P2 (learned from P1) -> P3 (converged from P2).

### 8. Run Tier Sizing Sweep

```bash
for pct in 0.05 0.10 0.20 0.30; do
  python -m benchmark.benchmark \
    --variant full_hybrid \
    --manifest-file benchmark/workloads/beir/nfcorpus/queries_manifest.jsonl \
    --top-k 5 \
    --collection-name YOUR_COLLECTION \
    --gate-threshold 0.95 \
    --cold-start \
    --tier2-percent $pct \
    --output-name nfcorpus_t2_${pct}_fh
done
```

## What to Expect

### Output

Results are saved to `benchmark/results/<output-name>/results.json` containing:

```json
{
  "per_query": [
    {
      "query_id": "...",
      "ndcg_at_5": 0.8,
      "source": "tier2_dynamic",
      "latency_ms": 2.1,
      ...
    }
  ],
  "summary": {
    "mean_ndcg_at_5": 0.366,
    "local_hit_rate": 5.9,
    "avg_latency_ms": 151,
    "dynamic_admissions_total": 423,
    "dynamic_evictions_total": 12,
    "anchor_system": {
      "total_anchors": 1,
      "anchor_types": {"weak": 1}
    }
  },
  "config": {
    "variant": "full_hybrid",
    "gate_threshold_override": 0.95,
    "cold_start": true,
    "tier2_capacity": 15000
  }
}
```

### Expected Runtimes

| Dataset | Queries | Cloud-only | Full Hybrid (tau=0.95) |
|---------|---------|------------|----------------------|
| NFCorpus | 323 | ~45s | ~50s |
| SciFact | 300 | ~45s | ~30s |
| QReCC | 3,481 | ~9min | ~6min |
| CAsT 2020 | 208 | ~35s | ~90s |
| TREC-COVID | 50 | ~8s | ~7s |

*Times depend on network latency to Qdrant cloud instance.*

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gate-threshold` | 0.9 | Cosine similarity threshold for local/cloud routing |
| `--tier2-percent` | 0.15 | Tier 2 capacity as fraction of corpus |
| `--variant` | full_hybrid | Retrieval variant (see above) |
| `--cold-start` | false | Clear Tier 2 before run (reproducible) |
| `--top-k` | 5 | Number of results per query |

## Reproducing Paper Results

The paper evaluates on 5 datasets across 4 experiment types:

1. **Main results** (Table 4): Run each variant at tau=0.90 with --cold-start
2. **Gate sweep** (Table 5): Sweep tau in [0.84, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99]
3. **Multipass learning** (Table 6): 3-pass runs for CAsT tau=0.95, QReCC RC tau=0.88, QReCC FH tau=0.95
4. **Tier sizing**: Sweep tier2-percent in [0.05, 0.10, 0.20, 0.30] at tau=0.95

All runs should use `--cold-start` for reproducibility.

## Notes for Reviewers

- The embedding model used is `intfloat/e5-base-v2` (pre-computed embeddings provided in workload directories)
- QReCC evaluation uses labeled queries only (1,569 of 3,481; unlabeled queries have no relevance judgments)
- TREC-COVID uses the BEIR v1.0 qrels; ensure your Qdrant collection is seeded with the correct corpus
- The gate threshold is global and static; future work may explore per-workload adaptive calibration
