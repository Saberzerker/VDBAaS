"""
Seed Tier 1 (permanent layer) using k-means centroids from the cloud corpus.

Instead of taking the first N vectors (sequential method), this script:
1. Fetches ALL vectors from the Qdrant cloud collection
2. Runs k-means clustering with k = tier1_size
3. Uses the centroids as Tier 1 vectors — maximally representative of the corpus

Based on: k-means++ initialization [Arthur & Vassilvitskii, 2007] via FAISS.
This ensures Tier 1 covers the full embedding space rather than being biased
toward whatever order Qdrant returns vectors in.

Usage:
    python scripts/seed_permanent_kmeans.py --workload-dir benchmark/workloads/beir/nfcorpus --tier1-size 36
    python scripts/seed_permanent_kmeans.py --workload-dir benchmark/workloads/beir/nfcorpus  # auto-size from metadata
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import os

from benchmark.benchmark_config import config as benchmark_config
from benchmark.workload_utils import (
    derive_collection_name,
    derive_local_data_dirs,
    resolve_workload_dir,
)
from src.qdrant_ids import external_doc_id

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Tier 1 using k-means centroids from cloud corpus."
    )
    parser.add_argument(
        "--workload-dir",
        type=str,
        help="Workload directory for per-workload collection and local paths.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        help="Optional Qdrant collection override.",
    )
    parser.add_argument(
        "--permanent-dir",
        type=str,
        help="Optional permanent layer output directory.",
    )
    parser.add_argument(
        "--tier1-size",
        type=int,
        default=None,
        help="Number of k-means centroids (= Tier 1 size). Auto-computed from metadata if omitted.",
    )
    parser.add_argument(
        "--kmeans-epochs",
        type=int,
        default=20,
        help="Number of k-means iterations (FAISS niter). Default: 20.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Qdrant scroll batch size for fetching vectors.",
    )
    args = parser.parse_args()

    resolved_workload_dir = resolve_workload_dir(workload_dir=args.workload_dir)
    if args.collection_name:
        collection_name = args.collection_name
    elif resolved_workload_dir:
        collection_name = derive_collection_name(resolved_workload_dir)
    else:
        collection_name = os.getenv("QDRANT_COLLECTION", "pubmed_qa_full")

    if args.permanent_dir:
        permanent_dir = Path(args.permanent_dir).expanduser().resolve()
    elif resolved_workload_dir:
        permanent_dir, _dynamic_dir = derive_local_data_dirs(resolved_workload_dir)
    else:
        permanent_dir = Path(os.getenv("PERMANENT_DIR", "./data/permanent"))

    # Per §22.6: Compute tier1_size from corpus metadata using percentage-based sizing.
    if args.tier1_size is not None:
        tier1_size = args.tier1_size
    elif resolved_workload_dir:
        metadata_path = resolved_workload_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                workload_meta = json.load(f)
            corpus_count = int(workload_meta.get("corpus_count", 0))
            tier1_size = max(10, int(corpus_count * benchmark_config.TIER1_PERCENT))
        else:
            tier1_size = 36  # fallback
    else:
        tier1_size = 36  # fallback

    print("=" * 70)
    print("SEEDING TIER 1 (PERMANENT) — K-MEANS CENTROIDS")
    print("=" * 70)
    print(f"Collection:    {collection_name}")
    print(f"Target:         {permanent_dir}")
    print(f"K-means k:      {tier1_size}")
    print(f"K-means epochs: {args.kmeans_epochs}")
    print("=" * 70)

    permanent_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch ALL vectors from cloud ──────────────────────────
    print("\n[1/4] Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    info = client.get_collection(collection_name)
    points_count = getattr(info, "vectors_count", None) or getattr(info, "points_count", 0)
    print(f"  Cloud has {points_count:,} vectors")

    if points_count == 0:
        print("Cloud is empty. Run seed_cloud.py first.")
        return

    # Validate completeness
    if resolved_workload_dir:
        metadata_path = resolved_workload_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                workload_meta = json.load(f)
            expected = int(workload_meta.get("corpus_count", 0))
            if expected and points_count < expected:
                raise ValueError(
                    f"Cloud collection '{collection_name}' is incomplete: "
                    f"{points_count:,}/{expected:,} documents. "
                    "Run seed_cloud.py first."
                )

    print(f"\n[2/4] Fetching ALL {points_count:,} vectors from cloud...")
    all_vectors: list[np.ndarray] = []
    all_ids: list[str] = []
    all_payloads: list[dict] = []
    offset = None

    with tqdm(total=points_count, desc="Fetching") as pbar:
        while len(all_vectors) < points_count:
            results, offset = client.scroll(
                collection_name=collection_name,
                limit=args.batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if not results:
                break

            for point in results:
                all_vectors.append(np.array(point.vector, dtype="float32"))
                payload = point.payload or {}
                permanent_id = external_doc_id(point.id, payload)
                all_ids.append(permanent_id)
                all_payloads.append(payload)
                pbar.update(1)

            if offset is None:
                break

    n_fetched = len(all_vectors)
    print(f"  Fetched {n_fetched:,} vectors")

    if n_fetched < tier1_size:
        print(f"  WARNING: Fewer vectors ({n_fetched}) than k-means k ({tier1_size}).")
        print(f"  Falling back to using all {n_fetched} vectors as Tier 1.")
        tier1_size = n_fetched

    # ── Step 2: Run k-means ───────────────────────────────────────────
    print(f"\n[3/4] Running k-means (k={tier1_size}, epochs={args.kmeans_epochs})...")

    vectors_array = np.vstack(all_vectors).astype("float32")
    dim = vectors_array.shape[1]
    print(f"  Vector dimension: {dim}")
    print(f"  Corpus size: {vectors_array.shape[0]:,}")

    # FAISS k-means: niter=20, redo=3 for robust centroids
    # Based on: k-means++ initialization [Arthur & Vassilvitskii, 2007]
    # FAISS uses nredo (not redo) for multiple restarts
    kmeans = faiss.Kmeans(d=dim, k=tier1_size, niter=args.kmeans_epochs, verbose=True, nredo=3)
    kmeans.train(vectors_array)

    # Centroids are the Tier 1 vectors
    centroids = kmeans.centroids.astype("float32")  # (tier1_size, dim)
    print(f"  K-means converged. Centroids shape: {centroids.shape}")

    # ── Step 3: Assign each centroid to its nearest corpus vector ─────
    # We can't use centroids directly as Tier 1 because we need real document IDs
    # for the metadata lookup. Find the nearest real vector to each centroid.
    print("  Assigning centroids to nearest corpus vectors...")

    # Build a flat index of all corpus vectors for nearest-neighbor lookup
    corpus_index = faiss.IndexFlatL2(dim)
    corpus_index.add(vectors_array)

    # For each centroid, find the nearest corpus vector
    _, nearest_indices = corpus_index.search(centroids, 1)  # (tier1_size, 1)
    nearest_indices = nearest_indices.flatten()

    # Deduplicate: if two centroids map to the same corpus vector, pick the next nearest
    used_indices = set()
    final_indices = []
    for i, idx in enumerate(nearest_indices):
        if idx not in used_indices:
            used_indices.add(idx)
            final_indices.append(idx)
        else:
            # Search for next nearest unused vector
            # Get top-50 neighbors and pick first unused
            _, top_k_indices = corpus_index.search(centroids[i:i+1], 50)
            found = False
            for candidate_idx in top_k_indices.flatten():
                if candidate_idx not in used_indices and candidate_idx < n_fetched:
                    used_indices.add(candidate_idx)
                    final_indices.append(candidate_idx)
                    found = True
                    break
            if not found:
                # Extremely unlikely — just use the centroid's nearest
                final_indices.append(idx)

    # ── Step 4: Build FAISS HNSW index and metadata ──────────────────
    print(f"\n[4/4] Building Tier 1 FAISS HNSW index...")

    # Use the ACTUAL corpus vectors (not centroids) for the index
    # This preserves real document IDs and exact vectors for search
    tier1_vectors = vectors_array[final_indices].astype("float32")
    tier1_ids = [all_ids[i] for i in final_indices]
    tier1_payloads = [all_payloads[i] for i in final_indices]

    # Build HNSW index
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 200
    index.add(tier1_vectors)
    print(f"  Created HNSW index with {index.ntotal:,} vectors")

    # Save index
    index_file = permanent_dir / "partition_0.index"
    faiss.write_index(index, str(index_file))
    print(f"  Index: {index_file}")

    # Build metadata
    metadata = {}
    for local_idx, (doc_id, payload) in enumerate(zip(tier1_ids, tier1_payloads)):
        preview = str(
            payload.get("question")
            or payload.get("text_preview")
            or payload.get("title")
            or ""
        )[:100]
        metadata[doc_id] = {
            "cloud_id": all_ids[local_idx],  # original cloud ID
            "preview": preview,
            "local_idx": local_idx,
            "partition_file": str((permanent_dir / "partition_0.index").resolve()),
            "seed_method": "kmeans_centroid",
            "centroid_index": local_idx,  # which centroid this vector represents
        }

    metadata_file = permanent_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_file}")

    # Save k-means assignment info for analysis
    kmeans_info = {
        "method": "kmeans",
        "k": tier1_size,
        "niter": args.kmeans_epochs,
        "nredo": 3,
        "corpus_size": n_fetched,
        "dimension": dim,
        "seed_method": "nearest_corpus_vector",
        "unique_vectors_used": len(set(final_indices)),
        "timestamp": time.time(),
    }
    kmeans_file = permanent_dir / "kmeans_info.json"
    with open(kmeans_file, "w", encoding="utf-8") as f:
        json.dump(kmeans_info, f, indent=2)
    print(f"  K-means info: {kmeans_file}")

    size_mb = (len(final_indices) * dim * 4 * 2) / (1024**2)
    print(f"\n  Tier 1 size: ~{size_mb:.1f} MB")
    print(f"  Vectors: {len(final_indices)}")
    print(f"  Method: k-means centroids (nearest corpus vectors)")

    print("\n" + "=" * 70)
    print("TIER 1 SEEDED (K-MEANS)")
    print("=" * 70)


if __name__ == "__main__":
    main()