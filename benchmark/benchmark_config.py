# benchmark/benchmark_config.py

"""
Benchmark-specific configuration.

Separate from src/config.py to keep middleware clean.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load credentials
load_dotenv()


class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # ═══════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════

    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent

    # Benchmark directories
    BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
    QUERIES_DIR = BENCHMARK_DIR / "queries"
    RESULTS_DIR = BENCHMARK_DIR / "results"

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    PERMANENT_DIR = DATA_DIR / "permanent"
    DYNAMIC_DIR = DATA_DIR / "dynamic"

    # Create directories
    for directory in [QUERIES_DIR, RESULTS_DIR, DATA_DIR, PERMANENT_DIR, DYNAMIC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # CLOUD CREDENTIALS
    # ═══════════════════════════════════════════════════════════

    # Qdrant Cloud
    QDRANT_URL = os.getenv(
        "QDRANT_URL", "https://your-cluster.gcp.cloud.qdrant.io:6333"
    )
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-api-key-here")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pubmed_qa_full")

    # ═══════════════════════════════════════════════════════════
    # DATASET
    # ═══════════════════════════════════════════════════════════

    DATASET_NAME = "pubmed_qa"
    DATASET_SUBSET = "pqa_labeled"

    # ═══════════════════════════════════════════════════════════
    # STORAGE CONFIGURATION
    # ═══════════════════════════════════════════════════════════

    # Per §22.6 + Step 5: Percentage-based tier sizing.
    # Increased from 1%/5% after Step 3 showed 99.9% neighborhood skip rate
    # on NFCorpus — T2 was too small to build a diverse working set.
    TIER1_PERCENT = 0.03   # 3% of corpus — permanent centroids
    TIER2_PERCENT = 0.15    # 15% of corpus — adaptive working set

    # Fallback fixed capacities (used when corpus size is unknown at init)
    # These are overridden by percentage-based sizing at runtime.
    TIER1_SIZE = 3000  # Will be overridden by TIER1_PERCENT * corpus_size
    TIER2_CAPACITY = 15000  # Will be overridden by TIER2_PERCENT * corpus_size

    # ═══════════════════════════════════════════════════════════
    # BENCHMARK SETTINGS
    # ═══════════════════════════════════════════════════════════

    # Test modes
    QUICK_TEST_SIZE = 100  # Fast validation
    FULL_TEST_SIZE = 1000  # Realistic workload

    # Query distribution (for full test)
    # Simulates medical chatbot usage pattern
    IN_DATASET_RATIO = 0.65  # 65% - Normal questions
    EDGE_CASE_RATIO = 0.25  # 25% - Paraphrased/similar
    OOD_RATIO = 0.10  # 10% - Novel questions

    # Timing
    QUERY_INTERVAL_MS = 500  # 500ms = 2 queries/sec (realistic)

    # ═══════════════════════════════════════════════════════════
    # MODEL
    # ═══════════════════════════════════════════════════════════

    EMBEDDING_MODEL = "intfloat/e5-base-v2"
    EMBEDDING_DIM = 768
    MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # Cache downloaded models

    # ═══════════════════════════════════════════════════════════
    # VDB OPTIMIZATIONS
    # ═══════════════════════════════════════════════════════════

    USE_HNSW = True  # Fast approximate search
    USE_QUANTIZATION = True  # INT8 compression (4× memory savings)
    PREFETCH_ENABLED = True  # Smart prefetching

    # Per Step 4: Per-anchor adaptive thresholds
    # When True, HybridGate uses AnchorAwareThreshold which maintains
    # separate adaptive thresholds per anchor basin, with global fallback
    # for new anchors (minimum 10 outcomes before anchor-specific threshold).
    PER_ANCHOR_THRESHOLD = True

    # HNSW parameters
    HNSW_M = 16  # Connections per layer
    HNSW_EF_CONSTRUCTION = 200  # Build quality
    HNSW_EF_SEARCH = 50  # Search quality

    # ═══════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════

    SAVE_PLOTS = True
    SAVE_JSON = True
    SAVE_CSV = True

    PLOT_DPI = 150  # Plot resolution

    # ═══════════════════════════════════════════════════════════
    # DISPLAY
    # ═══════════════════════════════════════════════════════════

    VERBOSE = True  # Print detailed progress
    SHOW_WARNINGS = True


# Create singleton instance
config = BenchmarkConfig()


# ═══════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════


def validate_config():
    """Validate configuration before running."""

    errors = []

    # Check credentials
    if "your-cluster" in config.QDRANT_URL:
        errors.append("[X] QDRANT_URL not set in .env")

    if "your-api-key" in config.QDRANT_API_KEY:
        errors.append("[X] QDRANT_API_KEY not set in .env")

    # Per §22.6: Display percentage-based sizing (actual sizes computed at runtime)
    print("=" * 70)
    print("STORAGE CONFIGURATION (percentage-based, per §22.6)")
    print("=" * 70)
    print(
        f"TIER 1 (Permanent): {config.TIER1_PERCENT*100:.0f}% of corpus "
        f"(fallback: {config.TIER1_SIZE:,} vectors)"
    )
    print(
        f"TIER 2 (Dynamic):   {config.TIER2_PERCENT*100:.0f}% of corpus "
        f"(fallback: {config.TIER2_CAPACITY:,} capacity)"
    )
    print("  NOTE: Actual sizes computed from corpus metadata at runtime.")
    print("=" * 70)

    if errors:
        print("\n[!] CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  {error}")
        print("\n[i] Create a .env file with:")
        print("   QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333")
        print("   QDRANT_API_KEY=your-api-key-here")
        print("   QDRANT_COLLECTION=pubmed_qa_full")
        return False

    print("[OK] Configuration valid")
    return True


if __name__ == "__main__":
    validate_config()
