# src/config.py
"""
Configuration for Hybrid VDB System

All tunable parameters in one place.
Defines the three-tier architecture parameters and learning thresholds.

Author: Saberzerker
Date: 2025-11-17
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Central configuration for hybrid VDB system."""

    # ═══════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════

    BASE_DIR = PROJECT_ROOT
    DATA_DIR = BASE_DIR / "data"

    BASE_LAYER_PATH = DATA_DIR / "permanent"
    DYNAMIC_LAYER_PATH = DATA_DIR / "dynamic"
    LOGS_DIR = BASE_DIR / "logs"

    # Create directories if they don't exist
    BASE_LAYER_PATH.mkdir(parents=True, exist_ok=True)
    DYNAMIC_LAYER_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # VECTOR DIMENSIONS
    # ═══════════════════════════════════════════════════════════

    VECTOR_DIMENSION = 768  # intfloat/e5-base-v2 dimension

    # ═══════════════════════════════════════════════════════════
    # STORAGE CAPACITY (THREE-TIER ARCHITECTURE)
    # ═══════════════════════════════════════════════════════════

    # Per §22.6 + Step 5: Percentage-based tier sizing.
    # Tier 1 = 3% of corpus (permanent centroids, motor memory)
    # Tier 2 = 15% of corpus (adaptive working set, working memory)
    # Tier 3 = 100% of corpus (cloud, unlimited)
    #
    # Increased from 1%/5% after Step 3 showed 99.9% neighborhood skip rate
    # on NFCorpus — T2 was too small to build a diverse working set.
    # 3%/15% gives NFCorpus: T1=109, T2=545 (enough room for diversity).
    # TREC-COVID: T1=5133, T2=25650 (still bounded, <30% of corpus).
    TIER1_PERCENT = 0.03   # 3% of corpus — permanent centroids
    TIER2_PERCENT = 0.15   # 15% of corpus — adaptive working set

    # Fallback fixed capacities (used when corpus size is unknown at init)
    PERMANENT_LAYER_CAPACITY = 3000  # Will be overridden by TIER1_PERCENT * corpus_size
    DYNAMIC_LAYER_CAPACITY = 15000   # Will be overridden by TIER2_PERCENT * corpus_size

    # Hot partition (in-memory portion of dynamic)
    HOT_PARTITION_RAM_LIMIT = 1500  # Half of dynamic in RAM for speed

    # TIER 3: Cloud VDB (Bakery - Canonical Truth)
    # No capacity limit (cloud has unlimited storage)

    # ═══════════════════════════════════════════════════════════
    # CLOUD VDB CONFIGURATION (TIER 3)
    # ═══════════════════════════════════════════════════════════

    CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "qdrant")
    CLOUD_URL = os.getenv(
        "QDRANT_URL",
        "https://6e6e7451-fa6d-4dcb-b987-49dba2bb7373.europe-west3-0.gcp.cloud.qdrant.io:6333",
    )
    CLOUD_API_KEY = os.getenv("QDRANT_API_KEY", "")
    CLOUD_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hybrid_vdb_test")

    # Cloud artificial latency for testing (when using stub)
    CLOUD_ARTIFICIAL_LATENCY_MS = 200
    CLOUD_TIMEOUT_SECONDS = 30.0  # Per pre-flight P0: increased from 10s to prevent cloud timeouts

    # ═══════════════════════════════════════════════════════════
    # SEARCH PARAMETERS
    # ═══════════════════════════════════════════════════════════

    DEFAULT_SEARCH_K = 5
    LOCAL_CONFIDENCE_THRESHOLD = 0.75  # Min score to trust local result (75%)

    # Similarity thresholds
    MIN_SIMILARITY_THRESHOLD = 0.50  # Below this = not relevant
    HIGH_SIMILARITY_THRESHOLD = 0.90  # Above this = very confident

    # ═══════════════════════════════════════════════════════════
    # PREFETCHING PARAMETERS (THE SMART LEARNING!)
    # ═══════════════════════════════════════════════════════════

    PREFETCH_ENABLED = True
    PREFETCH_K = 5  # Number of predictions to generate per query

    # Phase thresholds (determines prefetch strategy)
    COLD_START_QUERIES = 3  # Queries 1-3: Fill aggressively
    WARMUP_QUERIES = 20  # Queries 4-20: Refine accuracy
    # After query 20: Steady state (high accuracy maintenance)

    # Prediction matching threshold
    PREDICTION_SIMILARITY_THRESHOLD = 0.85  # 85% similar = prediction match

# ═══════════════════════════════════════════════════════════════
    # DECOUPLED NEIGHBORHOOD THRESHOLDS (Per §22 Step 1.5 / HAKES insight)
    # ═══════════════════════════════════════════════════════════════
    # HAKES (arXiv 2505.12524) decouples compression parameters for
    # insertion vs search. Our neighborhood_threshold was the same
    # for admission and search dedup — this is wrong.
    #
    # ADMISSION (loose, 0.85): When caching cloud results to Tier 2,
    # we WANT to admit diverse vectors. A loose threshold lets more
    # vectors in, building a richer working set. Only block vectors
    # that are near-identical to existing Tier 2 residents.
    #
    # SEARCH DEDUP (tight, 0.98): When serving query results to the
    # user, we must NOT return duplicate/near-duplicate documents.
    # A tight threshold ensures only near-exact dupes are filtered.
    # Per §19.2 D1: 0.98 = near-exact duplicate only.
    NEIGHBORHOOD_THRESHOLD_ADMISSION = 0.85   # Loose: admit diverse vectors to Tier 2
    NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP = 0.98  # Tight: only block near-exact dupes in results
    # Per §22 Step 1.5 fix: reactive cache bypasses neighborhood check entirely
    # (confirmed-relevant docs should always enter Tier 2; exact-ID dedup is sufficient).
    # Predictive prefetch uses a higher threshold (0.95) to avoid near-duplicate
    # speculative fetches while still admitting diverse predictions.
    NEIGHBORHOOD_THRESHOLD_PREFETCH_ADMISSION = 0.95  # Loose-but-not-bypass for prefetch

    # ═══════════════════════════════════════════════════════════════════════════════
    # TIER 2 EVICTION MODE
    # ═══════════════════════════════════════════════════════════════════════════════
    # "anchor" = evict by anchor confidence (cascade dead anchors first, then weakest)
    # "lru"    = strict LRU by last-access timestamp (no anchor involvement)
    EVICTION_MODE = "anchor"  # Default: anchor-weight eviction

    # ═══════════════════════════════════════════════════════════════════════════════
    # HYBRID GATE (Phase 5: BM25+cosine replaces cross-encoder)
    # ═══════════════════════════════════════════════════════════════════════════════
    # The cross-encoder (MS-MARCO-MiniLM-L6-v2) rejected 96.7% of local results
    # on NFCorpus (domain mismatch) and added 90ms latency. Replaced with a
    # lightweight BM25+cosine hybrid gate that runs in <0.1ms.
    #
    # gate_signal = alpha * cosine_similarity + (1 - alpha) * normalize(bm25_score)
    # If gate_signal >= adaptive_threshold: serve locally.
    # Else: fall through to cloud.
    #
    # Based on: BM25 [Robertson & Zaragoza, 2009]; hybrid fusion proven
    # effective in BEIR [Thakur et al., 2021] where sparse+dense > either alone.
    GATE_ALPHA = 0.7              # Weight for cosine similarity (0.7 = cosine-dominant)
    # Per Phase 5.7 analysis: gate signal on NFCorpus has mean=0.87, std=0.067.
    # A threshold of 0.70 is 2.5 std below mean — rejects essentially nothing.
    # Empirical analysis shows threshold must be at or above the signal mean
    # (0.88+) to meaningfully reject poor local results.
    #
    # The cold-start nDCG when serving cloud directly was 0.611, vs 0.145 when
    # serving locally at threshold 0.70. Raising the threshold forces cloud
    # fallbacks, which drives reactive admissions to Tier 2, which populates
    # Tier 2 with confirmed-relevant documents.
    #
    # Phase-aware step sizes:
    #   Cold/warmup: step_up=0.05 (aggressive — need signal fast)
    #   Steady:      step_up=0.02 (conservative — stable)
    # The AdaptiveThreshold._adapt() uses step_up/step_down from config.
    GATE_INITIAL_THRESHOLD = 0.90  # Per Phase 5.7: must be near signal mean
    GATE_PRECISION_WINDOW = 50     # Sliding window for adaptive threshold
    GATE_TARGET_PRECISION = 0.70   # Target local precision for adaptive threshold
    GATE_STEP_UP = 0.05           # Threshold increase when precision < target (was 0.02)
    GATE_STEP_DOWN = 0.01         # Threshold decrease when precision > target+0.05
    GATE_MIN_THRESHOLD = 0.50     # Floor for adaptive threshold
    GATE_MAX_THRESHOLD = 0.98     # Ceiling for adaptive threshold (was 0.95)

    # Per Step 4: Per-anchor adaptive thresholds
    # When True, HybridGate uses AnchorAwareThreshold which maintains
    # separate adaptive thresholds per anchor basin, with global fallback
    # for new anchors (minimum 10 outcomes before anchor-specific threshold).
    PER_ANCHOR_THRESHOLD = True

    # Per pre-flight P1: When False, _adapt() is a no-op — threshold stays at
    # GATE_INITIAL_THRESHOLD. record_outcome() still logs precision (diagnostics).
    # Used for clean ablation in benchmark sweeps. Set True for future directions.
    ADAPTIVE_GATE = False

    # ═══════════════════════════════════════════════════════════════════════════════
    # SHADOW PROBE CALIBRATION (Phase 5.7)
    # ═══════════════════════════════════════════════════════════════════════════════
    # The hybrid gate's adaptive threshold needs cloud ground truth to calibrate.
    # Without periodic cloud probes, the threshold stays at 0.70 forever (the
    # NFCorpus problem: 100% local hit rate, 63.7% below cloud nDCG@10).
    #
    # Phase definitions:
    #   Cold (queries 1-5):   Serve cloud results directly. Need ground truth fast.
    #   Warmup (queries 6-20): Serve local, shadow-probe cloud every 5th query.
    #   Steady (queries 21+):  Serve local, shadow-probe cloud every 10th query.
    #
    # "Shadow" means: fire cloud async, compare with local when cloud completes,
    # feed record_outcome(). The user sees local results immediately.
    # "Cold" is different: the user sees cloud results because we have no
    # calibration data yet and can't trust the gate.
    #
    # Based on: Adaptive threshold calibration via periodic ground-truth probes
    # (THEORY.md §20); cold-start probing per RaLMSpec [Zheng et al., ICML 2024].
    CALIBRATION_COLD_QUERIES = 5       # Queries 1-5: serve cloud directly
    CALIBRATION_WARMUP_QUERIES = 20    # Queries 6-20: shadow probe every Nth
    CALIBRATION_PROBE_INTERVAL_COLD = 1    # Cold: every query goes to cloud
    CALIBRATION_PROBE_INTERVAL_WARMUP = 5  # Warmup: every 5th query probes cloud
    CALIBRATION_PROBE_INTERVAL_STEADY = 10 # Steady: every 10th query probes cloud

    # Per Fix A: minimum quality floor for local serving.
    # If best_local_similarity < this threshold, force cloud fallback
    # regardless of gate decision. Catches queries where local results
    # are nearly irrelevant but BM25 pushes gate signal above threshold.
    MIN_LOCAL_QUALITY_FLOOR = 0.30

    # Phase-based prefetch thresholds (used for PREDICTION matching, not admission)
    # Per §19.2 D1: raised from 0.85/0.90/0.92 to 0.98 across all phases.
    NEIGHBORHOOD_THRESHOLD_COLD = 0.98  # Cold start — only block exact dupes
    NEIGHBORHOOD_THRESHOLD_WARMUP = 0.98  # Warmup — only block exact dupes
    NEIGHBORHOOD_THRESHOLD_STEADY = 0.98  # Steady state — only block exact dupes

    # Prediction generation noise
    NOISE_SCALE_COLD = 0.30  # High noise for exploration
    NOISE_SCALE_WARMUP = 0.15  # Medium noise
    NOISE_SCALE_STEADY = 0.05  # Low noise, exploit known paths

    # ═══════════════════════════════════════════════════════════
    # ANCHOR SYSTEM (V5: Retrieval-driven reinforcement)
    # ═══════════════════════════════════════════════════════════

    # V5 retrieval-driven signal scaling
    ANCHOR_SIGNAL_SCALE = 0.05       # Scaling for strengthen/weaken signals
    BASE_ABSORPTION_REWARD = 0.05    # Scaling for adaptive absorption reward

    # ═══════════════════════════════════════════════════════════
    # STEP 7: LOSS-MODULATED CENTROID UPDATES
    # ═══════════════════════════════════════════════════════════
    # Per Phase 3 / Step 7 design: replace 1/n incremental mean with
    # loss-modulated learning rate η = η_base × pred_err × (1-conf).
    # This is online gradient descent, NOT bandit (three violated assumptions
    # fixed — see Design Decisions Q7).
    #
    # Max possible step: η_base × 0.5 × 1.0 = 0.05 (centroid moves ≤5%).
    # Score penalty prevents ghost anchors that drift without consequence.
    # DORMANT state: soft eviction — skipped by matching/prediction but
    # Markov transitions preserved. Reactive cache catches fallout.

    ANCHOR_LEARNING_RATE_BASE = 0.1       # η_base — max learning rate per update
    CENTROID_SHIFT_PENALTY = 0.1          # score reduction per unit centroid shift
    ANCHOR_DEATH_FLOOR = 0.02             # confidence below → evict entirely

    # ═══════════════════════════════════════════════════════════
    # SEMANTIC CACHE (MOMENTUM-BASED CLUSTERING)
    # ═══════════════════════════════════════════════════════════

    SEMANTIC_DRIFT_THRESHOLD = 0.35  # Distance > 0.35 = new cluster created
    MOMENTUM_ALPHA = 0.9  # Centroid momentum (0.9 = 90% old, 10% new)
    MIN_CLUSTER_REINFORCEMENT_SCORE = 0.70  # Min score to reinforce cluster

    # Cluster lifecycle
    CLUSTER_TTL_SECONDS = 86400  # 24 hours before cluster can be evicted
    CLUSTER_MIN_ACCESS_COUNT = 3  # Min accesses to avoid eviction

    # ═══════════════════════════════════════════════════════════
    # BACKGROUND JOBS (MAINTENANCE)
    # ═══════════════════════════════════════════════════════════

    COMPACTION_INTERVAL_SECONDS = 3600  # Compact dynamic layer every 1 hour
    DECAY_INTERVAL_SECONDS = 1800  # Check for decay every 30 minutes
    EVICTION_INTERVAL_SECONDS = 3600  # Evict weak anchors every 1 hour

    # Auto-save interval
    AUTOSAVE_INTERVAL_SECONDS = 600  # Save state every 10 minutes

    # ═══════════════════════════════════════════════════════════
    # LOGGING & DEBUGGING
    # ═══════════════════════════════════════════════════════════

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "hybrid_vdb.log"

    # ═══════════════════════════════════════════════════════════
    # DEMO APP SETTINGS
    # ═══════════════════════════════════════════════════════════

    DEMO_HOST = "0.0.0.0"
    DEMO_PORT = 5000
    DEMO_DEBUG = False

    # LLM settings (for RAG)
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
    OLLAMA_TIMEOUT = 45

    # RAG settings
    RAG_CONTEXT_SIZE = 3  # Top-3 documents for context
    RAG_MAX_CONTEXT_LENGTH = 2000  # Max characters per document

    # ═══════════════════════════════════════════════════════════
    # METRICS & TELEMETRY
    # ═══════════════════════════════════════════════════════════

    METRICS_ENABLED = True
    METRICS_EXPORT_INTERVAL = 60  # Export metrics every minute
    METRICS_HISTORY_LIMIT = 1000  # Keep last 1000 events

    # ═══════════════════════════════════════════════════════════
    # DEVELOPMENT & TESTING
    # ═══════════════════════════════════════════════════════════

    USE_REAL_SYSTEM = os.getenv("USE_REAL_SYSTEM", "true").lower() == "true"
    ENABLE_PROFILING = False

    # Test mode settings
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
    TEST_QUERY_LIMIT = 100  # Max queries in test mode


# ═══════════════════════════════════════════════════════════
# CREATE SINGLETON & EXPORT ALL CONSTANTS
# ═══════════════════════════════════════════════════════════

# Create singleton instance
config = Config()

# Vector dimensions
VECTOR_DIMENSION = config.VECTOR_DIMENSION

# Paths
BASE_LAYER_PATH = config.BASE_LAYER_PATH
DYNAMIC_LAYER_PATH = config.DYNAMIC_LAYER_PATH
LOGS_DIR = config.LOGS_DIR
DATA_DIR = config.DATA_DIR

# Storage capacity
PERMANENT_LAYER_CAPACITY = config.PERMANENT_LAYER_CAPACITY
DYNAMIC_LAYER_CAPACITY = config.DYNAMIC_LAYER_CAPACITY
HOT_PARTITION_RAM_LIMIT = config.HOT_PARTITION_RAM_LIMIT

# Search parameters
DEFAULT_SEARCH_K = config.DEFAULT_SEARCH_K
LOCAL_CONFIDENCE_THRESHOLD = config.LOCAL_CONFIDENCE_THRESHOLD
MIN_SIMILARITY_THRESHOLD = config.MIN_SIMILARITY_THRESHOLD
HIGH_SIMILARITY_THRESHOLD = config.HIGH_SIMILARITY_THRESHOLD

# Prefetch parameters
PREFETCH_ENABLED = config.PREFETCH_ENABLED
PREFETCH_K = config.PREFETCH_K
COLD_START_QUERIES = config.COLD_START_QUERIES
WARMUP_QUERIES = config.WARMUP_QUERIES
PREDICTION_SIMILARITY_THRESHOLD = config.PREDICTION_SIMILARITY_THRESHOLD
NEIGHBORHOOD_THRESHOLD_ADMISSION = config.NEIGHBORHOOD_THRESHOLD_ADMISSION
NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP = config.NEIGHBORHOOD_THRESHOLD_SEARCH_DEDUP
NEIGHBORHOOD_THRESHOLD_PREFETCH_ADMISSION = config.NEIGHBORHOOD_THRESHOLD_PREFETCH_ADMISSION
NEIGHBORHOOD_THRESHOLD_COLD = config.NEIGHBORHOOD_THRESHOLD_COLD
NEIGHBORHOOD_THRESHOLD_WARMUP = config.NEIGHBORHOOD_THRESHOLD_WARMUP
NEIGHBORHOOD_THRESHOLD_STEADY = config.NEIGHBORHOOD_THRESHOLD_STEADY
# Per Phase 5: Hybrid gate config (BM25+cosine replaces cross-encoder)
GATE_ALPHA = config.GATE_ALPHA
GATE_INITIAL_THRESHOLD = config.GATE_INITIAL_THRESHOLD
GATE_PRECISION_WINDOW = config.GATE_PRECISION_WINDOW
GATE_TARGET_PRECISION = config.GATE_TARGET_PRECISION
GATE_STEP_UP = config.GATE_STEP_UP
GATE_STEP_DOWN = config.GATE_STEP_DOWN
GATE_MIN_THRESHOLD = config.GATE_MIN_THRESHOLD
GATE_MAX_THRESHOLD = config.GATE_MAX_THRESHOLD
PER_ANCHOR_THRESHOLD = config.PER_ANCHOR_THRESHOLD
ADAPTIVE_GATE = config.ADAPTIVE_GATE
# Per Phase 5.7: Shadow probe calibration constants
CALIBRATION_COLD_QUERIES = config.CALIBRATION_COLD_QUERIES
CALIBRATION_WARMUP_QUERIES = config.CALIBRATION_WARMUP_QUERIES
CALIBRATION_PROBE_INTERVAL_COLD = config.CALIBRATION_PROBE_INTERVAL_COLD
CALIBRATION_PROBE_INTERVAL_WARMUP = config.CALIBRATION_PROBE_INTERVAL_WARMUP
CALIBRATION_PROBE_INTERVAL_STEADY = config.CALIBRATION_PROBE_INTERVAL_STEADY
MIN_LOCAL_QUALITY_FLOOR = config.MIN_LOCAL_QUALITY_FLOOR
NOISE_SCALE_COLD = config.NOISE_SCALE_COLD
NOISE_SCALE_WARMUP = config.NOISE_SCALE_WARMUP
NOISE_SCALE_STEADY = config.NOISE_SCALE_STEADY

# Anchor system (V5: retrieval-driven)
ANCHOR_SIGNAL_SCALE = config.ANCHOR_SIGNAL_SCALE
BASE_ABSORPTION_REWARD = config.BASE_ABSORPTION_REWARD
# Per Step 7: Loss-modulated centroid update parameters
ANCHOR_LEARNING_RATE_BASE = config.ANCHOR_LEARNING_RATE_BASE
CENTROID_SHIFT_PENALTY = config.CENTROID_SHIFT_PENALTY
ANCHOR_DEATH_FLOOR = config.ANCHOR_DEATH_FLOOR

# Semantic cache
SEMANTIC_DRIFT_THRESHOLD = config.SEMANTIC_DRIFT_THRESHOLD
MOMENTUM_ALPHA = config.MOMENTUM_ALPHA
MIN_CLUSTER_REINFORCEMENT_SCORE = config.MIN_CLUSTER_REINFORCEMENT_SCORE
CLUSTER_TTL_SECONDS = config.CLUSTER_TTL_SECONDS
CLUSTER_MIN_ACCESS_COUNT = config.CLUSTER_MIN_ACCESS_COUNT

# Background jobs
COMPACTION_INTERVAL_SECONDS = config.COMPACTION_INTERVAL_SECONDS
DECAY_INTERVAL_SECONDS = config.DECAY_INTERVAL_SECONDS
EVICTION_INTERVAL_SECONDS = config.EVICTION_INTERVAL_SECONDS
AUTOSAVE_INTERVAL_SECONDS = config.AUTOSAVE_INTERVAL_SECONDS

# Logging
LOG_LEVEL = config.LOG_LEVEL
VERBOSE = config.VERBOSE
LOG_FORMAT = config.LOG_FORMAT
LOG_FILE = config.LOG_FILE

# Demo app
DEMO_HOST = config.DEMO_HOST
DEMO_PORT = config.DEMO_PORT
DEMO_DEBUG = config.DEMO_DEBUG
OLLAMA_HOST = config.OLLAMA_HOST
OLLAMA_MODEL = config.OLLAMA_MODEL
OLLAMA_TIMEOUT = config.OLLAMA_TIMEOUT
RAG_CONTEXT_SIZE = config.RAG_CONTEXT_SIZE
RAG_MAX_CONTEXT_LENGTH = config.RAG_MAX_CONTEXT_LENGTH

# Metrics
METRICS_ENABLED = config.METRICS_ENABLED
METRICS_EXPORT_INTERVAL = config.METRICS_EXPORT_INTERVAL
METRICS_HISTORY_LIMIT = config.METRICS_HISTORY_LIMIT

# Development
USE_REAL_SYSTEM = config.USE_REAL_SYSTEM
ENABLE_PROFILING = config.ENABLE_PROFILING
TEST_MODE = config.TEST_MODE
TEST_QUERY_LIMIT = config.TEST_QUERY_LIMIT

# Cloud (add this if missing from Config class)
CLOUD_TIMEOUT_SECONDS = config.CLOUD_TIMEOUT_SECONDS  # Per P0: 30s
CLOUD_PROVIDER = config.CLOUD_PROVIDER
CLOUD_URL = config.CLOUD_URL
CLOUD_API_KEY = config.CLOUD_API_KEY
CLOUD_COLLECTION_NAME = config.CLOUD_COLLECTION_NAME
CLOUD_ARTIFICIAL_LATENCY_MS = config.CLOUD_ARTIFICIAL_LATENCY_MS
ANCHOR_DECAY_CHECK_INTERVAL = config.DECAY_INTERVAL_SECONDS

# Eviction mode: "anchor" (confidence cascade) or "lru" (strict timestamp)
EVICTION_MODE = config.EVICTION_MODE


# Convenience function
def get_config():
    """Get configuration singleton."""
    return config


USE_HNSW = True  # Enable HNSW for fast approximate search

# HNSW parameters
HNSW_M = 16  # Connections per layer (higher = more accurate, slower build)
# 16 is optimal for most cases

HNSW_EF_CONSTRUCTION = 200  # Build quality (higher = better index, slower build)
# 200 is good default

HNSW_EF_SEARCH = 50  # Search quality (higher = more accurate, slower search)
# 50 gives 99%+ recall with good speed
# Increase to 100 for 99.9%+ recall

# ═══════════════════════════════════════════════════════════
# QUANTIZATION CONFIGURATION (OPTIMIZATION)
# ═══════════════════════════════════════════════════════════

USE_QUANTIZATION = True  # Enable INT8 quantization for 4× memory reduction

# Quantization trades memory for slight accuracy loss:
# - Memory: 1.5MB → 384KB (4× reduction)
# - Accuracy: >98% retained (cosine similarity)
# - Speed: Slightly slower (quantize/dequantize overhead ~5-10%)
#
# Set to False if you need 100% accuracy or have abundant memory
