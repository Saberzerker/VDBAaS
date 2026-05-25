# src/metrics.py
"""
Metrics Tracker - Comprehensive Telemetry for Hybrid VDB

Tracks ALL system events for performance analysis and visualization:
- Tier hit rates (tier1, tier2, tier3, offline)
- Latency distributions (p50, p95, p99)
- Prefetch statistics (cache hits, misses, efficiency)
- Anchor performance (prediction accuracy, strength distribution)
- Learning curve over time (TIER 2 prefetch effectiveness)

Thread-safe, production-ready implementation.

Author: Anonymous
Date: 2025-11-17
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import logging
import json
from pathlib import Path

from src.config import METRICS_HISTORY_LIMIT

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Thread-safe metrics tracking for hybrid VDB system.
    
    NEW FEATURES (2025-11-17):
    - TIER 2 prefetch effectiveness tracking (neighborhood coverage)
    - Separate retrieval vs total latency
    - Query count tracking
    - Hit count (X/Y format)
    
    Tracks:
    - Query sources (tier1, tier2, tier3, offline)
    - Latencies (per tier, percentiles)
    - Prefetch efficiency (hits, misses, cache rate)
    - Anchor prediction accuracy
    - Dynamic space fill rate over time
    - Learning curve progression (TIER 2 hits over time)
    
    All operations are thread-safe for production use.
    """
    
    def __init__(self, history_limit: int = METRICS_HISTORY_LIMIT):
        """
        Initialize metrics tracker.
        
        Args:
            history_limit: Max events to keep in memory (default 1000)
        """
        self.lock = threading.RLock()
        
        # Event storage
        # {event_type: deque([{timestamp, latency, metadata}, ...])}
        self.events = defaultdict(lambda: deque(maxlen=history_limit))
        
        # Query tracking
        self.total_queries = 0
        
        # Tier-specific tracking
        self.tier1_hits = 0
        self.tier2_hits = 0  # Prefetch hits (neighborhood coverage)
        self.tier3_hits = 0
        self.offline_fallbacks = 0
        
        # TIER 2 specific (for learning curve)
        # This tracks how many queries CHECKED TIER 2
        self.tier2_attempts = 0
        
        # Latency tracking (retrieval only, NOT including LLM)
        self.total_retrieval_latency_ms = 0
        self.latency_history = deque(maxlen=history_limit)  # [(timestamp, latency)]
        
        # Prefetch tracking
        self.prefetch_fetched = 0  # Fetched from cloud
        self.prefetch_skipped = 0  # Already cached
        
        # Prediction tracking
        self.prediction_hits = 0
        self.prediction_misses = 0
        
        # Time-series data for visualization
        self.hit_rate_history = deque(maxlen=history_limit)  # [(timestamp, tier2_hit_rate)]
        self.dynamic_fill_history = deque(maxlen=history_limit)  # [(timestamp, fill_rate)]
        
        # Learning curve data (TIER 2 prefetch effectiveness over time)
        self.learning_curve_data = []
        
        # Session start time
        self.session_start = time.time()
        
        logger.info("[METRICS] Initialized metrics tracker")
        logger.info(f"[METRICS] History limit: {history_limit} events")
    
    # ═══════════════════════════════════════════════════════════
    # EVENT LOGGING
    # ═══════════════════════════════════════════════════════════
    
    def record_query(
        self,
        source: str,
        latency_ms: float,
        prefetch_fetched: int = 0,
        prefetch_skipped: int = 0
    ):
        """
        Record query result.
        
        IMPORTANT: latency_ms should be RETRIEVAL time only (not including LLM)
        
        Args:
            source: Where result came from (tier1_permanent, tier2_dynamic, tier3_cloud, offline_fallback)
            latency_ms: Retrieval latency in milliseconds (NOT including LLM time)
            prefetch_fetched: Number of vectors fetched from cloud during prefetch
            prefetch_skipped: Number of vectors already cached (prefetch cache hit)
        """
        with self.lock:
            self.total_queries += 1
            
            # Record tier hit
            if source == 'tier1_permanent':
                self.tier1_hits += 1
                self.tier2_attempts += 1  # Checked TIER 2 first (missed)
            
            elif source == 'tier2_dynamic':
                self.tier2_hits += 1  # ⭐ Prefetch worked! Query was in neighborhood
                self.tier2_attempts += 1
            
            elif source == 'tier3_cloud':
                self.tier3_hits += 1
                self.tier2_attempts += 1  # Checked TIER 2 first (missed)
            
            elif source == 'offline_fallback':
                self.offline_fallbacks += 1
                self.tier2_attempts += 1

            elif source == 'local_only_empty':
                self.tier2_attempts += 1
            
            # Record latency (retrieval only)
            self.total_retrieval_latency_ms += latency_ms
            self.latency_history.append((time.time(), latency_ms))
            
            # Record prefetch stats
            self.prefetch_fetched += prefetch_fetched
            self.prefetch_skipped += prefetch_skipped
            
            # Calculate TIER 2 hit rate (prefetch effectiveness)
            tier2_hit_rate = (self.tier2_hits / self.tier2_attempts * 100) \
                if self.tier2_attempts > 0 else 0
            
            # Store for learning curve
            self.learning_curve_data.append({
                'query_num': self.total_queries,
                'tier2_hit_rate': tier2_hit_rate,
                'latency_ms': latency_ms,
                'source': source,
                'timestamp': time.time()
            })
            
            # Store hit rate history
            self.hit_rate_history.append((time.time(), tier2_hit_rate))
            
            # Log event
            self.log_event(
                event_type=source,
                latency=latency_ms,
                metadata={
                    'query_number': self.total_queries,
                    'tier2_hit_rate': tier2_hit_rate,
                    'prefetch_fetched': prefetch_fetched,
                    'prefetch_skipped': prefetch_skipped
                }
            )
            
            logger.debug(f"[METRICS] Query #{self.total_queries}: {source}, "
                        f"latency={latency_ms:.1f}ms, tier2_rate={tier2_hit_rate:.1f}%")
    
    def log_event(
        self, 
        event_type: str, 
        latency: Optional[float] = None, 
        metadata: Optional[Dict] = None
    ):
        """
        Log a system event.
        
        Event types:
        - "tier1_hit": Query answered from permanent layer
        - "tier2_hit": Query answered from dynamic layer (prefetch worked!)
        - "tier3_hit": Query answered from cloud
        - "offline_fallback": Query answered offline (network down)
        - "prefetch_operation": Background prefetch completed
        
        Args:
            event_type: Type of event
            latency: Latency in milliseconds (optional)
            metadata: Additional event data (optional)
        """
        with self.lock:
            event = {
                "timestamp": time.time(),
                "latency_ms": latency,
                "query_number": self.total_queries
            }
            
            if metadata:
                event.update(metadata)
            
            # Store event
            self.events[event_type].append(event)
    
    def log_prefetch_hit(self):
        """Log prefetch cache hit (skipped cloud fetch)."""
        with self.lock:
            self.prefetch_skipped += 1
    
    def log_prefetch_miss(self):
        """Log prefetch cache miss (had to fetch from cloud)."""
        with self.lock:
            self.prefetch_fetched += 1
    
    def log_prediction_hit(self):
        """Log anchor prediction hit (query matched prediction)."""
        with self.lock:
            self.prediction_hits += 1
    
    def log_prediction_miss(self):
        """Log anchor prediction miss (query didn't match any prediction)."""
        with self.lock:
            self.prediction_misses += 1
    
    def log_dynamic_fill_rate(self, fill_rate: float):
        """
        Log dynamic space fill rate for visualization.
        
        Args:
            fill_rate: Fill percentage (0-100)
        """
        with self.lock:
            self.dynamic_fill_history.append((time.time(), fill_rate))
    
    # ═══════════════════════════════════════════════════════════
    # STATISTICS & SUMMARIES
    # ═══════════════════════════════════════════════════════════
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dict with all key metrics:
            - Query counts by tier
            - Hit rates (overall and TIER 2 specific)
            - Latency statistics (retrieval only)
            - Prefetch efficiency
            - Prediction accuracy
            - Session stats
        """
        with self.lock:
            if self.total_queries == 0:
                return self._empty_summary()
            
            # Count queries by tier
            tier1_hits = self.tier1_hits
            tier2_hits = self.tier2_hits
            tier3_hits = self.tier3_hits
            offline_hits = self.offline_fallbacks
            
            # Calculate hit rates (as percentage of total queries)
            tier1_rate = (tier1_hits / self.total_queries * 100)
            tier2_rate = (tier2_hits / self.total_queries * 100)
            tier3_rate = (tier3_hits / self.total_queries * 100)
            offline_rate = (offline_hits / self.total_queries * 100)
            
            # Local hit rate = TIER 1 + TIER 2 (both are local)
            local_hits = tier1_hits + tier2_hits
            local_hit_rate = (local_hits / self.total_queries * 100)
            
            # TIER 2 prefetch effectiveness (what % of TIER 2 checks resulted in hits)
            # This is the KEY metric for learning curve!
            tier2_prefetch_rate = (self.tier2_hits / self.tier2_attempts * 100) \
                if self.tier2_attempts > 0 else 0
            
            # Latency statistics (retrieval only, not LLM)
            latency_stats = self._calculate_latency_stats()
            avg_latency = (self.total_retrieval_latency_ms / self.total_queries)
            
            # Prefetch statistics
            total_prefetch = self.prefetch_fetched + self.prefetch_skipped
            prefetch_cache_rate = (
                self.prefetch_skipped / total_prefetch * 100
                if total_prefetch > 0 else 0
            )
            
            # Prediction statistics
            total_predictions = self.prediction_hits + self.prediction_misses
            prediction_accuracy = (
                self.prediction_hits / total_predictions * 100
                if total_predictions > 0 else 0
            )
            
            # Session duration
            session_duration = time.time() - self.session_start
            queries_per_minute = (self.total_queries / session_duration * 60) if session_duration > 0 else 0
            
            return {
                # Query counts
                "total_queries": self.total_queries,
                "tier1_hits": tier1_hits,
                "tier2_hits": tier2_hits,
                "tier3_hits": tier3_hits,
                "offline_hits": offline_hits,
                "tier2_attempts": self.tier2_attempts,
                
                # Hit rates (percentages)
                "tier1_hit_rate": tier1_rate,
                "tier2_hit_rate": tier2_rate,
                "tier3_hit_rate": tier3_rate,
                "offline_rate": offline_rate,
                "local_hit_rate": local_hit_rate,
                
                # TIER 2 prefetch effectiveness (KEY LEARNING METRIC!)
                "tier2_prefetch_rate": tier2_prefetch_rate,
                
                # Latency statistics (retrieval only)
                "avg_latency": avg_latency,
                "p50_latency": latency_stats["p50"],
                "p95_latency": latency_stats["p95"],
                "p99_latency": latency_stats["p99"],
                "min_latency": latency_stats["min"],
                "max_latency": latency_stats["max"],
                
                # Prefetch statistics
                "prefetch_hits": self.prefetch_skipped,
                "prefetch_misses": self.prefetch_fetched,
                "prefetch_total": total_prefetch,
                "prefetch_cache_hit_rate": prefetch_cache_rate,
                "prefetch_fetched": self.prefetch_fetched,
                "prefetch_skipped": self.prefetch_skipped,
                
                # Prediction statistics
                "prediction_hits": self.prediction_hits,
                "prediction_misses": self.prediction_misses,
                "prediction_total": total_predictions,
                "prediction_accuracy": prediction_accuracy,
                
                # Session statistics
                "session_duration_seconds": session_duration,
                "queries_per_minute": queries_per_minute
            }
    
    def _empty_summary(self) -> Dict:
        """Return empty summary when no queries logged."""
        return {
            "total_queries": 0,
            "tier1_hits": 0,
            "tier2_hits": 0,
            "tier3_hits": 0,
            "offline_hits": 0,
            "tier2_attempts": 0,
            "tier1_hit_rate": 0.0,
            "tier2_hit_rate": 0.0,
            "tier3_hit_rate": 0.0,
            "offline_rate": 0.0,
            "local_hit_rate": 0.0,
            "tier2_prefetch_rate": 0.0,
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "min_latency": 0.0,
            "max_latency": 0.0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_total": 0,
            "prefetch_cache_hit_rate": 0.0,
            "prefetch_fetched": 0,
            "prefetch_skipped": 0,
            "prediction_hits": 0,
            "prediction_misses": 0,
            "prediction_total": 0,
            "prediction_accuracy": 0.0,
            "session_duration_seconds": 0.0,
            "queries_per_minute": 0.0
        }
    
    def _calculate_latency_stats(self) -> Dict:
        """Calculate latency percentiles."""
        all_latencies = []
        
        for event_list in self.events.values():
            for event in event_list:
                if event.get("latency_ms") is not None:
                    all_latencies.append(event["latency_ms"])
        
        if not all_latencies:
            return {
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        return {
            "avg": np.mean(all_latencies),
            "p50": np.percentile(all_latencies, 50),
            "p95": np.percentile(all_latencies, 95),
            "p99": np.percentile(all_latencies, 99),
            "min": np.min(all_latencies),
            "max": np.max(all_latencies)
        }
    
    # ═══════════════════════════════════════════════════════════
    # TIME-SERIES DATA (FOR VISUALIZATION)
    # ═══════════════════════════════════════════════════════════
    
    def get_latency_history(self, limit: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Get latency history for line chart.
        
        Args:
            limit: Max data points to return (default: all)
        
        Returns:
            List of (timestamp, latency_ms) tuples
        """
        with self.lock:
            history = list(self.latency_history)
            
            if limit and limit < len(history):
                # Return most recent N points
                return history[-limit:]
            
            return history
    
    def get_hit_rate_history(self, window_size: int = 10) -> List[Tuple[float, float]]:
        """
        Get rolling TIER 2 hit rate over time.
        
        Args:
            window_size: Number of queries to average over
        
        Returns:
            List of (timestamp, tier2_hit_rate) tuples
        """
        with self.lock:
            # Return stored hit rate history
            return list(self.hit_rate_history)
    
    def get_dynamic_fill_history(self) -> List[Tuple[float, float]]:
        """
        Get dynamic space fill rate over time.
        
        Returns:
            List of (timestamp, fill_rate) tuples
        """
        with self.lock:
            return list(self.dynamic_fill_history)
    
    def get_learning_curve(self) -> Dict:
        """
        Get learning curve data showing TIER 2 prefetch improvement over time.
        
        This is the KEY visualization showing how the system learns!
        
        Returns:
            Dict with:
            - has_data: bool
            - queries: List of query numbers
            - tier2_hit_rates: List of TIER 2 hit rates (neighborhood coverage effectiveness)
            - latencies: List of retrieval latencies
            - sources: List of sources (tier1/tier2/tier3)
            - initial_hit_rate: Starting hit rate
            - current_hit_rate: Latest hit rate
            - improvement: Percentage point improvement
            - trend: "improving", "stable", or "degrading"
        """
        with self.lock:
            if not self.learning_curve_data or len(self.learning_curve_data) < 2:
                return {
                    "has_data": False,
                    "initial_hit_rate": 0.0,
                    "current_hit_rate": 0.0,
                    "improvement": 0.0,
                    "trend": "insufficient_data"
                }
            
            # Extract data for visualization
            queries = [d['query_num'] for d in self.learning_curve_data]
            tier2_hit_rates = [d['tier2_hit_rate'] for d in self.learning_curve_data]
            latencies = [d['latency_ms'] for d in self.learning_curve_data]
            sources = [d['source'] for d in self.learning_curve_data]
            
            initial_hit_rate = tier2_hit_rates[0]
            current_hit_rate = tier2_hit_rates[-1]
            improvement = current_hit_rate - initial_hit_rate
            
            # Determine trend
            if improvement > 10:
                trend = "improving"
            elif improvement < -10:
                trend = "degrading"
            else:
                trend = "stable"
            
            return {
                "has_data": True,
                "queries": queries,
                "tier2_hit_rates": tier2_hit_rates,
                "latencies": latencies,
                "sources": sources,
                "initial_hit_rate": initial_hit_rate,
                "current_hit_rate": current_hit_rate,
                "improvement": improvement,
                "trend": trend,
                "data_points": len(self.learning_curve_data)
            }
    
    # ═══════════════════════════════════════════════════════════
    # EXPORT & PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    
    def export_to_json(self, filepath: Path) -> bool:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to save JSON
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                data = {
                    "summary": self.get_summary(),
                    "latency_history": self.get_latency_history(),
                    "hit_rate_history": self.get_hit_rate_history(),
                    "dynamic_fill_history": self.get_dynamic_fill_history(),
                    "learning_curve": self.get_learning_curve(),
                    "exported_at": time.time()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"[METRICS] Exported to {filepath}")
                return True
            
            except Exception as e:
                logger.error(f"[METRICS] Export failed: {e}")
                return False
    
    def reset(self):
        """Reset all metrics (start fresh session)."""
        with self.lock:
            self.events.clear()
            self.total_queries = 0
            self.tier1_hits = 0
            self.tier2_hits = 0
            self.tier3_hits = 0
            self.offline_fallbacks = 0
            self.tier2_attempts = 0
            self.total_retrieval_latency_ms = 0
            self.prefetch_fetched = 0
            self.prefetch_skipped = 0
            self.prediction_hits = 0
            self.prediction_misses = 0
            self.latency_history.clear()
            self.hit_rate_history.clear()
            self.dynamic_fill_history.clear()
            self.learning_curve_data.clear()
            self.session_start = time.time()
            
            logger.info("[METRICS] Reset all metrics")
