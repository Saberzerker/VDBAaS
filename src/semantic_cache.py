# src/semantic_cache.py
"""
Semantic Cache - Momentum-Based Clustering

Manages semantic clusters with momentum-based centroid updates.
Prevents noise from fragmenting clusters while adapting to genuine drift.

Key Features:
- Momentum-based centroid updates (smooth transitions)
- Semantic drift detection (new cluster creation)
- Cluster reinforcement (access tracking)
- TTL-based eviction (remove stale clusters)

Author: Anonymous
Date: 2025-11-17
"""

import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple, List

from src.config import (
    SEMANTIC_DRIFT_THRESHOLD,
    MOMENTUM_ALPHA,
    MIN_CLUSTER_REINFORCEMENT_SCORE,
    CLUSTER_TTL_SECONDS,
    CLUSTER_MIN_ACCESS_COUNT
)

logger = logging.getLogger(__name__)


class SemanticClusterCache:
    """
    Manages semantic clusters with momentum-based updates.
    
    Each cluster represents a "topic" or "semantic region":
    - Centroid: Weighted average of queries in this region
    - Access count: How often queries fall in this cluster
    - Created/accessed timestamps: For TTL eviction
    - Query IDs: Track which queries belong to cluster
    
    Momentum prevents noise:
    - Single off-topic query doesn't destroy cluster
    - Centroid shifts gradually toward genuine changes
    - Resistance to false drift
    """
    
    def __init__(self):
        """Initialize semantic cache."""
        self.clusters = {}  # {cluster_id: cluster_metadata}
        self.cluster_counter = 0
        self.query_to_cluster = {}  # {query_id: cluster_id}
        
        logger.info("[SEMANTIC] Initialized semantic cache")
        logger.info(f"[SEMANTIC] Drift threshold: {SEMANTIC_DRIFT_THRESHOLD}")
        logger.info(f"[SEMANTIC] Momentum alpha: {MOMENTUM_ALPHA}")
    
    def add_query(
        self,
        query_vector: np.ndarray,
        query_id: str
    ) -> Tuple[int, str, int]:
        """
        Add query to semantic cache.
        
        Flow:
        1. Find nearest cluster
        2. If close enough (< drift threshold): Reinforce cluster
        3. If too far: Create new cluster (drift detected)
        
        Args:
            query_vector: Query embedding (384-dim)
            query_id: Unique query identifier
        
        Returns:
            (cluster_id, action, access_count)
            
            action values:
            - "new_cluster": First cluster OR drift detected
            - "reinforced": Added to existing cluster
        """
        # First query ever?
        if not self.clusters:
            cluster_id = self._create_cluster(query_vector, query_id)
            return cluster_id, "new_cluster", 1
        
        # Find nearest cluster
        nearest_cluster_id, distance = self._find_nearest_cluster(query_vector)
        
        if distance < SEMANTIC_DRIFT_THRESHOLD:
            # REINFORCE existing cluster
            cluster = self.clusters[nearest_cluster_id]
            
            # Momentum-based centroid update
            # Formula: new_centroid = α * old + (1-α) * new
            # α=0.9 means 90% old, 10% new (smooth transition)
            old_centroid = cluster["centroid"]
            new_centroid = (
                MOMENTUM_ALPHA * old_centroid + 
                (1 - MOMENTUM_ALPHA) * query_vector
            )
            
            # Normalize to prevent drift in vector magnitude
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            
            # Update cluster
            cluster["centroid"] = new_centroid
            cluster["last_access"] = time.time()
            cluster["access_count"] += 1
            cluster["query_ids"].append(query_id)
            
            self.query_to_cluster[query_id] = nearest_cluster_id
            
            logger.debug(f"[SEMANTIC] Reinforced cluster {nearest_cluster_id} "
                        f"(distance={distance:.3f}, access={cluster['access_count']})")
            
            return nearest_cluster_id, "reinforced", cluster["access_count"]
        
        else:
            # DRIFT DETECTED → Create new cluster
            cluster_id = self._create_cluster(query_vector, query_id)
            
            logger.info(f"[SEMANTIC] Drift detected (distance={distance:.3f} > {SEMANTIC_DRIFT_THRESHOLD})")
            logger.info(f"[SEMANTIC] Created new cluster {cluster_id}")
            
            return cluster_id, "new_cluster", 1
    
    def _create_cluster(self, query_vector: np.ndarray, query_id: str) -> int:
        """
        Create new semantic cluster.
        
        Args:
            query_vector: Centroid for new cluster
            query_id: First query in cluster
        
        Returns:
            cluster_id: New cluster's ID
        """
        cluster_id = self.cluster_counter
        self.cluster_counter += 1
        
        # Normalize centroid
        centroid = query_vector / np.linalg.norm(query_vector)
        
        self.clusters[cluster_id] = {
            "id": cluster_id,
            "centroid": centroid,
            "created_at": time.time(),
            "last_access": time.time(),
            "access_count": 1,
            "query_ids": [query_id]
        }
        
        self.query_to_cluster[query_id] = cluster_id
        
        logger.info(f"[SEMANTIC] Created cluster {cluster_id}")
        
        return cluster_id
    
    def _find_nearest_cluster(self, query_vector: np.ndarray) -> Tuple[int, float]:
        """
        Find nearest cluster to query vector.
        
        Args:
            query_vector: Query to compare
        
        Returns:
            (nearest_cluster_id, distance)
        """
        min_distance = float('inf')
        nearest_id = None
        
        for cluster_id, cluster in self.clusters.items():
            distance = self._cosine_distance(query_vector, cluster["centroid"])
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = cluster_id
        
        return nearest_id, min_distance
    
    def get_centroid(self, cluster_id: int) -> Optional[np.ndarray]:
        """
        Get cluster centroid for trajectory guidance.
        
        Used by anchor system to bias predictions toward cluster.
        
        Args:
            cluster_id: Cluster to get centroid from
        
        Returns:
            Centroid vector (copy), or None if cluster doesn't exist
        """
        if cluster_id in self.clusters:
            return self.clusters[cluster_id]["centroid"].copy()
        
        logger.warning(f"[SEMANTIC] Cluster {cluster_id} not found")
        return None
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict]:
        """
        Get complete cluster information.
        
        Args:
            cluster_id: Cluster to query
        
        Returns:
            Cluster metadata dict, or None if not found
        """
        return self.clusters.get(cluster_id)
    
    def evict_stale_clusters(self) -> List[int]:
        """
        Evict clusters that haven't been accessed recently.
        
        Eviction criteria:
        - Idle time > CLUSTER_TTL_SECONDS
        - Access count < CLUSTER_MIN_ACCESS_COUNT
        
        Returns:
            List of evicted cluster IDs
        """
        current_time = time.time()
        evicted = []
        
        for cluster_id, cluster in list(self.clusters.items()):
            idle_time = current_time - cluster["last_access"]
            
            # Evict if:
            # 1. Idle for too long AND
            # 2. Not accessed enough (indicates low relevance)
            if (idle_time > CLUSTER_TTL_SECONDS and 
                cluster["access_count"] < CLUSTER_MIN_ACCESS_COUNT):
                
                # Remove cluster
                del self.clusters[cluster_id]
                
                # Remove query mappings
                for query_id in cluster["query_ids"]:
                    if query_id in self.query_to_cluster:
                        del self.query_to_cluster[query_id]
                
                evicted.append(cluster_id)
                
                logger.info(f"[SEMANTIC] Evicted cluster {cluster_id} "
                           f"(idle={idle_time:.0f}s, access={cluster['access_count']})")
        
        if evicted:
            logger.info(f"[SEMANTIC] Evicted {len(evicted)} stale clusters")
        
        return evicted
    
    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate cosine distance (1 - cosine similarity).
        
        Returns value in [0, 2]:
        - 0.0 = identical vectors
        - 1.0 = orthogonal vectors
        - 2.0 = opposite vectors
        """
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Cosine similarity
        similarity = np.dot(v1_norm, v2_norm)
        
        # Convert to distance
        distance = 1.0 - similarity
        
        return distance
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive semantic cache statistics.
        
        Returns:
            Dict with cluster counts, sizes, access patterns
        """
        if not self.clusters:
            return {
                "total_clusters": 0,
                "total_queries": 0,
                "avg_cluster_size": 0.0,
                "max_cluster_size": 0,
                "avg_access_count": 0.0,
                "max_access_count": 0
            }
        
        cluster_sizes = [len(c["query_ids"]) for c in self.clusters.values()]
        access_counts = [c["access_count"] for c in self.clusters.values()]
        
        return {
            "total_clusters": len(self.clusters),
            "total_queries": sum(cluster_sizes),
            "avg_cluster_size": np.mean(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
            "avg_access_count": np.mean(access_counts),
            "max_access_count": max(access_counts),
            "min_access_count": min(access_counts)
        }
    
    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        Get distribution of queries across clusters.
        
        Returns:
            {cluster_id: query_count}
        """
        return {
            cluster_id: len(cluster["query_ids"])
            for cluster_id, cluster in self.clusters.items()
        }
    
    def reset(self):
        """Reset semantic cache (clear all clusters)."""
        self.clusters.clear()
        self.query_to_cluster.clear()
        self.cluster_counter = 0
        logger.info("[SEMANTIC] Reset cache")