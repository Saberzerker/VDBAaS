# src/utils/embedding_model.py
"""
Embedding model wrapper for the hybrid VDB system.

Wraps sentence-transformers for consistent embedding generation.

Author: Saberzerker
Date: 2025-11-16
"""

import numpy as np
from typing import List, Union
import logging

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, VECTOR_DIMENSION

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for embedding generation.
    
    Provides:
    - Lazy loading (only loads model when first used)
    - Consistent normalization
    - Batch encoding with progress bar
    - Dimension validation
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model wrapper.
        
        Args:
            model_name: Model name (default: from config)
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.expected_dimension = VECTOR_DIMENSION
        
        self._model = None  # Lazy loading
        
        logger.info(f"[EMBEDDING] Model configured: {self.model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Get the underlying model, loading if necessary.
        
        Returns:
            SentenceTransformer instance
        """
        if self._model is None:
            logger.info(f"[EMBEDDING] Loading model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"[EMBEDDING] Model loaded (dimension: {self._model.get_sentence_embedding_dimension()})")
            
            # Validate dimension
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.expected_dimension:
                logger.warning(
                    f"[EMBEDDING] Dimension mismatch: expected {self.expected_dimension}, "
                    f"got {actual_dim}"
                )
        
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings to unit length (default: True)
            show_progress_bar: Show progress for batch encoding
            batch_size: Batch size for encoding
        
        Returns:
            Embeddings array:
            - Shape [dimension] for single text
            - Shape [n, dimension] for list of texts
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Encode
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        
        # Return single embedding if input was single text
        if single_text:
            return embeddings[0]
        
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


# Global singleton instance
_embedding_model_instance = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get global embedding model instance (singleton).
    
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model_instance
    
    if _embedding_model_instance is None:
        _embedding_model_instance = EmbeddingModel()
    
    return _embedding_model_instance