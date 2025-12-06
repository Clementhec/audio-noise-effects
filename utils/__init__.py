"""
Utilities Package

This package provides utility functions for embeddings, sound processing, and other helpers.
"""

from .embeddings_utils import (
    get_model,
    get_embedding,
    get_embeddings,
    cosine_similarity,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
    get_embedding_dimension,
)
from .logger import setup_logger, get_logger, project_logger

__all__ = [
    "get_model",
    "get_embedding",
    "get_embeddings",
    "cosine_similarity",
    "distances_from_embeddings",
    "indices_of_nearest_neighbors_from_distances",
    "get_embedding_dimension",
    "setup_logger",
    "get_logger",
    "project_logger",
]
