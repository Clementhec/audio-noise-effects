"""
Sound Embedding Package

Generates and loads semantic embeddings for sound effects.
Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) for local, offline embedding generation.

Main Components:
- SoundEmbedder: Generate embeddings from sound metadata (title, description, keywords)
- SoundEmbeddingLoader: Load and manage pre-computed embeddings

Compatible with speech embeddings for semantic matching.
"""

from .sound_embedder import (
    SoundEmbedder,
    process_sound_file
)

from .sound_embedding_loader import (
    SoundEmbeddingLoader,
    load_sound_embeddings
)

__all__ = [
    # Classes
    'SoundEmbedder',
    'SoundEmbeddingLoader',

    # Convenience functions
    'process_sound_file',
    'load_sound_embeddings'
]

__version__ = '1.0.0'
