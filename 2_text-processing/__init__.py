"""
Text Processing Pipeline for Speech Embeddings

This package provides tools for converting speech transcripts into semantic embeddings
compatible with sound effect embeddings for intelligent audio enhancement.
"""

from .speech_segmenter import SpeechSegmenter, load_stt_output
from .speech_embedder import SpeechEmbeddingPipeline, process_speech_file

__all__ = [
    'SpeechSegmenter',
    'load_stt_output',
    'SpeechEmbeddingPipeline',
    'process_speech_file'
]
