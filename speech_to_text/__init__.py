"""
Speech-to-Text Module

This module provides functions for transcribing audio files using ElevenLabs API.
"""

from .stt_elevenlabs import (
    transcribe_audio_elevenlabs,
    transcribe_audio_file
)

__all__ = [
    'transcribe_audio_elevenlabs',
    'transcribe_audio_file'
]
