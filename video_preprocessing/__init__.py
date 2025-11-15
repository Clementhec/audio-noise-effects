"""
Video Preprocessing Utilities

This package provides tools for preprocessing video files,
including audio extraction for further processing.
"""

from .video_to_audio import (
    extract_audio_from_video,
    extract_audio_ffmpeg_direct,
    batch_extract_audio
)

__all__ = [
    'extract_audio_from_video',
    'extract_audio_ffmpeg_direct',
    'batch_extract_audio'
]
