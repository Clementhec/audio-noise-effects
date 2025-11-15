"""
LLM Filtering Module

This module uses Google's Gemini LLM to intelligently filter similarity matches
and determine which sentences benefit most from sound effects.
"""

from .filtering import (
    filter_sounds,
    filter_from_file,
    get_gemini_model,
    create_prompt,
    clean_json_response
)

__all__ = [
    'filter_sounds',
    'filter_from_file',
    'get_gemini_model',
    'create_prompt',
    'clean_json_response'
]
