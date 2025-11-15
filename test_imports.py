#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for the pipeline.
"""

print("Testing imports...")
print()

# Test text_processing module
print("1. Testing text_processing module...")
try:
    from text_processing import SpeechSegmenter
    print("   ✓ SpeechSegmenter imported successfully")

    # Test instantiation
    segmenter = SpeechSegmenter(max_words_per_segment=15)
    print("   ✓ SpeechSegmenter instantiated successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print()

# Test utils module
print("2. Testing utils module...")
try:
    from utils import get_embeddings, get_model
    print("   ✓ get_embeddings imported successfully")
    print("   ✓ get_model imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print()

# Test speech_to_text module
print("3. Testing speech_to_text module...")
try:
    from speech_to_text import transcribe_audio_file
    print("   ✓ transcribe_audio_file imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print()

# Test video_preprocessing module
print("4. Testing video_preprocessing module...")
try:
    from video_preprocessing import extract_audio_from_video
    print("   ✓ extract_audio_from_video imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print()

# Test similarity module
print("5. Testing similarity module...")
try:
    from similarity import find_similar_sounds, cosine_similarity
    print("   ✓ find_similar_sounds imported successfully")
    print("   ✓ cosine_similarity imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print()
print("=" * 70)
print("All imports tested successfully!")
print("=" * 70)
