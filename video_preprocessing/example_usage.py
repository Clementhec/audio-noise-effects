"""
Example usage of video_to_audio utilities

This script demonstrates how to use the video preprocessing utilities
to extract audio from video files.
"""

from pathlib import Path
from video_to_audio import (
    extract_audio_from_video,
    extract_audio_ffmpeg_direct,
    batch_extract_audio
)


def example_single_video():
    """Example: Extract audio from a single video file"""
    print("=" * 70)
    print("Example 1: Extract audio from single video")
    print("=" * 70)
    print()

    # Replace with your actual video file path
    video_file = "path/to/your/video.mp4"

    if not Path(video_file).exists():
        print(f"⚠ Video file not found: {video_file}")
        print("Please update the video_file variable with an actual MP4 file path")
        return

    # Extract audio with default settings (16kHz, mono)
    audio_path = extract_audio_from_video(video_file)
    print(f"\n Audio extracted to: {audio_path}")


def example_custom_settings():
    """Example: Extract audio with custom sample rate and channels"""
    print("\n" + "=" * 70)
    print("Example 2: Extract audio with custom settings")
    print("=" * 70)
    print()

    video_file = "path/to/your/video.mp4"

    if not Path(video_file).exists():
        print(f"⚠ Video file not found: {video_file}")
        print("Please update the video_file variable with an actual MP4 file path")
        return

    # Extract with 44.1kHz stereo for higher quality
    audio_path = extract_audio_from_video(
        video_file,
        output_path="output/high_quality.wav",
        sample_rate=44100,
        channels=2
    )
    print(f"\n High quality audio extracted to: {audio_path}")


def example_ffmpeg_direct():
    """Example: Use direct ffmpeg method"""
    print("\n" + "=" * 70)
    print("Example 3: Extract using direct ffmpeg")
    print("=" * 70)
    print()

    video_file = "path/to/your/video.mp4"

    if not Path(video_file).exists():
        print(f"⚠ Video file not found: {video_file}")
        print("Please update the video_file variable with an actual MP4 file path")
        return

    # Use direct ffmpeg method (alternative to pydub)
    audio_path = extract_audio_ffmpeg_direct(
        video_file,
        sample_rate=16000,
        channels=1
    )
    print(f"\n Audio extracted to: {audio_path}")


def example_batch_processing():
    """Example: Batch process multiple videos"""
    print("\n" + "=" * 70)
    print("Example 4: Batch process multiple videos")
    print("=" * 70)
    print()

    video_dir = "path/to/video/folder"
    output_dir = "audio_output"

    if not Path(video_dir).exists():
        print(f"⚠ Video directory not found: {video_dir}")
        print("Please update the video_dir variable with an actual directory path")
        return

    # Process all MP4 files in directory
    audio_files = batch_extract_audio(
        video_dir=video_dir,
        output_dir=output_dir,
        pattern="*.mp4",
        sample_rate=16000,
        channels=1
    )

    print(f"\n Extracted {len(audio_files)} audio files")
    for audio_file in audio_files:
        print(f"  - {audio_file}")


def example_for_stt_pipeline():
    """Example: Extract audio for STT processing"""
    print("\n" + "=" * 70)
    print("Example 5: Extract for STT Pipeline (Recommended Settings)")
    print("=" * 70)
    print()

    video_file = "path/to/your/video.mp4"

    if not Path(video_file).exists():
        print(f"⚠ Video file not found: {video_file}")
        print("Please update the video_file variable with an actual MP4 file path")
        return

    # Recommended settings for STT pipeline
    # - 16kHz sample rate (Google STT standard)
    # - Mono channel (speech recognition doesn't need stereo)
    audio_path = extract_audio_from_video(
        video_file,
        output_path="STT/input_audio.wav",
        sample_rate=16000,
        channels=1
    )

    print(f"\n Audio ready for STT processing: {audio_path}")
    print("\nNext steps:")
    print("  1. Process with Google STT to get word_timing.json")
    print("  2. Run: uv run python process_stt_embeddings.py")
    print("  3. Run: uv run python semantic_matcher.py")


if __name__ == "__main__":
    print("Video to Audio Extraction Examples")
    print()
    print("This script contains several examples of how to use")
    print("the video_to_audio utilities. Edit the file paths and")
    print("uncomment the examples you want to run.")
    print()

    # Uncomment the examples you want to run:

    # example_single_video()
    # example_custom_settings()
    # example_ffmpeg_direct()
    # example_batch_processing()
    # example_for_stt_pipeline()

    print("\n" + "=" * 70)
    print("ℹ To run examples, uncomment them in the __main__ section")
    print("=" * 70)
