#!/usr/bin/env python3
"""
Main Video Preprocessing Pipeline

This script orchestrates the complete video-to-audio-to-embedding pipeline:
1. Extract audio from video (.mp4 -> .wav)
2. Save audio to speech_to_text/input/
3. Optionally run STT processing
4. Generate embeddings from transcription
5. Match with sound effects

Usage:
    python main.py video.mp4
    python main.py video.mp4 --run-stt
    python main.py video.mp4 --run-stt --run-embeddings
    python main.py video.mp4 --full-pipeline
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "video_preprocessing"))

from video_preprocessing import extract_audio_from_video


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "speech_to_text/input",
        "speech_to_text/output",
        "data"
    ]

    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def extract_audio_step(
    video_path: str,
    output_dir: str = "speech_to_text/input",
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    Step 1: Extract audio from video file.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save audio file
        sample_rate: Audio sample rate (default: 16000 Hz for STT)
        channels: Number of audio channels (default: 1 for mono)

    Returns:
        Path to extracted audio file
    """
    print("=" * 70)
    print("STEP 1: Extract Audio from Video")
    print("=" * 70)
    print()

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output path with same root name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_filename = video_path.stem + ".wav"
    audio_path = output_dir / audio_filename

    print(f"Video file: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Audio file: {audio_filename}")
    print()

    # Extract audio
    result_path = extract_audio_from_video(
        video_path,
        output_path=audio_path,
        sample_rate=sample_rate,
        channels=channels
    )

    print()
    print(f"✓ Audio extraction complete!")
    print(f"  Saved to: {result_path}")
    print()

    return result_path


def run_stt_step(audio_path: Path) -> tuple[Path, Path]:
    """
    Step 2: Run Speech-to-Text processing.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (transcription_path, word_timing_path)
    """
    print("=" * 70)
    print("STEP 2: Run Speech-to-Text (ElevenLabs)")
    print("=" * 70)
    print()

    print(f"Audio file: {audio_path}")
    print()

    try:
        from speech_to_text.stt_elevenlabs import transcribe_audio_elevenlabs
        from io import BytesIO

        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_data = BytesIO(f.read())

        print("Running STT transcription...")
        result = transcribe_audio_elevenlabs(
            audio_source=audio_data,
            save_to_json_file=True
        )

        transcription_path = Path("speech_to_text/output/full_transcription.json")
        word_timing_path = Path("speech_to_text/output/word_timing.json")

        print()
        print(f"✓ STT processing complete!")
        print(f"  Transcription: {transcription_path}")
        print(f"  Word timings: {word_timing_path}")
        print(f"  Full text: {result['full_transcript'][:100]}...")
        print()

        return transcription_path, word_timing_path

    except ImportError as e:
        print(f"✗ Error: Missing dependencies for STT")
        print(f"  {e}")
        print()
        print("Install required packages:")
        print("  pip install elevenlabs python-dotenv requests")
        raise
    except Exception as e:
        print(f"✗ Error during STT processing: {e}")
        raise


def run_embeddings_step(
    transcription_path: Path,
    word_timing_path: Path
) -> Path:
    """
    Step 3: Generate embeddings from transcription.

    Args:
        transcription_path: Path to full_transcription.json
        word_timing_path: Path to word_timing.json

    Returns:
        Path to generated embeddings CSV
    """
    print("=" * 70)
    print("STEP 3: Generate Speech Embeddings")
    print("=" * 70)
    print()

    print(f"Transcription file: {transcription_path}")
    print(f"Word timing file: {word_timing_path}")
    print()

    # Import and run process_stt_embeddings
    import json
    import pandas as pd
    import numpy as np

    sys.path.insert(0, str(project_root / "text-processing"))
    from speech_segmenter import SpeechSegmenter
    from utils.embeddings_utils import get_embeddings

    def parse_time_string(time_str: str) -> float:
        """Convert time string from STT format to float seconds."""
        return float(time_str.rstrip('s'))

    # Load STT data
    print("Loading STT output...")
    with open(word_timing_path, 'r') as f:
        raw_word_timings = json.load(f)

    with open(transcription_path, 'r') as f:
        transcription_data = json.load(f)

    # Handle both formats (from process_stt_embeddings.py or stt_elevenlabs.py)
    if isinstance(transcription_data, list):
        transcript = transcription_data[0]['transcription']
    elif 'full_transcript' in transcription_data:
        transcript = transcription_data['full_transcript']
    elif 'segment_result' in transcription_data:
        transcript = transcription_data['segment_result'][0]['transcription']
    else:
        raise ValueError("Unknown transcription format")

    # Convert word timings to expected format
    word_timings = []
    for wt in raw_word_timings:
        if wt['word'].strip():
            word_timings.append({
                'word': wt['word'].strip(),
                'start_time': parse_time_string(wt['startTime']),
                'end_time': parse_time_string(wt['endTime'])
            })

    print(f"  Transcript: \"{transcript[:80]}...\"")
    print(f"  Total words: {len(word_timings)}")
    print()

    # Segment the transcript
    print("Segmenting transcript...")
    segmenter = SpeechSegmenter(max_words_per_segment=15)
    segments = segmenter.segment_by_sentences(transcript, word_timings)
    print(f"  Created {len(segments)} segments")
    print()

    if not segments:
        raise ValueError("No segments created!")

    # Generate embeddings
    print("Generating embeddings...")
    segment_texts = [seg['text'] for seg in segments]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    embeddings = get_embeddings(
        segment_texts,
        model=EMBEDDING_MODEL,
        show_progress=True
    )
    print(f"  Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    print()

    # Create DataFrame
    print("Creating DataFrame...")
    output_data = []
    for i, segment in enumerate(segments):
        output_data.append({
            'segment_id': segment['segment_id'],
            'text': segment['text'],
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['end_time'] - segment['start_time'],
            'word_count': segment['word_count'],
            'embedding': embeddings[i],
            'embedding_model': EMBEDDING_MODEL
        })

    df = pd.DataFrame(output_data)

    # Save to CSV
    output_path = Path("data/video_speech_embeddings.csv")
    df_csv = df.copy()
    df_csv['embedding'] = df_csv['embedding'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    df_csv.to_csv(output_path, index=False)

    print(f"✓ Embeddings generated successfully!")
    print(f"  Saved to: {output_path}")
    print(f"  Total segments: {len(df)}")
    print()

    return output_path


def run_semantic_matching_step(embeddings_path: Path) -> Path:
    """
    Step 4: Match speech embeddings with sound effects.

    Args:
        embeddings_path: Path to speech embeddings CSV

    Returns:
        Path to generated timeline CSV
    """
    print("=" * 70)
    print("STEP 4: Semantic Matching with Sound Effects")
    print("=" * 70)
    print()

    sound_embeddings_path = Path("data/soundbible_embeddings.csv")

    if not sound_embeddings_path.exists():
        print(f"⚠ Sound embeddings not found: {sound_embeddings_path}")
        print("  Please generate sound embeddings first.")
        print("  Run: uv run python generate_sound_embeddings.py")
        print()
        return None

    print(f"Speech embeddings: {embeddings_path}")
    print(f"Sound embeddings: {sound_embeddings_path}")
    print()

    # Import semantic matcher
    print("Running semantic matching...")
    print("Note: Run this command manually with custom parameters if needed:")
    print(f"  uv run python semantic_matcher.py \\")
    print(f"    {embeddings_path} \\")
    print(f"    {sound_embeddings_path} \\")
    print(f"    --output data/video_timeline.csv")
    print()

    # For now, just return the expected output path
    output_path = Path("data/video_timeline.csv")
    return output_path


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Video preprocessing pipeline for audio-noise-effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract audio only
  python main.py video.mp4

  # Extract audio and run STT
  python main.py video.mp4 --run-stt

  # Run full pipeline (extract + STT + embeddings)
  python main.py video.mp4 --full-pipeline

  # Custom sample rate
  python main.py video.mp4 --sample-rate 44100 --channels 2
        """
    )

    parser.add_argument(
        "video",
        help="Path to input video file (.mp4)"
    )

    parser.add_argument(
        "--run-stt",
        action="store_true",
        help="Run Speech-to-Text processing after audio extraction"
    )

    parser.add_argument(
        "--run-embeddings",
        action="store_true",
        help="Generate embeddings from transcription (requires --run-stt)"
    )

    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run complete pipeline: extract + STT + embeddings + matching"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels (default: 1 for mono)"
    )

    parser.add_argument(
        "--output-dir",
        default="speech_to_text/input",
        help="Output directory for audio file (default: speech_to_text/input)"
    )

    args = parser.parse_args()

    # Full pipeline enables all steps
    if args.full_pipeline:
        args.run_stt = True
        args.run_embeddings = True

    # Validate dependencies
    if args.run_embeddings and not args.run_stt:
        print("Error: --run-embeddings requires --run-stt")
        sys.exit(1)

    print()
    print("=" * 70)
    print("VIDEO PREPROCESSING PIPELINE")
    print("=" * 70)
    print()
    print(f"Video file: {args.video}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Channels: {args.channels}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Setup directories
    setup_directories()
    print()

    try:
        # Step 1: Extract audio
        audio_path = extract_audio_step(
            args.video,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            channels=args.channels
        )

        # Step 2: Run STT (optional)
        if args.run_stt:
            transcription_path, word_timing_path = run_stt_step(audio_path)

            # Step 3: Generate embeddings (optional)
            if args.run_embeddings:
                embeddings_path = run_embeddings_step(
                    transcription_path,
                    word_timing_path
                )

                # Step 4: Semantic matching info
                timeline_path = run_semantic_matching_step(embeddings_path)

        # Final summary
        print("=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print()
        print("Generated files:")
        print(f"  ✓ Audio: {audio_path}")

        if args.run_stt:
            print(f"  ✓ Transcription: speech_to_text/output/full_transcription.json")
            print(f"  ✓ Word timings: speech_to_text/output/word_timing.json")

        if args.run_embeddings:
            print(f"  ✓ Embeddings: data/video_speech_embeddings.csv")

        print()

        if not args.run_stt:
            print("Next steps:")
            print("  1. Run STT processing:")
            print(f"     python main.py {args.video} --run-stt")
            print()
        elif not args.run_embeddings:
            print("Next steps:")
            print("  1. Generate embeddings:")
            print(f"     python main.py {args.video} --run-embeddings")
            print()
        else:
            print("Next steps:")
            print("  1. Run semantic matching:")
            print("     uv run python semantic_matcher.py \\")
            print("       data/video_speech_embeddings.csv \\")
            print("       data/soundbible_embeddings.csv \\")
            print("       --output data/video_timeline.csv")
            print()

        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Pipeline failed")
        print("=" * 70)
        print(f"{e}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
