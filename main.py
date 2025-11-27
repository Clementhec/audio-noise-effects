#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Video Preprocessing Pipeline
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

os.environ["PATH"] = "/Users/clementabiven/.local/bin:" + os.environ.get("PATH", "")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "video_preprocessing"))

from video_preprocessing import extract_audio_from_video


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ["speech_to_text/input", "speech_to_text/output", "data"]

    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")


def extract_audio(
    video_path: str,
    output_dir: str = "speech_to_text/input",
    sample_rate: int = 16000,
    channels: int = 1,
    extension: str = ".mp4",
) -> Path:
    """
    Extract audio from video file.

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

    if not video_path.suffix == extension:
        raise ValueError(
            f"Video extension is set to '{extension}' but got video '{video_path}'"
        )

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
        video_path, output_path=audio_path, sample_rate=sample_rate, channels=channels
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
        from speech_to_text import transcribe_audio_file

        print("Running STT transcription...")
        result = transcribe_audio_file(
            audio_file_path=audio_path,
            output_dir="speech_to_text/output",
            language_code="en",
        )

        # Get output file paths from result
        output_files = result.get("output_files", {})
        transcription_path = Path(
            output_files.get(
                "transcription", "speech_to_text/output/full_transcription.json"
            )
        )
        word_timing_path = Path(
            output_files.get("word_timing", "speech_to_text/output/word_timing.json")
        )

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
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        raise
    except Exception as e:
        print(f"✗ Error during STT processing: {e}")
        raise


def run_embeddings_step(
    transcription_path: Path, word_timing_path: Path, force_regenerate: bool = True
) -> Path:
    """
    Step 3: Generate embeddings from transcription.

    Args:
        transcription_path: Path to full_transcription.json
        word_timing_path: Path to word_timing.json
        force_regenerate: If True, regenerate embeddings even if file exists

    Returns:
        Path to generated embeddings CSV
    """
    print("=" * 70)
    print("Generate Speech Embeddings")
    print("=" * 70)
    print()

    output_path = Path("data/video_speech_embeddings.csv")

    # Check if embeddings already exist
    if output_path.exists() and not force_regenerate:
        print(f"✓ Embeddings already exist: {output_path}")
        print("  Skipping generation (use --force-regenerate to recreate)")
        print()
        return output_path

    print(f"Transcription file: {transcription_path}")
    print(f"Word timing file: {word_timing_path}")
    print()

    # Import required modules
    import json
    import pandas as pd
    import numpy as np
    from text_processing import SpeechSegmenter
    from utils import get_embeddings

    def parse_time_string(time_str: str) -> float:
        """Convert time string from STT format to float seconds."""
        return float(time_str.rstrip("s"))

    # Load STT data
    print("Loading STT output...")
    with open(word_timing_path, "r") as f:
        raw_word_timings = json.load(f)

    with open(transcription_path, "r") as f:
        transcription_data = json.load(f)

    # Handle both formats (from process_stt_embeddings.py or stt_elevenlabs.py)
    if isinstance(transcription_data, list):
        transcript = transcription_data[0]["transcription"]
    elif "full_transcript" in transcription_data:
        transcript = transcription_data["full_transcript"]
    elif "segment_result" in transcription_data:
        transcript = transcription_data["segment_result"][0]["transcription"]
    else:
        raise ValueError("Unknown transcription format")

    # Convert word timings to expected format
    word_timings = []
    for wt in raw_word_timings:
        if wt["word"].strip():
            word_timings.append(
                {
                    "word": wt["word"].strip(),
                    "start_time": parse_time_string(wt["startTime"]),
                    "end_time": parse_time_string(wt["endTime"]),
                }
            )

    print(f'  Transcript: "{transcript[:80]}..."')
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
    segment_texts = [seg["text"] for seg in segments]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    embeddings = get_embeddings(
        segment_texts, model=EMBEDDING_MODEL, show_progress=True
    )
    print(f"  Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    print()

    # Create DataFrame
    print("Creating DataFrame...")
    output_data = []
    for i, segment in enumerate(segments):
        output_data.append(
            {
                "segment_id": segment["segment_id"],
                "text": segment["text"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["end_time"] - segment["start_time"],
                "word_count": segment["word_count"],
                "embedding": embeddings[i],
                "embedding_model": EMBEDDING_MODEL,
            }
        )

    df = pd.DataFrame(output_data)

    # Save to CSV
    df_csv = df.copy()
    df_csv["embedding"] = df_csv["embedding"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    df_csv.to_csv(output_path, index=False)

    print(f"✓ Embeddings generated successfully!")
    print(f"  Saved to: {output_path}")
    print(f"  Total segments: {len(df)}")
    print()

    return output_path


def run_video_audio_merge_step(
    video_path: Path,
    filtered_results_path: Path,
    word_timing_path: Path,
    original_audio_path: Path,
    sound_intensity: float = 0.3,
    sound_duration: Optional[float] = None,
) -> Path:
    """
    Step 6: Merge sound effects with video.

    Args:
        video_path: Path to original video file
        filtered_results_path: Path to LLM filtered sounds JSON
        word_timing_path: Path to word timing JSON
        original_audio_path: Path to original audio from video
        sound_intensity: Volume level for sound effects (0.0-1.0, default: 0.3)
        sound_duration: Max duration for each sound effect in seconds (None = full sound)

    Returns:
        Path to final video with merged audio
    """
    print("=" * 70)
    print("STEP 6: Merge Sound Effects with Video")
    print("=" * 70)
    print()

    print(f"Video file: {video_path}")
    print(f"Filtered sounds: {filtered_results_path}")
    print(f"Word timings: {word_timing_path}")
    print(f"Original audio: {original_audio_path}")
    print(f"Sound intensity: {sound_intensity}")
    if sound_duration:
        print(f"Sound duration limit: {sound_duration}s")
    else:
        print("Sound duration: Full length")
    print()

    try:
        from video_audio_merger import run_complete_video_audio_merge

        output_video_path = Path("output/final_video_with_sounds.mp4")

        print("Starting video-audio merge pipeline...")
        print()

        final_video_path = run_complete_video_audio_merge(
            video_path=video_path,
            filtered_results_path=filtered_results_path,
            word_timing_path=word_timing_path,
            original_audio_path=original_audio_path,
            output_video_path=output_video_path,
            sound_intensity=sound_intensity,
            sound_duration=sound_duration,
        )

        if final_video_path:
            print()
            print(f"✓ Video-audio merge complete!")
            print(f"  Final video: {final_video_path}")
            print()
            return final_video_path
        else:
            print()
            print(f"⚠ Video-audio merge skipped (no sounds to add)")
            print()
            return None

    except ImportError as e:
        print(f"✗ Error: Missing dependencies for video-audio merge")
        print(f"  {e}")
        print()
        print("Install required packages:")
        print("  pip install pydub requests")
        print("  Also ensure ffmpeg is installed on your system")
        raise
    except Exception as e:
        print(f"✗ Error during video-audio merge: {e}")
        raise


def run_llm_filtering_step(
    similarity_results_path: Path, max_sounds: Optional[int] = None
) -> Path:
    """
    Step 5: Use LLM to intelligently filter and select best sound matches.

    Args:
        similarity_results_path: Path to similarity matching results JSON
        max_sounds: Maximum number of sentences to select for sound effects (None = LLM decides)

    Returns:
        Path to filtered results JSON
    """
    print("=" * 70)
    print("STEP 5: LLM Intelligent Filtering")
    print("=" * 70)
    print()

    print(f"Similarity results: {similarity_results_path}")
    if max_sounds:
        print(f"Max sounds to select: {max_sounds}")
    else:
        print("Max sounds: LLM will decide")
    print()

    try:
        import json
        from llm_filtering import filter_sounds

        # Load similarity results
        print("Loading similarity results...")
        with open(similarity_results_path, "r", encoding="utf-8") as f:
            similarity_data = json.load(f)

        print(f"  Found {len(similarity_data)} speech segments")
        print()

        # Run LLM filtering
        print("Running LLM analysis...")
        print("  The LLM will:")
        print("  - Determine which sentences benefit most from sound effects")
        print("  - Identify the specific word where to place each sound")
        print("  - Select the most appropriate sound from top matches")
        print()

        output_path = Path("output/video_filtered_sounds.json")

        result = filter_sounds(
            similarity_data=similarity_data,
            max_sounds=max_sounds,
            keep_only_with_sound=True,
            output_file=str(output_path),
        )

        filtered_count = len(result.get("filtered_sounds", []))

        print()
        print(f"✓ LLM filtering complete!")
        print(f"  Selected {filtered_count} segments for sound effects")
        print(f"  Results saved to: {output_path}")
        print()

        # Display sample results
        if result.get("filtered_sounds"):
            print("Sample filtered results:")
            print("-" * 70)
            for i, item in enumerate(result["filtered_sounds"][:3]):  # Show first 3
                print(
                    f'\nSegment {item["speech_index"]}: "{item["speech_text"][:60]}..."'
                )
                print(f"  Target word: '{item.get('target_word', 'N/A')}'")
                if item.get("selected_sound"):
                    sound = item["selected_sound"]
                    print(f"  Selected sound: {sound.get('sound_title', 'N/A')}")
                    print(f"  Reason: {sound.get('reason', 'N/A')[:80]}...")

            if filtered_count > 3:
                print(f"\n... and {filtered_count - 3} more segments")

        print()
        return output_path

    except ImportError as e:
        print(f"✗ Error: Missing dependencies for LLM filtering")
        print(f"  {e}")
        print()
        print("Install required packages:")
        print("  pip install google-generativeai")
        raise
    except Exception as e:
        print(f"✗ Error during LLM filtering: {e}")
        raise


def run_semantic_matching_step(
    embeddings_path: Path,
    top_k: int = 5,
    sound_embeddings_path=Path("data/soundbible_embeddings.csv"),
) -> Path:
    """
    Step 4: Match speech embeddings with sound effects.

    Args:
        embeddings_path: Path to speech embeddings CSV
        top_k: Number of top similar sounds to find for each segment

    Returns:
        Path to generated similarity results JSON
    """
    print("=" * 70)
    print("Semantic Matching with Sound Effects")
    print("=" * 70)
    print()

    if not sound_embeddings_path.exists():
        print(f"Sound embeddings not found: {sound_embeddings_path}")
        print("Please generate sound embeddings first.")
        print("Run: uv run python utils/sound_embedding/generate_embeddings.py")
        print()
        return None

    print(f"Speech embeddings: {embeddings_path}")
    print(f"Sound embeddings: {sound_embeddings_path}")
    print()

    try:
        import pandas as pd
        from ast import literal_eval
        from similarity import find_similar_sounds

        # Load speech embeddings
        print("Loading speech embeddings...")
        df_speech = pd.read_csv(embeddings_path)

        # Convert embeddings from string to list/array
        if "embedding" in df_speech.columns:
            if isinstance(df_speech["embedding"].iloc[0], str):
                df_speech["embedding"] = df_speech["embedding"].apply(literal_eval)

        print(f"  Loaded {len(df_speech)} speech segments")
        print()

        # Load sound embeddings
        print("Loading sound embeddings...")
        df_sounds = pd.read_csv(sound_embeddings_path)

        # Convert embeddings from string to list/array
        if "embedding" in df_sounds.columns:
            if isinstance(df_sounds["embedding"].iloc[0], str):
                df_sounds["embedding"] = df_sounds["embedding"].apply(literal_eval)

        print(f"  Loaded {len(df_sounds)} sound effects")
        print()

        # Run similarity matching
        print(f"Finding top {top_k} similar sounds for each speech segment...")
        # output_path = Path("output/video_similarity_matches.json")
        output_path = os.path.join("similarity/output", "similarity.json")

        results = find_similar_sounds(
            df_speech=df_speech,
            df_sounds=df_sounds,
            top_k=top_k,
            save_to_json_file=True,
            output_path=str(output_path),
        )

        print()
        print(f"✓ Similarity matching complete!")
        print(f"  Matched {len(results)} speech segments")
        print(f"  Results saved to: {output_path}")
        print()

        # Display sample results
        if results:
            print("Sample matches:")
            print("-" * 70)
            for i, result in enumerate(results[:3]):  # Show first 3
                print(
                    f'\nSegment {result["speech_index"]}: "{result["speech_text"][:60]}..."'
                )
                top_match = result["top_matches"][0]
                print(f"  Best match: {top_match['sound_title']}")
                print(f"  Similarity: {top_match['similarity']:.4f}")
                print(f"  Description: {top_match['sound_description'][:80]}...")

            if len(results) > 3:
                print(f"\n... and {len(results) - 3} more segments")

        print()
        return output_path

    except ImportError as e:
        print(f"✗ Error: Missing dependencies")
        print(f"  {e}")
        raise
    except Exception as e:
        print(f"✗ Error during similarity matching: {e}")
        raise


def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Video preprocessing pipeline for audio-noise-effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("video", help="Path to input video file (.mp4)")

    parser.add_argument(
        "--run-stt",
        action="store_true",
        help="Run Speech-to-Text processing after audio extraction",
    )

    parser.add_argument(
        "--run-embeddings",
        action="store_true",
        help="Generate embeddings from transcription (requires transcription files or --run-stt)",
    )

    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run complete pipeline: extract + STT + embeddings + matching + LLM filter + video merge",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels (default: 1 for mono)",
    )

    parser.add_argument(
        "--output-dir",
        default="speech_to_text/input",
        help="Output directory for audio file (default: speech_to_text/input)",
    )

    parser.add_argument(
        "--run-matching",
        action="store_true",
        help="Run similarity matching with sound effects (requires embeddings file or --run-embeddings)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similar sounds to find for each segment (default: 5)",
    )

    parser.add_argument(
        "--run-llm-filter",
        action="store_true",
        help="Use LLM to intelligently filter best sound matches (requires similarity results or --run-matching)",
    )

    parser.add_argument(
        "--max-sounds",
        type=int,
        default=None,
        help="Maximum number of sentences to select for sound effects (default: LLM decides)",
    )

    parser.add_argument(
        "--run-video-merge",
        action="store_true",
        help="Merge sound effects with video to create final output (requires filtered results, word timings, or --run-llm-filter)",
    )

    parser.add_argument(
        "--sound-intensity",
        type=float,
        default=0.3,
        help="Volume level for sound effects, 0.0-1.0 (default: 0.3)",
    )

    parser.add_argument(
        "--sound-duration",
        type=float,
        default=None,
        help="Maximum duration for each sound effect in seconds (default: full sound length)",
    )

    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of embeddings even if they already exist",
    )

    args = parser.parse_args()

    # Full pipeline enables all steps
    if args.full_pipeline:
        args.run_stt = True
        args.run_embeddings = True
        args.run_matching = True
        args.run_llm_filter = True
        args.run_video_merge = True

    # Validate sound intensity
    if not 0.0 <= args.sound_intensity <= 1.0:
        print("Error: --sound-intensity must be between 0.0 and 1.0")
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
        # Initialize paths from default locations
        audio_path = Path(args.output_dir) / (Path(args.video).stem + ".wav")
        transcription_path = Path("speech_to_text/output/full_transcription.json")
        word_timing_path = Path("speech_to_text/output/word_timing.json")
        embeddings_path = Path("data/video_speech_embeddings.csv")
        similarity_results_path = Path("similarity/output/similarity.json")
        filtered_results_path = Path("output/video_filtered_sounds.json")

        # Step 1: Extract audio
        audio_path = extract_audio_step(
            args.video,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            channels=args.channels,
        )

        if args.run_stt:
            transcription_path, word_timing_path = run_stt_step(audio_path)
        elif args.run_embeddings or args.run_video_merge:
            # Check if required files exist
            if not transcription_path.exists():
                print(
                    f"✗ Erreur : Le fichier de transcription est manquant : {transcription_path}"
                )
                print(
                    f"  Exécutez d'abord --run-stt ou assurez-vous que le fichier existe"
                )
                sys.exit(1)
            if not word_timing_path.exists():
                print(
                    f"✗ Erreur : Le fichier de timing des mots est manquant : {word_timing_path}"
                )
                print(
                    f"  Exécutez d'abord --run-stt ou assurez-vous que le fichier existe"
                )
                sys.exit(1)
            print(f"✓ Utilisation de la transcription existante : {transcription_path}")
            print(f"✓ Utilisation du timing existant : {word_timing_path}")
            print()

        if args.run_embeddings:
            embeddings_path = run_embeddings_step(
                transcription_path,
                word_timing_path,
                force_regenerate=args.force_regenerate,
            )
        elif args.run_matching:
            # Check if embeddings exist
            if not embeddings_path.exists():
                print(f"✗ Erreur : Les embeddings sont manquants : {embeddings_path}")
                print(
                    f"  Exécutez d'abord --run-embeddings ou assurez-vous que le fichier existe"
                )
                sys.exit(1)
            print(f"✓ Utilisation des embeddings existants : {embeddings_path}")
            print()

        if args.run_matching:
            similarity_results_path = run_semantic_matching_step(
                embeddings_path, top_k=args.top_k
            )
        elif args.run_llm_filter:
            # Check if similarity results exist
            if not similarity_results_path.exists():
                print(
                    f"✗ Erreur : Les résultats de similarité sont manquants : {similarity_results_path}"
                )
                print(
                    f"  Exécutez d'abord --run-matching ou assurez-vous que le fichier existe"
                )
                sys.exit(1)
            print(
                f"✓ Utilisation des résultats de similarité existants : {similarity_results_path}"
            )
            print()

        if args.run_llm_filter:
            filtered_results_path = run_llm_filtering_step(
                similarity_results_path, max_sounds=args.max_sounds
            )
        elif args.run_video_merge:
            # Check if filtered results exist
            if not filtered_results_path.exists():
                print(
                    f"✗ Erreur : Les résultats filtrés LLM sont manquants : {filtered_results_path}"
                )
                print(
                    f"  Exécutez d'abord --run-llm-filter ou assurez-vous que le fichier existe"
                )
                sys.exit(1)
            print(
                f"✓ Utilisation des résultats filtrés existants : {filtered_results_path}"
            )
            print()

        if args.run_video_merge:
            # Check if original audio exists
            if not audio_path.exists():
                print(f"✗ Erreur : L'audio original est manquant : {audio_path}")
                print(f"  Assurez-vous que l'extraction audio a été effectuée")
                sys.exit(1)

            final_video_path = run_video_audio_merge_step(
                video_path=Path(args.video),
                filtered_results_path=filtered_results_path,
                word_timing_path=word_timing_path,
                original_audio_path=audio_path,
                sound_intensity=args.sound_intensity,
                sound_duration=args.sound_duration,
            )

        print("=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print()
        print("Generated files:")
        print(f"Audio: {audio_path}")
        print(f"Final video path : {final_video_path}")

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
