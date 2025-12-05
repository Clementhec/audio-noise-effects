import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from ast import literal_eval


from utils import get_embeddings
from similarity import find_similar_sounds
from text_processing import SpeechSegmenter
from video_preprocessing import extract_audio_from_video
from video_audio_merger import run_complete_video_audio_merge


def setup_directories(output_dir: str = "data"):
    """Create necessary directories if they don't exist.

    Args:
        output_dir: Base output directory for all generated files
    """
    # Base output directory and subdirectories for pipeline stages
    directories = [
        output_dir,  # Base output directory
        f"{output_dir}/audio",  # Extracted audio files
        f"{output_dir}/soundbible",  # soundbible sounds metadata
        f"{output_dir}/soundbible/downloaded_sounds",  # soundbible download directory
        f"{output_dir}/speech_to_text",  # Transcription and word timing files
        f"{output_dir}/embeddings",  # Speech embeddings
        f"{output_dir}/similarity",  # Similarity matching results
        f"{output_dir}/filtered",  # LLM-filtered results
        f"{output_dir}/output",  # Final video output
    ]

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

    # Extract audio
    result_path = extract_audio_from_video(
        video_path, output_path=audio_path, sample_rate=sample_rate, channels=channels
    )

    print(f"Audio extraction complete!")
    print(f"Saved to: {result_path}")

    return result_path


def run_stt_step(
    audio_path: Path, transcription_path: Path, word_timing_path: Path
) -> tuple[Path, Path]:
    """
    Step 2: Run Speech-to-Text processing.

    Args:
        audio_path: Path to audio file
        transcription_path: Path where transcription should be saved
        word_timing_path: Path where word timing should be saved

    Returns:
        Tuple of (transcription_path, word_timing_path)
    """
    print("=" * 70)
    print("STEP 2: Run Speech-to-Text (ElevenLabs)")
    print("=" * 70)

    print(f"Audio file: {audio_path}")

    from speech_to_text import transcribe_audio_file

    print("Running STT transcription...")
    result = transcribe_audio_file(
        audio_file_path=audio_path,
        word_timing_path=word_timing_path,
        transcription_path=transcription_path,
        language_code="en",
    )

    # Get output file paths from result (use provided paths as fallback)
    output_files = result.get("output_files", {})
    result_transcription_path = Path(
        output_files.get("transcription", transcription_path)
    )
    result_word_timing_path = Path(output_files.get("word_timing", word_timing_path))

    print(f"STT processing complete!")
    print(f"Transcription: {result_transcription_path}")
    print(f"Word timings: {result_word_timing_path}")
    print(f"Full text: {result['full_transcript'][:100]}...")

    return result_transcription_path, result_word_timing_path


def run_embeddings_step(
    transcription_path: Path,
    word_timing_path: Path,
    embeddings_path: Path,
    force_regenerate: bool = True,
) -> Path:
    """
    Step 3: Generate embeddings from transcription.

    Args:
        transcription_path: Path to full_transcription.json
        word_timing_path: Path to word_timing.json
        embeddings_path: Path where embeddings should be saved
        force_regenerate: If True, regenerate embeddings even if file exists

    Returns:
        Path to generated embeddings CSV
    """
    print("=" * 70)
    print("Generate Speech Embeddings")
    print("=" * 70)

    output_path = embeddings_path

    # Check if embeddings already exist
    if output_path.exists() and not force_regenerate:
        print(f"Embeddings already exist: {output_path}")
        print("Skipping generation (use --force-regenerate to recreate)")
        return output_path

    print(f"Transcription file: {transcription_path}")
    print(f"Word timing file: {word_timing_path}")

    def parse_time_string(time_str: str) -> float:
        """Convert time string from STT format to float seconds."""
        return float(time_str.rstrip("s"))

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

    print(f'Transcript: "{transcript[:80]}..."')
    print(f"Total words: {len(word_timings)}")

    # Segment the transcript
    print("Segmenting transcript...")
    segmenter = SpeechSegmenter(max_words_per_segment=15)
    segments = segmenter.segment_by_sentences(transcript, word_timings)

    print(f"Created {len(segments)} segments")

    if not segments:
        raise ValueError("No segments created!")

    print("Generating embeddings...")
    segment_texts = [seg["text"] for seg in segments]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    embeddings = get_embeddings(
        segment_texts, model=EMBEDDING_MODEL, show_progress=True
    )
    print(f"Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")

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

    df["embedding"] = df["embedding"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    df.to_csv(output_path, index=False)

    print(f"Embeddings saved to: {output_path}")
    print(f"Total segments: {len(df)}")

    return output_path


def run_llm_filtering_step(
    similarity_results_path: Path,
    filtered_results_path: Path,
    max_sounds: Optional[int] = None,
) -> Path:
    """
    Use LLM to filter and select best sound matches.

    Args:
        similarity_results_path: Path to similarity matching results JSON
        filtered_results_path: Path where filtered results should be saved
        max_sounds: Maximum number of sentences to select for sound effects (None = LLM decides)

    Returns:
        Path to filtered results JSON
    """
    print("=" * 70)
    print("STEP 5: LLM Filtering")
    print("=" * 70)

    print(f"Similarity results: {similarity_results_path}")
    if max_sounds:
        print(f"Max sounds to select: {max_sounds}")
    else:
        print("Max sounds: LLM will decide")

    try:
        import json
        from llm_filtering import filter_sounds

        # Load similarity results
        print("Loading similarity results...")
        with open(similarity_results_path, "r", encoding="utf-8") as f:
            similarity_data = json.load(f)

        print(f"Found {len(similarity_data)} speech segments")

        # Run LLM filtering
        print("Running LLM analysis...")
        print("  The LLM will:")
        print("  - Determine which sentences benefit most from sound effects")
        print("  - Identify the specific word where to place each sound")
        print("  - Select the most appropriate sound from top matches")
        print()

        output_path = filtered_results_path

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = filter_sounds(
            similarity_data=similarity_data,
            max_sounds=max_sounds,
            # keep_only_with_sound=True,
            output_file=str(output_path),
        )

        filtered_count = len(result.get("filtered_sounds", []))

        print(f"LLM filtering complete!")
        print(f"Selected {filtered_count} segments for sound effects")
        print(f"Results saved to: {output_path}")

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

        return output_path

    except ImportError as e:
        print(f" Error: Missing dependencies for LLM filtering")
        print(f"  {e}")
        print()
        print("Install required packages:")
        print("  pip install google-generativeai")
        raise
    except Exception as e:
        print(f" Error during LLM filtering: {e}")
        raise


def run_semantic_matching_step(
    embeddings_path: Path,
    similarity_results_path: Path,
    sound_embeddings_path: Path,
    top_k: int = 5,
) -> Path:
    """
    Step 4: Match speech embeddings with sound effects.

    Args:
        embeddings_path: Path to speech embeddings CSV
        similarity_results_path: Path where similarity results should be saved
        top_k: Number of top similar sounds to find for each segment

    Returns:
        Path to generated similarity results JSON
    """
    print("=" * 70)
    print("Semantic Matching with Sound Effects")
    print("=" * 70)
    print()

    print(f"Speech embeddings: {embeddings_path}")
    print(f"Sound embeddings: {sound_embeddings_path}")
    print()

    # Load speech embeddings
    print("Loading speech embeddings...")
    df_speech = pd.read_csv(embeddings_path)

    # Convert embeddings from string to list/array
    if "embedding" in df_speech.columns:
        if isinstance(df_speech["embedding"].iloc[0], str):
            df_speech["embedding"] = df_speech["embedding"].apply(literal_eval)

    print(f"Loaded {len(df_speech)} speech segments")
    print()

    # Load sound embeddings
    print("Loading sound embeddings...")
    df_sounds = pd.read_csv(sound_embeddings_path)

    # Convert embeddings from string to list/array
    if "embedding" in df_sounds.columns:
        if isinstance(df_sounds["embedding"].iloc[0], str):
            df_sounds["embedding"] = df_sounds["embedding"].apply(literal_eval)

    print(f"Loaded {len(df_sounds)} sound effects")

    results = find_similar_sounds(
        df_speech=df_speech,
        df_sounds=df_sounds,
        top_k=top_k,
        save_to_json_file=True,
        output_path=str(similarity_results_path),
    )

    print(f"Matched {len(results)} speech segments")
    print(f"Results saved to: {similarity_results_path}")

    return similarity_results_path


def validate_args(args):
    # Validate sound intensity
    if not 0.0 <= args.sound_intensity <= 1.0:
        print("Error: --sound-intensity must be between 0.0 and 1.0")
        sys.exit(1)


def main():
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
        default="data",
        help="Output directory for intermediate files (default: data)",
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

    validate_args(args)
    setup_directories(output_dir=args.output_dir)

    # Initialize paths from default locations
    input_video_path = Path(args.video)
    base_name = input_video_path.stem
    audio_base_path = Path(args.output_dir) / "audio" / (Path(args.video).stem + ".wav")
    soundbible_embeddings_path = Path(args.output_dir) / "embeddings/soundbible.csv"
    soundbible_details_path = Path(args.output_dir) / "soundbible"
    soundbible_download_dir = Path(args.output_dir) / Path(
        "soundbible/downloaded_sounds"
    )
    sound_audio_urls_path = soundbible_details_path / "soundbible_audio_files.csv"
    transcription_path = Path(args.output_dir) / Path(
        f"speech_to_text/{base_name}_full_transcription.json"
    )
    word_timing_path = Path(args.output_dir) / Path(
        f"speech_to_text/{base_name}_word_timing.json"
    )
    embeddings_path = Path(args.output_dir) / Path(
        f"embeddings/{base_name}_video_speech_embeddings.csv"
    )
    similarity_results_path = Path(args.output_dir) / Path(
        f"similarity/{base_name}_similarity.json"
    )
    filtered_results_path = Path(args.output_dir) / Path(
        f"filtered/{base_name}_video_filtered_sounds.json"
    )
    output_video_path = Path(args.output_dir) / Path(
        f"output/{base_name}_soundeasy.mp4"
    )

    audio_path = extract_audio(
        args.video,
        output_dir=audio_base_path,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    if args.run_stt:
        transcription_path, word_timing_path = run_stt_step(
            audio_path=audio_path,
            transcription_path=transcription_path,
            word_timing_path=word_timing_path,
        )
    elif args.run_embeddings or args.run_video_merge:
        if not transcription_path.exists():
            print(f"Error: missing {transcription_path}")
            sys.exit(1)
        if not word_timing_path.exists():
            print(f"Error: missing {word_timing_path}")
            sys.exit(1)

    if args.run_embeddings:
        embeddings_path = run_embeddings_step(
            transcription_path,
            word_timing_path,
            embeddings_path,
            force_regenerate=args.force_regenerate,
        )
    elif args.run_matching:
        if not embeddings_path.exists():
            print(f"Missing video speech embeddings : {embeddings_path}")
            sys.exit(1)
        print(f"Use existing video speech embeddings : {embeddings_path}")

        from utils.sound_embedding.sound_embedder import SoundEmbedder
        from sounds.soundbible import (
            fetch_sound_hrefs,
            fetch_sound_details_from_hrefs,
            clean_sounds_description,
            fetch_audio_urls_from_details,
        )

        base_url = "https://soundbible.com/free-sound-effects-{}.html"
        sound_links_path = soundbible_details_path / "soundbible_links.csv"
        if not sound_links_path.exists():
            sound_hrefs = fetch_sound_hrefs(base_url)
            sound_hrefs.to_csv(sound_links_path, index=False)
        sound_details_path = soundbible_details_path / "soundbible_details.csv"
        if not sound_details_path.exists():
            sound_hrefs = pd.read_csv(sound_links_path)
            sound_details = fetch_sound_details_from_hrefs(sound_hrefs)
            sound_details.to_csv(sound_details_path, index=False)
        sound_details_clean_path = (
            soundbible_details_path / "soundbible_details_clean.csv"
        )
        if not sound_details_clean_path.exists():
            sound_details = pd.read_csv(sound_details_path)
            sound_details_clean = clean_sounds_description(sound_details)
            sound_details_clean.to_csv(sound_details_clean_path, index=False)
        if not sound_audio_urls_path.exists():
            sound_details = pd.read_csv(sound_details_path)
            sound_audio_urls = fetch_audio_urls_from_details(sound_details)
            sound_audio_urls.to_csv(sound_audio_urls_path, index=False)
        sound_audio_urls = pd.read_csv(sound_audio_urls_path)
        soundbible_details = pd.read_csv(sound_details_clean_path)
        print(soundbible_details.head(10))
        print()
        print(sound_audio_urls[["title", "audio_url_wav"]])
        soundbible_details_full = soundbible_details.merge(
            sound_audio_urls[["title", "audio_url_wav"]],
            on="title",
            how="left"
        )
        if not soundbible_embeddings_path.exists():
            SoundEmbedder().process_sound_dataframe(
                soundbible_details_full, output_path=soundbible_embeddings_path
            )

    if args.run_matching:
        similarity_results_path = run_semantic_matching_step(
            embeddings_path,
            similarity_results_path,
            top_k=args.top_k,
            sound_embeddings_path=soundbible_embeddings_path,
        )
    elif args.run_llm_filter:
        if not similarity_results_path.exists():
            print(f"Missing similarity results : {similarity_results_path}")
            sys.exit(1)
        print(f"Using existing similarity results : {similarity_results_path}")

    if args.run_llm_filter:
        filtered_results_path = run_llm_filtering_step(
            similarity_results_path,
            filtered_results_path,
            max_sounds=args.max_sounds,
        )
    elif args.run_video_merge:
        if not filtered_results_path.exists():
            print(f"Error : missing filtered results : {filtered_results_path}")
            sys.exit(1)
        print(f"Using existing filtered results: {filtered_results_path}")

    if args.run_video_merge:
        if not audio_path.exists():
            print(f"Missing audio path : {audio_path}")
            sys.exit(1)

        final_video_path = run_complete_video_audio_merge(
            video_path=input_video_path,
            filtered_results_path=filtered_results_path,
            speech_embedding_file=embeddings_path,
            word_timing_path=word_timing_path,
            original_audio_path=audio_path,
            output_video_path=output_video_path,
            sound_intensity=args.sound_intensity,
            sound_duration=args.sound_duration,
            metadata_path=sound_audio_urls_path,
            soundbible_download_dir=soundbible_download_dir,
        )

        print("Generated files:")
        print(f"Audio: {audio_path}")
        print(f"Final video path : {final_video_path}")


if __name__ == "__main__":
    main()
