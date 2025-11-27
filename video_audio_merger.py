#!/usr/bin/env python3
"""
Video Audio Merger

This module handles the final step of the pipeline:
1. Map sound titles from LLM filtered results to sound file URLs from metadata
2. Download sound effects if needed
3. Find exact word timings for target words
4. Merge sound effects with original audio track
5. Combine merged audio with video to create final output
"""

import json
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import ffmpeg
from merging_audio.audio_merger import merge_audio_files, AudioEntry
from pydub import AudioSegment


def load_sound_metadata(metadata_path: str = "data/soundbible_metadata.csv") -> pd.DataFrame:
    """
    Load sound effects metadata containing URLs.

    Args:
        metadata_path: Path to metadata CSV file

    Returns:
        DataFrame with sound metadata
    """
    return pd.read_csv(metadata_path)


def find_sound_url(sound_title: str, metadata_df: pd.DataFrame) -> Optional[str]:
    """
    Find the WAV download URL for a given sound title.

    Args:
        sound_title: Title of the sound effect
        metadata_df: DataFrame containing sound metadata

    Returns:
        WAV file URL or None if not found
    """
    matches = metadata_df[metadata_df['title'] == sound_title]

    if len(matches) == 0:
        print(f"   Sound not found in metadata: {sound_title}")
        return None

    # Prefer WAV format, fallback to MP3
    if 'audio_url_wav' in matches.columns and pd.notna(matches.iloc[0]['audio_url_wav']):
        return matches.iloc[0]['audio_url_wav']
    elif 'audio_url' in matches.columns and pd.notna(matches.iloc[0]['audio_url']):
        return matches.iloc[0]['audio_url']

    return None


def download_sound_effect(
    url: str,
    output_path: Path,
    force_download: bool = False
) -> bool:
    """
    Download a sound effect file from URL.

    Args:
        url: URL to download from
        output_path: Local path to save file
        force_download: Re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    if output_path.exists() and not force_download:
        print(f"   Already downloaded: {output_path.name}")
        return True

    try:
        print(f"   Downloading: {output_path.name}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

        print(f"   Downloaded: {output_path.name}")
        return True

    except Exception as e:
        print(f"   Download failed: {e}")
        return False


def parse_time_string(time_str: str) -> float:
    """Convert time string (e.g., '1.23s') to float seconds."""
    return float(time_str.rstrip('s'))


def find_word_timing(
    target_word: str,
    speech_text: str,
    word_timings: List[Dict]
) -> Optional[float]:
    """
    Find the start time of a target word in the speech.

    Args:
        target_word: The word to find timing for
        speech_text: The full speech text containing the word
        word_timings: List of word timing dictionaries from STT

    Returns:
        Start time in seconds, or None if not found
    """
    # Normalize target word (remove punctuation, lowercase)
    target_normalized = target_word.strip().lower().rstrip('.,!?;:')

    # Find the word in the speech text to get approximate position
    n = len(word_timings)
    k = 0
    while k < n:
        wt = word_timings[k]
        if target_normalized == wt['word'].strip().lower().rstrip('.,!?;:'):
            return parse_time_string(wt['startTime'])
        k += 1

    print(f"Timing not found for word {target_word}")


def prepare_sound_effects(
    filtered_results: Dict,
    metadata_df: pd.DataFrame,
    word_timings: List[Dict],
    download_dir: Path = Path("merging_audio/downloaded_sounds")
) -> List[Tuple[Path, float, str]]:
    """
    Prepare sound effects: download files and find timings.

    Args:
        filtered_results: LLM filtered results with sound selections
        metadata_df: Sound metadata DataFrame
        word_timings: Word timing data from STT
        download_dir: Directory to store downloaded sounds

    Returns:
        List of tuples: (sound_file_path, start_time, sound_title)
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    prepared_sounds = []

    print("Preparing sound effects...")
    print()

    for item in filtered_results.get('filtered_sounds', []):
        if not item.get('should_add_sound', False):
            continue

        selected_sound = item.get('selected_sound')
        if not selected_sound:
            continue

        sound_title = selected_sound.get('sound_title')
        target_word = item.get('target_word')
        speech_text = item.get('speech_text')

        if not all([sound_title, target_word, speech_text]):
            continue

        print(f"Processing: '{target_word}' → {sound_title}")

        # Find sound URL
        sound_url = find_sound_url(sound_title, metadata_df)
        if not sound_url:
            continue

        # Determine file extension and output path
        file_ext = '.wav' if sound_url.endswith('.wav') else '.mp3'
        safe_filename = "".join(c for c in sound_title if c.isalnum() or c in (' ', '-', '_'))
        safe_filename = safe_filename.replace(' ', '_')
        output_path = download_dir / f"{safe_filename}{file_ext}"

        # Download sound effect
        if not download_sound_effect(sound_url, output_path):
            continue

        # Convert MP3 to WAV if needed
        if file_ext == '.mp3':
            wav_path = output_path.with_suffix('.wav')
            if not wav_path.exists():
                try:
                    print(f"   Converting to WAV...")
                    audio = AudioSegment.from_mp3(output_path)
                    audio.export(wav_path, format="wav")
                    output_path = wav_path
                    print(f"   Converted to WAV")
                except Exception as e:
                    print(f"   Conversion failed: {e}")
                    continue
            else:
                output_path = wav_path

        # Find word timing
        start_time = find_word_timing(target_word, speech_text, word_timings)
        if start_time is None:
            continue

        prepared_sounds.append((output_path, start_time, sound_title))
        print(f"   Ready: {sound_title} at {start_time:.2f}s")
        print()

    return prepared_sounds


def merge_sounds_with_original_audio(
    original_audio_path: Path,
    prepared_sounds: List[Tuple[Path, float, str]],
    output_path: Path,
    sound_intensity: float = 0.3,
    sound_duration: Optional[float] = None
) -> Path:
    """
    Merge sound effects with the original audio track.

    Args:
        original_audio_path: Path to original audio extracted from video
        prepared_sounds: List of (sound_path, start_time, title) tuples
        output_path: Path for merged audio output
        sound_intensity: Volume level for sound effects (0.0-1.0)
        sound_duration: Max duration for each sound (None = full sound)

    Returns:
        Path to merged audio file
    """
    print("Merging sound effects with original audio...")
    print()

    # Load original audio
    original_audio = AudioSegment.from_file(original_audio_path)
    original_duration = len(original_audio) / 1000.0  # Convert to seconds

    print(f"Original audio duration: {original_duration:.2f}s")
    print(f"Sound effects to add: {len(prepared_sounds)}")
    print()

    # Create audio entries for merging
    audio_entries = []

    # Add original audio as first entry (full duration, full intensity)
    audio_entries.append(AudioEntry(
        StartTime=0.0,
        Duration=original_duration,
        Intensity=1.0,
        Data=original_audio
    ))

    # Add each sound effect
    for sound_path, start_time, sound_title in prepared_sounds:
        try:
            sound = AudioSegment.from_file(sound_path)
            sound_length = len(sound) / 1000.0

            # Use specified duration or full sound length
            duration = sound_duration if sound_duration else sound_length
            duration = min(duration, sound_length)  # Cap at actual sound length

            audio_entries.append(AudioEntry(
                StartTime=start_time,
                Duration=duration,
                Intensity=sound_intensity,
                Data=sound
            ))

            print(f"   Added: {sound_title} at {start_time:.2f}s (duration: {duration:.2f}s)")

        except Exception as e:
            print(f"   Failed to add {sound_title}: {e}")

    print()
    print("Performing audio merge...")

    # Merge all audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_audio = merge_audio_files(
        audio_entries=audio_entries,
        output_path=str(output_path),
        fade_in_duration=0.05,   # Short fade to avoid clicks
        fade_out_duration=0.05
    )

    print(f" Merged audio saved to: {output_path}")
    print()

    return output_path


def combine_audio_with_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path
) -> Path:
    """
    Combine audio track with video file using ffmpeg.

    Args:
        video_path: Path to original video file
        audio_path: Path to merged audio file
        output_path: Path for final video output

    Returns:
        Path to final video file
    """
    print("Combining audio with video...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg-python to replace audio in video
    # Use video from first input and audio from second input
    # Copy video codec (no re-encoding), encode audio as AAC
    # Match shortest stream duration

    try:
        print(f"Running ffmpeg...")
        print(f"  Video: {video_path.name}")
        print(f"  Audio: {audio_path.name}")
        print(f"  Output: {output_path.name}")
        print()

        video_input = ffmpeg.input(str(video_path))
        audio_input = ffmpeg.input(str(audio_path))

        output = ffmpeg.output(
            video_input.video,
            audio_input.audio,
            str(output_path),
            vcodec='copy',        # Copy video stream (no re-encoding)
            acodec='aac',         # Encode audio as AAC
            audio_bitrate='192k', # Audio bitrate
            shortest=None         # Match shortest stream duration
        )

        ffmpeg.run(output, overwrite_output=True)

        print(f"Video created successfully: {output_path}")

        return output_path

    except ffmpeg.Error as e:
        print(f"ffmpeg failed:")
        print(f"{e.stderr.decode() if e.stderr else 'Unknown error'}")
        raise
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg")
        raise


def run_complete_video_audio_merge(
    video_path: Path,
    filtered_results_path: Path,
    word_timing_path: Path,
    original_audio_path: Path,
    output_video_path: Path,
    metadata_path: str = "data/soundbible_metadata.csv",
    sound_intensity: float = 0.3,
    sound_duration: Optional[float] = None
) -> Path:
    """
    Complete pipeline to merge sound effects with video.

    Args:
        video_path: Original video file
        filtered_results_path: LLM filtered sounds JSON
        word_timing_path: STT word timing JSON
        original_audio_path: Original audio extracted from video
        output_video_path: Final output video path
        metadata_path: Sound metadata CSV
        sound_intensity: Volume level for sound effects (0.0-1.0)
        sound_duration: Max duration for each sound (None = full)

    Returns:
        Path to final video file
    """
    # Load data
    print("Loading data...")
    with open(filtered_results_path, 'r', encoding='utf-8') as f:
        filtered_results = json.load(f)

    with open(word_timing_path, 'r', encoding='utf-8') as f:
        word_timings = json.load(f)

    metadata_df = load_sound_metadata(metadata_path)
    print(f"Loaded {len(filtered_results.get('filtered_sounds', []))} filtered sounds")
    print(f"Loaded {len(word_timings)} word timings")
    print(f"Loaded {len(metadata_df)} sound metadata entries")

    # Prepare sound effects (download and find timings)
    prepared_sounds = prepare_sound_effects(
        filtered_results,
        metadata_df,
        word_timings
    )

    if not prepared_sounds:
        print("No sound effects prepared. Skipping merge.")
        return

    print(f" Prepared {len(prepared_sounds)} sound effects")

    # Merge sounds with original audio
    merged_audio_path = Path("output/merged_audio.wav")
    merge_sounds_with_original_audio(
        original_audio_path,
        prepared_sounds,
        merged_audio_path,
        sound_intensity=sound_intensity,
        sound_duration=sound_duration
    )

    # Combine merged audio with video
    final_video_path = combine_audio_with_video(
        video_path,
        merged_audio_path,
        output_video_path
    )

    return final_video_path
