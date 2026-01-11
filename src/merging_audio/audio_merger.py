from typing import List, Union
from dataclasses import dataclass
import math

# Import audioop before pydub for Python 3.13 compatibility (audioop-lts provides this)
try:
    import audioop
except ImportError:
    pass
from pydub import AudioSegment


@dataclass
class AudioEntry:
    """Represents an audio entry to be merged into the final audio file.

    Attributes:
        StartTime: Start time in seconds where this audio should begin
        Duration: Duration in seconds to play (capped at audio file length if exceeds)
        Intensity: Volume multiplier from 0.0 to 1.0
        Data: Either a file path (str) or an AudioSegment object
    """

    StartTime: float
    Duration: float
    Intensity: float
    Data: Union[str, AudioSegment]


def merge_audio_files(
    audio_entries: List[AudioEntry],
    output_path: str = "merged_output.wav",
    fade_in_duration: float = 0.1,
    fade_out_duration: float = 0.1,
) -> AudioSegment:
    """
    Merge multiple audio files into one final WAV file.

    This function takes a list of audio entries, each with a start time, duration,
    intensity (volume), and either a file path or AudioSegment object. Overlapping
    audio segments are mixed together. Each audio entry will have a smooth fade in
    at the start and fade out at the end.

    Args:
        audio_entries: List of AudioEntry objects specifying audio to merge
        output_path: Path where the merged audio file will be saved (default: "merged_output.wav")
        fade_in_duration: Duration in seconds for fade in effect (default: 0.1 seconds)
        fade_out_duration: Duration in seconds for fade out effect (default: 0.1 seconds)

    Returns:
        AudioSegment: The merged audio segment

    Raises:
        ValueError: If audio_entries is empty or if intensity is not in [0.0, 1.0]
        FileNotFoundError: If a file path doesn't exist
    """
    if not audio_entries:
        raise ValueError("audio_entries cannot be empty")

    # Validate intensity values
    for entry in audio_entries:
        if not 0.0 <= entry.Intensity <= 1.0:
            raise ValueError(
                f"Intensity must be between 0.0 and 1.0, got {entry.Intensity}"
            )

    # First pass: Load all audio files to determine actual durations
    # This ensures we calculate total duration based on actual audio lengths, not requested durations
    actual_durations = []
    for entry in audio_entries:
        # Load audio from file path or use AudioSegment directly
        if isinstance(entry.Data, str):
            # Load from file path
            audio = AudioSegment.from_file(entry.Data)
        elif isinstance(entry.Data, AudioSegment):
            # Use AudioSegment directly
            audio = entry.Data
        else:
            raise TypeError(
                f"Data must be either str (file path) or AudioSegment, got {type(entry.Data)}"
            )

        # Get actual audio length in seconds
        audio_length_seconds = len(audio) / 1000.0
        # Use the minimum of requested duration and actual audio length
        actual_duration = min(entry.Duration, audio_length_seconds)
        actual_durations.append(actual_duration)

    # Calculate total duration needed based on actual durations
    max_end_time = 0.0
    for entry, actual_duration in zip(audio_entries, actual_durations):
        end_time = entry.StartTime + actual_duration
        if end_time > max_end_time:
            max_end_time = end_time

    # Convert to milliseconds for Pydub
    total_duration_ms = int(max_end_time * 1000)

    # Create silent base audio
    merged_audio = AudioSegment.silent(duration=total_duration_ms)

    # Process each audio entry
    for entry, actual_duration in zip(audio_entries, actual_durations):
        # Load audio from file path or use AudioSegment directly
        if isinstance(entry.Data, str):
            # Load from file path
            audio = AudioSegment.from_file(entry.Data)
        elif isinstance(entry.Data, AudioSegment):
            # Use AudioSegment directly
            audio = entry.Data
        else:
            raise TypeError(
                f"Data must be either str (file path) or AudioSegment, got {type(entry.Data)}"
            )

        # Apply intensity (volume adjustment)
        # Convert intensity (0.0-1.0) to dB gain
        # 0.0 = -inf dB (silent), 1.0 = 0 dB (original volume)
        if entry.Intensity == 0.0:
            # Make silent
            audio = audio - 200  # Very low volume (effectively silent)
        elif entry.Intensity < 1.0:
            # Calculate dB gain: intensity 0.5 = -6dB, intensity 0.25 = -12dB, etc.
            # Formula: gain_db = 20 * log10(intensity)
            gain_db = 20 * math.log10(entry.Intensity) if entry.Intensity > 0 else -200
            audio = audio + gain_db

        # Handle duration: trim to actual duration (capped at audio file length)
        trim_duration_ms = int(actual_duration * 1000)
        audio = audio[:trim_duration_ms]

        # Apply fade in and fade out effects
        # Convert fade durations to milliseconds
        fade_in_ms = int(fade_in_duration * 1000)
        fade_out_ms = int(fade_out_duration * 1000)

        # Ensure fade durations don't exceed audio length
        current_audio_length_ms = len(audio)
        if fade_in_ms + fade_out_ms >= current_audio_length_ms:
            # If fades would overlap, reduce them proportionally
            # Keep fade_in and fade_out balanced
            max_fade = current_audio_length_ms // 2
            fade_in_ms = min(fade_in_ms, max_fade)
            fade_out_ms = min(fade_out_ms, max_fade)

        # Apply fade in at the start
        if fade_in_ms > 0 and fade_in_ms < current_audio_length_ms:
            audio = audio.fade_in(fade_in_ms)

        # Apply fade out at the end
        if fade_out_ms > 0 and fade_out_ms < current_audio_length_ms:
            audio = audio.fade_out(fade_out_ms)

        # Calculate start position in milliseconds
        start_position_ms = int(entry.StartTime * 1000)

        # Overlay audio at the specified start time (mixes with existing audio)
        merged_audio = merged_audio.overlay(audio, position=start_position_ms)

    # Export to WAV file
    merged_audio.export(output_path, format="wav")

    return merged_audio
