"""
Video to Audio Extraction Utilities

This module provides functions to extract audio from video files (MP4)
and save them as WAV files for further processing.

Requirements:
- pydub (already in requirements.txt)
- ffmpeg must be installed on the system

Install ffmpeg:
- Ubuntu/Debian: sudo apt-get install ffmpeg
- macOS: brew install ffmpeg
- Windows: Download from https://ffmpeg.org/download.html
"""

import subprocess
from pathlib import Path
from typing import Optional, Union
from pydub import AudioSegment


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    audio_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    Extract audio from a video file and save as audio file.

    Uses pydub with ffmpeg backend to extract audio from video files.

    Args:
        video_path: Path to the input video file (.mp4)
        output_path: Path for the output audio file. If None, uses video name with .wav extension
        audio_format: Output audio format (default: "wav")
        sample_rate: Sample rate in Hz (default: 16000, good for speech recognition)
        channels: Number of audio channels (default: 1 for mono)

    Returns:
        Path object pointing to the created audio file

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If audio extraction fails

    Example:
        >>> extract_audio_from_video("my_video.mp4")
        PosixPath('my_video.wav')

        >>> extract_audio_from_video("my_video.mp4", "output/audio.wav", sample_rate=44100)
        PosixPath('output/audio.wav')
    """
    video_path = Path(video_path)

    # Validate input file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output path
    if output_path is None:
        output_path = video_path.with_suffix(f".{audio_format}")
    else:
        output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Extracting audio from: {video_path}")
        print(f"Output file: {output_path}")

        # Load video file and extract audio
        audio = AudioSegment.from_file(str(video_path), format="mp4")

        # Convert to specified sample rate and channels
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(channels)

        # Export as WAV
        audio.export(
            str(output_path),
            format=audio_format,
            parameters=["-ac", str(channels)]
        )

        print(f"✓ Audio extracted successfully: {output_path}")
        print(f"  Duration: {len(audio) / 1000:.2f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")

        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {str(e)}") from e


def extract_audio_ffmpeg_direct(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    Extract audio from video using ffmpeg directly via subprocess.

    This is an alternative method that calls ffmpeg directly without pydub.
    Useful as a fallback if pydub has issues.

    Args:
        video_path: Path to the input video file (.mp4)
        output_path: Path for the output audio file. If None, uses video name with .wav extension
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)

    Returns:
        Path object pointing to the created audio file

    Raises:
        FileNotFoundError: If video file doesn't exist or ffmpeg not installed
        RuntimeError: If audio extraction fails

    Example:
        >>> extract_audio_ffmpeg_direct("my_video.mp4")
        PosixPath('my_video.wav')
    """
    video_path = Path(video_path)

    # Validate input file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output path
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(video_path),           # Input file
        "-vn",                           # Disable video
        "-acodec", "pcm_s16le",         # Audio codec (16-bit PCM for WAV)
        "-ar", str(sample_rate),        # Sample rate
        "-ac", str(channels),           # Number of channels
        "-y",                           # Overwrite output file
        str(output_path)
    ]

    try:
        print(f"Extracting audio from: {video_path}")
        print(f"Output file: {output_path}")
        print(f"Running: {' '.join(cmd)}")

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"✓ Audio extracted successfully: {output_path}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")

        return output_path

    except FileNotFoundError:
        raise FileNotFoundError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed with error:\n{e.stderr}"
        ) from e


def batch_extract_audio(
    video_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.mp4",
    sample_rate: int = 16000,
    channels: int = 1
) -> list[Path]:
    """
    Extract audio from all video files in a directory.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory for output audio files
        pattern: Glob pattern for video files (default: "*.mp4")
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)

    Returns:
        List of Path objects for created audio files

    Example:
        >>> batch_extract_audio("videos/", "audio/", pattern="*.mp4")
        [PosixPath('audio/video1.wav'), PosixPath('audio/video2.wav')]
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = sorted(video_dir.glob(pattern))

    if not video_files:
        print(f"No video files found in {video_dir} matching pattern {pattern}")
        return []

    print(f"Found {len(video_files)} video files")
    print("=" * 70)

    extracted_files = []

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")

        # Generate output path with same name but .wav extension
        output_path = output_dir / video_path.with_suffix(".wav").name

        try:
            result_path = extract_audio_from_video(
                video_path,
                output_path,
                sample_rate=sample_rate,
                channels=channels
            )
            extracted_files.append(result_path)

        except Exception as e:
            print(f"✗ Failed to process {video_path.name}: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"✓ Batch extraction complete: {len(extracted_files)}/{len(video_files)} successful")

    return extracted_files


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_to_audio.py <video_file.mp4> [output_file.wav]")
        print("\nExample:")
        print("  python video_to_audio.py my_video.mp4")
        print("  python video_to_audio.py my_video.mp4 audio/output.wav")
        sys.exit(1)

    video_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = extract_audio_from_video(video_file, output_file)
        print(f"\n✓ Success! Audio saved to: {result}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
