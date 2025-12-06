from pathlib import Path
from typing import Optional, Union
from pydub import AudioSegment


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    audio_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
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

    print(f"Extracting audio from: {video_path}")
    print(f"Output file: {output_path}")

    # Load video file and extract audio
    audio = AudioSegment.from_file(str(video_path), format="mp4")

    # Convert to specified sample rate and channels
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(channels)

    # Export as WAV
    audio.export(
        str(output_path), format=audio_format, parameters=["-ac", str(channels)]
    )

    print(f"Audio extracted successfully: {output_path}")
    print(f"Duration: {len(audio) / 1000:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")

    return output_path


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract audio from video files")
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output audio file path")
    parser.add_argument("--format", default="wav", help="Audio format (default: wav)")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Sample rate in Hz"
    )
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")

    args = parser.parse_args()

    try:
        result = extract_audio_from_video(
            args.video_file,
            args.output,
            audio_format=args.format,
            sample_rate=args.sample_rate,
            channels=args.channels,
        )
        print(f"\n Success! Audio saved to: {result}")
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
