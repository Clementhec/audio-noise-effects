# Video Preprocessing Utilities

Utilities for extracting audio from video files for further processing in the Audio-noise-effects pipeline.

## Features

- Extract audio from MP4 videos to WAV format
- Configurable sample rate and channels
- Batch processing support
- Two extraction methods (pydub and direct ffmpeg)

## Requirements

Install ffmpeg on your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

Python dependencies (already in requirements.txt):
- pydub

## Usage

### Basic Usage

```python
from video_preprocessing import extract_audio_from_video

# Extract audio from video (creates my_video.wav)
audio_path = extract_audio_from_video("my_video.mp4")

# Specify output path
audio_path = extract_audio_from_video("my_video.mp4", "output/audio.wav")

# Custom sample rate and channels
audio_path = extract_audio_from_video(
    "my_video.mp4",
    sample_rate=44100,  # 44.1 kHz
    channels=2          # Stereo
)
```

### Command Line Usage

```bash
# Extract audio from a single video
python video_preprocessing/video_to_audio.py my_video.mp4

# Specify output path
python video_preprocessing/video_to_audio.py my_video.mp4 output/audio.wav
```

### Batch Processing

```python
from video_preprocessing import batch_extract_audio

# Extract audio from all MP4 files in a directory
audio_files = batch_extract_audio(
    video_dir="videos/",
    output_dir="audio/",
    pattern="*.mp4"
)

print(f"Extracted {len(audio_files)} audio files")
```

### Alternative Method (Direct ffmpeg)

If pydub has issues, use the direct ffmpeg method:

```python
from video_preprocessing import extract_audio_ffmpeg_direct

audio_path = extract_audio_ffmpeg_direct("my_video.mp4")
```

## Function Reference

### extract_audio_from_video()

Extract audio from video using pydub.

**Parameters:**
- `video_path` (str | Path): Path to input video file (.mp4)
- `output_path` (str | Path, optional): Output audio file path. Defaults to video name with .wav extension
- `audio_format` (str): Output format (default: "wav")
- `sample_rate` (int): Sample rate in Hz (default: 16000, good for STT)
- `channels` (int): Number of channels (default: 1 for mono)

**Returns:** Path to created audio file

### extract_audio_ffmpeg_direct()

Extract audio using ffmpeg directly via subprocess.

**Parameters:**
- `video_path` (str | Path): Path to input video file (.mp4)
- `output_path` (str | Path, optional): Output audio file path
- `sample_rate` (int): Sample rate in Hz (default: 16000)
- `channels` (int): Number of channels (default: 1)

**Returns:** Path to created audio file

### batch_extract_audio()

Extract audio from multiple videos in a directory.

**Parameters:**
- `video_dir` (str | Path): Directory containing video files
- `output_dir` (str | Path): Directory for output audio files
- `pattern` (str): Glob pattern for video files (default: "*.mp4")
- `sample_rate` (int): Sample rate in Hz (default: 16000)
- `channels` (int): Number of channels (default: 1)

**Returns:** List of paths to created audio files

## Integration with STT Pipeline

After extracting audio, use it with the STT pipeline:

```python
from video_preprocessing import extract_audio_from_video

# 1. Extract audio from video
audio_file = extract_audio_from_video("my_video.mp4", sample_rate=16000)

# 2. Process with STT (run your STT script)
# This will create word_timing.json and full_transcription.json

# 3. Generate embeddings
# uv run python process_stt_embeddings.py

# 4. Match with sound effects
# uv run python semantic_matcher.py
```

## Sample Rates

Choose appropriate sample rate for your use case:

- **16000 Hz**: Standard for speech recognition (Google STT, Whisper)
- **22050 Hz**: Acceptable quality for speech
- **44100 Hz**: CD quality, good for music
- **48000 Hz**: Professional audio/video standard

For this pipeline (STT + semantic matching), **16000 Hz mono** is recommended.

## Troubleshooting

### ffmpeg not found

Install ffmpeg on your system (see Requirements section above).

### Permission errors

Ensure you have read permissions for the video file and write permissions for the output directory.

### Memory issues with large files

For very large video files, consider using `extract_audio_ffmpeg_direct()` which streams the data instead of loading it all into memory.
