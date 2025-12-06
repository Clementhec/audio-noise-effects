# Video Preprocessing Utilities

Utilities for extracting audio from video files.

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

Example sample rates :

- **16000 Hz**: Standard for speech recognition (Google STT, Whisper)
- **22050 Hz**: Acceptable quality for speech
- **44100 Hz**: CD quality, good for music
- **48000 Hz**: Professional audio/video standard
