# Speech-to-Text Module

This module provides functions for transcribing audio files using the ElevenLabs API.

## Features

- Transcribe audio files (WAV, MP3, etc.) to text
- Word-level timing information
- Configurable output directory
- Support for both URL and local file inputs
- Automatic JSON output saving

## Functions

### `transcribe_audio_file()` - Recommended

Convenience function that accepts file paths directly.

```python
from speech_to_text import transcribe_audio_file

# Transcribe an audio file
result = transcribe_audio_file(
    audio_file_path="speech_to_text/input/video.wav",
    output_dir="speech_to_text/output"  # Optional, defaults to speech_to_text/output
)

print(result['full_transcript'])
print(result['output_files']['transcription'])  # Path to saved JSON
print(result['output_files']['word_timing'])    # Path to word timing JSON
```

**Parameters:**
- `audio_file_path` (str | Path): Path to the audio file
- `output_dir` (str | Path, optional): Directory to save output JSON files
- `api_key` (str, optional): ElevenLabs API key (uses .env if None)
- `model_id` (str): Model to use (default: "scribe_v1")
- `tag_audio_events` (bool): Tag audio events like laughter (default: False)
- `language_code` (str, optional): Language code (auto-detect if None)
- `diarize` (bool): Annotate who is speaking (default: False)

**Returns:**
```python
{
    'full_transcript': str,           # Complete transcription text
    'segment_result': [...],          # List of segments with timing
    'word_timings': [...],            # Word-level timing data
    'output_files': {                 # Paths to saved files
        'transcription': str,
        'word_timing': str
    }
}
```

### `transcribe_audio_elevenlabs()` - Low-level API

Direct API function that accepts URLs or BytesIO objects.

```python
from speech_to_text import transcribe_audio_elevenlabs
from io import BytesIO

# With BytesIO
with open('audio.wav', 'rb') as f:
    audio_data = BytesIO(f.read())

result = transcribe_audio_elevenlabs(
    audio_source=audio_data,
    output_dir="custom/output/dir"
)

# With URL
result = transcribe_audio_elevenlabs(
    audio_source="https://example.com/audio.mp3"
)
```

**Parameters:**
- `audio_source` (str | BytesIO): URL or BytesIO object containing audio
- `output_dir` (str, optional): Output directory for JSON files
- `api_key` (str, optional): ElevenLabs API key
- `model_id` (str): Model to use (default: "scribe_v1")
- `tag_audio_events` (bool): Tag audio events (default: False)
- `language_code` (str, optional): Language code
- `diarize` (bool): Speaker diarization (default: False)
- `output_format` (str): "segments", "words", or "both" (default: "both")
- `save_to_json_file` (bool): Save to JSON files (default: True)

## Setup

### 1. Install Dependencies

```bash
pip install elevenlabs python-dotenv requests
```

### 2. Configure API Key

Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

Get your API key from: https://elevenlabs.io/

## Output Files

The module saves two JSON files:

### `full_transcription.json`
```json
{
  "full_transcript": "Complete transcription text...",
  "segment_result": [
    {
      "transcription": "Complete transcription text...",
      "startTime": "0.0s",
      "endTime": "14.779s"
    }
  ]
}
```

### `word_timing.json`
```json
[
  {
    "word": "With",
    "startTime": "0.099s",
    "endTime": "0.259s"
  },
  {
    "word": "a",
    "startTime": "0.299s",
    "endTime": "0.36s"
  }
  ...
]
```

## Integration with Pipeline

This module is integrated into the main preprocessing pipeline:

```bash
# Extract audio and run STT
uv run python main.py video.mp4 --run-stt
```

Or use directly in Python:

```python
from speech_to_text import transcribe_audio_file

# Transcribe audio from video preprocessing
result = transcribe_audio_file(
    audio_file_path="speech_to_text/input/video.wav",
    output_dir="speech_to_text/output"
)

# Result files will be at:
# - speech_to_text/output/full_transcription.json
# - speech_to_text/output/word_timing.json
```

## Advanced Usage

### Custom Model

```python
result = transcribe_audio_file(
    audio_file_path="audio.wav",
    model_id="scribe_v2"  # Use different model
)
```

### Tag Audio Events

```python
result = transcribe_audio_file(
    audio_file_path="audio.wav",
    tag_audio_events=True  # Tag laughter, applause, etc.
)
```

### Specify Language

```python
result = transcribe_audio_file(
    audio_file_path="audio.wav",
    language_code="en"  # Force English
)
```

### Speaker Diarization

```python
result = transcribe_audio_file(
    audio_file_path="audio.wav",
    diarize=True  # Identify different speakers
)
```

## Error Handling

```python
from speech_to_text import transcribe_audio_file

try:
    result = transcribe_audio_file("audio.wav")
except FileNotFoundError:
    print("Audio file not found")
except Exception as e:
    print(f"Transcription failed: {e}")
```

## Next Steps After Transcription

After transcribing, use the output with the embedding pipeline:

```bash
# Generate embeddings from transcription
uv run python process_stt_embeddings.py

# Match with sound effects
uv run python semantic_matcher.py \
  data/video_speech_embeddings.csv \
  data/soundbible_embeddings.csv \
  --output data/video_timeline.csv
```
