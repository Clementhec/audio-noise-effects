# Video Preprocessing Pipeline - Usage Guide

## Quick Start

### Extract Audio Only
Extract audio from a video and save to `speech_to_text/input/`:

```bash
uv run python main.py path/to/video.mp4
```

This will create `speech_to_text/input/video.wav` with:
- 16kHz sample rate (optimal for STT)
- Mono channel

### Extract Audio with Custom Settings

```bash
uv run python main.py video.mp4 --sample-rate 44100 --channels 2
```

### Run Full Pipeline

Extract audio, run STT, and generate embeddings:

```bash
uv run python main.py video.mp4 --full-pipeline
```

## Pipeline Steps

The pipeline consists of 4 steps:

### Step 1: Extract Audio (Always runs)
- Converts MP4 video to WAV audio
- Saves to `speech_to_text/input/`
- Uses same filename as video with `.wav` extension

### Step 2: Speech-to-Text (Optional: `--run-stt`)
- Transcribes audio using ElevenLabs API
- Requires API key in `.env` file
- Generates:
  - `speech_to_text/output/full_transcription.json`
  - `speech_to_text/output/word_timing.json`

### Step 3: Generate Embeddings (Optional: `--run-embeddings`)
- Segments transcription into chunks
- Generates semantic embeddings using `all-MiniLM-L6-v2`
- Saves to `data/video_speech_embeddings.csv`

### Step 4: Semantic Matching (Manual)
- Match speech segments with sound effects
- Run separately:
  ```bash
  uv run python semantic_matcher.py \
    data/video_speech_embeddings.csv \
    data/soundbible_embeddings.csv \
    --output data/video_timeline.csv
  ```

## Command Line Options

```
uv run python main.py [OPTIONS] VIDEO_FILE

Required:
  VIDEO_FILE                Path to input MP4 video

Optional:
  --run-stt                 Run STT after audio extraction
  --run-embeddings          Generate embeddings (requires --run-stt)
  --full-pipeline           Run all steps automatically
  --sample-rate RATE        Sample rate in Hz (default: 16000)
  --channels NUM            Number of channels (default: 1)
  --output-dir DIR          Output directory (default: speech_to_text/input)
```

## Examples

### 1. Just Extract Audio

```bash
uv run python main.py my_video.mp4
```

**Output:**
- `speech_to_text/input/my_video.wav`

### 2. Extract + Transcribe

```bash
uv run python main.py my_video.mp4 --run-stt
```

**Output:**
- `speech_to_text/input/my_video.wav`
- `speech_to_text/output/full_transcription.json`
- `speech_to_text/output/word_timing.json`

### 3. Full Pipeline

```bash
uv run python main.py my_video.mp4 --full-pipeline
```

**Output:**
- `speech_to_text/input/my_video.wav`
- `speech_to_text/output/full_transcription.json`
- `speech_to_text/output/word_timing.json`
- `data/video_speech_embeddings.csv`

### 4. High Quality Audio (44.1kHz Stereo)

```bash
uv run python main.py concert.mp4 --sample-rate 44100 --channels 2
```

### 5. Custom Output Directory

```bash
uv run python main.py video.mp4 --output-dir audio_files/
```

## Prerequisites

### 1. FFmpeg
Must be installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Python Dependencies
Already in `requirements.txt`:
- pydub
- sentence-transformers
- pandas
- numpy

### 3. ElevenLabs API Key (for STT)
Create a `.env` file in the project root:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

## Directory Structure

After running the pipeline:

```
Audio-noise-effects/
├── main.py
├── speech_to_text/
│   ├── input/           # Extracted WAV files
│   │   └── video.wav
│   └── output/          # STT results
│       ├── full_transcription.json
│       └── word_timing.json
├── data/
│   ├── video_speech_embeddings.csv
│   ├── soundbible_embeddings.csv
│   └── video_timeline.csv
└── video_preprocessing/
    ├── video_to_audio.py
    └── ...
```

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg (see Prerequisites above).

### "ModuleNotFoundError"
Ensure you're using `uv run`:
```bash
uv run python main.py video.mp4
```

### "ELEVENLABS_API_KEY not found"
Create a `.env` file with your API key (see Prerequisites).

### "Video file not found"
Check the path to your video file. Use absolute or relative path.

## Integration with Existing Workflow

### Manual Workflow (Before)

```bash
# 1. Extract audio manually
ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav

# 2. Run STT
python speech_to_text/stt_elevenlabs.py

# 3. Generate embeddings
python process_stt_embeddings.py

# 4. Match with sounds
python semantic_matcher.py
```

### Automated Workflow (Now)

```bash
# All in one command
uv run python main.py video.mp4 --full-pipeline

# Then run semantic matching
uv run python semantic_matcher.py \
  data/video_speech_embeddings.csv \
  data/soundbible_embeddings.csv \
  --output data/video_timeline.csv
```

## Next Steps After Pipeline

1. Review the generated timeline: `data/video_timeline.csv`
2. Download matched sound effects
3. Mix audio with video using the timeline
4. Export final video with sound effects

## Advanced Usage

### Process Multiple Videos

Create a bash script:

```bash
#!/bin/bash
for video in videos/*.mp4; do
    echo "Processing: $video"
    uv run python main.py "$video" --full-pipeline
done
```

### Custom Embedding Model

Edit `main.py` and change:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

to another sentence-transformers model.
