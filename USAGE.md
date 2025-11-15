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

Extract audio, run STT, generate embeddings, and match with sound effects:

```bash
uv run python main.py video.mp4 --full-pipeline
```

This will:
1. Extract audio to WAV
2. Transcribe with word-level timing
3. Generate semantic embeddings
4. Find similar sound effects for each speech segment

## Pipeline Steps

The pipeline consists of 5 steps:

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

### Step 4: Similarity Matching (Optional: `--run-matching`)
- Matches speech segments with sound effects semantically
- Uses cosine similarity between embeddings
- Finds top K most similar sounds for each segment
- Saves to `output/video_similarity_matches.json`

### Step 5: LLM Intelligent Filtering (Optional: `--run-llm-filter`)
- Uses Google Gemini LLM to intelligently select best sounds
- Determines which sentences benefit most from sound effects
- Identifies specific target word for each sound placement
- Selects most appropriate sound from top-K candidates
- Saves to `output/video_filtered_sounds.json`
- Requires: `GOOGLE_API_KEY` in `.env` file

## Command Line Options

```
uv run python main.py [OPTIONS] VIDEO_FILE

Required:
  VIDEO_FILE                Path to input MP4 video

Optional:
  --run-stt                 Run STT after audio extraction
  --run-embeddings          Generate embeddings (requires --run-stt)
  --run-matching            Match with sound effects (requires --run-embeddings)
  --run-llm-filter          LLM intelligent filtering (requires --run-matching)
  --full-pipeline           Run all steps automatically
  --sample-rate RATE        Sample rate in Hz (default: 16000)
  --channels NUM            Number of channels (default: 1)
  --output-dir DIR          Output directory (default: speech_to_text/input)
  --top-k NUM               Number of top similar sounds per segment (default: 5)
  --max-sounds NUM          Max sentences to select for sounds (default: LLM decides)
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
- `output/video_similarity_matches.json`

### 4. Full Pipeline with LLM Filtering

Complete pipeline with intelligent sound selection:

```bash
uv run python main.py my_video.mp4 --full-pipeline
```

**Output:**
- All previous files plus:
- `output/video_filtered_sounds.json` - LLM-selected sounds with target words

### 5. Custom LLM Filtering

Control max sounds and top-k:

```bash
# Find 10 candidates, let LLM select best 5
uv run python main.py my_video.mp4 --full-pipeline --top-k 10 --max-sounds 5

# Find 15 candidates, LLM decides how many to use
uv run python main.py my_video.mp4 --full-pipeline --top-k 15
```

### 6. High Quality Audio (44.1kHz Stereo)

```bash
uv run python main.py concert.mp4 --sample-rate 44100 --channels 2
```

### 7. Custom Output Directory

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

### 4. Google API Key (for LLM Filtering - Optional)
Add to `.env` file for intelligent sound filtering:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

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
│   └── soundbible_embeddings.csv
├── output/
│   ├── video_similarity_matches.json
│   └── video_filtered_sounds.json
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
