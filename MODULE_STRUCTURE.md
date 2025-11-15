# Module Structure and Organization

This document describes the organization of the codebase and how to import modules correctly.

## Directory Structure

```
Audio-noise-effects/
├── main.py                      # Main pipeline orchestrator
├── video_preprocessing/         # Video to audio extraction
│   ├── __init__.py
│   ├── video_to_audio.py
│   └── README.md
├── speech_to_text/              # STT using ElevenLabs
│   ├── __init__.py
│   ├── stt_elevenlabs.py
│   └── README.md
├── text_processing/             # Speech segmentation and embedding
│   ├── __init__.py
│   ├── speech_segmenter.py
│   ├── speech_embedder.py
│   └── README.md
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── embeddings_utils.py
│   └── sound_embedding/
└── data/                        # Data files (CSV, embeddings)
```

## Module Imports

### ✅ Correct Import Patterns

All modules have proper `__init__.py` files for clean imports:

```python
# Video preprocessing
from video_preprocessing import extract_audio_from_video, batch_extract_audio

# Speech-to-text
from speech_to_text import transcribe_audio_file

# Text processing (segmentation)
from text_processing import SpeechSegmenter

# Utilities (embeddings)
from utils import get_embeddings, get_model
```

### ❌ Incorrect Import Patterns (Deprecated)

These patterns are no longer needed and should be avoided:

```python
# DON'T DO THIS - No need for sys.path manipulation
import sys
sys.path.insert(0, "text-processing")  # Wrong directory name
sys.path.insert(0, str(project_root / "text-processing"))

# DON'T DO THIS - Import from submodules directly
from speech_segmenter import SpeechSegmenter  # Missing parent package
from utils.embeddings_utils import get_embeddings  # Use utils instead
```

## Module Details

### 1. video_preprocessing

**Purpose:** Extract audio from video files

**Exports:**
- `extract_audio_from_video(video_path, output_path, sample_rate, channels)`
- `extract_audio_ffmpeg_direct(video_path, output_path, sample_rate, channels)`
- `batch_extract_audio(video_dir, output_dir, pattern)`

**Usage:**
```python
from video_preprocessing import extract_audio_from_video

audio_path = extract_audio_from_video("video.mp4", sample_rate=16000)
```

### 2. speech_to_text

**Purpose:** Transcribe audio using ElevenLabs API

**Exports:**
- `transcribe_audio_file(audio_file_path, output_dir, api_key, ...)`
- `transcribe_audio_elevenlabs(audio_source, ...)` (low-level API)

**Usage:**
```python
from speech_to_text import transcribe_audio_file

result = transcribe_audio_file(
    audio_file_path="speech_to_text/input/video.wav",
    output_dir="speech_to_text/output"
)
```

### 3. text_processing

**Purpose:** Segment speech into chunks with timing

**Exports:**
- `SpeechSegmenter` - Main segmentation class
- `load_stt_output(stt_result)` - Parse STT output
- `SpeechEmbeddingPipeline` - Complete embedding pipeline
- `process_speech_file(...)` - High-level processing

**Usage:**
```python
from text_processing import SpeechSegmenter

segmenter = SpeechSegmenter(max_words_per_segment=15)
segments = segmenter.segment_by_sentences(transcript, word_timings)
```

### 4. utils

**Purpose:** Embedding generation and utilities

**Exports:**
- `get_model(model_name)` - Load embedding model
- `get_embedding(text, model)` - Get single embedding
- `get_embeddings(texts, model, show_progress)` - Batch embeddings
- `cosine_similarity(a, b)` - Similarity calculation
- `get_embedding_dimension(model)` - Get model dimensions

**Usage:**
```python
from utils import get_embeddings

texts = ["hello world", "goodbye world"]
embeddings = get_embeddings(texts, model="all-MiniLM-L6-v2")
```

## Testing Imports

Run the import test script to verify all modules are working:

```bash
uv run python test_imports.py
```

Expected output:
```
1. Testing text_processing module...
   ✓ SpeechSegmenter imported successfully
   ✓ SpeechSegmenter instantiated successfully

2. Testing utils module...
   ✓ get_embeddings imported successfully
   ✓ get_model imported successfully

3. Testing speech_to_text module...
   ✓ transcribe_audio_file imported successfully

4. Testing video_preprocessing module...
   ✓ extract_audio_from_video imported successfully
```

## Common Issues and Solutions

### Issue: "No module named 'speech_segmenter'"

**Cause:** Importing from the submodule directly instead of the package.

**Solution:**
```python
# Wrong
from speech_segmenter import SpeechSegmenter

# Correct
from text_processing import SpeechSegmenter
```

### Issue: "No module named 'text-processing'" (with hyphen)

**Cause:** The directory is `text_processing` with underscore, not hyphen.

**Solution:**
```python
# Wrong
sys.path.insert(0, "text-processing")

# Correct - No sys.path needed, just import directly
from text_processing import SpeechSegmenter
```

### Issue: "ModuleNotFoundError" in general

**Solution:**
1. Ensure you're running from the project root directory
2. Use `uv run python` instead of `python` directly
3. Check that `__init__.py` files exist in all package directories

## Best Practices

1. **Always import from package level**, not submodules:
   ```python
   from text_processing import SpeechSegmenter  # ✓
   from speech_segmenter import SpeechSegmenter  # ✗
   ```

2. **Don't manipulate sys.path** in application code:
   ```python
   sys.path.insert(0, ...)  # ✗ Not needed with proper package structure
   ```

3. **Use relative imports within packages**:
   ```python
   # Inside text_processing/speech_embedder.py
   from .speech_segmenter import SpeechSegmenter  # ✓
   ```

4. **Import only what you need**:
   ```python
   from utils import get_embeddings  # ✓ Specific import
   import utils  # ✗ Import entire module
   ```

## Package Maintenance

When adding new modules or functions:

1. Create the module file in the appropriate directory
2. Add exports to the package's `__init__.py`
3. Update this documentation
4. Add tests to `test_imports.py`
5. Update relevant README files

## Migration Guide

If you have old code using sys.path manipulation, update it as follows:

**Before:**
```python
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "text-processing"))

from speech_segmenter import SpeechSegmenter
from utils.embeddings_utils import get_embeddings
```

**After:**
```python
from text_processing import SpeechSegmenter
from utils import get_embeddings
```

Much cleaner and more maintainable!
