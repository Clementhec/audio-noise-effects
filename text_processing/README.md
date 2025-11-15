# Text Processing Pipeline

This module implements **Step 2 and Step 3** of the audio enhancement pipeline: converting speech transcripts into vector embeddings compatible with sound effect embeddings.

## Overview

The text processing pipeline takes STT (Speech-to-Text) output from Google Cloud Speech API and generates semantic embeddings that can be matched against sound effect embeddings for intelligent sound placement.

**Key Design Principles:**
- **Compatible Embeddings**: Uses `all-MiniLM-L6-v2` from HuggingFace - the same model used for sound embeddings
- **Local Processing**: No API keys required - runs entirely on your machine
- **Timing Preservation**: Maintains word-level timestamps for precise audio synchronization
- **Information Preservation**: No pre-filtering - all segments are embedded for score-based matching
- **Flexible Segmentation**: Supports sentence-based and time-window based segmentation strategies

## Architecture

```
STT Output (transcript + word timings)
    ↓
[Speech Segmentation]
    ↓
Segments with timing info
    ↓
[Batch Embedding Generation]
    ↓
Embedded segments (ready for matching)
```

## Modules

### `speech_segmenter.py`

Segments transcripts into meaningful chunks while preserving timing information.

**Features:**
- Sentence-based segmentation with automatic long-sentence splitting
- Time-window based segmentation (alternative approach)
- Configurable maximum words per segment (default: 15)
- Preserves word-level timing data for each segment

**Example:**
```python
from speech_segmenter import SpeechSegmenter, load_stt_output

# Load STT output from stt_google.py
transcript, word_timings = load_stt_output(stt_result)

# Create segmenter
segmenter = SpeechSegmenter(max_words_per_segment=15)

# Segment by sentences
segments = segmenter.segment_by_sentences(transcript, word_timings)

# Each segment contains:
# {
#   'segment_id': 0,
#   'text': 'The dog is barking loudly.',
#   'start_time': 1.2,
#   'end_time': 3.5,
#   'word_count': 5,
#   'words': [...]  # Original word timing data
# }
```

### `speech_embedder.py`

Main pipeline that generates embeddings for speech segments.

**Features:**
- Uses `all-MiniLM-L6-v2` from HuggingFace (384 dimensions)
- Local processing - no API keys required
- Batch processing for efficiency
- Outputs to CSV and JSON formats
- Preserves all timing metadata

**Example:**
```python
from speech_embedder import SpeechEmbeddingPipeline

# Initialize pipeline
pipeline = SpeechEmbeddingPipeline(
    segmentation_method="sentences",
    output_dir="./text-processing/output"
)

# Process STT output
df = pipeline.process_stt_output(stt_result)

# DataFrame contains:
# - segment_id: Unique identifier
# - text: The text segment
# - start_time: Beginning timestamp (seconds)
# - end_time: Ending timestamp (seconds)
# - duration: Segment length (seconds)
# - word_count: Number of words
# - embedding: 384-dim numpy array
# - embedding_model: "all-MiniLM-L6-v2"
```

## Usage

### Command Line

```bash
# Process an STT result file
python text-processing/speech_embedder.py path/to/stt_result.json

# Specify output directory
python text-processing/speech_embedder.py path/to/stt_result.json \
    --output-dir ./my_output

# Use time-window segmentation instead
python text-processing/speech_embedder.py path/to/stt_result.json \
    --method time_windows
```

### Programmatic Usage

```python
import sys
sys.path.append('.')

from stt_google import transcribe_audio
from text_processing.speech_embedder import SpeechEmbeddingPipeline

# Step 1: Transcribe audio
stt_result = transcribe_audio("audio.wav", language_code="en-US")

# Step 2: Generate embeddings
pipeline = SpeechEmbeddingPipeline(output_dir="./output")
df = pipeline.process_stt_output(stt_result)

# Step 3: Use embeddings for matching
# (Next step: compare with sound embeddings)
print(f"Generated {len(df)} embedded segments")
print(df[['segment_id', 'text', 'start_time', 'end_time']])
```

## Output Formats

### CSV Format
```csv
segment_id,text,start_time,end_time,duration,word_count,embedding,embedding_model
0,"The dog is barking.",1.2,3.5,2.3,4,"[0.023, -0.045, ...]",all-MiniLM-L6-v2
1,"Thunder echoes loudly.",3.6,6.1,2.5,3,"[-0.012, 0.089, ...]",all-MiniLM-L6-v2
```

### JSON Format
```json
{
  "metadata": {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "segmentation_method": "sentences",
    "total_segments": 42
  },
  "segments": [
    {
      "segment_id": 0,
      "text": "The dog is barking.",
      "start_time": 1.2,
      "end_time": 3.5,
      "duration": 2.3,
      "word_count": 4,
      "embedding": [0.023, -0.045, ...],
      "embedding_model": "all-MiniLM-L6-v2"
    }
  ]
}
```

## Segmentation Strategies

### Sentence-based (Default)
- Splits on sentence boundaries (. ! ?)
- Limits segments to max words (default: 15)
- Best for narrative content with clear sentence structure
- Preserves semantic coherence

### Time-window based
- Fixed-duration windows (e.g., 5 seconds)
- Optional overlap between windows
- Best for continuous speech without clear punctuation
- Ensures uniform temporal distribution

## Integration with Sound Matching

The output from this pipeline feeds directly into the semantic matching step:

```python
import pandas as pd
import numpy as np
from utils.embeddings_utils import cosine_similarity

# Load speech embeddings
speech_df = pd.read_csv("text-processing/output/speech_embeddings.csv")
speech_df['embedding'] = speech_df['embedding'].apply(eval).apply(np.array)

# Load sound embeddings (pre-computed with same model)
sounds_df = pd.read_csv("data/soundbible_details_from_section_with_embeddings.csv")
sounds_df['embedding'] = sounds_df['embedding'].apply(eval).apply(np.array)

# For each speech segment, find matching sounds
for idx, row in speech_df.iterrows():
    speech_emb = row['embedding']

    # Calculate similarity with all sounds
    similarities = sounds_df['embedding'].apply(
        lambda sound_emb: cosine_similarity(speech_emb, sound_emb)
    )

    # Filter by threshold
    THRESHOLD = 0.75
    matches = sounds_df[similarities >= THRESHOLD]

    if len(matches) > 0:
        best_match = matches.loc[similarities.idxmax()]
        print(f"Segment: '{row['text']}' at {row['start_time']}s")
        print(f"  → Match: {best_match['title']} (score: {similarities.max():.3f})")
```

## Configuration

### Environment Variables
No API keys or environment variables required! The HuggingFace model runs locally.

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segmentation_method` | `"sentences"` | Segmentation strategy |
| `max_words_per_segment` | `15` | Maximum words per segment |
| `output_dir` | `None` | Directory for output files |

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (HuggingFace sentence-transformers)
- **Dimensions**: 384
- **Critical**: Must match the model used for sound embeddings
- **Advantages**: Local processing, no API keys, free, fast

## Performance Considerations

- **Batch Processing**: Uses `get_embeddings()` for efficient local processing
- **No API Calls**: Runs entirely on your machine - no rate limits or costs
- **Speed**: Processes ~32 segments per batch by default (configurable)
- **First Run**: Model downloads automatically from HuggingFace (~80MB)
- **GPU Support**: Automatically uses GPU if available (CUDA/MPS) for faster processing

## Testing

Create a test with sample STT output:

```python
# test_pipeline.py
from text_processing.speech_embedder import SpeechEmbeddingPipeline

# Mock STT output
stt_result = {
    'results': {
        'transcript': 'The dog barked loudly. Thunder echoed across the valley.',
        'confidence': 0.95
    },
    'words_timings': [
        {'word': 'The', 'start_time': 0.0, 'end_time': 0.2},
        {'word': 'dog', 'start_time': 0.2, 'end_time': 0.5},
        {'word': 'barked', 'start_time': 0.5, 'end_time': 0.9},
        {'word': 'loudly', 'start_time': 0.9, 'end_time': 1.3},
        {'word': 'Thunder', 'start_time': 1.5, 'end_time': 1.9},
        {'word': 'echoed', 'start_time': 1.9, 'end_time': 2.3},
        {'word': 'across', 'start_time': 2.3, 'end_time': 2.6},
        {'word': 'the', 'start_time': 2.6, 'end_time': 2.8},
        {'word': 'valley', 'start_time': 2.8, 'end_time': 3.2}
    ]
}

pipeline = SpeechEmbeddingPipeline(output_dir="./test_output")
df = pipeline.process_stt_output(stt_result)

print(f"Created {len(df)} segments")
print(df[['text', 'start_time', 'end_time']])
```

## Next Steps

After generating speech embeddings:
1. Load sound embeddings from `data/soundbible_details_from_section_with_embeddings.csv`
2. Calculate similarity scores (see Integration section above)
3. Filter by threshold (e.g., ≥ 0.75)
4. Select best matches for each segment
5. Generate audio mixing timeline

See main `README.md` for the complete end-to-end workflow.
