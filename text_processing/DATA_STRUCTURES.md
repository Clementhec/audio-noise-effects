# Data Structures Reference

This document details the exact data structures used throughout the text-processing pipeline.

## 1. Input: STT Output from `stt_google.py`

```python
{
    'results': {
        'transcript': str,     # Full transcript text
        'confidence': float    # Overall confidence score (0-1)
    },
    'words_timings': [
        {
            'word': str,           # Individual word
            'start_time': float,   # Word start time in seconds
            'end_time': float      # Word end time in seconds
        },
        ...
    ]
}
```

**Example:**
```python
{
    'results': {
        'transcript': 'The dog barked loudly.',
        'confidence': 0.95
    },
    'words_timings': [
        {'word': 'The', 'start_time': 0.0, 'end_time': 0.2},
        {'word': 'dog', 'start_time': 0.2, 'end_time': 0.5},
        {'word': 'barked', 'start_time': 0.5, 'end_time': 0.9},
        {'word': 'loudly', 'start_time': 0.9, 'end_time': 1.3}
    ]
}
```

---

## 2. Intermediate: Segmented Speech (from `speech_segmenter.py`)

```python
[
    {
        'segment_id': int,         # Unique identifier (0-indexed)
        'text': str,               # Segment text content
        'start_time': float,       # Segment start time (seconds)
        'end_time': float,         # Segment end time (seconds)
        'word_count': int,         # Number of words in segment
        'words': [                 # Word-level detail (preserved from STT)
            {
                'word': str,
                'start_time': float,
                'end_time': float
            },
            ...
        ]
    },
    ...
]
```

**Example:**
```python
[
    {
        'segment_id': 0,
        'text': 'The dog barked loudly.',
        'start_time': 0.0,
        'end_time': 1.3,
        'word_count': 4,
        'words': [
            {'word': 'The', 'start_time': 0.0, 'end_time': 0.2},
            {'word': 'dog', 'start_time': 0.2, 'end_time': 0.5},
            {'word': 'barked', 'start_time': 0.5, 'end_time': 0.9},
            {'word': 'loudly', 'start_time': 0.9, 'end_time': 1.3}
        ]
    },
    {
        'segment_id': 1,
        'text': 'Thunder echoed across the valley.',
        'start_time': 1.5,
        'end_time': 3.2,
        'word_count': 5,
        'words': [...]
    }
]
```

---

## 3. Output: Embedded Speech Segments (from `speech_embedder.py`)

### As pandas DataFrame

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'segment_id': int,             # Unique identifier
    'text': str,                   # Segment text
    'start_time': float,           # Start timestamp (seconds)
    'end_time': float,             # End timestamp (seconds)
    'duration': float,             # Segment duration (seconds)
    'word_count': int,             # Number of words
    'embedding': np.ndarray,       # 384-dimensional vector
    'embedding_model': str         # "all-MiniLM-L6-v2"
})
```

**Example:**
```python
   segment_id                              text  start_time  end_time  duration  word_count                                          embedding     embedding_model
0           0          The dog barked loudly.         0.0       1.3       1.3           4  [0.023, -0.045, 0.012, ..., -0.033]  all-MiniLM-L6-v2
1           1  Thunder echoed across the valley.         1.5       3.2       1.7           5  [-0.012, 0.089, -0.056, ..., 0.021]  all-MiniLM-L6-v2
```

### As CSV File

```csv
segment_id,text,start_time,end_time,duration,word_count,embedding,embedding_model
0,"The dog barked loudly.",0.0,1.3,1.3,4,"[0.023, -0.045, 0.012, ..., -0.033]",all-MiniLM-L6-v2
1,"Thunder echoed across the valley.",1.5,3.2,1.7,5,"[-0.012, 0.089, -0.056, ..., 0.021]",all-MiniLM-L6-v2
```

**Loading CSV:**
```python
df = pd.read_csv("speech_embeddings.csv")
# Convert string representation to numpy array
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
```

### As JSON File

```json
{
  "metadata": {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "segmentation_method": "sentences",
    "total_segments": 2
  },
  "segments": [
    {
      "segment_id": 0,
      "text": "The dog barked loudly.",
      "start_time": 0.0,
      "end_time": 1.3,
      "duration": 1.3,
      "word_count": 4,
      "embedding": [0.023, -0.045, 0.012, ..., -0.033],
      "embedding_model": "all-MiniLM-L6-v2"
    },
    {
      "segment_id": 1,
      "text": "Thunder echoed across the valley.",
      "start_time": 1.5,
      "end_time": 3.2,
      "duration": 1.7,
      "word_count": 5,
      "embedding": [-0.012, 0.089, -0.056, ..., 0.021],
      "embedding_model": "all-MiniLM-L6-v2"
    }
  ]
}
```

**Loading JSON:**
```python
import json
with open("speech_embeddings.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['segments'])
df['embedding'] = df['embedding'].apply(np.array)
```

---

## 4. Integration: Sound Embeddings Format

The sound embeddings (from `data/soundbible_details_from_section_with_embeddings.csv`) have compatible structure:

```python
df_sounds = pd.DataFrame({
    'title': str,                  # Sound name
    'description': str,            # Sound description
    'keywords': str,               # Comma-separated keywords
    'audio_url': str,              # URL to audio file
    'length': float,               # Sound duration (seconds)
    'embedding': np.ndarray,       # 384-dimensional vector (same model!)
    # ... other metadata columns
})
```

**Critical: Both speech and sound embeddings use `all-MiniLM-L6-v2` model**

---

## 5. Matching Output: Sound Timeline

After matching speech embeddings with sound embeddings, the output structure for audio mixing:

```python
[
    {
        'segment_id': int,             # Speech segment identifier
        'segment_text': str,           # Original speech text
        'segment_start_time': float,   # Speech start time
        'segment_end_time': float,     # Speech end time
        'sound_id': str,               # Sound identifier
        'sound_title': str,            # Sound name
        'sound_url': str,              # URL to sound file
        'sound_duration': float,       # Sound length
        'similarity_score': float,     # Cosine similarity (0-1)
        'insert_time': float,          # When to insert sound (seconds)
        'volume_adjustment': float     # Optional volume scaling
    },
    ...
]
```

**Example:**
```python
[
    {
        'segment_id': 0,
        'segment_text': 'The dog barked loudly.',
        'segment_start_time': 0.0,
        'segment_end_time': 1.3,
        'sound_id': 'dog_bark_001',
        'sound_title': 'Dog Barking',
        'sound_url': 'https://soundbible.com/dog_bark.wav',
        'sound_duration': 1.2,
        'similarity_score': 0.89,
        'insert_time': 0.5,  # Insert during "barked"
        'volume_adjustment': 0.7
    },
    {
        'segment_id': 1,
        'segment_text': 'Thunder echoed across the valley.',
        'segment_start_time': 1.5,
        'segment_end_time': 3.2,
        'sound_id': 'thunder_001',
        'sound_title': 'Thunder Clap',
        'sound_url': 'https://soundbible.com/thunder.wav',
        'sound_duration': 2.5,
        'similarity_score': 0.92,
        'insert_time': 1.5,
        'volume_adjustment': 0.8
    }
]
```

---

## Type Hints Reference

For use in type-annotated Python code:

```python
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Type aliases
STTResult = Dict[str, Any]
WordTiming = Dict[str, Any]
Segment = Dict[str, Any]
Embedding = np.ndarray  # Shape: (384,)

# Function signatures
def load_stt_output(stt_result: STTResult) -> tuple[str, List[WordTiming]]:
    """Extract transcript and word timings."""
    ...

def segment_by_sentences(
    transcript: str,
    word_timings: List[WordTiming]
) -> List[Segment]:
    """Segment transcript with timing info."""
    ...

def process_stt_output(stt_result: STTResult) -> pd.DataFrame:
    """Generate embeddings for speech segments."""
    ...
```

---

## Data Flow Diagram

```
STT Output (dict)
    │
    ├─── results.transcript (str)
    └─── words_timings (List[Dict])
          │
          ▼
    [Segmentation]
          │
          ▼
    Segments (List[Dict])
    - segment_id, text, timing, words
          │
          ▼
    [Batch Embedding]
          │
          ▼
    DataFrame / CSV / JSON
    - segment_id, text, timing, embedding
          │
          ▼
    [Similarity Matching with Sounds]
          │
          ▼
    Sound Timeline (List[Dict])
    - segment info + matched sound info + score
          │
          ▼
    [Audio Mixing]
          │
          ▼
    Enhanced Audio File
```
