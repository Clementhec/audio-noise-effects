# Agentic Sound Editing Pipeline

An intelligent audio processing system that automatically enhances audio and video content by adding contextually relevant sound effects based on speech content analysis.


## Overview

This project implements an AI-driven pipeline that analyzes narrated speech, understands the semantic context, and intelligently inserts sound effects from a library of 2,120+ sounds at the most appropriate moments. The system combines speech recognition, natural language understanding, vector embeddings, and audio processing to create dynamic and engaging audio experiences.

**Key Capabilities:**
- Speech-to-text transcription with word-level timing precision
- Semantic embedding of speech content and sound metadata in compatible vector space
- Vector similarity matching between speech context and sound effects
- Score-based filtering to select optimal sound placements
- Automated audio mixing and synchronization

## Architecture

The pipeline follows a microservice-oriented architecture with the following flow:

```
Input (Audio/Video)
    ↓
[Audio Extraction] → Extract audio track from video
    ↓
[Speech-to-Text] → Transcribe speech with word-level timing
    ↓
[Speech Embedding] → Generate embeddings for speech segments
    ↓
[Sound Embedding] → Pre-computed embeddings for sound metadata
    ↓
[Vector Matching] → Calculate similarity scores between speech and sounds
    ↓
[Score-based Filtering] → Select best matches based on similarity thresholds
    ↓
[Audio Mixing] → Combine original audio with selected sound effects
    ↓
Output (Enhanced Audio/Video)
```

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

**Key Principle**: Both speech content and sound metadata must use **compatible embeddings** (same model, same embedding space) to enable meaningful similarity comparisons. Filtering decisions are made based on **actual similarity scores**, not pre-selection.

**Technology Stack:**
- **Python 3.x** - Core language
- **Google Cloud Speech-to-Text** - Speech recognition with timing
- **OpenAI API** - Embeddings (text-embedding-3-small) and LLM analysis (o3-mini)
- **NumPy/SciPy** - Vector operations and similarity metrics
- **Pandas** - Data manipulation and sound library management

## References

- [Existing Eleven labs project](https://videotosfx.elevenlabs.io/)

## Project Structure

```
Audio-noise-effects/
├── data/
│   ├── soundbible_details_from_section.csv              # Sound library metadata (2,120 sounds)
│   └── soundbible_details_from_section_with_embeddings.csv  # Pre-computed embeddings
├── utils/
│   └── embeddings_utils.py                              # Embedding and similarity utilities
├── stt_google.py                                        # Speech-to-text module
├── embeddings.py                                        # Sound metadata embedding generator
├── semantic-search.py                                   # Vector similarity search
├── LLM-sentence-highlight.py                            # (Exploratory) LLM-based pre-filtering
├── requirements.txt                                     # Python dependencies
└── README.md
```

### Module Descriptions

- **stt_google.py**: Transcribes audio using Google Cloud Speech API, extracting full transcript and word-level timings
- **embeddings.py**: Generates vector embeddings for sound effect descriptions using OpenAI's text-embedding-3-small model
- **semantic-search.py**: Performs vector similarity search using cosine distance to calculate scores between speech embeddings and sound embeddings
- **utils/embeddings_utils.py**: Utility functions for generating compatible embeddings, calculating distance metrics (cosine, L1, L2, Linf), and visualization

**Note on Architecture**: The current `LLM-sentence-highlight.py` module represents an exploratory approach that pre-filters sentences before matching. The recommended architecture uses **score-based filtering** instead - calculate similarity scores for all speech segments against all sounds, then filter by threshold. This preserves all information and makes decisions based on actual compatibility.

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Cloud account with Speech-to-Text API enabled
- OpenAI API key

## Configuration

### API Key Setup

Add to `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your key from: [Google AI Studio](https://makersuite.google.com/app/apikey)


### Install Dependencies

```bash
pip install -r requirements.txt
```

```bash
sudo apt-get install portaudio19-dev
```

### Configuration

#### 1. Google Cloud Speech API Setup

1. Create a Google Cloud project and enable the Speech-to-Text API
2. Create a service account and download the JSON credentials file
3. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

#### 2. OpenAI API Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 3. Sound Library

The sound library dataset should be located in the `data/` directory. Pre-computed embeddings are available in `soundbible_details_from_section_with_embeddings.csv`.

## Module Documentation

### Speech-to-Text (stt_google.py)

Transcribes audio files using the Google Cloud Speech-to-Text Python API with word-level timing information.

**Usage:**

```python
from stt_google import transcribe_audio

# Transcribe an audio file
result = transcribe_audio("path/to/audio.wav", language_code="fr-FR")

# Access transcript
transcript = result['results']['transcript']
confidence = result['results']['confidence']

# Access word-level timings
for word_info in result['words_timings']:
    word = word_info['word']
    start_time = word_info['start_time']  # In seconds
    end_time = word_info['end_time']
    print(f"{word}: {start_time}s - {end_time}s")
```

**Features:**
- Automatic sample rate detection for WAV files
- Multi-language support (default: French "fr-FR")
- Word-level confidence scores
- Precise timing for synchronization

---

### Sound Embeddings (embeddings.py)

Generates vector embeddings for sound effect descriptions using OpenAI's embedding model.

**Usage:**

```python
import pandas as pd
from embeddings import process_sound_library

# Load sound library
sounds_df = pd.read_csv("data/soundbible_details_from_section.csv")

# Generate embeddings for descriptions
sounds_with_embeddings = process_sound_library(sounds_df)

# Save enriched dataset
sounds_with_embeddings.to_csv("data/sounds_embedded.csv")
```

**Output:**
- CSV file with original metadata plus `embedding` column containing vector representations

---

### Semantic Search (semantic-search.py)

Performs vector similarity search to find sounds matching speech content.

**Usage:**

```python
from semantic_search import search_sounds
import pandas as pd

# Load sound library with embeddings
sounds_df = pd.read_csv("data/soundbible_details_from_section_with_embeddings.csv")

# Search for matching sounds
query = "The dog is barking loudly in the backyard"
matches = search_sounds(sounds_df, query, n=5)

# Results include similarity scores and metadata
for idx, row in matches.iterrows():
    print(f"Sound: {row['title']}")
    print(f"Similarity: {row['similarity_score']}")
    print(f"Audio URL: {row['audio_url']}")
```

**Algorithm:**
- Uses cosine similarity for vector matching
- Returns top-N most relevant sounds
- Includes metadata (title, description, audio URL, duration)

---

### LLM Sentence Highlighting (LLM-sentence-highlight.py)

**⚠️ DEPRECATED APPROACH**: This module represents an exploratory technique that pre-filters sentences before similarity matching. This loses information because it makes selection decisions without knowing actual compatibility scores. **The recommended approach is score-based filtering** (see workflow example above).

Uses LLM to identify which sentences in the transcript would benefit from sound effect enhancement.

**Usage (for reference only):**

```python
from LLM_sentence_highlight import highlight_sentences

# Analyze transcript
transcript = "The storm approached rapidly. Lightning struck nearby. Thunder echoed across the valley."
highlighted = highlight_sentences(transcript, max_highlights=4)

# Returns exact sentence fragments (≤15 words each)
for sentence in highlighted:
    print(f"Enhance: {sentence}")
```

**Why This Approach Is Not Recommended:**
- Pre-filters sentences **before** calculating similarity scores with sounds
- Makes blind decisions without knowing which sounds are actually compatible
- Loses information that could lead to better matches
- The LLM has no access to the sound library or embedding similarity metrics

**Better Alternative:**
1. Embed ALL speech segments
2. Calculate similarity scores with ALL sounds
3. Filter by score threshold (e.g., cosine similarity ≥ 0.75)
4. Select best matches based on actual compatibility

---

### Embedding Utilities (utils/embeddings_utils.py)

Comprehensive utility library for embedding operations.

**Key Functions:**

```python
from utils.embeddings_utils import (
    get_embedding,
    get_embeddings,
    cosine_similarity,
    distances_from_embeddings
)

# Single embedding
embedding = get_embedding("text to embed", model="text-embedding-3-small")

# Batch embeddings (up to 2,048 texts)
embeddings = get_embeddings(["text1", "text2", "text3"])

# Calculate cosine similarity
similarity = cosine_similarity(embedding1, embedding2)

# Distance metrics (cosine, L1, L2, Linf)
distances = distances_from_embeddings(
    query_embedding,
    embeddings_list,
    distance_metric="cosine"
)
```

**Additional Features:**
- Retry logic with exponential backoff (tenacity)
- PCA and t-SNE dimensionality reduction
- 2D/3D visualization tools (Plotly)
- Async embedding functions for performance

## Usage Examples

### Recommended End-to-End Workflow (Score-Based Matching)

```python
import pandas as pd
import numpy as np
from stt_google import transcribe_audio
from utils.embeddings_utils import get_embeddings, cosine_similarity

# Step 1: Transcribe audio with word-level timing
audio_file = "input_audio.wav"
transcript_data = transcribe_audio(audio_file, language_code="en-US")
transcript = transcript_data['results']['transcript']
word_timings = transcript_data['words_timings']

# Step 2: Segment transcript into sentences/phrases
# (Simple sentence splitting - could use more sophisticated segmentation)
sentences = transcript.split('.')
sentences = [s.strip() for s in sentences if s.strip()]

# Step 3: Generate embeddings for ALL speech segments (compatible with sound embeddings)
speech_embeddings = get_embeddings(sentences, model="text-embedding-3-small")

# Step 4: Load pre-computed sound embeddings (same model!)
sounds_df = pd.read_csv("data/soundbible_details_from_section_with_embeddings.csv")
# Parse embedding strings to numpy arrays
sounds_df['embedding'] = sounds_df['embedding'].apply(eval).apply(np.array)

# Step 5: Calculate similarity scores between ALL speech segments and ALL sounds
sound_timeline = []
for idx, sentence in enumerate(sentences):
    speech_emb = speech_embeddings[idx]

    # Calculate cosine similarity with all sounds
    similarities = sounds_df['embedding'].apply(
        lambda sound_emb: cosine_similarity(speech_emb, sound_emb)
    )
    sounds_df['similarity_score'] = similarities

    # Step 6: Filter by threshold and select best match
    SIMILARITY_THRESHOLD = 0.75  # Adjust based on desired selectivity
    candidates = sounds_df[sounds_df['similarity_score'] >= SIMILARITY_THRESHOLD]

    if len(candidates) > 0:
        best_match = candidates.nlargest(1, 'similarity_score').iloc[0]

        # Map sentence to timestamp using word_timings
        # (Implementation depends on word alignment)

        sound_timeline.append({
            'text': sentence,
            'sound_url': best_match['audio_url'],
            'sound_title': best_match['title'],
            'similarity_score': best_match['similarity_score'],
            'timestamp': None  # Calculate from word_timings
        })

# Step 7: Mix audio (implementation in development)
# mix_audio_with_effects(audio_file, sound_timeline, output_file="enhanced_audio.wav")
```

**Key Advantages of This Approach:**
- Uses **compatible embeddings** (same model for speech and sounds)
- Makes decisions based on **actual similarity scores**, not blind pre-filtering
- Preserves all information until filtering stage
- Threshold-based filtering is transparent and tunable

### Custom Sound Selection

```python
# Filter sounds by keywords or duration
sounds_df = pd.read_csv("data/soundbible_details_from_section_with_embeddings.csv")

# Find short sounds only (< 5 seconds)
short_sounds = sounds_df[sounds_df['length'] < 5.0]

# Search within specific category
query = "ocean waves crashing"
matches = search_sounds(short_sounds, query, n=10)
```

## Project Guidelines

### Microservice Architecture
Each pipeline step is designed as an independent module with clear inputs and outputs, enabling:
- Parallel development
- Easy testing and debugging
- Flexible deployment options
- Modular replacement of components

### Data Flow Conventions
- **Input/Output Folders**: Each module can dump intermediate results to designated folders for inspection
- **Audio Format**: Use `.wav` format for all audio processing (lossless, widely supported)
- **Metadata Format**: CSV files for sound libraries, JSON for configuration
- **Embeddings**: Store as CSV columns with array representations

### Adding New Sounds
To extend the sound library:

```python
import pandas as pd
from embeddings import generate_embedding

# Load existing library
sounds = pd.read_csv("data/soundbible_details_from_section_with_embeddings.csv")

# Add new sound
new_sound = {
    'title': 'Thunder Crash',
    'description': 'Loud thunder clap during storm',
    'audio_url': 'https://example.com/thunder.wav',
    'keywords': 'thunder, storm, weather',
    'length': 3.2
}

# Generate embedding
new_sound['embedding'] = generate_embedding(new_sound['description'])

# Append to library
sounds = sounds.append(new_sound, ignore_index=True)
sounds.to_csv("data/soundbible_details_from_section_with_embeddings.csv")
```

## Current Status & Roadmap

The core components for speech analysis and semantic matching are functional. Current development focuses on:
- Audio/video extraction and format conversion
- Real-time audio mixing and synchronization
- Timeline generation and event scheduling
- End-to-end pipeline integration
- Performance optimization for large audio files

**Note**: Some audio manipulation features are in active development. The semantic matching and speech analysis components are fully operational.

## Contributing

### Code Structure Guidelines
- Follow PEP 8 style conventions
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new modules
- Document configuration parameters

### Extending Modules
Each module exposes simple interfaces:
- Input: Clear parameters or file paths
- Output: Structured data (dicts, DataFrames, arrays)
- Error handling: Graceful degradation with warnings

### Testing
Place test files in module directories:
```python
# test_stt_google.py
from stt_google import transcribe_audio

def test_transcription():
    result = transcribe_audio("test.wav")
    assert 'results' in result
    assert 'words_timings' in result
```

## License

This project uses sound effects from [SoundBible.com](http://soundbible.com/) under various Creative Commons licenses. Please check individual sound licenses before commercial use.

## Credits

- **Sound Library**: SoundBible.com community contributors
- **Speech Recognition**: Google Cloud Speech-to-Text API
- **Embeddings & LLM**: OpenAI API
- **Vector Operations**: NumPy, SciPy, scikit-learn

---

**Questions or Issues?** Open an issue or contribute to the project development.
