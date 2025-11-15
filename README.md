# Agentic Sound Editing Pipeline

An intelligent audio processing system that automatically enhances audio and video content by adding contextually relevant sound effects based on speech content analysis.

## Overview

This project implements an AI-driven pipeline that analyzes narrated speech, understands the semantic context, and intelligently inserts sound effects from a library of 2,120+ sounds at the most appropriate moments. The system combines speech recognition, natural language understanding, vector embeddings, and audio processing to create dynamic and engaging audio experiences.

**Key Capabilities:**
- Speech-to-text transcription with word-level timing precision
- Semantic analysis of speech content using embeddings
- Intelligent matching between speech context and sound effects
- LLM-powered selection of optimal placement moments
- Automated audio mixing and synchronization

## Architecture

The pipeline follows a microservice-oriented architecture with the following flow:

```
Input (Audio/Video)
    ↓
[Audio Extraction] → Extract audio track from video
    ↓
[Speech-to-Text] → Transcribe speech with Google Cloud Speech API
    ↓
[Semantic Analysis] → Generate embeddings for transcribed content
    ↓
[Vector Matching] → Match speech context to sound effect descriptions
    ↓
[LLM Filtering] → Select optimal sentences for enhancement
    ↓
[Audio Mixing] → Combine original audio with sound effects
    ↓
Output (Enhanced Audio/Video)
```

**Technology Stack:**
- **Python 3.x** - Core language
- **Google Cloud Speech-to-Text** - Speech recognition with timing
- **OpenAI API** - Embeddings (text-embedding-3-small) and LLM analysis (o3-mini)
- **NumPy/SciPy** - Vector operations and similarity metrics
- **Pandas** - Data manipulation and sound library management

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
├── LLM-sentence-highlight.py                            # Intelligent placement selection
├── requirements.txt                                     # Python dependencies
└── README.md
```

### Module Descriptions

- **stt_google.py**: Transcribes audio using Google Cloud Speech API, extracting full transcript and word-level timings
- **embeddings.py**: Generates vector embeddings for sound effect descriptions in the library
- **semantic-search.py**: Performs vector similarity search to match speech content with relevant sounds
- **LLM-sentence-highlight.py**: Uses LLM to identify sentences that would benefit from sound enhancement
- **utils/embeddings_utils.py**: Utility functions for embeddings, distance metrics, and visualization

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Cloud account with Speech-to-Text API enabled
- OpenAI API key

### Install Dependencies

```bash
pip install -r requirements.txt
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

Uses LLM to identify which sentences in the transcript would benefit from sound effect enhancement.

**Usage:**

```python
from LLM_sentence_highlight import highlight_sentences

# Analyze transcript
transcript = "The storm approached rapidly. Lightning struck nearby. Thunder echoed across the valley."
highlighted = highlight_sentences(transcript, max_highlights=4)

# Returns exact sentence fragments (≤15 words each)
for sentence in highlighted:
    print(f"Enhance: {sentence}")
```

**Features:**
- Structured output using Pydantic models
- Returns exact text fragments (no rephrasing)
- Configurable number of highlights
- Optimized for impactful moment detection

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

### Basic End-to-End Workflow

```python
import pandas as pd
from stt_google import transcribe_audio
from semantic_search import search_sounds
from LLM_sentence_highlight import highlight_sentences

# Step 1: Transcribe audio
audio_file = "input_audio.wav"
transcript_data = transcribe_audio(audio_file, language_code="en-US")
transcript = transcript_data['results']['transcript']
word_timings = transcript_data['words_timings']

# Step 2: Identify key moments for enhancement
highlighted_sentences = highlight_sentences(transcript, max_highlights=5)

# Step 3: Load sound library
sounds_df = pd.read_csv("data/soundbible_details_from_section_with_embeddings.csv")

# Step 4: Find matching sounds for each highlighted sentence
sound_timeline = []
for sentence in highlighted_sentences:
    matches = search_sounds(sounds_df, sentence, n=3)
    best_match = matches.iloc[0]

    # Find timing for this sentence in the transcript
    # (Implementation depends on sentence position matching)

    sound_timeline.append({
        'text': sentence,
        'sound_url': best_match['audio_url'],
        'sound_title': best_match['title'],
        'timestamp': None  # Calculate from word_timings
    })

# Step 5: Mix audio (implementation in development)
# mix_audio_with_effects(audio_file, sound_timeline, output_file="enhanced_audio.wav")
```

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
