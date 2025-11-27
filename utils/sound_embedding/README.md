# Sound Embedding Module

This module generates and manages semantic embeddings for sound effects, enabling intelligent matching with speech segments.

## Overview

The sound embedding pipeline converts sound effect metadata (title, description, keywords) into vector representations that capture semantic meaning. These embeddings are then used to find sounds that match the content of speech segments.

**Key Features:**
- **Rich Semantic Representation**: Combines title, description, and keywords for comprehensive meaning capture
- **Compatible with Speech Embeddings**: Uses same `all-MiniLM-L6-v2` model (384 dimensions)
- **Local Processing**: No API keys required - runs entirely offline
- **Efficient Batch Processing**: Handles 2,000+ sounds in seconds
- **Flexible Loading**: Cache support, validation, filtering capabilities

## Architecture

```
Sound CSV (title, description, keywords)
    ↓
[Text Combination] → "{title}: {description} [{keywords}]"
    ↓
[Batch Embedding Generation]
    ↓
Embedded sounds (384-dim vectors)
    ↓
[Save to CSV]
    ↓
[Load & Validate]
    ↓
Ready for semantic matching with speech
```

## Modules

### `sound_embedder.py`

Generates embeddings for sound effect metadata.

**Main Class:** `SoundEmbedder`

**Features:**
- Combines title + description + keywords into embedding text
- Batch processing with configurable batch size (default: 32)
- Progress tracking for large datasets
- Outputs CSV and metadata JSON

**Example:**
```python
from sound_embedding import SoundEmbedder

# Create embedder
embedder = SoundEmbedder(batch_size=32, show_progress=True)

# Load sound data
import pandas as pd
df = pd.read_csv("data/soundbible_details_from_section.csv", sep=';')

# Generate embeddings
df_embedded = embedder.process_sound_dataframe(
    df,
    save_output=True,
    output_path="data/soundbible_embeddings.csv"
)

# Result: CSV with 384-dim embeddings for each sound
```

### `sound_embedding_loader.py`

Loads and manages pre-computed sound embeddings.

**Main Class:** `SoundEmbeddingLoader`

**Features:**
- Load embeddings from CSV files
- Parse string representations to numpy arrays
- Validate embedding dimensions and format
- Filter by keywords, duration
- Caching for performance

**Example:**
```python
from sound_embedding import SoundEmbeddingLoader

# Create loader
loader = SoundEmbeddingLoader(cache_enabled=True)

# Load embeddings
sounds_df = loader.load_embeddings("data/soundbible_embeddings.csv")

# Filter by keywords
dog_sounds = loader.filter_by_keywords(sounds_df, ['dog', 'bark'])

# Filter by duration
short_sounds = loader.filter_by_duration(sounds_df, max_seconds=3.0)
```

## Usage

### Command Line

#### Generate Sound Embeddings

```bash
# Generate embeddings for all sounds
uv run python -m sound_embedding.sound_embedder

# With custom paths
uv run python -m sound_embedding.sound_embedder \
    --input data/my_sounds.csv \
    --output data/my_embeddings.csv \
    --batch-size 64
```

#### Load and Validate Embeddings

```bash
# Load and validate
uv run python -m sound_embedding.sound_embedding_loader \
    data/soundbible_embeddings.csv

# Filter by keywords
uv run python -m sound_embedding.sound_embedding_loader \
    data/soundbible_embeddings.csv \
    --filter-keywords dog bark animal

# Filter by duration
uv run python -m sound_embedding.sound_embedding_loader \
    data/soundbible_embeddings.csv \
    --min-duration 1.0 \
    --max-duration 5.0
```

### Programmatic Usage

#### End-to-End: Generate and Save

```python
from sound_embedding import process_sound_file

# One-liner to generate embeddings
df = process_sound_file(
    input_path="data/soundbible_details_from_section.csv",
    output_path="data/soundbible_embeddings.csv",
    batch_size=32
)

print(f"Generated embeddings for {len(df)} sounds")
```

#### Load and Use

```python
from sound_embedding import load_sound_embeddings

# Load embeddings
sounds_df = load_sound_embeddings("data/soundbible_embeddings.csv")

# Access embeddings
for idx, row in sounds_df.head().iterrows():
    print(f"{row['title']}: {row['embedding'].shape}")
```

#### Integration with Semantic Matching

```python
from sound_embedding import load_sound_embeddings
from semantic_matcher import SemanticMatcher
import pandas as pd
import numpy as np

# Load sound embeddings
sounds_df = load_sound_embeddings("data/soundbible_embeddings.csv")

# Load speech embeddings (from text-processing pipeline)
speech_df = pd.read_csv("text-processing/output/speech_embeddings.csv")
speech_df['embedding'] = speech_df['embedding'].apply(eval).apply(np.array)

# Match speech to sounds
matcher = SemanticMatcher(similarity_threshold=0.75, top_k=3)
matches = matcher.match_speech_to_sounds(speech_df, sounds_df)

print(f"Found {len(matches)} sound matches")
```

## Data Formats

### Input: Sound Metadata CSV

Semicolon-delimited CSV with columns:

```csv
title;href;url;description;keywords;length;audio_url
Airplane Landing Airport;/123.html;https://...;Large commercial airplane...;['airport', 'runway', 'landing'];2.5;https://...mp3
```

### Output: Sound Embeddings CSV

Comma-delimited CSV with embeddings:

```csv
title,href,url,description,keywords,length,audio_url,embedding,embedding_model,embedding_text
Airplane Landing Airport,/123.html,https://...,Large commercial airplane...,['airport', 'runway', 'landing'],2.5,https://...mp3,"[0.157, -0.010, ...]",all-MiniLM-L6-v2,"Airplane Landing Airport: Large commercial airplane... [airport, runway, landing]"
```

### Metadata JSON

Accompanying metadata file:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "total_sounds": 2120,
  "batch_size": 32,
  "text_format": "{title}: {description} [{keywords}]"
}
```

## Embedding Text Format

Sounds are embedded using a combined text format:

```
{title}: {description} [{keywords}]
```

**Example:**
```
Thunder: Sound of thunder crackling through the air followed by a large thunder clap... [weather, storm, thunder storm, lightning, explosions, rain]
```

This format provides:
- **Title**: Primary identifier
- **Description**: Rich contextual information
- **Keywords**: Additional semantic tags

## Configuration

### Environment Variables

No environment variables or API keys required! The model runs entirely locally.

### Model Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Model | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers |
| Dimensions | 384 | Must match speech embeddings |
| Normalization | Yes | Recommended for cosine similarity |
| Batch Size | 32 | Configurable (16-128) |

### Performance

- **Processing Speed**: ~4 seconds for 2,120 sounds (batch size 32)
- **File Size**: ~19MB for 2,120 embeddings (CSV format)
- **Memory**: Model uses ~80MB, embeddings use ~3MB in memory
- **GPU Support**: Automatic if CUDA/MPS available

## Integration Points

### With Text Processing Pipeline

Sound embeddings are designed to work seamlessly with speech embeddings:

```python
# Both use the same model!
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions

# Speech embeddings (from text-processing/speech_embedder.py)
speech_embeddings = process_stt_output(stt_result)

# Sound embeddings (from sound_embedding/sound_embedder.py)
sound_embeddings = process_sound_file("data/sounds.csv")

# Compatible for cosine similarity matching!
```

### With Semantic Matcher

The semantic matcher uses sound embeddings to find relevant sounds:

```python
from semantic_matcher import SemanticMatcher

matcher = SemanticMatcher(similarity_threshold=0.75)
matches = matcher.match_speech_to_sounds(
    speech_embeddings,
    sound_embeddings
)
```

## API Reference

### SoundEmbedder

```python
class SoundEmbedder:
    def __init__(self, batch_size: int = 32, show_progress: bool = True)

    def create_embedding_text(self, title: str, description: str, keywords: str) -> str

    def process_sound_dataframe(
        self,
        df: pd.DataFrame,
        save_output: bool = True,
        output_path: Optional[str] = None
    ) -> pd.DataFrame
```

### SoundEmbeddingLoader

```python
class SoundEmbeddingLoader:
    def __init__(self, cache_enabled: bool = True)

    def load_embeddings(
        self,
        filepath: str,
        validate: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame

    def filter_by_keywords(
        self,
        df: pd.DataFrame,
        keywords: List[str],
        match_any: bool = True
    ) -> pd.DataFrame

    def filter_by_duration(
        self,
        df: pd.DataFrame,
        min_seconds: Optional[float] = None,
        max_seconds: Optional[float] = None
    ) -> pd.DataFrame
```

### Convenience Functions

```python
def process_sound_file(
    input_path: str = "data/soundbible_details_from_section.csv",
    output_path: str = "data/soundbible_embeddings.csv",
    batch_size: int = 32
) -> pd.DataFrame

def load_sound_embeddings(filepath: str) -> pd.DataFrame
```

## Troubleshooting

### Import Errors

If you get import errors:

```python
# Make sure to import from the package
from sound_embedding import SoundEmbedder, load_sound_embeddings

# Not:
# from sound_embedder import SoundEmbedder  # ❌ Wrong
```

### Dimension Mismatch

If embeddings have wrong dimensions:

```python

### File Format Issues

If loading fails:

```bash
# Check CSV delimiter (semicolon vs comma)
head -1 data/soundbible_details_from_section.csv

# For semicolon-delimited:
df = pd.read_csv(filepath, sep=';')

# For comma-delimited:
df = pd.read_csv(filepath)
```

## Testing

Run the module tests:

```bash
# Test sound embedder
uv run python -m sound_embedding.sound_embedder \
    --input data/soundbible_details_from_section.csv

# Test loader
uv run python -m sound_embedding.sound_embedding_loader \
    data/soundbible_embeddings.csv
```

## Next Steps

After generating sound embeddings:

1. Load speech embeddings (from `text-processing` module)
2. Use `semantic_matcher.py` to match speech to sounds
3. Generate sound timeline for audio mixing
4. Proceed to Step #5 (Audio Mixing)

See main project README for the complete pipeline workflow.
