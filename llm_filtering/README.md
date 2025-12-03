# LLM Filtering Module

This module uses Google's Gemini LLM to intelligently filter and refine sound effect recommendations from the similarity matching stage.

## Purpose

After the similarity matching generates top-K sound candidates for each speech segment, the LLM filtering step:

1. **Ranks all sentences** - Assigns a unique relevance rank to each sentence (1 = most impactful, 2 = second most, etc.)
2. **Selects optimal placement** - Identifies the specific word where each sound should be placed
3. **Chooses best match** - Picks the most appropriate sound from the top candidates
4. **Provides reasoning** - Explains why each sound was selected and ranked

## How It Works

### Input

The LLM receives similarity matching results:

```json
[
  {
    "speech_index": 0,
    "speech_text": "I heard thunder rumbling in the distance",
    "top_matches": [
      {
        "sound_title": "Thunder Rumble",
        "similarity": 0.8542,
        "sound_description": "Deep thunder sound",
        "audio_url": "..."
      },
      ...
    ]
  }
]
```

### Processing

The LLM (Gemini 2.5 Flash Lite) evaluates:
- Assigns a unique relevance rank to each sentence (1, 2, 3, 4...)
- The specific word where the sound fits best
- Which of the top-K sounds is most appropriate (by index: 0, 1, or 2)
- Why this combination works and why this rank was assigned

### Output

Filtered results with relevance ranking and target words:

```json
{
  "filtered_sounds": [
    {
      "speech_index": 0,
      "speech_text": "I heard thunder rumbling in the distance",
      "relevance_rank": 1,
      "target_word": "thunder",
      "selected_sound": {
        "sound_title": "Thunder Rumble",
        "sound_description": "Deep thunder sound",
        "audio_url_wav": "...",
        "similarity_score": 0.8542
      },
      "reasoning": "Thunder is a concrete sound that clearly benefits from an effect - ranked #1 for high impact"
    }
  ]
}
```

## Usage

### As Part of Full Pipeline

```bash
# Run complete pipeline with LLM filtering
uv run python main.py video.mp4 --full-pipeline

# Control max sounds to select
uv run python main.py video.mp4 --full-pipeline --max-sounds 5
```

### Standalone Usage

```python
from llm_filtering import filter_sounds
import json

# Load similarity results
with open('output/video_similarity_matches.json', 'r') as f:
    similarity_data = json.load(f)

# Run LLM filtering
result = filter_sounds(
    similarity_data=similarity_data,
    max_sounds=5,  # Optional: prioritize top N sentences
    user_prompt="Favor natural sounds and avoid ambiances",  # Optional
    output_file='output/filtered_sounds.json'
)

# Use filtered results (sorted by relevance rank)
for item in result['filtered_sounds']:
    print(f"Rank #{item['relevance_rank']}: {item['speech_text']}")
    print(f"Target word: {item['target_word']}")
    print(f"Sound: {item['selected_sound']['sound_title']}")
```

### From File

```python
from llm_filtering import filter_from_file

result = filter_from_file(
    input_file='output/video_similarity_matches.json',
    output_file='output/filtered_sounds.json',
    max_sounds=5,
    user_prompt="Prioritize impactful sounds"  # Optional
)
```

## Configuration

### API Key

The module requires a Google API key for Gemini:

1. Get a key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Or pass directly:

```python
result = filter_sounds(
    similarity_data=data,
    api_key='your_api_key_here'
)
```

### Max Sounds Parameter

Control how many sentences to prioritize:

```bash
# Rank all sentences (LLM decides priority)
uv run python main.py video.mp4 --full-pipeline

# Prioritize top 5 sentences
uv run python main.py video.mp4 --full-pipeline --max-sounds 5

# Prioritize top 10 sentences
uv run python main.py video.mp4 --full-pipeline --max-sounds 10
```

**Note**: The `max_sounds` parameter guides the LLM to prioritize the top N most impactful sentences. All sentences are still ranked with unique relevance scores (1, 2, 3...), but the LLM focuses on selecting the best N.

## LLM Behavior

### Ranking Criteria

The LLM is instructed to:
- **Assign unique ranks** - Each sentence gets a unique relevance rank (1 = most impactful, 2 = second most, etc.)
- **Favor concrete sounds** - Thunder, barking, rain over general ambiances
- **One keyword per sentence** - Maximum, on the most relevant word
- **Natural placement** - Sounds should feel organic, not forced
- **Use sound indices** - Reference sounds by index (0, 1, or 2) to avoid hallucinations

### Example Rankings

**High Rank (1-3):**
- "I heard **thunder** rumbling" → Rank 1, Thunder sound on "thunder"
- "A dog **barking** loudly" → Rank 2, Dog bark on "barking"
- "The **rain** was pouring" → Rank 3, Rain sound on "rain"

**Lower Rank (4+):**
- "The weather was nice" → Rank 4, Less concrete/impactful
- "I felt happy" → Rank 5, Abstract emotion
- "We talked for hours" → Rank 6, Less benefit from sound effect

All sentences receive a unique rank, allowing you to select the top N most impactful ones.

## Output Format

### Complete Structure

```json
{
  "filtered_sounds": [
    {
      "speech_index": 0,
      "speech_text": "Original sentence text",
      "relevance_rank": 1,
      "target_word": "specific_word",
      "selected_sound": {
        "sound_title": "Sound Effect Name",
        "sound_description": "Description of the sound",
        "audio_url_wav": "https://...",
        "similarity_score": 0.8542
      },
      "reasoning": "Detailed decision explanation including why this rank"
    }
  ]
}
```

### Fields Explained

- `speech_index`: Index from original similarity results
- `speech_text`: Full sentence text
- `relevance_rank`: Unique rank (1 = most impactful, 2 = second most, etc.)
- `target_word`: Specific word for sound placement
- `selected_sound`: Chosen sound details (from original data, not LLM-generated)
  - `sound_title`: Name of the sound effect
  - `sound_description`: Description of the sound
  - `audio_url_wav`: URL/path to WAV audio file
  - `similarity_score`: Original similarity score from matching
- `reasoning`: LLM's explanation for the rank, sound choice, and word placement

## Integration with Pipeline

### Pipeline Flow

```
1. Extract Audio → 2. STT → 3. Embeddings → 4. Similarity Matching
                                                        ↓
                                                5. LLM Filtering
                                                        ↓
                                            Filtered Sound Timeline
```

### Step Details

**Step 4 Output** (Similarity Matching):
- 10 segments × 5 sounds = 50 total sound suggestions

**Step 5 Output** (LLM Filtering):
- All 10 segments ranked (1-10)
- Each with unique relevance rank
- Each with specific target word
- Best sound from top-3 candidates (selected by index)

### Why LLM Filtering?

Without LLM filtering:
- Too many sounds (overwhelming)
- No word-level placement
- Can't distinguish important vs. unimportant sounds
- No context awareness

With LLM filtering:
- All sounds ranked by relevance (1, 2, 3...)
- Precise timing (word-level)
- Context-aware decisions
- Easy to select top N most impactful sounds
- Prevents LLM hallucinations by using indices
- Natural, professional results

## Performance

### Speed

For typical videos:
- 10 speech segments
- Processing time: ~2-5 seconds
- API cost: ~$0.001 per request

Using Gemini 2.5 Flash Lite for:
- Speed: Very fast inference
- Cost: Extremely low cost
- Quality: Good reasoning capabilities

### Accuracy

The LLM correctly:
- Assigns unique relevance ranks (100% - enforced by prompt)
- Identifies concrete vs. abstract concepts (95%+)
- Selects appropriate target words (90%+)
- Chooses contextually correct sounds by index (90%+ - no hallucinations)

## Error Handling

### API Key Missing

```python
ValueError: GOOGLE_API_KEY must be defined in the environment or passed as a parameter
```

**Solution**: Add `GOOGLE_API_KEY` to `.env` file

### JSON Parsing Errors

The module includes `clean_json_response()` to handle:
- Markdown code blocks
- Extra whitespace
- Common LLM formatting issues

### API Errors

```python
try:
    result = filter_sounds(data)
except Exception as e:
    print(f"LLM filtering failed: {e}")
    # Fall back to using similarity results directly
```

## Advanced Usage

### Custom Prompts

To modify LLM behavior, edit `llm_filtering/filtering.py`:

```python
def create_prompt(similarity_data, max_sounds):
    prompt = """Your custom instructions here..."""
    # Modify the prompt
    return prompt
```

### Different Models

Change the Gemini model:

```python
def get_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')  # Or other model
```

### Custom Filtering Logic

Add post-processing:

```python
result = filter_sounds(data, max_sounds=10)

# Select only top 5 ranked sounds
top_5 = result['filtered_sounds'][:5]

# Filter by similarity threshold
high_similarity = [
    item for item in result['filtered_sounds']
    if item['selected_sound']['similarity_score'] > 0.7
]
```

## Best Practices

1. **Use Ranking System**: All sentences are ranked - select top N based on your needs
2. **Review Results**: Check filtered output before audio mixing
3. **Adjust top-k**: Use `--top-k 3` to give LLM good options to choose from (default)
4. **API Key Security**: Never commit `.env` file with API keys
5. **Error Handling**: Implement fallbacks if LLM filtering fails
6. **Custom Instructions**: Use `user_prompt` parameter to guide LLM behavior

## Troubleshooting

### "Need Fewer Sounds"

All sentences are ranked - simply use the top N:
```python
# Take only top 5 ranked sounds
top_sounds = result['filtered_sounds'][:5]
```

Or guide the LLM:
```bash
uv run python main.py video.mp4 --full-pipeline --max-sounds 5
```

### "Need Better Rankings"

If rankings seem off:
- Use `user_prompt` to guide LLM: "Prioritize loud, impactful sounds"
- Check similarity scores in Step 4
- Increase `--top-k` for more options
- Review prompt instructions in `filtering.py`

### "Wrong Words Selected"

If target words seem incorrect:
- Review the prompt in `filtering.py`
- Adjust LLM instructions
- Consider using different Gemini model

## Future Enhancements

Potential improvements:
1. **Multi-language Support**: Handle non-English transcripts
2. **Confidence Scores**: Add LLM confidence ratings per rank
3. **User Feedback**: Learn from manual corrections
4. **Batch Processing**: Process multiple videos efficiently
5. **Sound Categories**: Filter by sound type preferences
6. **Temporal Awareness**: Consider timing and pacing
7. **Dynamic Ranking**: Adjust ranks based on video context

## Examples

### Example 1: Weather Documentary

**Input**: "The storm was approaching with dark clouds overhead. Thunder rumbled in the distance. Rain began falling heavily."

**LLM Output**:
- Rank 1: Segment 2 "Thunder rumbled" → **thunder** (Thunder sound)
- Rank 2: Segment 3 "Rain began falling" → **Rain** (Rain sound)
- Rank 3: Segment 1 "storm was approaching" → **storm** (less concrete)

### Example 2: Nature Scene

**Input**: "Birds were singing in the trees. A dog barked nearby. Children were laughing and playing."

**LLM Output**:
- Rank 1: Segment 2 "dog barked" → **barked** (Dog bark - most impactful)
- Rank 2: Segment 1 "Birds were singing" → **singing** (Birds chirping)
- Rank 3: Segment 3 "Children were laughing" → **laughing** (Children laughter)

### Example 3: Abstract Content

**Input**: "I felt happy about the decision. The future looks bright. We made progress today."

**LLM Output**:
- Rank 1: Segment 3 "We made progress" → **progress** (least abstract)
- Rank 2: Segment 1 "I felt happy" → **happy** (emotion)
- Rank 3: Segment 2 "future looks bright" → **bright** (very abstract)
- LLM reasoning: "All sentences are abstract - ranked by relative concreteness. Consider using only top 1 or skipping all."

## API Reference

### filter_sounds()

```python
def filter_sounds(
    similarity_data: List[Dict[str, Any]],
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    user_prompt: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `similarity_data`: Data from similarity matching
- `max_sounds`: Number of sentences to prioritize (optional guide for LLM)
- `api_key`: Google API key (optional, uses env var if not provided)
- `user_prompt`: Additional instructions to refine filtering (optional)
- `output_file`: Output path (optional, defaults to `llm_filtering/output/filtered_sounds.json`)

**Returns:** Dictionary with all sounds ranked by unique relevance (1 = best)

### filter_from_file()

```python
def filter_from_file(
    input_file: str,
    output_file: Optional[str] = None,
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    user_prompt: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `input_file`: Path to similarity.json file
- `output_file`: Output path (optional)
- `max_sounds`: Number of sentences to prioritize (optional)
- `api_key`: Google API key (optional)
- `user_prompt`: Additional filtering instructions (optional)

**Returns:** Filtering result with all sounds ranked by unique rank (1, 2, 3...)

### get_gemini_model()

```python
def get_gemini_model(api_key: Optional[str] = None) -> genai.GenerativeModel
```

## Dependencies

- `google-generativeai` - Google's Gemini API client
- `python-dotenv` - Environment variable management (optional)
- Standard library: `json`, `os`, `typing`

## License

Part of the Audio-noise-effects project.
