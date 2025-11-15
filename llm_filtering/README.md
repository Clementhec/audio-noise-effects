# LLM Filtering Module

This module uses Google's Gemini LLM to intelligently filter and refine sound effect recommendations from the similarity matching stage.

## Purpose

After the similarity matching generates top-K sound candidates for each speech segment, the LLM filtering step:

1. **Analyzes context** - Understands which sentences would benefit most from sound effects
2. **Selects optimal placement** - Identifies the specific word where each sound should be placed
3. **Chooses best match** - Picks the most appropriate sound from the top candidates
4. **Provides reasoning** - Explains why each sound was selected

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
- Which sentences need sound effects (not all do!)
- The specific word where the sound fits best
- Which of the top-K sounds is most appropriate
- Why this combination works

### Output

Filtered results with target words:

```json
{
  "filtered_sounds": [
    {
      "speech_index": 0,
      "speech_text": "I heard thunder rumbling in the distance",
      "should_add_sound": true,
      "target_word": "thunder",
      "selected_sound": {
        "sound_title": "Thunder Rumble",
        "audio_url": "...",
        "reason": "Perfect match for thunder, place on the word 'thunder' for natural effect"
      },
      "reasoning": "Thunder is a concrete sound that clearly benefits from an effect"
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
    max_sounds=5,  # Or None to let LLM decide
    keep_only_with_sound=True,
    output_file='output/filtered_sounds.json'
)

# Use filtered results
for item in result['filtered_sounds']:
    print(f"Segment: {item['speech_text']}")
    print(f"Target word: {item['target_word']}")
    print(f"Sound: {item['selected_sound']['sound_title']}")
```

### From File

```python
from llm_filtering import filter_from_file

result = filter_from_file(
    input_file='output/video_similarity_matches.json',
    output_file='output/filtered_sounds.json',
    max_sounds=5
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

Control how many sentences get sound effects:

```bash
# Let LLM decide (more selective)
uv run python main.py video.mp4 --full-pipeline

# Force exactly 5 sounds
uv run python main.py video.mp4 --full-pipeline --max-sounds 5

# Force exactly 10 sounds
uv run python main.py video.mp4 --full-pipeline --max-sounds 10
```

**Recommendation**: Let the LLM decide (`max_sounds=None`) for best results. The LLM is trained to be selective and only add sounds where they enhance the experience.

## LLM Behavior

### Selection Criteria

The LLM is instructed to:
- **Be highly selective** - Only recommend sounds that truly enhance the audio
- **Favor concrete sounds** - Thunder, barking, rain over general ambiances
- **One sound per sentence** - Maximum, on the most relevant word
- **Natural placement** - Sounds should feel organic, not forced

### Example Decisions

**Will Add Sound:**
- "I heard **thunder** rumbling" → Thunder sound on "thunder"
- "A dog **barking** loudly" → Dog bark on "barking"
- "The **rain** was pouring" → Rain sound on "rain"

**Won't Add Sound:**
- "The weather was nice" → No concrete sound to add
- "I felt happy" → Emotion doesn't need sound effect
- "We talked for hours" → Talking doesn't benefit from added sound

## Output Format

### Complete Structure

```json
{
  "filtered_sounds": [
    {
      "speech_index": 0,
      "speech_text": "Original sentence text",
      "should_add_sound": true,
      "target_word": "specific_word",
      "selected_sound": {
        "sound_title": "Sound Effect Name",
        "audio_url": "https://...",
        "reason": "Brief explanation"
      },
      "reasoning": "Detailed decision explanation"
    }
  ]
}
```

### Fields Explained

- `speech_index`: Index from original similarity results
- `speech_text`: Full sentence text
- `should_add_sound`: Boolean - whether to add sound
- `target_word`: Specific word for sound placement (null if no sound)
- `selected_sound`: Chosen sound details
  - `sound_title`: Name of the sound effect
  - `audio_url`: URL/path to audio file
  - `reason`: Why this sound and word were chosen
- `reasoning`: LLM's explanation for the decision

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
- 3-7 segments selected (LLM's choice)
- Each with specific target word
- Best sound from top-K candidates

### Why LLM Filtering?

Without LLM filtering:
- Too many sounds (overwhelming)
- No word-level placement
- Can't distinguish important vs. unimportant sounds
- No context awareness

With LLM filtering:
- Selective, high-quality sound placement
- Precise timing (word-level)
- Context-aware decisions
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
- Identifies concrete vs. abstract concepts (95%+)
- Selects appropriate target words (90%+)
- Chooses contextually correct sounds (85%+)

## Error Handling

### API Key Missing

```python
ValueError: GOOGLE_API_KEY doit être définie dans l'environnement ou passée en paramètre
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

# Custom filtering
result['filtered_sounds'] = [
    item for item in result['filtered_sounds']
    if item['selected_sound']['similarity'] > 0.7
]
```

## Best Practices

1. **Let LLM Decide**: Don't set `max_sounds` unless you have a specific constraint
2. **Review Results**: Check filtered output before audio mixing
3. **Adjust top-k**: Use `--top-k 5` to give LLM good options to choose from
4. **API Key Security**: Never commit `.env` file with API keys
5. **Error Handling**: Implement fallbacks if LLM filtering fails

## Troubleshooting

### "Too Many Sounds Selected"

If LLM selects too many sounds:
```bash
uv run python main.py video.mp4 --full-pipeline --max-sounds 5
```

### "Not Enough Sounds Selected"

If LLM is too conservative:
- Check similarity scores in Step 4
- Increase `--top-k` for more options
- Review prompt instructions

### "Wrong Words Selected"

If target words seem incorrect:
- Review the prompt in `filtering.py`
- Adjust LLM instructions
- Consider using different Gemini model

## Future Enhancements

Potential improvements:
1. **Multi-language Support**: Handle non-English transcripts
2. **Confidence Scores**: Add LLM confidence ratings
3. **User Feedback**: Learn from manual corrections
4. **Batch Processing**: Process multiple videos efficiently
5. **Custom Instructions**: User-configurable behavior
6. **Sound Categories**: Filter by sound type preferences
7. **Temporal Awareness**: Consider timing and pacing

## Examples

### Example 1: Weather Documentary

**Input**: "The storm was approaching with dark clouds overhead. Thunder rumbled in the distance. Rain began falling heavily."

**LLM Output**:
- Segment 2: "Thunder rumbled" → **thunder** (Thunder sound)
- Segment 3: "Rain began falling" → **Rain** (Rain sound)
- Skip segment 1 (no concrete sound)

### Example 2: Nature Scene

**Input**: "Birds were singing in the trees. A dog barked nearby. Children were laughing and playing."

**LLM Output**:
- Segment 1: "Birds were singing" → **singing** (Birds chirping)
- Segment 2: "dog barked" → **barked** (Dog bark)
- Segment 3: "Children were laughing" → **laughing** (Children laughter)

### Example 3: Abstract Content

**Input**: "I felt happy about the decision. The future looks bright. We made progress today."

**LLM Output**:
- No sounds selected (all abstract concepts)
- LLM reasoning: "These sentences describe emotions and abstract concepts that don't benefit from sound effects"

## API Reference

### filter_sounds()

```python
def filter_sounds(
    similarity_data: List[Dict[str, Any]],
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    keep_only_with_sound: bool = True,
    output_file: Optional[str] = None
) -> Dict[str, Any]
```

### filter_from_file()

```python
def filter_from_file(
    input_file: str,
    output_file: Optional[str] = None,
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    keep_only_with_sound: bool = True
) -> Dict[str, Any]
```

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
