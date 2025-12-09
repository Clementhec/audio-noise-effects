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


1. Get a key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
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
