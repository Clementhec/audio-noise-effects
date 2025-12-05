"""
Prompt templates for LLM-based sound filtering.
"""

FILTER_PROMPT_HEADER = """You are an expert in sound design for audio. Your role is to analyze sentences with their suggested corresponding sounds and determine:

1. A unique relevance rank for each sentence (1 = most relevant, 2 = second most relevant, etc.)
2. On which specific word(s) to place the sound effect for a natural result
3. Which sound among the suggestions (by index) is most appropriate

IMPORTANT RULES:
- You MUST assign a UNIQUE rank to each sentence (no duplicates: 1, 2, 3, 4, 5...)
- Rank 1 = most impactful/relevant, higher numbers = less relevant
- Favor concrete sounds (thunder, barking, rain) rather than general ambiances
- Maximum one keyword per sentence (the most relevant one)
- Use the sound_index (0, 1, or 2) to reference which sound from the suggestions you select{max_sounds_instruction}{user_context}

Here is the data to analyze:

"""

FILTER_PROMPT_FOOTER = """

RESPOND ONLY with valid JSON in the following format (no markdown, no ```json):
Your overall response should be a valid JSON string AS IS.

CRITICAL: Use ONLY the sound_index (0, 1, or 2) to reference sounds. DO NOT copy titles or URLs.

{{
  "filtered_sounds": [
    {{
      "speech_index": 0,
      "speech_text": "original text",
      "should_add_sound": true/false,
      "target_word": "specific word where to place the sound (null if should_add_sound=false)",
      "selected_sound_index" : "EXACT sound index identifier",
      "reasoning": "explanation of the decision",
      "relevance_rank": "integer numeric relevance, (most is 1)"
    }}
  ]
}}

ALL sentences must be included with UNIQUE ranks (1, 2, 3, 4...). Order by rank ascending (1 first).
"""
