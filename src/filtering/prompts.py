"""
Prompt templates for LLM-based sound filtering.
"""

FILTER_PROMPT_HEADER = """
You are an expert in sound design for audio. 
Your role is to analyze sentences with their suggested corresponding sounds and determine:

1. A unique relevance rank for each sentence (1 = most relevant, 2 = second most relevant, etc.)
2. On which specific word(s) to place the sound effect for a natural result
3. Which sound among the suggestions (by index) is most appropriate

IMPORTANT RULES:
- You MUST assign a UNIQUE rank to each sentence (no duplicates: 1, 2, 3, 4, 5...)
- Rank 1 = most impactful/relevant, higher numbers = less relevant
- Favor concrete sounds (thunder, barking, rain) rather than general ambiances
- Maximum one keyword per sentence (the most relevant one)
- Use the sound_index (0, 1, or 2) to reference which sound from the suggestions you select

{max_sounds_instruction}

{user_context}

Here is the data to analyze :
<DATA>
"""

FILTER_PROMPT_FOOTER = """
</DATA>

For each sentence, provide:
- speech_index: the index of the sentence
- speech_text: the original text of the sentence
- should_add_sound: whether a sound should be added (true/false)
- target_word: the specific word where to place the sound (null if should_add_sound=false)
- selected_sound_index: the index (0, 1, or 2) of the selected sound from the suggestions
- reasoning: explanation of your decision
- relevance_rank: unique integer rank (1 = most relevant, 2 = second most relevant, etc.)

CRITICAL:
- Use ONLY the sound_index (0, 1, or 2) to reference sounds
- ALL sentences must be included with UNIQUE ranks (1, 2, 3, 4...)
- Order by rank ascending (1 first)
"""
