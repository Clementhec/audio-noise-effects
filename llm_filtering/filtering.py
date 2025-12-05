import json
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv


def get_gemini_model(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Configure and return a Gemini model.
    
    Args:
        api_key: Google API key. If None, uses the GOOGLE_API_KEY environment variable
        
    Returns:
        Configured Gemini model instance
    """
    load_dotenv()
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be defined in the environment or passed as a parameter")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash-lite')


def create_prompt(similarity_data: List[Dict[str, Any]], max_sounds: Optional[int] = None, user_prompt: Optional[str] = None) -> str:
    """
    Create the prompt for the LLM.
    
    Args:
        similarity_data: Similarity data between sentences and sounds
        max_sounds: Maximum number of sentences to select (optional)
        user_prompt: Additional user instructions to refine filtering (optional)
        
    Returns:
        Formatted prompt for the LLM
    """
    max_sounds_instruction = ""
    if max_sounds is not None:
        max_sounds_instruction = f"\n- You should prioritize the top {max_sounds} most impactful sentences"
    
    user_context = ""
    if user_prompt:
        user_context = f"\n\nUSER SPECIFIC INSTRUCTIONS:\n{user_prompt}\n"
    
    prompt = f"""You are an expert in sound design for audio. Your role is to analyze sentences with their suggested corresponding sounds and determine:

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
    
    for item in similarity_data:
        prompt += f"\n--- Sentence {item['speech_index']} ---\n"
        prompt += f"Text: \"{item['speech_text']}\"\n"
        prompt += f"Suggested sounds:\n"
        for i, match in enumerate(item['top_matches'][:3], 1):  # Top 3 only
            prompt += f"  {i}. {match['sound_title']} (similarity: {match['similarity']:.2f})\n"
            prompt += f"     Description: {match['sound_description']}\n"
            prompt += f"     URL: {match['audio_url_wav']}\n"
    prompt += """

RESPOND ONLY with valid JSON in the following format (no markdown, no ```json):
Your overall response should be a valid JSON string AS IS.

CRITICAL: Use ONLY the sound_index (0, 1, or 2) to reference sounds. DO NOT copy titles or URLs.

{
  "filtered_sounds": [
    {
      "speech_index": 0,
      "speech_text": "original text",
      "should_add_sound": true/false,
      "target_word": "specific word where to place the sound (null if should_add_sound=false)",
      "selected_sound": {
        "sound_title": "title of chosen sound",
        "audio_url_wav": "EXACT URL from the data above - DO NOT MODIFY",
        "reason": "brief explanation of why this sound and this word"
      },
      "reasoning": "explanation of the decision"
    }
  ]
}

ALL sentences must be included with UNIQUE ranks (1, 2, 3, 4...). Order by rank ascending (1 first).
"""
    return prompt


def clean_json_response(response_text: str) -> str:
    """
    Clean the LLM response to extract pure JSON.
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Cleaned JSON
    """
    response_text = response_text.strip()
    
    # Clean markdown if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1])
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()
    
    return response_text


def filter_sounds(
    similarity_data: List[Dict[str, Any]], 
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    user_prompt: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter sounds using the LLM with a relevance ranking system.
    
    Args:
        similarity_data: Data from the similarity.json file
        max_sounds: Number of sentences to prioritize (optional, serves as a guide for the LLM)
        api_key: Google API key (optional)
        user_prompt: Additional instructions to refine filtering (optional)
        output_file: Output file path. If None, uses 'llm_filtering/output/filtered_sounds.json'
        
    Returns:
        Dictionary with all sounds ranked by unique rank (1 = best, 2 = 2nd best, etc.)
    """
    model = get_gemini_model(api_key)
    prompt = create_prompt(similarity_data, max_sounds, user_prompt)
    response = model.generate_content(prompt)
    response_text = clean_json_response(response.text)
    print(response)
    print()
    print(response_text)
    try:
        llm_result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print("Error decoding LLM filter response")
        raise e
    # Reconstruct complete data from indexes
    # to avoid LLM hallucinations
    result = {"filtered_sounds": []}
    
    for item in llm_result['filtered_sounds']:
        speech_idx = item['speech_index']
        sound_idx = item['selected_sound_index']
        
        # Retrieve original data
        original_data = similarity_data[speech_idx]
        selected_sound = original_data['top_matches'][sound_idx]
        
        # Build entry with real data (not generated by the LLM)
        result['filtered_sounds'].append({
            'speech_index': speech_idx,
            'speech_text': original_data['speech_text'],
            'relevance_rank': item['relevance_rank'],
            'target_word': item['target_word'],
            'selected_sound': {
                'sound_title': selected_sound['sound_title'],
                'sound_description': selected_sound['sound_description'],
                'audio_url_wav': selected_sound['audio_url_wav'],
                'similarity_score': selected_sound['similarity']
            },
            'reasoning': item['reasoning']
        })
    
    # Sort by ascending relevance rank (1 = best)
    result['filtered_sounds'].sort(key=lambda x: x['relevance_rank'])
    
    # Automatic save to output/
    if output_file is None:
        output_file = 'llm_filtering/output/filtered_sounds.json'
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Result saved to {output_file}")
    
    return result


def filter_from_file(
    input_file: str, 
    output_file: Optional[str] = None, 
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    user_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter sounds from a similarity.json file.
    
    Args:
        input_file: Path to the similarity.json file
        output_file: Output path (optional)
        max_sounds: Number of sentences to prioritize (optional, serves as a guide for the LLM)
        api_key: Google API key (optional)
        user_prompt: Additional instructions to refine filtering (optional)
        
    Returns:
        Filtering result with all sounds ranked by unique rank (1, 2, 3...)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        similarity_data = json.load(f)
    
    result = filter_sounds(similarity_data, max_sounds, api_key, user_prompt, output_file)
    
    return result



def print_summary(result: Dict[str, Any]) -> None:
    """
    Display a summary of the filtering results.
    
    Args:
        result: Filtering result
    """
    print("\n=== FILTERING SUMMARY ===")
    print(f"Total number of ranked sounds: {len(result['filtered_sounds'])}")
    print("\nTop 5 most relevant sounds:")
    for item in result['filtered_sounds'][:5]:
        print(f"\nRank #{item['relevance_rank']}")
        print(f"   Sentence: \"{item['speech_text']}\"")
        print(f"   Target word: \"{item['target_word']}\"")
        print(f"   Sound: {item['selected_sound']['sound_title']}")
        print(f"   Reasoning: {item['reasoning']}")


def main():
    """Usage example."""
    input_file = "similarity/output/similarity.json"
    output_file = "llm_filtering/output/filtered_sounds.json"
    max_sounds = 3
    user_prompt = None  # Example: "Favor natural sounds and avoid ambiances"
    
    print(f"Analyzing sounds with LLM (prioritizing the {max_sounds} best sentences)..." if max_sounds else "Analyzing sounds with LLM...")
    if user_prompt:
        print(f"User instructions: {user_prompt}")
    
    result = filter_from_file(
        input_file, 
        output_file, 
        max_sounds=max_sounds,
        user_prompt=user_prompt
    )
    print_summary(result)


if __name__ == "__main__":
    main()

