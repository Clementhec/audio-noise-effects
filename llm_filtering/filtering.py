import json
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv


def get_gemini_model(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Configure et retourne un modèle Gemini.
    
    Args:
        api_key: Clé API Google. Si None, utilise la variable d'environnement GOOGLE_API_KEY
        
    Returns:
        Instance du modèle Gemini configuré
    """
    load_dotenv()
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY should be defined in env or passed as parameter")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash-lite')


def create_prompt(similarity_data: List[Dict[str, Any]], max_sounds: Optional[int] = None) -> str:
    """
    Crée le prompt pour le LLM.
    
    Args:
        similarity_data: Données de similarité entre phrases et sons
        max_sounds: Nombre maximum de phrases à sélectionner (optionnel)
        
    Returns:
        Prompt formaté pour le LLM
    """
    max_sounds_instruction = ""
    if max_sounds is not None:
        max_sounds_instruction = f"\n- CRITICAL: You MUST select EXACTLY {max_sounds} sentences (no more, no less) that would benefit most from sound effects"
    
    prompt = f"""You are an expert in sound design for audio. Your role is to analyze sentences with their suggested corresponding sounds and determine:

1. Which sentences would benefit MOST from a sound effect (prioritize the most impactful ones)
2. On which specific word(s) to place the sound effect for a natural result
3. Which sound among the suggestions is most appropriate

IMPORTANT RULES:
- Be highly selective: only recommend sounds that truly enhance the audio experience in a natural way
- Favor concrete sounds (thunder, barking, rain) rather than general ambiances
- Maximum one keyword per sentence (the most relevant one)
- If a sentence should not have a sound, set "should_add_sound": false{max_sounds_instruction}

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

CRITICAL: You MUST use the EXACT audio_url_wav provided in the data above. DO NOT create or modify URLs.

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
"""
    return prompt


def clean_json_response(response_text: str) -> str:
    """
    Nettoie la réponse du LLM pour extraire le JSON pur.
    
    Args:
        response_text: Réponse brute du LLM
        
    Returns:
        JSON nettoyé
    """
    response_text = response_text.strip()
    
    # Nettoie le markdown si présent
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
    keep_only_with_sound: bool = True,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filtre les sons en utilisant le LLM.
    
    Args:
        similarity_data: Données du fichier similarity.json
        max_sounds: Nombre maximum de phrases à sélectionner pour des effets sonores.
                   Si None, le LLM décide librement. Si spécifié, le LLM sélectionnera
                   exactement ce nombre de phrases parmi les plus pertinentes.
        api_key: Clé API Google (optionnel)
        keep_only_with_sound: Si True, ne garde que les résultats avec should_add_sound=true (défaut: True)
        output_file: Chemin du fichier de sortie. Si None, utilise 'llm_filtering/output/filtered_sounds.json'
        
    Returns:
        Dictionnaire avec les sons filtrés et les mots cibles
    """
    model = get_gemini_model(api_key)
    prompt = create_prompt(similarity_data, max_sounds)
    response = model.generate_content(prompt)
    response_text = clean_json_response(response.text)
    result = json.loads(response_text)
    
    # Filtre pour ne garder que les sons à ajouter
    if keep_only_with_sound:
        result['filtered_sounds'] = [
            item for item in result['filtered_sounds'] 
            if item.get('should_add_sound', False)
        ]
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved in {output_file}")
    
    return result


def filter_from_file(
    input_file: str, 
    output_file: Optional[str] = None, 
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    keep_only_with_sound: bool = True
) -> Dict[str, Any]:
    """
    Filtre les sons à partir d'un fichier similarity.json.
    
    Args:
        input_file: Chemin vers le fichier similarity.json
        output_file: Chemin de sortie (optionnel)
        max_sounds: Nombre maximum de phrases à sélectionner pour des effets sonores (optionnel)
        api_key: Clé API Google (optionnel)
        keep_only_with_sound: Si True, ne garde que les résultats avec should_add_sound=true (défaut: True)
        
    Returns:
        Résultat du filtrage
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        similarity_data = json.load(f)
    
    result = filter_sounds(similarity_data, max_sounds, api_key, keep_only_with_sound)
    
    if output_file:
        # Crée le répertoire de sortie si nécessaire
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Résultat sauvegardé dans {output_file}")
    
    return result



def main():
    """Exemple d'utilisation."""
    input_file = "similarity/output/similarity.json"
    output_file = "llm_filtering/output/filtered_sounds.json"
    max_sounds = 3
    
    print(f"Analyse des sons avec le LLM (max {max_sounds} phrases)..." if max_sounds else "Analyse des sons avec le LLM...")
    result = filter_from_file(input_file, output_file, max_sounds=max_sounds)
    print_summary(result)


if __name__ == "__main__":
    main()

