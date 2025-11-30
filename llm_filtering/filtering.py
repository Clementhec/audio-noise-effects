import json
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai


def get_gemini_model(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Configure et retourne un modèle Gemini.
    
    Args:
        api_key: Clé API Google. Si None, utilise la variable d'environnement GOOGLE_API_KEY
        
    Returns:
        Instance du modèle Gemini configuré
    """
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY doit être définie dans l'environnement ou passée en paramètre")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash-lite')


def create_prompt(similarity_data: List[Dict[str, Any]], max_sounds: Optional[int] = None, user_prompt: Optional[str] = None) -> str:
    """
    Crée le prompt pour le LLM.
    
    Args:
        similarity_data: Données de similarité entre phrases et sons
        max_sounds: Nombre maximum de phrases à sélectionner (optionnel)
        user_prompt: Instructions supplémentaires de l'utilisateur pour affiner le filtrage (optionnel)
        
    Returns:
        Prompt formaté pour le LLM
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
            prompt += f"  [Index {i-1}] {match['sound_title']} (similarity: {match['similarity']:.2f})\n"
            prompt += f"              Description: {match['sound_description']}\n"
    
    prompt += """

RESPOND ONLY with valid JSON in the following format (no markdown, no ```json):

CRITICAL: Use ONLY the sound_index (0, 1, or 2) to reference sounds. DO NOT copy titles or URLs.

{
  "filtered_sounds": [
    {
      "speech_index": 0,
      "relevance_rank": 1,
      "target_word": "specific word where to place the sound",
      "selected_sound_index": 0,
      "reasoning": "brief explanation of why this sound, this word, and this rank"
    }
  ]
}

ALL sentences must be included with UNIQUE ranks (1, 2, 3, 4...). Order by rank ascending (1 first).
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
    user_prompt: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filtre les sons en utilisant le LLM avec un système de classement par pertinence.
    
    Args:
        similarity_data: Données du fichier similarity.json
        max_sounds: Nombre de phrases à prioriser (optionnel, sert de guide au LLM)
        api_key: Clé API Google (optionnel)
        user_prompt: Instructions supplémentaires pour affiner le filtrage (optionnel)
        output_file: Chemin du fichier de sortie. Si None, utilise 'llm_filtering/output/filtered_sounds.json'
        
    Returns:
        Dictionnaire avec tous les sons classés par rang unique (1 = meilleur, 2 = 2ème meilleur, etc.)
    """
    model = get_gemini_model(api_key)
    prompt = create_prompt(similarity_data, max_sounds, user_prompt)
    response = model.generate_content(prompt)
    response_text = clean_json_response(response.text)
    llm_result = json.loads(response_text)
    
    # Reconstruction des données complètes à partir des index
    # pour éviter les hallucinations du LLM
    result = {"filtered_sounds": []}
    
    for item in llm_result['filtered_sounds']:
        speech_idx = item['speech_index']
        sound_idx = item['selected_sound_index']
        
        # Récupère les données originales
        original_data = similarity_data[speech_idx]
        selected_sound = original_data['top_matches'][sound_idx]
        
        # Construit l'entrée avec les données réelles (pas générées par le LLM)
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
    
    # Trie par rang de pertinence croissant (1 = meilleur)
    result['filtered_sounds'].sort(key=lambda x: x['relevance_rank'])
    
    # Sauvegarde automatique dans output/
    if output_file is None:
        output_file = 'llm_filtering/output/filtered_sounds.json'
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Résultat sauvegardé dans {output_file}")
    
    return result


def filter_from_file(
    input_file: str, 
    output_file: Optional[str] = None, 
    max_sounds: Optional[int] = None,
    api_key: Optional[str] = None,
    user_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filtre les sons à partir d'un fichier similarity.json.
    
    Args:
        input_file: Chemin vers le fichier similarity.json
        output_file: Chemin de sortie (optionnel)
        max_sounds: Nombre de phrases à prioriser (optionnel, sert de guide au LLM)
        api_key: Clé API Google (optionnel)
        user_prompt: Instructions supplémentaires pour affiner le filtrage (optionnel)
        
    Returns:
        Résultat du filtrage avec tous les sons classés par rang unique (1, 2, 3...)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        similarity_data = json.load(f)
    
    result = filter_sounds(similarity_data, max_sounds, api_key, user_prompt, output_file)
    
    return result



def print_summary(result: Dict[str, Any]) -> None:
    """
    Affiche un résumé des résultats du filtrage.
    
    Args:
        result: Résultat du filtrage
    """
    print("\n=== RÉSUMÉ DU FILTRAGE ===")
    print(f"Nombre total de sons classés: {len(result['filtered_sounds'])}")
    print("\nTop 5 des sons les plus pertinents:")
    for item in result['filtered_sounds'][:5]:
        print(f"\nRang #{item['relevance_rank']}")
        print(f"   Phrase: \"{item['speech_text']}\"")
        print(f"   Mot cible: \"{item['target_word']}\"")
        print(f"   Son: {item['selected_sound']['sound_title']}")
        print(f"   Raison: {item['reasoning']}")


def main():
    """Exemple d'utilisation."""
    input_file = "similarity/output/similarity.json"
    output_file = "llm_filtering/output/filtered_sounds.json"
    max_sounds = 3
    user_prompt = None  # Exemple: "Privilégier les sons naturels et éviter les ambiances"
    
    print(f"Analyse des sons avec le LLM (priorisation des {max_sounds} meilleures phrases)..." if max_sounds else "Analyse des sons avec le LLM...")
    if user_prompt:
        print(f"Instructions utilisateur: {user_prompt}")
    
    result = filter_from_file(
        input_file, 
        output_file, 
        max_sounds=max_sounds,
        user_prompt=user_prompt
    )
    print_summary(result)


if __name__ == "__main__":
    main()

