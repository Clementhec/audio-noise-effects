"""
Script de test pour le système de filtrage LLM.
"""

import json
import os
from filtering import filter_sounds


def test_with_sample_data():
    """Test avec quelques exemples du fichier similarity.json."""
    
    # Charge les données d'exemple
    with open('similarity/output/similarity.json', 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    print("Test du système de filtrage LLM...\n")
    
    # Filtre les sons (garde uniquement ceux avec should_add_sound=true)
    result = filter_sounds(sample_data, max_sounds=2, keep_only_with_sound=True)
    
    # Affiche les résultats
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Sauvegarde dans le répertoire llm_filtering/output
    output_file = 'llm_filtering/output/filtered_sounds.json'
    os.makedirs('llm_filtering/output', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    test_with_sample_data()

