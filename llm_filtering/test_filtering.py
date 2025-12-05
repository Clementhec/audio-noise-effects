"""
Script de test pour le système de filtrage LLM.
"""

import json
from filtering import filter_sounds

# Charge les données d'exemple
with open("similarity/output/similarity.json", "r", encoding="utf-8") as f:
    sample_data = json.load(f)

print("Test du système de filtrage LLM...\n")

# Filtre les sons (garde uniquement ceux avec should_add_sound=true)
# Le fichier JSON est automatiquement généré dans llm_filtering/output/filtered_sounds.json
result = filter_sounds(sample_data, max_sounds=2, keep_only_with_sound=True)

# Affiche les résultats
print(json.dumps(result, indent=2, ensure_ascii=False))
