# -*- coding: utf-8 -*-
"""
Script de test pour le module similarity.
Charge les données et exécute la fonction de similarité.
"""
import os
import sys
from pathlib import Path
import pandas as pd
from ast import literal_eval

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from similarity import find_similar_sounds
from typing import List, Dict, Any


def load_embeddings_csv(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV contenant des embeddings et convertit les embeddings en arrays numpy.
    
    Args:
        filepath: Chemin vers le fichier CSV
    
    Returns:
        DataFrame avec les embeddings convertis
    """
    df = pd.read_csv(filepath)
    
    # Convertir les embeddings de string vers array numpy
    if 'embedding' in df.columns:
        df['embedding'] = df.embedding.apply(literal_eval)
    
    return df


def print_similarity_results(results: List[Dict[str, Any]], 
                            max_description_length: int = 100):
    """
    Affiche les résultats de similarité de manière formatée.
    
    Args:
        results: Résultats retournés par find_similar_sounds()
        max_description_length: Longueur maximale de la description à afficher
    """
    print(f"Nombre de segments analysés: {len(results)}")
    print()
    
    for result in results:
        print(f"Segment {result['speech_index']}: '{result['speech_text']}'")
        print(f"Top {len(result['top_matches'])} sons similaires:")
        
        for i, match in enumerate(result['top_matches'], 1):
            print(f"  {i}. {match['sound_title']} (similarité: {match['similarity']:.4f})")
            description = match['sound_description']
            if len(description) > max_description_length:
                description = description[:max_description_length] + '...'
            print(f"     Description: {description}")
        
        print()
        print("-" * 80)
        print()


def main():
    """
    Fonction principale pour tester la similarité entre embeddings de parole et de sons.
    """
    # Définir les chemins des fichiers
    project_root = Path(__file__).parent.parent
    test_speech_path = project_root / "data" / "test_speech_embeddings.csv"
    sound_embeddings_path = project_root / "data" / "soundbible_embeddings.csv"
    
    # Vérifier que les fichiers existent
    if not test_speech_path.exists():
        print(f"Erreur: Le fichier {test_speech_path} n'existe pas.")
        return
    
    if not sound_embeddings_path.exists():
        print(f"Erreur: Le fichier {sound_embeddings_path} n'existe pas.")
        return
    
    # Charger les données
    print("Chargement des données...")
    df_speech = load_embeddings_csv(str(test_speech_path))
    df_sounds = load_embeddings_csv(str(sound_embeddings_path))
    
    print(f"Nombre de segments de parole: {len(df_speech)}")
    print(f"Nombre de sons dans soundbible: {len(df_sounds)}")
    print()
    
    # Exécuter la fonction de similarité
    print("Calcul des similarités...")
    results = find_similar_sounds(df_speech, df_sounds, top_k=5)
    
    # Afficher les résultats
    print()
    print("=" * 80)
    print("RÉSULTATS")
    print("=" * 80)
    print()
    print_similarity_results(results)



if __name__ == "__main__":
    main()

