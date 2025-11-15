# -*- coding: utf-8 -*-
"""
Module pour calculer la similarité entre des embeddings de parole et des embeddings de sons.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity score (0 to 1 for normalized vectors)
    """
    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_similar_sounds(df_speech: pd.DataFrame, 
                       df_sounds: pd.DataFrame,
                       top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Trouve les sons les plus similaires pour chaque segment de parole.
    
    Args:
        df_speech: DataFrame contenant les embeddings de parole avec colonnes:
                   - 'embedding': embedding vectoriel (numpy array ou liste)
                   - 'text': texte du segment de parole
        df_sounds: DataFrame contenant les embeddings de sons avec colonnes:
                   - 'embedding': embedding vectoriel (numpy array ou liste)
                   - 'title': titre du son
                   - 'description': description du son
                   - 'audio_url': URL du fichier audio
        top_k: Nombre de sons les plus similaires à retourner pour chaque segment
    
    Returns:
        Liste de dictionnaires contenant les résultats pour chaque segment de parole.
        Chaque dictionnaire contient:
        - 'speech_index': index du segment de parole
        - 'speech_text': texte du segment
        - 'top_matches': liste des top_k sons les plus similaires avec leurs scores
    """
    # S'assurer que les embeddings sont des arrays numpy
    df_speech = df_speech.copy()
    df_sounds = df_sounds.copy()
    
    if not isinstance(df_speech['embedding'].iloc[0], np.ndarray):
        df_speech['embedding'] = df_speech['embedding'].apply(
            lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else x
        )
    
    if not isinstance(df_sounds['embedding'].iloc[0], np.ndarray):
        df_sounds['embedding'] = df_sounds['embedding'].apply(
            lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else x
        )
    
    results = []
    
    # Pour chaque segment de parole, trouver les sons les plus similaires
    for idx, speech_row in df_speech.iterrows():
        speech_embedding = speech_row['embedding']
        speech_text = speech_row.get('text', f'Segment {idx}')
        
        # Calculer la similarité avec tous les sons
        similarities = []
        for sound_idx, sound_row in df_sounds.iterrows():
            sound_embedding = sound_row['embedding']
            similarity = cosine_similarity(speech_embedding, sound_embedding)
            similarities.append({
                'sound_title': sound_row.get('title', 'N/A'),
                'sound_description': sound_row.get('description', 'N/A'),
                'similarity': float(similarity),
                'audio_url': sound_row.get('audio_url', 'N/A')
            })
        
        # Trier par similarité décroissante
        similarities_sorted = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        
        # Ajouter les résultats
        results.append({
            'speech_index': idx,
            'speech_text': speech_text,
            'top_matches': similarities_sorted[:top_k]
        })
    
    return results


