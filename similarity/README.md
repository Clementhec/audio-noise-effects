# Module de Similarité

Ce module permet de calculer la similarité entre des embeddings de parole et des embeddings de sons.

## Structure

- `similarity.py` : Module principal contenant les fonctions réutilisables
- `similarity_test.py` : Script de test qui charge les données et exécute les fonctions
- `__init__.py` : Fichier d'initialisation du package

## Fonctions principales

### Dans `similarity.py`

#### `find_similar_sounds(df_speech, df_sounds, top_k=5)`

Trouve les sons les plus similaires pour chaque segment de parole.

**Paramètres :**
- `df_speech` : DataFrame contenant les embeddings de parole avec colonnes :
  - `embedding` : embedding vectoriel (numpy array ou liste)
  - `text` : texte du segment de parole
- `df_sounds` : DataFrame contenant les embeddings de sons avec colonnes :
  - `embedding` : embedding vectoriel (numpy array ou liste)
  - `title` : titre du son
  - `description` : description du son
  - `audio_url` : URL du fichier audio
- `top_k` : Nombre de sons les plus similaires à retourner (défaut: 5)

**Retourne :**
Une liste de dictionnaires contenant les résultats pour chaque segment.

### Dans `similarity_test.py`

#### `print_similarity_results(results, max_description_length=100)`

Affiche les résultats de similarité de manière formatée.

**Paramètres :**
- `results` : Résultats retournés par `find_similar_sounds()`
- `max_description_length` : Longueur maximale de la description à afficher

## Utilisation

### Exemple basique

```python
import pandas as pd
from similarity import find_similar_sounds

# Charger vos DataFrames
df_speech = pd.read_csv("votre_fichier_speech.csv")
df_sounds = pd.read_csv("votre_fichier_sounds.csv")

# Calculer les similarités
results = find_similar_sounds(df_speech, df_sounds, top_k=5)

# Utiliser les résultats comme vous le souhaitez
for result in results:
    print(f"Segment: {result['speech_text']}")
    print(f"Meilleur match: {result['top_matches'][0]['sound_title']}")
```

### Exécuter le test

Pour tester le module avec les données d'exemple :

```bash
# Activer l'environnement virtuel
source .venv/bin/activate

# Exécuter le script de test
python3 similarity/similarity_test.py
```

## Format des données

Les DataFrames doivent contenir une colonne `embedding` avec des embeddings vectoriels. Les embeddings peuvent être :
- Des listes Python : `[0.1, 0.2, 0.3, ...]`
- Des chaînes de caractères représentant des listes : `"[0.1, 0.2, 0.3, ...]"`
- Des arrays numpy : `np.array([0.1, 0.2, 0.3, ...])`

Le module se charge automatiquement de la conversion.

## Dépendances

- pandas
- numpy
- typing (pour les annotations de type)

