# Guide d'utilisation du flag --mode

## Vue d'ensemble

Le pipeline de traitement audio supporte maintenant deux modes d'exécution :
- **`--mode dev`** (par défaut) : Mode développement avec fichiers intermédiaires
- **`--mode prod`** : Mode production avec résultats JSON uniquement

## Mode DEV (Développement)

```bash
python main.py video.mp4 --full-pipeline --mode dev
```

**Comportement :**
- ✅ Crée tous les fichiers intermédiaires JSON/CSV
- ✅ Télécharge les sons depuis SoundBible si nécessaire
- ✅ Génère les embeddings de sons
- ✅ Fusionne les sons avec la vidéo
- ✅ Produit une vidéo finale avec effets sonores

**Fichiers créés :**
```
data/
├── audio/
│   └── video.wav
├── speech_to_text/
│   ├── video_full_transcription.json
│   └── video_word_timing.json
├── embeddings/
│   └── video_video_speech_embeddings.csv
├── similarity/
│   └── video_similarity.json
├── filtered/
│   └── video_video_filtered_sounds.json
├── sounds/
│   └── soundbible/
│       └── [fichiers audio téléchargés]
└── output/
    └── video_soundeasy.mp4
```

## Mode PROD (Production)

```bash
python main.py video.mp4 --full-pipeline --mode prod
```

**Comportement :**
- ❌ Ne crée PAS de fichiers intermédiaires JSON/CSV
- ❌ Ne télécharge PAS les sons depuis SoundBible
- ❌ Ne fusionne PAS les sons avec la vidéo
- ✅ Retourne uniquement un résultat JSON sur stdout

**Prérequis :**
En mode prod, les embeddings de sons doivent déjà exister. Exécutez une fois en mode dev pour les générer :
```bash
# Première fois : générer les embeddings de sons
python main.py dummy_video.mp4 --run-embeddings --mode dev

# Ensuite : utiliser le mode prod
python main.py video.mp4 --full-pipeline --mode prod
```

**Résultat JSON :**
```json
{
  "filtered": {
    "filtered_sounds": [
      {
        "speech_index": 0,
        "speech_text": "Hello world",
        "relevance_rank": 1,
        "target_word": "world",
        "should_add_sound": true,
        "selected_sound": {
          "sound_title": "World Ambience",
          "sound_description": "Atmospheric world sound",
          "audio_url_wav": "/path/to/sound.wav",
          "similarity_score": 0.85
        },
        "reasoning": "This sound matches perfectly..."
      }
    ]
  }
}
```

## Exemples d'utilisation

### Pipeline complet en DEV
```bash
python main.py video.mp4 --full-pipeline --mode dev
```

### Pipeline complet en PROD
```bash
python main.py video.mp4 --full-pipeline --mode prod > results.json
```

### Étapes individuelles en PROD
```bash
# Transcription uniquement
python main.py video.mp4 --run-stt --mode prod

# Transcription + Embeddings
python main.py video.mp4 --run-stt --run-embeddings --mode prod

# Pipeline jusqu'au filtrage LLM (sans merge vidéo)
python main.py video.mp4 --run-stt --run-embeddings --run-matching --run-llm-filter --mode prod
```

## Différences entre les modes

| Fonctionnalité | Mode DEV | Mode PROD |
|----------------|----------|-----------|
| Fichiers JSON intermédiaires | ✅ Créés | ❌ Non créés |
| Fichiers CSV embeddings | ✅ Créés | ❌ Non créés |
| Téléchargement SoundBible | ✅ Automatique | ❌ Ignoré |
| Fusion vidéo | ✅ Exécutée | ❌ Ignorée |
| Résultat final | Vidéo MP4 | JSON stdout |
| Fichier audio extrait | ✅ Créé | ✅ Créé* |

*L'extraction audio est toujours effectuée car nécessaire pour le traitement, mais c'est temporaire.

## Notes importantes

1. **Performance** : Le mode prod est plus rapide car il évite les écritures disque
2. **Debug** : Utilisez le mode dev pour débugger le pipeline
3. **API** : Utilisez le mode prod pour intégrer dans une API REST
4. **Embeddings** : Les embeddings de sons doivent exister avant d'utiliser le mode prod

## Intégration API

Exemple d'utilisation dans une API FastAPI :

```python
import subprocess
import json

@app.post("/process-video")
async def process_video(video_file: UploadFile):
    # Sauvegarder temporairement
    temp_path = f"/tmp/{video_file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await video_file.read())
    
    # Exécuter en mode prod
    result = subprocess.run(
        ["python", "main.py", temp_path, "--full-pipeline", "--mode", "prod"],
        capture_output=True,
        text=True
    )
    
    # Parser le JSON de sortie
    output_lines = result.stdout.split("\n")
    json_start = output_lines.index("=" * 70) + 2
    json_output = "\n".join(output_lines[json_start:])
    
    return json.loads(json_output)
```

