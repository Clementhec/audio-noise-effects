"""
Exemple d'utilisation simple de la fonction transcribe_audio_elevenlabs.
Ce fichier montre comment utiliser la fonction dans un cas d'usage réel.
"""

import json
from stt_elevenlabs import transcribe_audio_elevenlabs


def main():
    """
    Exemple d'utilisation de base de la fonction de transcription.
    """
    print("🎙️  Exemple d'utilisation de transcribe_audio_elevenlabs\n")
    print("=" * 80)
    
    # Configuration de base (identique au fichier original)
    audio_url = "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"
    
    print(f"\n📥 Téléchargement et transcription de l'audio depuis :")
    print(f"   {audio_url}\n")
    
    # Appeler la fonction avec les paramètres du fichier original
    result = transcribe_audio_elevenlabs(
        audio_source=audio_url,
        api_key=None,              # Utilise ELEVENLABS_API_KEY depuis .env
        model_id="scribe_v1",      # Modèle à utiliser
        tag_audio_events=False,    # Pas de détection d'événements audio
        language=None,             # Détection automatique de la langue
        diarize=False,             # Pas de diarisation
        output_format="both"       # Retourner segments et mots
    )
    
    # Afficher les résultats
    print("--- Résultat par segment (ElevenLabs) ---\n")
    print(json.dumps(result["segment_result"], indent=2, ensure_ascii=False))
    
    print("\n--- Timings par mot ---\n")
    print(json.dumps(result["word_timings"], indent=2, ensure_ascii=False))
    
    # Sauvegarder les résultats (optionnel)
    print("\n💾 Sauvegarde des résultats...")
    
    with open("full_transcription.json", "w", encoding="utf-8") as f:
        json.dump(result["segment_result"], f, indent=2, ensure_ascii=False)
    print("   ✅ full_transcription.json créé")
    
    with open("word_timing.json", "w", encoding="utf-8") as f:
        json.dump(result["word_timings"], f, indent=2, ensure_ascii=False)
    print("   ✅ word_timing.json créé")
    
    print("\n" + "=" * 80)
    print("✅ Transcription terminée avec succès!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


