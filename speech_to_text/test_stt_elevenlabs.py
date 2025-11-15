import json
from stt_elevenlabs import transcribe_audio_elevenlabs

# URL audio à transcrire
audio_url = "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"

print("🎙️  Transcription en cours...\n")

# Appeler la fonction
result = transcribe_audio_elevenlabs(
    audio_source=audio_url,
    model_id="scribe_v1",
    language_code="en",
    tag_audio_events=False,
    diarize=False
)

# Afficher les résultats
print("--- Résultat par segment ---\n")
print(json.dumps(result["segment_result"], indent=2, ensure_ascii=False))

print("\n--- Timings par mot ---\n")
print(json.dumps(result["word_timings"], indent=2, ensure_ascii=False))

