import os
from google.cloud import speech

import wave

def transcribe_audio(audio_file_path, language_code="fr-FR"):
    """
    Transcrit un fichier audio en texte avec Google Cloud Speech-to-Text.
    Args:
        audio_file_path (str): Chemin vers le fichier audio (WAV/FLAC/LINEAR16).
        language_code (str): Code langue (ex: 'fr-FR' pour le français).
    Returns:
        str: Texte transcrit.
    """
    client = speech.SpeechClient()

    # Détecter le sample rate si WAV
    sample_rate = 16000
    if audio_file_path.lower().endswith(".wav"):
        try:
            with wave.open(audio_file_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de lire le sample rate du wav, utilisation de 16000 Hz par défaut.\nDétail : {e}")

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        enable_word_confidence=True,
    )

    response = client.recognize(config=config, audio=audio)

    results = []
    words_timings = []
    for result in response.results:
        alternative = result.alternatives[0]
        words = alternative.words
        if words:
            start_time = words[0].start_time.total_seconds()
            end_time = words[-1].end_time.total_seconds()
        else:
            start_time = None
            end_time = None
        results.append({
            "alternatives": [{
                "transcript": alternative.transcript,
                "confidence": alternative.confidence
            }],
            "startTime": f"{start_time}s" if start_time is not None else None,
            "endTime": f"{end_time}s" if end_time is not None else None
        })
        # Ajoute les timings par mot
        for w in words:
            words_timings.append({
                "word": w.word,
                "startTime": f"{w.start_time.total_seconds()}s" if w.start_time else None,
                "endTime": f"{w.end_time.total_seconds()}s" if w.end_time else None
            })
    return {"results": results, "words_timings": words_timings}

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Transcription audio avec Google Cloud STT")
    parser.add_argument("audio_file", help="Chemin vers le fichier audio (WAV/FLAC/LINEAR16)")
    parser.add_argument("--lang", default="fr-FR", help="Code langue (ex: fr-FR)")
    args = parser.parse_args()

    # Vérifie la variable d'environnement GOOGLE_APPLICATION_CREDENTIALS
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("[ERREUR] Vous devez définir la variable d'environnement GOOGLE_APPLICATION_CREDENTIALS vers votre fichier clé JSON de service Google Cloud.")
        exit(1)

    result = transcribe_audio(args.audio_file, args.lang)
    #print(json.dumps(result["results"], indent=2, ensure_ascii=False))
    #print(json.dumps(result["words_timings"], indent=2, ensure_ascii=False))