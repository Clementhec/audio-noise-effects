import os
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO
import requests
import json
from elevenlabs.client import ElevenLabs
from typing import Optional, Union, Dict, Any


def transcribe_audio_elevenlabs(
    audio_source: Union[str, BytesIO],
    transcription_path: Path,
    word_timing_path: Path,
    api_key: Optional[str] = None,
    model_id: str = "scribe_v1",
    tag_audio_events: bool = False,
    language_code: Optional[str] = None,
    diarize: bool = False,
) -> Dict[str, Any]:
    """
    Transcrit un fichier audio en utilisant l'API ElevenLabs.

    Args:
        audio_source: URL du fichier audio ou objet BytesIO contenant les données audio
        api_key: Clé API ElevenLabs (si None, utilise ELEVENLABS_API_KEY depuis .env)
        model_id: ID du modèle à utiliser (défaut: "scribe_v1")
        tag_audio_events: Marquer les événements audio comme les rires, applaudissements, etc.
        language_code: Langue du fichier audio (si None, détection automatique)
        diarize: Annoter qui parle
        output_format: Format de sortie - "segments", "words", ou "both"
        save_to_json_file: Sauvegarder les résultats en fichiers JSON
        output_dir: Répertoire de sortie (si None, utilise speech_to_text/output)

    Returns:
        Dictionnaire contenant:
        - segment_result: Liste des segments avec transcription et timings
        - word_timings: Liste des mots avec leurs timings individuels
        - full_transcript: Texte complet de la transcription
    """
    load_dotenv()

    if api_key is None:
        api_key = os.getenv("ELEVENLABS_API_KEY")

    elevenlabs = ElevenLabs(api_key=api_key)

    if isinstance(audio_source, str):
        response = requests.get(audio_source)
        audio_data = BytesIO(response.content)
    else:
        audio_data = audio_source

    transcription = elevenlabs.speech_to_text.convert(
        file=audio_data,
        model_id=model_id,
        tag_audio_events=tag_audio_events,
        language_code=language_code,
        diarize=diarize,
    )

    full_transcript = transcription.text if hasattr(transcription, "text") else ""
    words = transcription.words if hasattr(transcription, "words") else []

    start_time = None
    end_time = None

    if words and len(words) > 0:
        start_time = words[0].start if hasattr(words[0], "start") else None
        end_time = words[-1].end if hasattr(words[-1], "end") else None

    segment_result = [
        {
            "transcription": full_transcript,
            "startTime": f"{start_time}s" if start_time is not None else "0.0s",
            "endTime": f"{end_time}s" if end_time is not None else None,
        }
    ]

    word_timings = []
    for word_info in words:
        word_text = word_info.text if hasattr(word_info, "text") else ""
        word_start = word_info.start if hasattr(word_info, "start") else None
        word_end = word_info.end if hasattr(word_info, "end") else None

        word_timings.append(
            {
                "word": word_text,
                "startTime": f"{word_start}s" if word_start is not None else None,
                "endTime": f"{word_end}s" if word_end is not None else None,
            }
        )

    result = {
        "full_transcript": full_transcript,
        "segment_result": segment_result,
        "word_timings": word_timings,
    }

    if transcription_path:
        with open(transcription_path, "w", encoding="utf-8") as f:
            json.dump(
                {"full_transcript": full_transcript, "segment_result": segment_result},
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(word_timing_path, "w", encoding="utf-8") as f:
            json.dump(word_timings, f, ensure_ascii=False, indent=2)

        result["output_files"] = {
            "transcription": transcription_path,
            "word_timing": word_timing_path,
        }

    return result


def transcribe_audio_file(
    audio_file_path: Union[str, Path],
    transcription_path: Path,
    word_timing_path: Path,
    api_key: Optional[str] = None,
    model_id: str = "scribe_v1",
    tag_audio_events: bool = False,
    language_code: Optional[str] = None,
    diarize: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe an audio file using ElevenLabs API (wrapper for file paths).

    This is a convenience function that accepts file paths directly instead of
    requiring BytesIO objects.

    Args:
        audio_file_path: Path to the audio file (WAV, MP3, etc.)
        output_dir: Directory to save output JSON files (default: speech_to_text/output)
        api_key: ElevenLabs API key (if None, uses ELEVENLABS_API_KEY from .env)
        model_id: Model ID to use (default: "scribe_v1")
        tag_audio_events: Tag audio events like laughter, applause, etc.
        language_code: Language of the audio (if None, auto-detect)
        diarize: Annotate who is speaking

    Returns:
        Dictionary containing:
        - full_transcript: Full transcription text
        - segment_result: List of segments with transcription and timing
        - word_timings: List of words with individual timing
        - output_files: Paths to saved JSON files (if save_to_json_file=True)

    Raises:
        FileNotFoundError: If audio file doesn't exist

    Example:
        >>> result = transcribe_audio_file("speech_to_text/input/video.wav")
        >>> print(result['full_transcript'])
        >>> print(result['output_files']['transcription'])
    """
    audio_file_path = Path(audio_file_path)

    # Validate file exists
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Read audio file into BytesIO
    with open(audio_file_path, "rb") as f:
        audio_data = BytesIO(f.read())

    # Convert output_dir to string if it's a Path
    if output_dir is not None:
        output_dir = str(output_dir)

    # Call the main transcription function
    result = transcribe_audio_elevenlabs(
        audio_source=audio_data,
        api_key=api_key,
        model_id=model_id,
        tag_audio_events=tag_audio_events,
        language_code=language_code,
        diarize=diarize,
        output_format="both",
        word_timing_path=word_timing_path,
        transcription_path=transcription_path,
    )

    return result
