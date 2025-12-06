import os
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from typing import Optional
import hashlib


def generate_sound_effect(
    sound_description: str,
    sound_folder: Path,
    api_key: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    prompt_influence: Optional[float] = None,
) -> Path:
    """
    Generate a sound effect from a text description using ElevenLabs API.

    Args:
        sound_description: Text description of the sound to generate
        sound_folder: Directory where the generated sound file will be saved
        api_key: ElevenLabs API key (if None, uses ELEVENLABS_API_KEY from .env)
        duration_seconds: Duration of the generated sound (optional)
        prompt_influence: How closely to follow the prompt (0.0-1.0, optional)

    Returns:
        Path to the generated .wav file

    Example:
        >>> from pathlib import Path
        >>> output_path = generate_sound_effect(
        ...     "dog barking",
        ...     Path("output/sounds")
        ... )
        >>> print(output_path)
        output/sounds/dog_barking_a1b2c3.wav
    """
    load_dotenv()

    if api_key is None:
        api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise ValueError(
            "ElevenLabs API key not found. Set ELEVENLABS_API_KEY in .env or pass api_key parameter."
        )

    # Create output directory if it doesn't exist
    sound_folder = Path(sound_folder)
    sound_folder.mkdir(parents=True, exist_ok=True)

    # Initialize ElevenLabs client
    client = ElevenLabs(api_key=api_key)

    # Generate the sound effect
    kwargs = {"text": sound_description}
    if duration_seconds is not None:
        kwargs["duration_seconds"] = duration_seconds
    if prompt_influence is not None:
        kwargs["prompt_influence"] = prompt_influence

    result = client.text_to_sound_effects.convert(**kwargs)

    # Create a safe filename from the description
    # Use first few words and add a hash for uniqueness
    safe_description = "".join(
        c if c.isalnum() or c.isspace() else "_" for c in sound_description
    )
    safe_description = "_".join(safe_description.split()[:5])  # First 5 words

    # Add hash of full description for uniqueness
    desc_hash = hashlib.md5(sound_description.encode()).hexdigest()[:6]
    filename = f"{safe_description}_{desc_hash}.wav"

    output_path = sound_folder / filename

    # Save the audio content to file
    with open(output_path, "wb") as f:
        for chunk in result:
            if isinstance(chunk, bytes):
                f.write(chunk)
            elif hasattr(chunk, "audio"):
                f.write(chunk.audio)

    return output_path


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--sound-folder", type=Path, default=Path("../data/sounds"))
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--prompt-influence", type=float, default=1.0)
    args = parser.parse_args()
    generate_sound_effect(
        sound_description=args.text,
        sound_folder=args.sound_folder,
        duration_seconds=args.duration,
        prompt_influence=args.prompt_influence,
    )
