"""
Process STT Output to Speech Embeddings

Loads real STT output (word timings + transcription), segments it,
generates embeddings, and saves to CSV.

This demonstrates the complete pipeline from STT output to embedded segments.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths for imports
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "text-processing"))

from speech_segmenter import SpeechSegmenter
from utils.embeddings_utils import get_embeddings


def parse_time_string(time_str: str) -> float:
    """
    Convert time string from STT format to float seconds.

    Args:
        time_str: Time in format "1.234s"

    Returns:
        Float seconds (e.g., 1.234)
    """
    return float(time_str.rstrip('s'))


def load_stt_data(word_timing_path: str, transcription_path: str):
    """
    Load STT output files and convert to pipeline format.

    Args:
        word_timing_path: Path to word_timing.json
        transcription_path: Path to full_transcription.json

    Returns:
        Tuple of (transcript, word_timings)
    """
    # Load word timings
    with open(word_timing_path, 'r') as f:
        raw_word_timings = json.load(f)

    # Load full transcription
    with open(transcription_path, 'r') as f:
        transcription_data = json.load(f)

    transcript = transcription_data[0]['transcription']

    # Convert word timings to expected format
    word_timings = []
    for wt in raw_word_timings:
        # Skip whitespace-only words
        if wt['word'].strip():
            word_timings.append({
                'word': wt['word'].strip(),
                'start_time': parse_time_string(wt['startTime']),
                'end_time': parse_time_string(wt['endTime'])
            })

    return transcript, word_timings


def main():
    print("=" * 70)
    print("Process STT Output to Speech Embeddings")
    print("=" * 70)
    print()

    # Paths
    word_timing_path = "STT/word_timing.json"
    transcription_path = "STT/full_transcription.json"
    output_path = "data/stt_speech_embeddings.csv"

    # Step 1: Load STT data
    print("Step 1: Loading STT output...")
    transcript, word_timings = load_stt_data(word_timing_path, transcription_path)

    print(f"  Transcript: \"{transcript[:80]}...\"")
    print(f"  Total words: {len(word_timings)}")
    print(f"  Duration: {word_timings[-1]['end_time']:.2f} seconds")
    print()

    # Step 2: Segment the transcript
    print("Step 2: Segmenting transcript...")
    segmenter = SpeechSegmenter(max_words_per_segment=15)
    segments = segmenter.segment_by_sentences(transcript, word_timings)

    print(f"  Created {len(segments)} segments")
    print()

    if not segments:
        print("ERROR: No segments created!")
        return

    # Display segment preview
    print("  Sample segments:")
    for i in range(min(3, len(segments))):
        seg = segments[i]
        print(f"    {i}: \"{seg['text'][:50]}...\" ({seg['start_time']:.2f}s - {seg['end_time']:.2f}s)")
    print()

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    segment_texts = [seg['text'] for seg in segments]

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Processing {len(segment_texts)} segments...")

    embeddings = get_embeddings(
        segment_texts,
        model=EMBEDDING_MODEL,
        show_progress=True
    )

    print(f"  Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    print()

    # Step 4: Create DataFrame
    print("Step 4: Creating structured DataFrame...")
    output_data = []

    for i, segment in enumerate(segments):
        output_data.append({
            'segment_id': segment['segment_id'],
            'text': segment['text'],
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'duration': segment['end_time'] - segment['start_time'],
            'word_count': segment['word_count'],
            'embedding': embeddings[i],
            'embedding_model': EMBEDDING_MODEL
        })

    df = pd.DataFrame(output_data)
    print(f"  DataFrame shape: {df.shape}")
    print()

    # Step 5: Save to CSV
    print("Step 5: Saving to CSV...")

    # Convert embeddings to list format for CSV storage
    df_csv = df.copy()
    df_csv['embedding'] = df_csv['embedding'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    df_csv.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  File contains {len(df_csv)} embedded segments")
    print()

    # Step 6: Display results
    print("=" * 70)
    print("Embedded Segments Summary")
    print("=" * 70)
    print()

    print(f"Total segments: {len(df)}")
    print(f"Total duration: {df['duration'].sum():.2f} seconds")
    print(f"Average segment length: {df['word_count'].mean():.1f} words")
    print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")
    print()

    # Show all segments
    print("All segments:")
    print("-" * 70)
    for idx in range(len(df)):
        row = df.iloc[idx]
        print(f"\nSegment {row['segment_id']}:")
        print(f"  Text: \"{row['text']}\"")
        print(f"  Time: {row['start_time']:.2f}s - {row['end_time']:.2f}s ({row['duration']:.2f}s)")
        print(f"  Words: {row['word_count']}")

    print()
    print("=" * 70)
    print("✓ STT embedding processing complete!")
    print()
    print("Next steps:")
    print("  1. Load sound embeddings: data/soundbible_embeddings.csv")
    print("  2. Match using: semantic_matcher.py")
    print(f"     uv run python semantic_matcher.py \\")
    print(f"       {output_path} \\")
    print(f"       data/soundbible_embeddings.csv \\")
    print(f"       --output data/stt_timeline.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
