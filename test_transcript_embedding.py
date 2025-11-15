"""
Test Transcript Embedding Pipeline

Demonstrates the full speech embedding workflow:
1. Generate sample transcript text
2. Create simulated word timings (mimicking STT output)
3. Segment the transcript into meaningful chunks
4. Generate embeddings for each segment
5. Save to CSV in data folder

This simulates what would happen with a real .wav file after STT processing.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add root directory and text-processing directory to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "text-processing"))

from speech_segmenter import SpeechSegmenter
from utils.embeddings_utils import get_embeddings


def create_sample_transcript():
    """
    Create a sample transcript with realistic content.

    This simulates what you would get from Google Speech-to-Text.
    """
    transcript = """
    The weather today is absolutely beautiful with clear blue skies.
    I heard thunder rumbling in the distance earlier this morning.
    A dog started barking loudly when the mailman arrived at the door.
    The airplane flew overhead making a tremendous roaring sound.
    Rain was pouring down heavily during the afternoon storm.
    Birds were chirping cheerfully in the garden at sunrise.
    A car door slammed shut in the parking lot nearby.
    The church bells rang out across the quiet village.
    Wind was howling through the trees during the night.
    Children were laughing and playing in the schoolyard.
    """

    return transcript.strip()


def create_word_timings(transcript: str):
    """
    Create simulated word-level timings for the transcript.

    This simulates what Google Speech-to-Text provides with word-level timing.
    Each word gets a start_time and end_time in seconds.

    Args:
        transcript: The full transcript text

    Returns:
        List of word timing dictionaries
    """
    words = transcript.split()
    word_timings = []

    # Simulate realistic timing: ~0.3 seconds per word on average
    current_time = 0.0

    for word in words:
        # Clean punctuation for word
        clean_word = word.strip('.,!?;:')

        # Variable word duration based on length
        word_duration = 0.2 + (len(clean_word) * 0.05)

        word_timings.append({
            'word': clean_word,
            'start_time': current_time,
            'end_time': current_time + word_duration
        })

        current_time += word_duration + 0.1  # Add small pause between words

    return word_timings


def main():
    print("=" * 70)
    print("Test Transcript Embedding Pipeline")
    print("=" * 70)
    print()

    # Step 1: Create sample transcript
    print("Step 1: Creating sample transcript...")
    transcript = create_sample_transcript()
    print(f"  Transcript length: {len(transcript)} characters")
    print(f"  Preview: {transcript[:100]}...")
    print()

    # Step 2: Create simulated word timings
    print("Step 2: Creating simulated word timings...")
    word_timings = create_word_timings(transcript)
    print(f"  Total words: {len(word_timings)}")
    print(f"  Total duration: {word_timings[-1]['end_time']:.1f} seconds")
    print(f"  Sample timings:")
    for wt in word_timings[:5]:
        print(f"    '{wt['word']}' @ {wt['start_time']:.2f}s - {wt['end_time']:.2f}s")
    print()

    # Step 3: Segment the transcript
    print("Step 3: Segmenting transcript into chunks...")
    segmenter = SpeechSegmenter(max_words_per_segment=15)
    segments = segmenter.segment_by_sentences(transcript, word_timings)
    print(f"  Created {len(segments)} segments")
    print()

    # Step 4: Generate embeddings
    print("Step 4: Generating embeddings for segments...")
    segment_texts = [seg['text'] for seg in segments]

    # Use the same model as sound embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Processing {len(segment_texts)} segments...")

    embeddings = get_embeddings(
        segment_texts,
        model=EMBEDDING_MODEL,
        show_progress=True
    )

    print(f"  Generated {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {len(embeddings[0])}")
    print()

    # Step 5: Create DataFrame
    print("Step 5: Creating structured DataFrame...")
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

    # Step 6: Save to CSV
    print("Step 6: Saving to CSV in data folder...")
    output_path = "data/test_speech_embeddings.csv"

    # Convert embeddings to list format for CSV storage
    df_csv = df.copy()
    df_csv['embedding'] = df_csv['embedding'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    df_csv.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  File contains {len(df_csv)} embedded segments")
    print()

    # Step 7: Display sample results
    print("=" * 70)
    print("Sample Embedded Segments:")
    print("=" * 70)

    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        embedding = row['embedding']
        # Convert to numpy array if it's a list for display
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        print(f"\nSegment {row['segment_id']}:")
        print(f"  Text: \"{row['text'][:60]}...\"")
        print(f"  Time: {row['start_time']:.2f}s - {row['end_time']:.2f}s ({row['duration']:.2f}s)")
        print(f"  Words: {row['word_count']}")
        print(f"  Embedding: {len(embedding)}-dim vector")
        print(f"  Embedding preview: [{embedding[0]:.3f}, {embedding[1]:.3f}, ...]")

    print()
    print("=" * 70)
    print("✓ Transcript embedding test complete!")
    print()
    print("Next steps:")
    print("  1. Load sound embeddings: data/soundbible_embeddings.csv")
    print("  2. Match speech to sounds using semantic_matcher.py")
    print("  3. Generate sound timeline for audio mixing")
    print("=" * 70)


if __name__ == "__main__":
    main()
