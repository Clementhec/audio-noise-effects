"""
Test Semantic Matching

Quick test of the semantic matcher with sample speech segments.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.embeddings_utils import get_embeddings
from sound_embedding.sound_embedding_loader import SoundEmbeddingLoader
from semantic_matcher import SemanticMatcher


def create_sample_speech_segments():
    """Create sample speech segments for testing."""

    # Sample speech texts that should match sound effects
    sample_texts = [
        "The dog started barking loudly",
        "Thunder rumbled across the sky",
        "An airplane flew overhead",
        "Rain was pouring down heavily",
        "A car door slammed shut",
        "Birds were chirping in the morning"
    ]

    print("Generating embeddings for sample speech segments...")
    embeddings = get_embeddings(sample_texts, model="all-MiniLM-L6-v2", show_progress=True)

    # Create DataFrame in the format expected by semantic_matcher
    speech_data = []
    for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        speech_data.append({
            'segment_id': i,
            'text': text,
            'start_time': i * 2.0,  # Each segment 2 seconds apart
            'end_time': i * 2.0 + 1.5,  # Each segment 1.5 seconds long
            'embedding': np.array(embedding)
        })

    return pd.DataFrame(speech_data)


def main():
    print("=" * 70)
    print("Testing Semantic Matcher")
    print("=" * 70)
    print()

    # Step 1: Create sample speech segments
    print("Step 1: Creating sample speech segments...")
    speech_df = create_sample_speech_segments()
    print(f"  Created {len(speech_df)} speech segments")
    print()

    # Step 2: Load sound embeddings
    print("Step 2: Loading sound embeddings...")
    loader = SoundEmbeddingLoader()
    sounds_df = loader.load_embeddings("data/soundbible_embeddings.csv")
    print(f"  Loaded {len(sounds_df)} sound embeddings")
    print()

    # Step 3: Match speech to sounds
    print("Step 3: Matching speech segments to sounds...")
    matcher = SemanticMatcher(
        similarity_threshold=0.3,  # Lower threshold for testing
        top_k=3
    )
    matches = matcher.match_speech_to_sounds(speech_df, sounds_df)
    print()

    # Step 4: Analyze results
    print("Step 4: Analyzing results...")
    analysis = matcher.analyze_matches(matches)

    print()
    print("=" * 70)
    print("Results:")
    print(f"  Total matches: {analysis['total_matches']}")
    print(f"  Unique segments matched: {analysis['unique_segments_matched']}")
    print(f"  Unique sounds used: {analysis['unique_sounds_used']}")
    print(f"  Avg similarity score: {analysis['avg_similarity_score']:.3f}")
    print(f"  Min similarity: {analysis['min_similarity_score']:.3f}")
    print(f"  Max similarity: {analysis['max_similarity_score']:.3f}")
    print("=" * 70)
    print()

    # Step 5: Show detailed matches for each segment
    print("Detailed matches by segment:")
    print("=" * 70)

    timeline_df = matcher.create_timeline_dataframe(matches)

    for segment_id in speech_df['segment_id']:
        segment = speech_df[speech_df['segment_id'] == segment_id].iloc[0]
        segment_matches = timeline_df[timeline_df['segment_id'] == segment_id]

        print(f"\nSegment {segment_id}: \"{segment['text']}\"")
        print(f"  Time: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")

        if len(segment_matches) > 0:
            print(f"  Top {len(segment_matches)} matches:")
            for idx, match in segment_matches.iterrows():
                print(f"    {match['similarity_score']:.3f} - {match['sound_title']}")
        else:
            print("  No matches found (below threshold)")

    print()
    print("=" * 70)

    # Step 6: Save timeline
    print("\nSaving timeline to test_timeline.csv...")
    matcher.save_timeline(matches, "test_timeline.csv", format='csv')
    matcher.save_timeline(matches, "test_timeline.json", format='json')

    print("\n✓ Test complete!")
    print()


if __name__ == "__main__":
    main()
