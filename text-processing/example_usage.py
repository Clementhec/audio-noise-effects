"""
Example Usage of Text Processing Pipeline

Demonstrates how to use the speech embedding pipeline with sample data.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from text_processing.speech_embedder import SpeechEmbeddingPipeline


def create_sample_stt_output():
    """
    Create a sample STT output for demonstration purposes.

    This mimics the output structure from stt_google.transcribe_audio()
    """
    return {
        'results': {
            'transcript': (
                "The storm approached rapidly from the west. "
                "Lightning struck nearby with a brilliant flash. "
                "Thunder echoed across the valley. "
                "Dogs began barking in the distance. "
                "Rain started pouring down heavily."
            ),
            'confidence': 0.94
        },
        'words_timings': [
            # Sentence 1: "The storm approached rapidly from the west."
            {'word': 'The', 'start_time': 0.0, 'end_time': 0.2},
            {'word': 'storm', 'start_time': 0.2, 'end_time': 0.6},
            {'word': 'approached', 'start_time': 0.6, 'end_time': 1.1},
            {'word': 'rapidly', 'start_time': 1.1, 'end_time': 1.5},
            {'word': 'from', 'start_time': 1.5, 'end_time': 1.7},
            {'word': 'the', 'start_time': 1.7, 'end_time': 1.8},
            {'word': 'west', 'start_time': 1.8, 'end_time': 2.1},

            # Sentence 2: "Lightning struck nearby with a brilliant flash."
            {'word': 'Lightning', 'start_time': 2.4, 'end_time': 2.9},
            {'word': 'struck', 'start_time': 2.9, 'end_time': 3.3},
            {'word': 'nearby', 'start_time': 3.3, 'end_time': 3.7},
            {'word': 'with', 'start_time': 3.7, 'end_time': 3.9},
            {'word': 'a', 'start_time': 3.9, 'end_time': 4.0},
            {'word': 'brilliant', 'start_time': 4.0, 'end_time': 4.5},
            {'word': 'flash', 'start_time': 4.5, 'end_time': 4.9},

            # Sentence 3: "Thunder echoed across the valley."
            {'word': 'Thunder', 'start_time': 5.2, 'end_time': 5.7},
            {'word': 'echoed', 'start_time': 5.7, 'end_time': 6.1},
            {'word': 'across', 'start_time': 6.1, 'end_time': 6.5},
            {'word': 'the', 'start_time': 6.5, 'end_time': 6.6},
            {'word': 'valley', 'start_time': 6.6, 'end_time': 7.0},

            # Sentence 4: "Dogs began barking in the distance."
            {'word': 'Dogs', 'start_time': 7.3, 'end_time': 7.6},
            {'word': 'began', 'start_time': 7.6, 'end_time': 7.9},
            {'word': 'barking', 'start_time': 7.9, 'end_time': 8.3},
            {'word': 'in', 'start_time': 8.3, 'end_time': 8.4},
            {'word': 'the', 'start_time': 8.4, 'end_time': 8.5},
            {'word': 'distance', 'start_time': 8.5, 'end_time': 9.0},

            # Sentence 5: "Rain started pouring down heavily."
            {'word': 'Rain', 'start_time': 9.3, 'end_time': 9.6},
            {'word': 'started', 'start_time': 9.6, 'end_time': 10.0},
            {'word': 'pouring', 'start_time': 10.0, 'end_time': 10.4},
            {'word': 'down', 'start_time': 10.4, 'end_time': 10.7},
            {'word': 'heavily', 'start_time': 10.7, 'end_time': 11.2},
        ]
    }


def main():
    """Run the example pipeline."""
    print("=" * 70)
    print("Text Processing Pipeline - Example Usage")
    print("=" * 70)
    print()

    # Step 1: Create sample STT output
    print("Step 1: Creating sample STT output...")
    stt_result = create_sample_stt_output()
    print(f"  Transcript: '{stt_result['results']['transcript'][:50]}...'")
    print(f"  Total words: {len(stt_result['words_timings'])}")
    print()

    # Optional: Save sample STT output for future use
    sample_dir = "./text-processing/output/samples"
    os.makedirs(sample_dir, exist_ok=True)
    sample_stt_path = os.path.join(sample_dir, "sample_stt_output.json")

    with open(sample_stt_path, 'w') as f:
        json.dump(stt_result, f, indent=2)
    print(f"  Saved sample STT output to: {sample_stt_path}")
    print()

    # Step 2: Initialize the pipeline
    print("Step 2: Initializing speech embedding pipeline...")
    pipeline = SpeechEmbeddingPipeline(
        segmentation_method="sentences",
        max_words_per_segment=15,
        output_dir="./text-processing/output"
    )
    print(f"  Model: {pipeline.EMBEDDING_MODEL}")
    print(f"  Embedding dimension: {pipeline.EMBEDDING_DIMENSION}")
    print()

    # Step 3: Process the STT output
    print("Step 3: Processing STT output...")
    print("  (This will make API calls to OpenAI - ensure OPENAI_API_KEY is set)")
    print()

    try:
        df = pipeline.process_stt_output(
            stt_result,
            save_output=True,
            output_filename="example_speech_embeddings"
        )

        print()
        print("=" * 70)
        print("Results Summary")
        print("=" * 70)
        print(f"Total segments created: {len(df)}")
        print(f"Total duration: {df['duration'].sum():.2f} seconds")
        print(f"Average segment length: {df['word_count'].mean():.1f} words")
        print()

        # Display segment details
        print("Segment Details:")
        print("-" * 70)
        for idx, row in df.iterrows():
            print(f"Segment {row['segment_id']}:")
            print(f"  Text: '{row['text']}'")
            print(f"  Time: {row['start_time']:.2f}s - {row['end_time']:.2f}s ({row['duration']:.2f}s)")
            print(f"  Words: {row['word_count']}")
            print(f"  Embedding shape: {row['embedding'].shape}")
            print()

        print("=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print("1. Load sound embeddings from:")
        print("   data/soundbible_details_from_section_with_embeddings.csv")
        print()
        print("2. Calculate similarity scores between speech and sound embeddings")
        print()
        print("3. Filter by threshold (e.g., cosine similarity ≥ 0.75)")
        print()
        print("4. Generate audio mixing timeline")
        print()
        print("See README.md for integration examples.")
        print("=" * 70)

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Make sure:")
        print("  1. OPENAI_API_KEY environment variable is set")
        print("  2. You have an active OpenAI API account")
        print("  3. The utils/embeddings_utils.py module is available")


if __name__ == "__main__":
    main()
