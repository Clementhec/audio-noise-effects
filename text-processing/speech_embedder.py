"""
Speech Embedding Pipeline

Generates vector embeddings for speech segments using OpenAI's text-embedding-3-small model.
This ensures compatibility with sound effect embeddings for semantic matching.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.embeddings_utils import get_embeddings
from text_processing.speech_segmenter import SpeechSegmenter, load_stt_output


class SpeechEmbeddingPipeline:
    """
    End-to-end pipeline for processing speech transcripts into embeddings.

    Key Features:
    - Uses text-embedding-3-small for compatibility with sound embeddings
    - Preserves timing information for audio synchronization
    - Outputs structured data ready for semantic matching
    """

    # Model must match the one used for sound embeddings!
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536

    def __init__(
        self,
        segmentation_method: str = "sentences",
        max_words_per_segment: int = 15,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the pipeline.

        Args:
            segmentation_method: "sentences" or "time_windows"
            max_words_per_segment: Maximum words per segment (for sentence method)
            output_dir: Directory to save output files (optional)
        """
        self.segmentation_method = segmentation_method
        self.segmenter = SpeechSegmenter(max_words_per_segment=max_words_per_segment)
        self.output_dir = output_dir

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def process_stt_output(
        self,
        stt_result: Dict,
        save_output: bool = True,
        output_filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete pipeline: STT output → Segmentation → Embedding → Structured output.

        Args:
            stt_result: Output from stt_google.transcribe_audio()
            save_output: Whether to save results to disk
            output_filename: Custom filename for output (auto-generated if None)

        Returns:
            DataFrame with columns:
                - segment_id: Unique identifier
                - text: The text segment
                - start_time: Beginning timestamp (seconds)
                - end_time: Ending timestamp (seconds)
                - word_count: Number of words in segment
                - embedding: Vector representation (1536-dim array)
                - embedding_model: Model used (for tracking)
        """
        print(f"[SpeechEmbeddingPipeline] Processing STT output...")

        # Step 1: Extract transcript and word timings
        transcript, word_timings = load_stt_output(stt_result)
        print(f"  Transcript length: {len(transcript)} characters")
        print(f"  Word timings: {len(word_timings)} words")

        # Step 2: Segment the transcript
        if self.segmentation_method == "sentences":
            segments = self.segmenter.segment_by_sentences(transcript, word_timings)
        elif self.segmentation_method == "time_windows":
            segments = self.segmenter.segment_by_time_windows(
                transcript, word_timings, window_seconds=5.0
            )
        else:
            raise ValueError(f"Unknown segmentation method: {self.segmentation_method}")

        print(f"  Created {len(segments)} segments")

        if not segments:
            print("  WARNING: No segments created!")
            return pd.DataFrame()

        # Step 3: Generate embeddings for all segments
        segment_texts = [seg['text'] for seg in segments]
        print(f"  Generating embeddings using {self.EMBEDDING_MODEL}...")

        embeddings = get_embeddings(
            segment_texts,
            model=self.EMBEDDING_MODEL
        )

        print(f"  Generated {len(embeddings)} embeddings")

        # Step 4: Create structured output
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
                'embedding_model': self.EMBEDDING_MODEL
            })

        df = pd.DataFrame(output_data)

        # Step 5: Save output if requested
        if save_output and self.output_dir:
            self._save_output(df, output_filename)

        print(f"[SpeechEmbeddingPipeline] Processing complete!")
        return df

    def _save_output(self, df: pd.DataFrame, filename: Optional[str] = None):
        """Save the embedded segments to CSV and JSON formats."""
        if filename is None:
            filename = "speech_embeddings"

        # Save as CSV (embeddings as string representation)
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df_csv = df.copy()
        df_csv['embedding'] = df_csv['embedding'].apply(lambda x: x.tolist())
        df_csv.to_csv(csv_path, index=False)
        print(f"  Saved CSV to: {csv_path}")

        # Save as JSON (more structured format)
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        output_json = {
            'metadata': {
                'embedding_model': self.EMBEDDING_MODEL,
                'embedding_dimension': self.EMBEDDING_DIMENSION,
                'segmentation_method': self.segmentation_method,
                'total_segments': len(df)
            },
            'segments': df.to_dict(orient='records')
        }

        # Convert numpy arrays to lists for JSON serialization
        for segment in output_json['segments']:
            if isinstance(segment['embedding'], np.ndarray):
                segment['embedding'] = segment['embedding'].tolist()

        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f"  Saved JSON to: {json_path}")

    def load_embeddings(self, filepath: str) -> pd.DataFrame:
        """
        Load previously saved embeddings from CSV or JSON.

        Args:
            filepath: Path to saved embeddings file (.csv or .json)

        Returns:
            DataFrame with embeddings as numpy arrays
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            # Convert string representation back to numpy array
            df['embedding'] = df['embedding'].apply(eval).apply(np.array)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data['segments'])
            df['embedding'] = df['embedding'].apply(np.array)
        else:
            raise ValueError("File must be .csv or .json")

        return df


def process_speech_file(
    stt_result_path: str,
    output_dir: str = "./text-processing/output",
    segmentation_method: str = "sentences"
) -> pd.DataFrame:
    """
    Convenience function to process a saved STT result file.

    Args:
        stt_result_path: Path to JSON file containing STT output
        output_dir: Directory to save embeddings
        segmentation_method: "sentences" or "time_windows"

    Returns:
        DataFrame with embedded segments
    """
    # Load STT result
    with open(stt_result_path, 'r') as f:
        stt_result = json.load(f)

    # Create pipeline and process
    pipeline = SpeechEmbeddingPipeline(
        segmentation_method=segmentation_method,
        output_dir=output_dir
    )

    df = pipeline.process_stt_output(stt_result)
    return df


if __name__ == "__main__":
    """
    Example usage:

    python text-processing/speech_embedder.py <stt_result.json>
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings for speech segments from STT output"
    )
    parser.add_argument(
        "stt_file",
        help="Path to STT result JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="./text-processing/output",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--method",
        choices=["sentences", "time_windows"],
        default="sentences",
        help="Segmentation method"
    )

    args = parser.parse_args()

    print(f"Processing: {args.stt_file}")
    df = process_speech_file(
        args.stt_file,
        output_dir=args.output_dir,
        segmentation_method=args.method
    )

    print(f"\nResults:")
    print(f"  Total segments: {len(df)}")
    print(f"  Total duration: {df['duration'].sum():.2f} seconds")
    print(f"  Average segment length: {df['word_count'].mean():.1f} words")
