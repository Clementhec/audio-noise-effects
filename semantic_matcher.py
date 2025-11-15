"""
Semantic Matcher

Matches speech segments to sound effects using semantic similarity (cosine similarity).
Produces a timeline of sound placements for audio mixing.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))

from utils.embeddings_utils import cosine_similarity
from sound_embedding.sound_embedding_loader import SoundEmbeddingLoader


class SemanticMatcher:
    """
    Match speech segments to sound effects using semantic similarity.

    Features:
    - Cosine similarity-based matching
    - Configurable similarity threshold
    - Top-K ranking for each segment
    - Sound timeline output for audio mixing
    - Optional filtering by duration, keywords
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        top_k: int = 3,
        min_sound_duration: Optional[float] = None,
        max_sound_duration: Optional[float] = None
    ):
        """
        Initialize the semantic matcher.

        Args:
            similarity_threshold: Minimum cosine similarity (0-1) to consider a match
            top_k: Maximum number of sound matches per speech segment
            min_sound_duration: Minimum sound duration in seconds (optional filter)
            max_sound_duration: Maximum sound duration in seconds (optional filter)
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.min_sound_duration = min_sound_duration
        self.max_sound_duration = max_sound_duration

    def match_speech_to_sounds(
        self,
        speech_df: pd.DataFrame,
        sounds_df: pd.DataFrame,
        return_all_scores: bool = False
    ) -> List[Dict]:
        """
        Match speech segments to sound effects.

        Args:
            speech_df: DataFrame with speech embeddings (from speech_embedder.py)
                      Required columns: segment_id, text, start_time, end_time, embedding
            sounds_df: DataFrame with sound embeddings (from sound_embedder.py)
                      Required columns: title, description, audio_url, embedding
            return_all_scores: If True, return all similarity scores for analysis

        Returns:
            List of match dictionaries in Sound Timeline format:
            [
                {
                    'segment_id': int,
                    'segment_text': str,
                    'segment_start_time': float,
                    'segment_end_time': float,
                    'sound_id': int,
                    'sound_title': str,
                    'sound_url': str,
                    'sound_duration': float,
                    'similarity_score': float,
                    'insert_time': float,
                    'volume_adjustment': float
                },
                ...
            ]
        """
        print(f"[SemanticMatcher] Matching {len(speech_df)} speech segments to {len(sounds_df)} sounds...")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print(f"  Top-K matches per segment: {self.top_k}")

        # Validate input dataframes
        self._validate_dataframes(speech_df, sounds_df)

        # Extract embeddings
        speech_embeddings = np.array(speech_df['embedding'].tolist())
        sound_embeddings = np.array(sounds_df['embedding'].tolist())

        print(f"  Speech embeddings shape: {speech_embeddings.shape}")
        print(f"  Sound embeddings shape: {sound_embeddings.shape}")

        # Calculate all similarities
        print("  Calculating similarity scores...")
        all_matches = []

        for idx, speech_row in speech_df.iterrows():
            segment_embedding = speech_row['embedding']

            # Calculate similarity with all sounds
            similarities = []
            for sound_idx, sound_row in sounds_df.iterrows():
                sound_embedding = sound_row['embedding']
                similarity = cosine_similarity(segment_embedding, sound_embedding)
                similarities.append((sound_idx, similarity))

            # Filter by threshold and sort by similarity (descending)
            filtered_matches = [
                (sound_idx, score) for sound_idx, score in similarities
                if score >= self.similarity_threshold
            ]
            filtered_matches.sort(key=lambda x: x[1], reverse=True)

            # Take top-K matches
            top_matches = filtered_matches[:self.top_k]

            # Create match records
            for sound_idx, similarity_score in top_matches:
                sound_row = sounds_df.iloc[sound_idx]

                # Determine insert time (at start of segment by default)
                insert_time = speech_row['start_time']

                # Default volume adjustment (can be customized based on similarity)
                volume_adjustment = self._calculate_volume_adjustment(similarity_score)

                match_record = {
                    'segment_id': speech_row.get('segment_id', idx),
                    'segment_text': speech_row['text'],
                    'segment_start_time': speech_row['start_time'],
                    'segment_end_time': speech_row['end_time'],
                    'sound_id': sound_idx,
                    'sound_title': sound_row['title'],
                    'sound_url': sound_row.get('audio_url', ''),
                    'sound_duration': sound_row.get('length', 0.0),
                    'similarity_score': similarity_score,
                    'insert_time': insert_time,
                    'volume_adjustment': volume_adjustment
                }

                # Add optional fields if available
                if 'description' in sound_row:
                    match_record['sound_description'] = sound_row['description']
                if 'keywords' in sound_row:
                    match_record['sound_keywords'] = sound_row['keywords']

                all_matches.append(match_record)

        print(f"  Found {len(all_matches)} matches (threshold >= {self.similarity_threshold})")

        return all_matches

    def _validate_dataframes(self, speech_df: pd.DataFrame, sounds_df: pd.DataFrame):
        """Validate input dataframes have required columns."""
        # Check speech dataframe
        required_speech_cols = ['text', 'start_time', 'end_time', 'embedding']
        missing_speech = [col for col in required_speech_cols if col not in speech_df.columns]
        if missing_speech:
            raise ValueError(f"Speech DataFrame missing columns: {missing_speech}")

        # Check sound dataframe
        required_sound_cols = ['title', 'embedding']
        missing_sound = [col for col in required_sound_cols if col not in sounds_df.columns]
        if missing_sound:
            raise ValueError(f"Sound DataFrame missing columns: {missing_sound}")

        # Check embeddings are numpy arrays
        if not isinstance(speech_df['embedding'].iloc[0], np.ndarray):
            raise ValueError("Speech embeddings must be numpy arrays")
        if not isinstance(sounds_df['embedding'].iloc[0], np.ndarray):
            raise ValueError("Sound embeddings must be numpy arrays")

    def _calculate_volume_adjustment(self, similarity_score: float) -> float:
        """
        Calculate volume adjustment based on similarity score.

        Higher similarity = higher volume (more prominent)
        Lower similarity = lower volume (more subtle)

        Args:
            similarity_score: Cosine similarity (0-1)

        Returns:
            Volume adjustment factor (0-1)
        """
        # Linear mapping: similarity 0.75-1.0 → volume 0.5-1.0
        if similarity_score >= 0.9:
            return 1.0
        elif similarity_score >= 0.8:
            return 0.8
        else:
            return 0.6

    def create_timeline_dataframe(self, matches: List[Dict]) -> pd.DataFrame:
        """
        Convert match list to a pandas DataFrame for easier analysis.

        Args:
            matches: List of match dictionaries

        Returns:
            DataFrame with all match information
        """
        if not matches:
            return pd.DataFrame()

        df = pd.DataFrame(matches)

        # Sort by segment start time, then by similarity score
        df = df.sort_values(['segment_start_time', 'similarity_score'],
                           ascending=[True, False])

        return df

    def save_timeline(
        self,
        matches: List[Dict],
        output_path: str,
        format: str = 'csv'
    ):
        """
        Save the sound timeline to disk.

        Args:
            matches: List of match dictionaries
            output_path: Path to save the timeline
            format: Output format ('csv' or 'json')
        """
        if format == 'csv':
            df = self.create_timeline_dataframe(matches)
            df.to_csv(output_path, index=False)
            print(f"  Saved timeline to: {output_path} ({len(df)} matches)")

        elif format == 'json':
            output_data = {
                'metadata': {
                    'total_matches': len(matches),
                    'similarity_threshold': self.similarity_threshold,
                    'top_k': self.top_k
                },
                'timeline': matches
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved timeline to: {output_path} ({len(matches)} matches)")

        else:
            raise ValueError(f"Unknown format: {format}")

    def analyze_matches(self, matches: List[Dict]) -> Dict:
        """
        Analyze the match results.

        Args:
            matches: List of match dictionaries

        Returns:
            Dictionary with analysis statistics
        """
        if not matches:
            return {'total_matches': 0}

        df = self.create_timeline_dataframe(matches)

        # Calculate statistics
        analysis = {
            'total_matches': len(matches),
            'unique_segments_matched': df['segment_id'].nunique(),
            'unique_sounds_used': df['sound_id'].nunique(),
            'avg_similarity_score': df['similarity_score'].mean(),
            'min_similarity_score': df['similarity_score'].min(),
            'max_similarity_score': df['similarity_score'].max(),
            'matches_per_segment': {
                'mean': df.groupby('segment_id').size().mean(),
                'min': df.groupby('segment_id').size().min(),
                'max': df.groupby('segment_id').size().max()
            }
        }

        return analysis


def match_speech_to_sounds(
    speech_embeddings_path: str,
    sound_embeddings_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.75,
    top_k: int = 3
) -> List[Dict]:
    """
    Convenience function to match speech to sounds.

    Args:
        speech_embeddings_path: Path to speech embeddings CSV
        sound_embeddings_path: Path to sound embeddings CSV
        output_path: Optional path to save timeline
        threshold: Similarity threshold
        top_k: Max matches per segment

    Returns:
        List of match dictionaries
    """
    print(f"Loading speech embeddings from: {speech_embeddings_path}")
    speech_df = pd.read_csv(speech_embeddings_path)
    speech_df['embedding'] = speech_df['embedding'].apply(eval).apply(np.array)

    print(f"Loading sound embeddings from: {sound_embeddings_path}")
    loader = SoundEmbeddingLoader()
    sounds_df = loader.load_embeddings(sound_embeddings_path)

    # Create matcher and match
    matcher = SemanticMatcher(similarity_threshold=threshold, top_k=top_k)
    matches = matcher.match_speech_to_sounds(speech_df, sounds_df)

    # Save if output path provided
    if output_path:
        matcher.save_timeline(matches, output_path, format='csv')

    return matches


if __name__ == "__main__":
    """
    Example usage:

    python semantic_matcher.py <speech.csv> <sounds.csv>
    python semantic_matcher.py speech.csv sounds.csv --output timeline.csv --threshold 0.8
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Match speech segments to sound effects using semantic similarity"
    )
    parser.add_argument(
        "speech_embeddings",
        help="Path to speech embeddings CSV file"
    )
    parser.add_argument(
        "sound_embeddings",
        help="Path to sound embeddings CSV file"
    )
    parser.add_argument(
        "--output",
        default="sound_timeline.csv",
        help="Path to save the sound timeline"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum similarity threshold (0-1)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum matches per speech segment"
    )
    parser.add_argument(
        "--format",
        choices=['csv', 'json'],
        default='csv',
        help="Output format"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Semantic Matcher - Speech to Sound Matching")
    print("=" * 70)
    print(f"Speech embeddings: {args.speech_embeddings}")
    print(f"Sound embeddings: {args.sound_embeddings}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Top-K matches: {args.top_k}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print()

    # Perform matching
    matches = match_speech_to_sounds(
        args.speech_embeddings,
        args.sound_embeddings,
        output_path=args.output,
        threshold=args.threshold,
        top_k=args.top_k
    )

    # Analyze results
    matcher = SemanticMatcher(similarity_threshold=args.threshold, top_k=args.top_k)
    analysis = matcher.analyze_matches(matches)

    print()
    print("=" * 70)
    print("Matching Results:")
    print(f"  Total matches: {analysis['total_matches']}")
    print(f"  Unique segments matched: {analysis['unique_segments_matched']}")
    print(f"  Unique sounds used: {analysis['unique_sounds_used']}")
    print(f"  Avg similarity score: {analysis['avg_similarity_score']:.3f}")
    print(f"  Min similarity: {analysis['min_similarity_score']:.3f}")
    print(f"  Max similarity: {analysis['max_similarity_score']:.3f}")
    print(f"  Matches per segment (avg): {analysis['matches_per_segment']['mean']:.1f}")
    print("=" * 70)

    # Show sample matches
    if matches:
        print("\nSample matches:")
        df = matcher.create_timeline_dataframe(matches)
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            print(f"\n  Segment {row['segment_id']}: \"{row['segment_text'][:50]}...\"")
            print(f"    → Sound: {row['sound_title']}")
            print(f"    → Similarity: {row['similarity_score']:.3f}")
            print(f"    → Insert at: {row['insert_time']:.2f}s")
