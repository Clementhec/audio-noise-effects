"""
Sound Embedding Loader

Loads and manages pre-computed sound embeddings for semantic matching.
Handles parsing, validation, and provides utility functions for filtering.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import ast


class SoundEmbeddingLoader:
    """
    Load and manage sound embeddings for efficient matching.

    Features:
    - Load embeddings from CSV files
    - Parse string representations to numpy arrays
    - Validate embedding dimensions and format
    - Optional filtering by duration, keywords, etc.
    - Caching for performance
    """

    EXPECTED_DIMENSION = 384  # all-MiniLM-L6-v2
    EXPECTED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the loader.

        Args:
            cache_enabled: Whether to cache loaded dataframes
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_embeddings(
        self,
        filepath: str,
        validate: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load sound embeddings from CSV file.

        Args:
            filepath: Path to CSV file with embeddings
            validate: Whether to validate embeddings after loading
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with embeddings as numpy arrays
        """
        # Check cache first
        if use_cache and self.cache_enabled and filepath in self._cache:
            print(f"[SoundEmbeddingLoader] Loading from cache: {filepath}")
            return self._cache[filepath].copy()

        print(f"[SoundEmbeddingLoader] Loading embeddings from: {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        # Load CSV
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} sound embeddings")

        # Check if embedding column exists
        if 'embedding' not in df.columns:
            raise ValueError("CSV must contain 'embedding' column")

        # Parse embeddings from string to numpy array
        print("  Parsing embeddings...")
        df['embedding'] = df['embedding'].apply(self._parse_embedding)

        # Validate if requested
        if validate:
            self._validate_embeddings(df)

        # Cache if enabled
        if self.cache_enabled:
            self._cache[filepath] = df.copy()

        print(f"[SoundEmbeddingLoader] Loading complete!")
        return df

    def _parse_embedding(self, embedding_str) -> np.ndarray:
        """
        Parse embedding from string or list to numpy array.

        Args:
            embedding_str: String representation or list

        Returns:
            Numpy array
        """
        if isinstance(embedding_str, np.ndarray):
            return embedding_str

        if isinstance(embedding_str, list):
            return np.array(embedding_str)

        if isinstance(embedding_str, str):
            try:
                # Try to parse as Python literal
                embedding_list = ast.literal_eval(embedding_str)
                return np.array(embedding_list)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Failed to parse embedding string: {e}")

        raise ValueError(f"Unknown embedding type: {type(embedding_str)}")

    def _validate_embeddings(self, df: pd.DataFrame):
        """
        Validate that embeddings have correct format and dimensions.

        Args:
            df: DataFrame with embeddings

        Raises:
            ValueError: If validation fails
        """
        print("  Validating embeddings...")

        # Check that all embeddings are numpy arrays
        non_arrays = df['embedding'].apply(lambda x: not isinstance(x, np.ndarray))
        if non_arrays.any():
            raise ValueError(f"Found {non_arrays.sum()} non-array embeddings")

        # Check dimensions
        dimensions = df['embedding'].apply(len)
        unique_dims = dimensions.unique()

        if len(unique_dims) > 1:
            raise ValueError(f"Inconsistent embedding dimensions: {unique_dims}")

        actual_dim = unique_dims[0]
        if actual_dim != self.EXPECTED_DIMENSION:
            print(f"  WARNING: Expected {self.EXPECTED_DIMENSION} dimensions, got {actual_dim}")
            print(f"  This may indicate a model mismatch!")

        print(f"   All embeddings validated ({actual_dim} dimensions)")

        # Check model if column exists
        if 'embedding_model' in df.columns:
            models = df['embedding_model'].unique()
            if len(models) == 1:
                model = models[0]
                print(f"   Embedding model: {model}")
                if model != self.EXPECTED_MODEL:
                    print(f"    WARNING: Expected {self.EXPECTED_MODEL}")
            else:
                print(f"  WARNING: Multiple embedding models found: {models}")

    def get_sound_by_id(
        self,
        df: pd.DataFrame,
        sound_id: int
    ) -> Optional[pd.Series]:
        """
        Get a specific sound by its index/ID.

        Args:
            df: DataFrame with sound embeddings
            sound_id: Index of the sound

        Returns:
            Sound data as Series, or None if not found
        """
        if sound_id < 0 or sound_id >= len(df):
            return None
        return df.iloc[sound_id]

    def filter_by_keywords(
        self,
        df: pd.DataFrame,
        keywords: List[str],
        match_any: bool = True
    ) -> pd.DataFrame:
        """
        Filter sounds by keywords.

        Args:
            df: DataFrame with sound embeddings
            keywords: List of keywords to search for
            match_any: If True, match any keyword; if False, match all

        Returns:
            Filtered DataFrame
        """
        if 'keywords' not in df.columns:
            print("  WARNING: 'keywords' column not found, returning all sounds")
            return df

        def matches_keywords(sound_keywords):
            """Check if sound keywords match the search keywords."""
            # Parse keywords if string
            if isinstance(sound_keywords, str):
                try:
                    sound_kw_list = ast.literal_eval(sound_keywords)
                    if not isinstance(sound_kw_list, list):
                        sound_kw_list = [sound_keywords]
                except:
                    sound_kw_list = [sound_keywords]
            else:
                sound_kw_list = sound_keywords

            # Convert to lowercase for case-insensitive matching
            sound_kw_lower = [kw.lower() for kw in sound_kw_list]
            search_kw_lower = [kw.lower() for kw in keywords]

            if match_any:
                # Match if any keyword matches
                return any(kw in sound_kw_lower for kw in search_kw_lower)
            else:
                # Match if all keywords match
                return all(kw in sound_kw_lower for kw in search_kw_lower)

        mask = df['keywords'].apply(matches_keywords)
        filtered_df = df[mask]
        print(f"  Filtered to {len(filtered_df)} sounds (from {len(df)})")
        return filtered_df

    def filter_by_duration(
        self,
        df: pd.DataFrame,
        min_seconds: Optional[float] = None,
        max_seconds: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter sounds by duration.

        Args:
            df: DataFrame with sound embeddings
            min_seconds: Minimum duration (inclusive)
            max_seconds: Maximum duration (inclusive)

        Returns:
            Filtered DataFrame
        """
        if 'length' not in df.columns:
            print("  WARNING: 'length' column not found, returning all sounds")
            return df

        filtered_df = df.copy()

        # Convert length to numeric if needed
        filtered_df['length'] = pd.to_numeric(filtered_df['length'], errors='coerce')

        if min_seconds is not None:
            filtered_df = filtered_df[filtered_df['length'] >= min_seconds]

        if max_seconds is not None:
            filtered_df = filtered_df[filtered_df['length'] <= max_seconds]

        print(f"  Filtered to {len(filtered_df)} sounds (from {len(df)})")
        return filtered_df

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        print("[SoundEmbeddingLoader] Cache cleared")

    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        return {
            'cache_enabled': self.cache_enabled,
            'cached_files': list(self._cache.keys()),
            'total_cached_sounds': sum(len(df) for df in self._cache.values())
        }


def load_sound_embeddings(filepath: str) -> pd.DataFrame:
    """
    Convenience function to load sound embeddings.

    Args:
        filepath: Path to embeddings CSV

    Returns:
        DataFrame with embeddings as numpy arrays
    """
    loader = SoundEmbeddingLoader()
    return loader.load_embeddings(filepath)


if __name__ == "__main__":
    """
    Test the loader with sample data.

    Usage:
        python sound_embedding_loader.py <embeddings.csv>
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and validate sound embeddings"
    )
    parser.add_argument(
        "embeddings_file",
        nargs='?',
        default="data/soundbible_embeddings.csv",
        help="Path to embeddings CSV file"
    )
    parser.add_argument(
        "--filter-keywords",
        nargs='+',
        help="Filter by keywords"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum sound duration (seconds)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Maximum sound duration (seconds)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Sound Embedding Loader Test")
    print("=" * 70)
    print()

    # Load embeddings
    loader = SoundEmbeddingLoader()
    df = loader.load_embeddings(args.embeddings_file)

    print()
    print("=" * 70)
    print("Loaded Data Summary:")
    print(f"  Total sounds: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    if len(df) > 0:
        print(f"  Embedding dimension: {len(df['embedding'].iloc[0])}")
        if 'embedding_model' in df.columns:
            print(f"  Embedding model: {df['embedding_model'].iloc[0]}")
    print("=" * 70)

    # Apply filters if requested
    if args.filter_keywords:
        print(f"\nFiltering by keywords: {args.filter_keywords}")
        df = loader.filter_by_keywords(df, args.filter_keywords)

    if args.min_duration is not None or args.max_duration is not None:
        print(f"\nFiltering by duration: {args.min_duration}-{args.max_duration} seconds")
        df = loader.filter_by_duration(df, args.min_duration, args.max_duration)

    # Show samples
    if len(df) > 0:
        print("\nSample sounds:")
        for idx in range(min(3, len(df))):
            sound = df.iloc[idx]
            print(f"\n  [{idx}] {sound['title']}")
            if 'description' in sound:
                print(f"      {sound['description'][:100]}...")
            if 'embedding_text' in sound:
                print(f"      Embedding text: {sound['embedding_text'][:100]}...")
            print(f"      Embedding shape: {sound['embedding'].shape}")

    print()
