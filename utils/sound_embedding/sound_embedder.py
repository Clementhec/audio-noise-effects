"""
Sound Embedding Generator

Generates semantic embeddings for sound effects by combining title, description, and keywords.
Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) for local, offline embedding generation.

This ensures compatibility with speech embeddings for semantic matching.
"""

import json
import pandas as pd
from typing import Optional
import ast

from utils.embeddings_utils import get_embeddings


class SoundEmbedder:
    """
    Generate embeddings for sound effects metadata.

    Key Features:
    - Combines title, description, and keywords for rich semantic representation
    - Uses all-MiniLM-L6-v2 (384 dimensions) - same as speech embeddings
    - Batch processing for efficiency
    - Progress tracking for large datasets
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Must match speech embedding model!
    EMBEDDING_DIMENSION = 384

    def __init__(self, batch_size: int = 32, show_progress: bool = True):
        """
        Initialize the sound embedder.

        Args:
            batch_size: Number of sounds to process at once
            show_progress: Whether to display progress bar
        """
        self.batch_size = batch_size
        self.show_progress = show_progress

    def create_embedding_text(self, title: str, description: str, keywords: str) -> str:
        """
        Combine sound metadata into embedding text.

        Format: "{title}: {description} [{keywords}]"

        Args:
            title: Sound effect name
            description: Text description
            keywords: Keywords (as string representation of list)

        Returns:
            Combined text for embedding
        """
        # Parse keywords if it's a string representation of a list
        if isinstance(keywords, str):
            try:
                # Try to parse as Python literal
                keywords_list = ast.literal_eval(keywords)
                if isinstance(keywords_list, list):
                    keywords_text = ", ".join(keywords_list)
                else:
                    keywords_text = keywords
            except (ValueError, SyntaxError):
                # If parsing fails, use as-is
                keywords_text = keywords
        else:
            keywords_text = str(keywords)

        # Combine into single text
        embedding_text = f"{title}: {description} [{keywords_text}]"
        return embedding_text

    def process_sound_dataframe(
        self,
        df: pd.DataFrame,
        save_output: bool = True,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate embeddings for all sounds in the dataframe.

        Args:
            df: DataFrame with columns: title, description, keywords
            save_output: Whether to save results to disk
            output_path: Custom output path (auto-generated if None)

        Returns:
            DataFrame with added 'embedding' and 'embedding_model' columns
        """
        print(f"[SoundEmbedder] Processing {len(df)} sounds...")

        # Validate required columns
        required_cols = ["title", "description", "keywords"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create embedding texts
        print("Creating embedding texts...")
        embedding_texts = []
        for idx, row in df.iterrows():
            text = self.create_embedding_text(
                row["title"], row["description"], row["keywords"]
            )
            embedding_texts.append(text)

        print(f"  Generated {len(embedding_texts)} embedding texts")

        # Generate embeddings in batch
        print(f"Generating embeddings using {self.EMBEDDING_MODEL}...")
        print(f"Batch size: {self.batch_size}")

        embeddings = get_embeddings(
            embedding_texts,
            model=self.EMBEDDING_MODEL,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
        )

        print(f"  Generated {len(embeddings)} embeddings")

        # Add embeddings to dataframe
        df_result = df.copy()
        df_result["embedding"] = embeddings
        df_result["embedding_model"] = self.EMBEDDING_MODEL
        df_result["embedding_text"] = (
            embedding_texts  # Store the combined text for reference
        )

        # Save output if requested
        if save_output:
            output_path = output_path or "data/soundbible_embeddings.csv"
            self._save_output(df_result, output_path)

        print(f"[SoundEmbedder] Processing complete!")
        return df_result

    def _save_output(self, df: pd.DataFrame, output_path: str):
        """Save the embedded sounds to CSV."""
        # Convert embeddings to list format for CSV storage
        df_csv = df.copy()
        df_csv["embedding"] = df_csv["embedding"].apply(
            lambda x: x if isinstance(x, list) else x.tolist()
        )

        df_csv.to_csv(output_path, index=False)
        print(f"Saved embeddings to: {output_path}")

        # Also save metadata as JSON
        metadata_path = str(output_path).replace(".csv", "_metadata.json")
        metadata = {
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "total_sounds": len(df),
            "batch_size": self.batch_size,
            "text_format": "{title}: {description} [{keywords}]",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings for sound effects from CSV file"
    )
    parser.add_argument(
        "--input",
        default="data/soundbible_details_from_section.csv",
        help="Path to input sound CSV file",
    )
    parser.add_argument(
        "--output",
        default="data/soundbible_embeddings.csv",
        help="Path to save embeddings",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Number of sounds to process at once"
    )

    args = parser.parse_args()
    batch_size = 32

    print("=" * 70)
    print("Sound Embedding Generator")
    print("=" * 70)
    print(f"Model: {SoundEmbedder.EMBEDDING_MODEL}")
    print(f"Dimensions: {SoundEmbedder.EMBEDDING_DIMENSION}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    print()

    df = pd.read_csv(args.input)

    embedder = SoundEmbedder(batch_size=batch_size, show_progress=True)
    df_embedded = embedder.process_sound_dataframe(
        df, save_output=True, output_path=args.output
    )

    print()
    print("=" * 70)
    print("Summary:")
    print(f"Total sounds embedded: {len(df_embedded)}")
    print(
        f"Embedding dimension: {len(df_embedded['embedding'].iloc[0]) if len(df_embedded) > 0 else 'N/A'}"
    )
    print(
        f"Model: {df_embedded['embedding_model'].iloc[0] if len(df_embedded) > 0 else 'N/A'}"
    )
    print("=" * 70)
