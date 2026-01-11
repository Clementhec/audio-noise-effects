"""
Convert MP3 files from a folder to WAV format.
Uses pydub and requires ffmpeg to be installed.
"""

import os
import sys
from pathlib import Path

# Import audioop before pydub for Python 3.13 compatibility (audioop-lts provides this)
try:
    import audioop
except ImportError:
    pass
from pydub import AudioSegment


def convert_mp3_to_wav(
    input_file: Path, output_file: Path = None, overwrite: bool = False
) -> bool:
    """
    Convert a single MP3 file to WAV format.

    Args:
        input_file: Path to the input MP3 file
        output_file: Path to the output WAV file (if None, uses same name with .wav extension)
        overwrite: If True, overwrite existing output files

    Returns:
        True if conversion successful, False otherwise
    """
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return False

    if not str(input_file).lower().endswith(".mp3"):
        print(f"[WARNING] File is not an MP3: {input_file}")
        return False

    # Determine output file path
    if output_file is None:
        output_file = input_file.with_suffix(".wav")
    else:
        output_file = Path(output_file)

    # Check if output file exists
    if output_file.exists() and not overwrite:
        print(f"[SKIP] Output file already exists: {output_file.name}")
        return False

    try:
        print(f"Converting: {input_file.name} -> {output_file.name}")

        # Load MP3 file
        audio = AudioSegment.from_mp3(str(input_file))

        # Export as WAV
        audio.export(str(output_file), format="wav")

        # Get file sizes for comparison
        input_size = input_file.stat().st_size / (1024 * 1024)  # MB
        output_size = output_file.stat().st_size / (1024 * 1024)  # MB
        duration = len(audio) / 1000.0  # seconds

        print(f"  [SUCCESS] Converted successfully!")
        print(f"    Duration: {duration:.2f} seconds")
        print(f"    Input size: {input_size:.2f} MB")
        print(f"    Output size: {output_size:.2f} MB")
        print(f"    Sample rate: {audio.frame_rate} Hz")
        print(f"    Channels: {audio.channels}")

        return True

    except FileNotFoundError as e:
        error_msg = str(e)
        if "ffmpeg" in error_msg.lower() or "introuvable" in error_msg.lower():
            print(f"  [ERROR] ffmpeg is required but not found!")
            print(f"    Install ffmpeg from: https://ffmpeg.org/download.html")
            print(f"    Or use: choco install ffmpeg (if you have Chocolatey)")
        else:
            print(f"  [ERROR] FileNotFoundError: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to convert: {e}")
        return False


def convert_folder_mp3_to_wav(
    folder_path: str,
    output_folder: str = None,
    overwrite: bool = False,
    recursive: bool = False,
) -> dict:
    """
    Convert all MP3 files in a folder to WAV format.

    Args:
        folder_path: Path to the folder containing MP3 files
        output_folder: Path to output folder (if None, saves in same folder as input)
        overwrite: If True, overwrite existing output files
        recursive: If True, search subdirectories recursively

    Returns:
        Dictionary with conversion statistics
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}")
        return {"success": 0, "failed": 0, "skipped": 0}

    if not folder.is_dir():
        print(f"[ERROR] Path is not a directory: {folder}")
        return {"success": 0, "failed": 0, "skipped": 0}

    # Determine output folder
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = folder

    # Find all MP3 files (deduplicate for case-insensitive filesystems)
    if recursive:
        mp3_files_raw = list(folder.rglob("*.mp3")) + list(folder.rglob("*.MP3"))
    else:
        mp3_files_raw = list(folder.glob("*.mp3")) + list(folder.glob("*.MP3"))

    # Deduplicate (Windows filesystem is case-insensitive)
    seen = set()
    mp3_files = []
    for mp3_file in mp3_files_raw:
        normalized = str(mp3_file).lower()
        if normalized not in seen:
            seen.add(normalized)
            mp3_files.append(mp3_file)

    if not mp3_files:
        print(f"[INFO] No MP3 files found in: {folder}")
        return {"success": 0, "failed": 0, "skipped": 0}

    print("=" * 60)
    print(f"Converting {len(mp3_files)} MP3 file(s) from: {folder}")
    if output_folder:
        print(f"Output folder: {output_path}")
    print("=" * 60)
    print()

    stats = {"success": 0, "failed": 0, "skipped": 0}

    for i, mp3_file in enumerate(mp3_files, 1):
        print(f"[{i}/{len(mp3_files)}] Processing: {mp3_file.name}")

        # Determine output file path
        if output_folder:
            # Preserve relative path structure if recursive
            if recursive:
                relative_path = mp3_file.relative_to(folder)
                output_file = output_path / relative_path.with_suffix(".wav")
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = output_path / mp3_file.with_suffix(".wav").name
        else:
            output_file = None  # Will use same folder as input

        # Convert file
        result = convert_mp3_to_wav(mp3_file, output_file, overwrite)

        if result:
            stats["success"] += 1
        elif output_file and output_file.exists():
            stats["skipped"] += 1
        else:
            stats["failed"] += 1

        print()  # Empty line between files

    # Print summary
    print("=" * 60)
    print("Conversion Summary:")
    print(f"  Successfully converted: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped (already exists): {stats['skipped']}")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert MP3 files from a folder to WAV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all MP3 files in current folder
  python convert_mp3_to_wav.py bank
  
  # Convert and save to different folder
  python convert_mp3_to_wav.py bank --output converted
  
  # Convert recursively (including subfolders)
  python convert_mp3_to_wav.py bank --recursive
  
  # Overwrite existing WAV files
  python convert_mp3_to_wav.py bank --overwrite
        """,
    )

    parser.add_argument("folder", help="Path to folder containing MP3 files")

    parser.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        help="Output folder for WAV files (default: same as input folder)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing WAV files"
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for MP3 files recursively in subdirectories",
    )

    args = parser.parse_args()

    # Convert files
    stats = convert_folder_mp3_to_wav(
        folder_path=args.folder,
        output_folder=args.output_folder,
        overwrite=args.overwrite,
        recursive=args.recursive,
    )

    # Exit with error code if any conversions failed
    if stats["failed"] > 0:
        sys.exit(1)
    elif stats["success"] == 0 and stats["skipped"] == 0:
        sys.exit(1)
    else:
        sys.exit(0)
