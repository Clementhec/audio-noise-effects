"""
Test script for audio_merger.py
Tests the merge_audio_files function with all sounds from the bank folder.
"""
import os
import subprocess
from pathlib import Path
from audio_merger import AudioEntry, merge_audio_files


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available in the system PATH.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      timeout=2)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_audio_files_from_bank(bank_folder: str = "bank") -> list:
    """
    Get all audio files from the bank folder.
    
    Args:
        bank_folder: Path to the bank folder relative to this script
    
    Returns:
        List of audio file paths
    """
    script_dir = Path(__file__).parent
    bank_path = script_dir / bank_folder
    
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank folder not found: {bank_path}")
    
    # Supported audio formats
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
    
    audio_files = []
    seen_files = set()  # Track files to avoid duplicates (Windows is case-insensitive)
    for ext in audio_extensions:
        # Search for both lowercase and uppercase extensions
        for file_path in bank_path.glob(f'*{ext}'):
            # Use normalized path to avoid duplicates on case-insensitive filesystems
            normalized = str(file_path).lower()
            if normalized not in seen_files:
                seen_files.add(normalized)
                audio_files.append(file_path)
        for file_path in bank_path.glob(f'*{ext.upper()}'):
            normalized = str(file_path).lower()
            if normalized not in seen_files:
                seen_files.add(normalized)
                audio_files.append(file_path)
    
    return sorted(audio_files)


def test_merge_all_sounds():
    """
    Test merging all sounds from the bank folder with various configurations.
    """
    print("Loading audio files from bank folder...")
    audio_files = get_audio_files_from_bank()
    
    if not audio_files:
        print("No audio files found in bank folder!")
        return
    
    # Check if ffmpeg is available
    ffmpeg_available = check_ffmpeg_available()
    
    # Filter out MP3 files if ffmpeg is not available
    if not ffmpeg_available:
        mp3_files = [f for f in audio_files if str(f).lower().endswith('.mp3')]
        if mp3_files:
            print(f"\n[WARNING] ffmpeg not found. Skipping {len(mp3_files)} MP3 file(s):")
            for mp3_file in mp3_files:
                print(f"   - {mp3_file.name}")
            print("   Install ffmpeg to process MP3 files.")
            audio_files = [f for f in audio_files if not str(f).lower().endswith('.mp3')]
            if not audio_files:
                print("\n[ERROR] No audio files remaining after filtering MP3 files!")
                return
    
    print(f"Found {len(audio_files)} audio file(s):")
    for audio_file in audio_files:
        print(f"  - {audio_file.name}")
    
    # Create audio entries with different start times, durations, and intensities
    # This will create an interesting mix with overlapping sounds
    audio_entries = []
    
    # Calculate spacing - each sound starts 2 seconds after the previous one
    # But we'll also add some overlapping sounds for testing
    current_time = 0.0
    
    for i, audio_file in enumerate(audio_files):
        # Vary intensity: first sound at 1.0, second at 0.8, third at 0.6, etc.
        intensity = max(0.3, 1.0 - (i * 0.2))
        
        # Use a duration of 3 seconds for each sound (or full length if shorter)
        duration = 3.0
        
        entry = AudioEntry(
            StartTime=current_time,
            Duration=duration,
            Intensity=intensity,
            Data=str(audio_file)
        )
        audio_entries.append(entry)
        
        print(f"\nEntry {i+1}:")
        print(f"  File: {audio_file.name}")
        print(f"  StartTime: {current_time}s")
        print(f"  Duration: {duration}s")
        print(f"  Intensity: {intensity}")
        
        # Next sound starts 2 seconds after current one (creates 1 second overlap)
        current_time += 2.0
    
    # Add some additional overlapping sounds for more complex mixing
    if len(audio_files) > 1:
        # Add first sound again at a later time with lower intensity
        audio_entries.append(AudioEntry(
            StartTime=current_time + 1.0,
            Duration=2.0,
            Intensity=0.5,
            Data=str(audio_files[0])
        ))
        print(f"\nEntry {len(audio_entries)} (overlapping):")
        print(f"  File: {audio_files[0].name}")
        print(f"  StartTime: {current_time + 1.0}s")
        print(f"  Duration: 2.0s")
        print(f"  Intensity: 0.5")
    
    # Merge all audio files
    print("\n" + "="*50)
    print("Merging audio files...")
    output_path = "test_merged_output.wav"
    
    # Check if we have MP3 files and warn about ffmpeg
    has_mp3 = any(str(f).lower().endswith('.mp3') for f in audio_files)
    if has_mp3:
        print("\n[NOTE] MP3 files require ffmpeg to be installed.")
        print("   If you encounter errors, install ffmpeg from: https://ffmpeg.org/download.html")
        print("   Or use: choco install ffmpeg (if you have Chocolatey)")
    
    try:
        merged_audio = merge_audio_files(audio_entries, output_path)
        
        # Get output file info
        output_file = Path(output_path)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        duration_seconds = len(merged_audio) / 1000.0
        
        print(f"\n[SUCCESS] Successfully merged {len(audio_entries)} audio entries!")
        print(f"  Output file: {output_path}")
        print(f"  Duration: {duration_seconds:.2f} seconds")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Sample rate: {merged_audio.frame_rate} Hz")
        print(f"  Channels: {merged_audio.channels}")
        
    except FileNotFoundError as e:
        error_msg = str(e)
        if "ffmpeg" in error_msg.lower() or "ffprobe" in error_msg.lower() or "introuvable" in error_msg.lower():
            print(f"\n[ERROR] ffmpeg is required to process MP3 files but was not found.")
            print("\nTo fix this:")
            print("  1. Download ffmpeg from: https://ffmpeg.org/download.html")
            print("  2. Extract and add to your system PATH")
            print("  3. Or install via Chocolatey: choco install ffmpeg")
            print("\nAlternatively, convert MP3 files to WAV format first.")
        else:
            print(f"\n[ERROR] FileNotFoundError: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Error during merging: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise


def test_with_audio_segment():
    """
    Test merging with AudioSegment objects (not just file paths).
    """
    print("\n" + "="*50)
    print("Testing with AudioSegment objects...")
    
    from pydub import AudioSegment
    
    audio_files = get_audio_files_from_bank()
    if not audio_files:
        print("[ERROR] No audio files found for AudioSegment test!")
        return
    
    # Filter to only WAV files (they don't require ffmpeg)
    wav_files = [f for f in audio_files if str(f).lower().endswith('.wav')]
    if not wav_files:
        print("[WARNING] No WAV files found. This test requires WAV files (MP3 requires ffmpeg).")
        return
    
    # Load first WAV file as AudioSegment
    audio_segment = AudioSegment.from_file(str(wav_files[0]))
    
    # Create entry with AudioSegment instead of file path
    entry = AudioEntry(
        StartTime=0.0,
        Duration=2.0,
        Intensity=0.7,
        Data=audio_segment  # Using AudioSegment directly
    )
    
    try:
        merged_audio = merge_audio_files([entry], "test_audiosegment_output.wav")
        print(f"[SUCCESS] Successfully merged AudioSegment object!")
        print(f"  Output file: test_audiosegment_output.wav")
        print(f"  Duration: {len(merged_audio) / 1000.0:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        raise


if __name__ == "__main__":
    print("="*50)
    print("Audio Merger Test Suite")
    print("="*50)
    
    # Test 1: Merge all sounds from bank folder
    test_merge_all_sounds()
    
    # Test 2: Test with AudioSegment objects
    test_with_audio_segment()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)

