"""
Audio Job Worker

Monitors the jobs folder for new job directories, processes them using the audio merger,
and renames completed jobs with "_completed" suffix.
"""
import json
import time
import os
from pathlib import Path
from typing import List, Optional
from audio_merger import AudioEntry, merge_audio_files


def load_job_metadata(metadata_path: Path) -> List[dict]:
    """
    Load job metadata from a JSON file.
    
    Args:
        metadata_path: Path to the metadata.json file
        
    Returns:
        List of metadata entries
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If metadata file is invalid JSON
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if not isinstance(metadata, list):
        raise ValueError(f"Metadata must be a list, got {type(metadata)}")
    
    return metadata


def create_audio_entries(metadata: List[dict], job_folder: Path) -> List[AudioEntry]:
    """
    Create AudioEntry objects from metadata.
    
    Args:
        metadata: List of metadata dictionaries
        job_folder: Path to the job folder containing audio files
        
    Returns:
        List of AudioEntry objects
        
    Raises:
        ValueError: If metadata is invalid
        FileNotFoundError: If audio file doesn't exist
    """
    audio_entries = []
    
    for i, entry in enumerate(metadata):
        # Validate required fields
        required_fields = ['startTime', 'duration', 'intensity', 'audioFile']
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Entry {i} missing required field: {field}")
        
        # Get audio file path
        audio_file_path = entry['audioFile']
        
        # Handle relative paths - if it's relative, resolve it relative to job folder
        if os.path.isabs(audio_file_path):
            audio_path = Path(audio_file_path)
        else:
            # Try relative to job folder first
            audio_path = job_folder / audio_file_path
            # If not found, try relative to current working directory
            if not audio_path.exists():
                audio_path = Path(audio_file_path)
        
        # Check if file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create AudioEntry
        audio_entry = AudioEntry(
            StartTime=float(entry['startTime']),
            Duration=float(entry['duration']),
            Intensity=float(entry['intensity']),
            Data=str(audio_path)
        )
        
        audio_entries.append(audio_entry)
    
    return audio_entries


def process_job(job_folder: Path, jobs_base_path: Path) -> bool:
    """
    Process a single job folder.
    
    Args:
        job_folder: Path to the job folder to process
        jobs_base_path: Base path to the jobs directory
        
    Returns:
        True if job was processed successfully, False otherwise
    """
    job_name = job_folder.name
    print(f"\n{'='*60}")
    print(f"Processing job: {job_name}")
    print(f"{'='*60}")
    
    # Check for metadata.json
    metadata_path = job_folder / "metadata.json"
    if not metadata_path.exists():
        print(f"[WARNING] No metadata.json found in {job_name}, skipping...")
        return False
    
    try:
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        metadata = load_job_metadata(metadata_path)
        print(f"Found {len(metadata)} audio entries in metadata")
        
        # Create audio entries
        print("Creating audio entries...")
        audio_entries = create_audio_entries(metadata, job_folder)
        
        # Determine output filename (use job name + "_merged.wav")
        output_filename = f"{job_name}_merged.wav"
        output_path = job_folder / output_filename
        
        # Merge audio files
        print(f"Merging audio files to {output_filename}...")
        merge_audio_files(audio_entries, str(output_path))
        
        print(f"[SUCCESS] Job {job_name} processed successfully!")
        print(f"  Output file: {output_path}")
        
        # Rename folder with "_completed" suffix
        completed_folder_name = f"{job_name}_completed"
        completed_folder_path = jobs_base_path / completed_folder_name
        
        # Check if completed folder already exists
        if completed_folder_path.exists():
            print(f"[WARNING] Completed folder already exists: {completed_folder_name}")
            print(f"  Skipping rename to avoid overwriting existing folder")
            return True
        
        # Rename the folder
        print(f"Renaming folder to {completed_folder_name}...")
        job_folder.rename(completed_folder_path)
        print(f"[SUCCESS] Folder renamed to {completed_folder_name}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return False
    except ValueError as e:
        print(f"[ERROR] Invalid metadata: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error processing job {job_name}: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def find_pending_jobs(jobs_path: Path) -> List[Path]:
    """
    Find all job folders that haven't been completed yet.
    
    Args:
        jobs_path: Path to the jobs directory
        
    Returns:
        List of Path objects for pending job folders
    """
    pending_jobs = []
    
    if not jobs_path.exists():
        return pending_jobs
    
    # Iterate through all directories in jobs folder
    for item in jobs_path.iterdir():
        if item.is_dir():
            # Skip folders that already have "_completed" in the name
            if "_completed" not in item.name:
                pending_jobs.append(item)
    
    return pending_jobs


def watch_jobs_folder(
    jobs_folder: str = "jobs",
    poll_interval: float = 5.0,
    run_once: bool = False
):
    """
    Monitor the jobs folder for new jobs and process them.
    
    This function continuously monitors the jobs folder for new job directories.
    When a new job is found, it processes it using the audio merger and renames
    the folder with "_completed" suffix.
    
    Args:
        jobs_folder: Path to the jobs folder (relative to script location or absolute)
        poll_interval: How often to check for new jobs in seconds (default: 5.0)
        run_once: If True, process all pending jobs once and exit. If False, run continuously.
    """
    # Get the script directory and resolve jobs path
    script_dir = Path(__file__).parent
    jobs_path = (script_dir / jobs_folder).resolve()
    
    print(f"Audio Job Worker")
    print(f"{'='*60}")
    print(f"Jobs folder: {jobs_path}")
    print(f"Poll interval: {poll_interval} seconds")
    print(f"Mode: {'Run once' if run_once else 'Continuous monitoring'}")
    print(f"{'='*60}")
    
    if not jobs_path.exists():
        print(f"[ERROR] Jobs folder does not exist: {jobs_path}")
        print(f"  Creating jobs folder...")
        jobs_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    try:
        while True:
            # Find pending jobs
            pending_jobs = find_pending_jobs(jobs_path)
            
            if pending_jobs:
                print(f"\n[INFO] Found {len(pending_jobs)} pending job(s)")
                for job in pending_jobs:
                    success = process_job(job, jobs_path)
                    if success:
                        processed_count += 1
            else:
                if run_once:
                    print(f"\n[INFO] No pending jobs found. Exiting...")
                    break
                else:
                    print(f"[INFO] No pending jobs. Waiting {poll_interval}s...", end='\r')
            
            if run_once:
                break
            
            # Wait before next check
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print(f"\n\n[INFO] Worker stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n[INFO] Processed {processed_count} job(s) total")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Job Worker - Process audio merging jobs")
    parser.add_argument(
        "--jobs-folder",
        type=str,
        default="jobs",
        help="Path to the jobs folder (default: 'jobs')"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process all pending jobs once and exit (default: run continuously)"
    )
    
    args = parser.parse_args()
    
    watch_jobs_folder(
        jobs_folder=args.jobs_folder,
        poll_interval=args.poll_interval,
        run_once=args.once
    )

