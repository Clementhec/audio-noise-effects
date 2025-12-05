import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from main import (
    setup_directories,
    extract_audio,
    run_stt_step,
    run_embeddings_step,
    run_semantic_matching_step,
    run_llm_filtering_step,
)

app = FastAPI(
    title="Audio Noise Effects API",
    description="API for audio processing and adding sound effects",
    version="1.0.0"
)


@app.get("/")
async def hello_world():
    """
    Simple test endpoint that returns a Hello World message
    """
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    """
    API health check endpoint
    """
    return {"status": "healthy"}


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """
    Endpoint to upload a .mp4 video
    
    Args:
        video: Video file in .mp4 format
        
    Returns:
        JSON indicating whether the video was received or not
    """
    try:
        # Check if a file was sent
        if not video:
            return JSONResponse(
                status_code=400,
                content={"status": "video not received", "error": "No file provided"}
            )
        
        # Check file extension
        if not video.filename.endswith('.mp4'):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "video not received", 
                    "error": "File must be in .mp4 format",
                    "filename": video.filename
                }
            )
        
        # Check MIME type
        if video.content_type not in ["video/mp4", "application/octet-stream"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "video not received",
                    "error": f"Invalid file type: {video.content_type}",
                    "filename": video.filename
                }
            )
        
        # Read content to verify the file is not empty
        contents = await video.read()
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "video not received",
                    "error": "File is empty",
                    "filename": video.filename
                }
            )
        
        # If everything is OK, return success
        return JSONResponse(
            status_code=200,
            content={
                "status": "video received",
                "filename": video.filename,
                "size_bytes": len(contents),
                "content_type": video.content_type
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "video not received",
                "error": f"Error during processing: {str(e)}"
            }
        )


def cleanup_temp_dir(temp_dir: str):
    """Clean up the temporary directory after processing."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temporary directory: {e}")


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    top_k: int = Query(default=5, description="Number of similar sounds to find per segment"),
    max_sounds: Optional[int] = Query(default=None, description="Maximum number of sounds to select (None = LLM decides)"),
):
    """
    Endpoint to process a video and get suggested sound effects.
    
    Pipeline executed:
    1. Audio extraction from video
    2. Speech-to-Text (transcription with word-level timing)
    3. Speech embeddings generation
    4. Semantic matching with sound effects
    5. LLM filtering to select the best matches
    
    Args:
        video: Video file in .mp4 format
        top_k: Number of similar sounds to find for each segment (default: 5)
        max_sounds: Maximum number of sounds to select (None = LLM decides)
        
    Returns:
        JSON with LLM filtering results containing suggested sound effects
    """
    temp_dir = None
    
    try:
        # Video file validation
        if not video:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not video.filename.endswith('.mp4'):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be in .mp4 format, received: {video.filename}"
            )
        
        if video.content_type not in ["video/mp4", "application/octet-stream"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {video.content_type}"
            )
        
        # Read video content
        video_contents = await video.read()
        if len(video_contents) == 0:
            raise HTTPException(status_code=400, detail="Video file is empty")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="audio_noise_effects_")
        output_dir = Path(temp_dir) / "data"
        
        # Save video temporarily
        temp_video_path = Path(temp_dir) / video.filename
        with open(temp_video_path, "wb") as f:
            f.write(video_contents)
        
        # Configure directories
        setup_directories(output_dir=str(output_dir))
        
        # Define intermediate file paths
        base_name = temp_video_path.stem
        audio_output_dir = output_dir / "audio"
        transcription_path = output_dir / f"speech_to_text/{base_name}_full_transcription.json"
        word_timing_path = output_dir / f"speech_to_text/{base_name}_word_timing.json"
        embeddings_path = output_dir / f"embeddings/{base_name}_video_speech_embeddings.csv"
        similarity_results_path = output_dir / f"similarity/{base_name}_similarity.json"
        filtered_results_path = output_dir / f"filtered/{base_name}_video_filtered_sounds.json"
        
        # Pre-computed sound embeddings paths (in main data directory)
        main_data_dir = Path("data")
        soundbible_embeddings_path = main_data_dir / "embeddings/soundbible.csv"
        
        # Check that sound embeddings exist
        if not soundbible_embeddings_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Pre-computed sound embeddings are missing: {soundbible_embeddings_path}"
            )
        
        # STEP 1: Audio extraction
        print("=" * 70)
        print("STEP 1: Audio extraction from video")
        print("=" * 70)
        audio_path = extract_audio(
            str(temp_video_path),
            output_dir=str(audio_output_dir),
            sample_rate=16000,
            channels=1,
        )
        
        # STEP 2: Speech-to-Text
        print("=" * 70)
        print("STEP 2: Speech-to-Text")
        print("=" * 70)
        transcription_path, word_timing_path = run_stt_step(
            audio_path=audio_path,
            transcription_path=transcription_path,
            word_timing_path=word_timing_path,
        )
        
        # STEP 3: Speech embeddings generation
        print("=" * 70)
        print("STEP 3: Speech embeddings generation")
        print("=" * 70)
        embeddings_path = run_embeddings_step(
            transcription_path=transcription_path,
            word_timing_path=word_timing_path,
            embeddings_path=embeddings_path,
            force_regenerate=True,
        )
        
        # STEP 4: Semantic matching
        print("=" * 70)
        print("STEP 4: Semantic matching with sound effects")
        print("=" * 70)
        similarity_results_path = run_semantic_matching_step(
            embeddings_path=embeddings_path,
            similarity_results_path=similarity_results_path,
            sound_embeddings_path=soundbible_embeddings_path,
            soundbible_metadata_path=main_data_dir / "input/soundbible_metadata.csv",
            top_k=top_k,
        )
        
        # STEP 5: LLM filtering
        print("=" * 70)
        print("STEP 5: LLM filtering")
        print("=" * 70)
        filtered_results_path = run_llm_filtering_step(
            similarity_results_path=similarity_results_path,
            filtered_results_path=filtered_results_path,
            max_sounds=max_sounds,
        )
        
        # Load and return filtered results
        with open(filtered_results_path, "r", encoding="utf-8") as f:
            filtered_results = json.load(f)
        
        # Schedule temporary directory cleanup in background
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": video.filename,
                "pipeline_steps": [
                    "audio_extraction",
                    "speech_to_text",
                    "speech_embedding",
                    "semantic_matching",
                    "llm_filtering"
                ],
                "parameters": {
                    "top_k": top_k,
                    "max_sounds": max_sounds
                },
                "results": filtered_results
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_dir:
            cleanup_temp_dir(temp_dir)
        raise
    except Exception as e:
        # Clean up on error
        if temp_dir:
            cleanup_temp_dir(temp_dir)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Error processing video: {str(e)}"
            }
        )
