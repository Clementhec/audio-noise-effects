import os
import json
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from main import SoundEasy
import argparse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    user_prompt: str = Form(""),
    max_sounds: int = Form(None),
    sound_intensity: float = Form(0.3),
):
    try:
        # Save uploaded video
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / video.filename
        
        print(f"Processing video: {video.filename}")
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Setup args - Run pipeline until LLM filtering (NO video merge)
        args = argparse.Namespace(
            video=str(video_path),
            user_prompt=user_prompt,
            full_pipeline=False,
            run_stt=True,
            run_embeddings=True,
            run_matching=True,
            run_llm_filter=True,
            run_video_merge=False,  # Don't merge - return JSON instead
            output_dir="data",
            top_k=5,
            sample_rate=16000,
            channels=1,
            max_sounds=max_sounds,
            sound_intensity=sound_intensity,
            sound_duration=None,
            force_regenerate=True,
        )
        
        # Run pipeline (stops after LLM filtering)
        pipe = SoundEasy(
            video_path=video_path,
            data_dir=Path(args.output_dir),
        )
        pipe.run(args)
        
        # Load filtered results and word timings to build audio blocks JSON
        with open(pipe.filtered_results_path, "r", encoding="utf-8") as f:
            filtered_results = json.load(f)
        
        with open(pipe.word_timing_path, "r", encoding="utf-8") as f:
            word_timings = json.load(f)
            
        # Load embeddings to get segment timing
        embeddings_df = pd.read_csv(pipe.speech_embeddings_path)
        
        # Convert filtered sounds to AudioBlock format for frontend
        audio_blocks = []
        for idx, item in enumerate(filtered_results.get("filtered_sounds", [])):
            if not item.get("should_add_sound", False):
                continue
                
            selected_sound = item.get("selected_sound")
            if not selected_sound:
                continue
            
            sound_title = selected_sound.get("sound_title")
            target_word = item.get("target_word")
            speech_text = item.get("speech_text")
            segment_id = item.get("speech_index", 0)
            
            # Get segment timing from embeddings
            segment_id = min(segment_id, len(embeddings_df) - 1)
            segment_row = embeddings_df[embeddings_df["segment_id"] == segment_id]
            
            if len(segment_row) == 0:
                continue
                
            sentence_start_time = segment_row["start_time"].iloc[0]
            sentence_end_time = segment_row["end_time"].iloc[0]
            
            # Find word timing for precise placement
            target_normalized = target_word.strip().lower().rstrip(".,!?;:")
            start_time = sentence_start_time  # Default to sentence start
            
            def parse_time_string(time_str: str) -> float:
                return float(time_str.rstrip("s"))
            
            # Find exact word timing
            for wt in word_timings:
                wt_start = parse_time_string(wt["startTime"])
                if wt_start >= sentence_start_time and wt_start <= sentence_end_time:
                    if target_normalized == wt["word"].strip().lower().rstrip(".,!?;:"):
                        start_time = wt_start
                        break
            
            # Get sound URL for download on demand
            sound_url = selected_sound.get("audio_url_wav", "")
            
            audio_blocks.append({
                "id": f"sound-{idx}",
                "name": sound_title or f"Sound {idx + 1}",
                "start": round(start_time, 2),
                "duration": 3.0,  # Default duration, user can adjust in editor
                "volume": int(sound_intensity * 100),
                "audioUrl": sound_url,
                "targetWord": target_word,
                "speechText": speech_text,
            })
        
        print(f"✓ Generated {len(audio_blocks)} audio blocks for timeline editor")
        
        response = {
            "success": True,
            "videoFile": video.filename,
            "audioBlocks": audio_blocks,
            "metadata": {
                "totalSounds": len(audio_blocks),
                "videoDuration": embeddings_df["end_time"].max() if len(embeddings_df) > 0 else 0,
            }
        }
        
        # Log the complete HTTP response for debugging
        print("\n" + "="*70)
        print("HTTP RESPONSE - Audio Blocks JSON")
        print("="*70)
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("="*70 + "\n")
        
        return response
        
    except ValueError as e:
        # User-facing errors (empty transcript, no speech, etc.)
        error_msg = str(e)
        print(f"⚠️  User error in process_video: {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": error_msg}
        )
    except Exception as e:
        # System errors
        print(f"❌ ERROR in process_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Internal server error: {str(e)}"}
        )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

