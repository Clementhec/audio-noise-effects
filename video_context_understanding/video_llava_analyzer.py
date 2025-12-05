#!/usr/bin/env python3
"""
Video Understanding using HuggingFace Video-LLaVA model.

This script takes an MP4 video file as input and uses the Video-LLaVA model
to generate a concise text description of the video content.

Usage:
    python video_llava_analyzer.py <path_to_video.mp4> [--prompt "custom prompt"]

Example:
    python video_llava_analyzer.py ./example.mp4
    python video_llava_analyzer.py ./example.mp4 --prompt "Describe the main activity in this video"
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor


def read_video_pyav(container, indices):
    """
    Read specific frames from a video container.

    Args:
        container: PyAV video container
        indices: Frame indices to extract

    Returns:
        np.ndarray: Stacked RGB frames
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_model(
    model_name: str = "LanguageBind/Video-LLaVA-7B-hf", device: str = "auto"
):
    """
    Load the Video-LLaVA model and processor.

    Args:
        model_name: HuggingFace model identifier
        device: Device placement ("auto", "cuda", "cpu")

    Returns:
        tuple: (model, processor)
    """
    print(f"Loading Video-LLaVA model: {model_name}...")

    # Determine dtype based on device availability
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True
    )

    processor = VideoLlavaProcessor.from_pretrained(model_name)

    print(f"Model loaded successfully on {device}")
    return model, processor


def process_video(
    video_path: str,
    model,
    processor,
    prompt: Optional[str] = None,
    num_frames: int = 8,
    max_new_tokens: int = 200,
) -> str:
    """
    Process a video file and generate a text description.

    Args:
        video_path: Path to the MP4 video file
        model: Video-LLaVA model
        processor: Video-LLaVA processor
        prompt: Custom prompt for the model (optional)
        num_frames: Number of frames to sample from the video
        max_new_tokens: Maximum tokens to generate

    Returns:
        str: Generated text description
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Processing video: {video_path}")

    # Load video and sample frames uniformly
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames

    print(f"Total frames in video: {total_frames}")
    print(f"Sampling {num_frames} frames uniformly...")

    # Sample frames uniformly across the video
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    indices = indices[:num_frames]  # Ensure exactly num_frames

    video_frames = read_video_pyav(container, indices)
    print(f"Extracted {len(video_frames)} frames")

    # Default prompt if none provided
    if prompt is None:
        prompt = (
            "USER: <video>\nDescribe the content of this video in detail. ASSISTANT:"
        )
    else:
        prompt = f"USER: <video>\n{prompt} ASSISTANT:"

    print("Generating description...")

    # Process inputs
    inputs = processor(text=prompt, videos=video_frames, return_tensors="pt")

    # Move inputs to the same device as model
    if torch.cuda.is_available():
        inputs = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

    # Generate description
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic output
            temperature=0.0,
        )

    # Decode output
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]

    # Extract only the assistant's response
    if "ASSISTANT:" in generated_text:
        generated_text = generated_text.split("ASSISTANT:")[-1].strip()

    container.close()
    return generated_text


def save_output(
    video_path: str,
    description: str,
    output_dir: str = "output",
    prompt: Optional[str] = None,
) -> str:
    """
    Save the video description to a JSON file in the output directory.

    Args:
        video_path: Original video file path
        description: Generated description text
        output_dir: Output directory name
        prompt: The prompt used (optional)

    Returns:
        str: Path to the saved output file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Generate output filename based on input video
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{video_name}_analysis_{timestamp}.json"

    # Prepare metadata
    metadata = {
        "video_path": str(Path(video_path).absolute()),
        "video_name": video_name,
        "timestamp": timestamp,
        "prompt": prompt or "Default description prompt",
        "description": description,
        "model": "LanguageBind/Video-LLaVA-7B-hf",
    }

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nOutput saved to: {output_file}")
    return str(output_file)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze video content using Video-LLaVA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_llava_analyzer.py video.mp4
  python video_llava_analyzer.py video.mp4 --prompt "What is happening in this video?"
  python video_llava_analyzer.py video.mp4 --output-dir ./results --frames 16
        """,
    )

    parser.add_argument("video_path", type=str, help="Path to the input MP4 video file")

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for video analysis (default: general description)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: ./output)",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Number of frames to sample from video (default: 8)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="LanguageBind/Video-LLaVA-7B-hf",
        help="HuggingFace model name (default: LanguageBind/Video-LLaVA-7B-hf)",
    )

    args = parser.parse_args()

    try:
        # Load model
        model, processor = load_model(args.model)

        # Process video
        description = process_video(
            args.video_path,
            model,
            processor,
            prompt=args.prompt,
            num_frames=args.frames,
            max_new_tokens=args.max_tokens,
        )

        # Print description
        print("\n" + "=" * 80)
        print("VIDEO DESCRIPTION:")
        print("=" * 80)
        print(description)
        print("=" * 80 + "\n")

        # Save output
        output_file = save_output(
            args.video_path, description, args.output_dir, args.prompt
        )

        print(f" Analysis complete!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
