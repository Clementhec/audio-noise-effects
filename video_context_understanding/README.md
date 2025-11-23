# Video Understanding with Video-LLaVA

This module provides a standalone script to analyze MP4 video files using the HuggingFace Video-LLaVA model, generating concise text descriptions of video content.

## Features

- Process MP4 video files to extract semantic understanding
- Uses state-of-the-art Video-LLaVA multimodal model
- Generates concise, detailed text descriptions
- Saves metadata and results to JSON format
- Customizable prompts for specific analysis needs
- Automatic frame sampling for efficient processing

## Usage

### Basic Usage

Analyze a video with default settings:

```bash
python video_llava_analyzer.py path/to/video.mp4
```

and provide a specific question or analysis prompt:

```bash
python video_llava_analyzer.py video.mp4 --prompt "What is the main activity in this video?"
```

### Advanced Options

```bash
python video_llava_analyzer.py video.mp4 \
    --prompt "Describe the scene and actions" \
    --output-dir ./my_results \
    --frames 16 \
    --max-tokens 300
```

## Output Format

Results are saved as JSON files in the output directory with the following structure:

```json
{
  "video_path": "/absolute/path/to/video.mp4",
  "video_name": "video",
  "timestamp": "20250115_143022",
  "prompt": "Describe the content of this video in detail.",
  "description": "The video shows...",
  "model": "LanguageBind/Video-LLaVA-7B-hf"
}
```

Output files are named: `{video_name}_analysis_{timestamp}.json`

## How It Works

1. **Video Loading**: Opens the MP4 file using PyAV
2. **Frame Sampling**: Extracts a specified number of frames uniformly distributed across the video
3. **Model Processing**: Feeds frames to Video-LLaVA model with the prompt
4. **Text Generation**: Model generates a natural language description
5. **Output Saving**: Saves results to JSON in the output directory


## Alternative Models

You can specify different Video-LLaVA variants:

```bash
# LLaVA-Next-Video (state-of-the-art)
python video_llava_analyzer.py video.mp4 --model "llava-hf/LLaVA-NeXT-Video-7B-hf"

# LLaVA-Video with Qwen2 (longer context)
python video_llava_analyzer.py video.mp4 --model "lmms-lab/LLaVA-Video-7B-Qwen2" --frames 64
```

## References and guides

- [Video-LLaVA Documentation](https://huggingface.co/docs/transformers/en/model_doc/video_llava)
- [Video-LLaVA GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA)
- [Model on HuggingFace](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)

Google Cloud : 
- [introduction to the CLI](https://docs.cloud.google.com/storage/docs/discover-object-storage-gcloud)
