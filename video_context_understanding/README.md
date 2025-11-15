# Video Understanding with Video-LLaVA

This module provides a standalone script to analyze MP4 video files using the HuggingFace Video-LLaVA model, generating concise text descriptions of video content.

## Features

- Process MP4 video files to extract semantic understanding
- Uses state-of-the-art Video-LLaVA multimodal model
- Generates concise, detailed text descriptions
- Saves metadata and results to JSON format
- Customizable prompts for specific analysis needs
- Automatic frame sampling for efficient processing

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. System Requirements

- **GPU**: Recommended (CUDA-compatible). The model will use CPU if GPU is unavailable, but processing will be slower.
- **RAM**: Minimum 16GB recommended for the 7B model
- **Storage**: ~14GB for the model weights

### 3. First Run

On first run, the script will automatically download the Video-LLaVA model (~14GB). This may take some time depending on your internet connection.

## Usage

### Basic Usage

Analyze a video with default settings:

```bash
python video_llava_analyzer.py path/to/video.mp4
```

Or use the executable directly:

```bash
./video_llava_analyzer.py path/to/video.mp4
```

### Custom Prompt

Provide a specific question or analysis prompt:

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

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `video_path` | str | (required) | Path to the input MP4 video file |
| `--prompt` | str | None | Custom prompt for analysis |
| `--output-dir` | str | `output` | Directory to save results |
| `--frames` | int | 8 | Number of frames to sample |
| `--max-tokens` | int | 200 | Maximum tokens to generate |
| `--model` | str | `LanguageBind/Video-LLaVA-7B-hf` | HuggingFace model identifier |

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

## Examples

### Example 1: General Description

```bash
python video_llava_analyzer.py cooking_tutorial.mp4
```

### Example 2: Specific Question

```bash
python video_llava_analyzer.py sports_clip.mp4 --prompt "What sport is being played and what is happening?"
```

### Example 3: Detailed Analysis with More Frames

```bash
python video_llava_analyzer.py nature_documentary.mp4 \
    --frames 16 \
    --max-tokens 400 \
    --prompt "Describe the wildlife and environment shown"
```

### Example 4: Custom Output Location

```bash
python video_llava_analyzer.py meeting_recording.mp4 \
    --output-dir ./meeting_analyses \
    --prompt "Summarize the key points discussed"
```

## How It Works

1. **Video Loading**: Opens the MP4 file using PyAV
2. **Frame Sampling**: Extracts a specified number of frames uniformly distributed across the video
3. **Model Processing**: Feeds frames to Video-LLaVA model with the prompt
4. **Text Generation**: Model generates a natural language description
5. **Output Saving**: Saves results to JSON in the output directory

## Troubleshooting

### Out of Memory Error

If you encounter OOM errors:

```bash
# Reduce number of frames
python video_llava_analyzer.py video.mp4 --frames 4

# Use a smaller model (if available)
# Or ensure other applications aren't using GPU memory
```

### Slow Processing

- First run will download the model (~14GB)
- CPU processing is much slower than GPU
- Longer videos with more frames take more time

### CUDA Not Available

The script will automatically fall back to CPU if CUDA is not available. For GPU support, ensure:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Model Information

**Default Model**: `LanguageBind/Video-LLaVA-7B-hf`

- **Parameters**: 7 billion
- **Architecture**: Video-LLaVA (unified visual representation)
- **Capabilities**: Video and image understanding
- **Context**: Processes 8 frames by default

## Alternative Models

You can specify different Video-LLaVA variants:

```bash
# LLaVA-Next-Video (state-of-the-art)
python video_llava_analyzer.py video.mp4 --model "llava-hf/LLaVA-NeXT-Video-7B-hf"

# LLaVA-Video with Qwen2 (longer context)
python video_llava_analyzer.py video.mp4 --model "lmms-lab/LLaVA-Video-7B-Qwen2" --frames 64
```

## License

This script uses the Video-LLaVA model from HuggingFace. Please refer to the model's license on HuggingFace for usage terms.

## References

- [Video-LLaVA Documentation](https://huggingface.co/docs/transformers/en/model_doc/video_llava)
- [Video-LLaVA GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA)
- [Model on HuggingFace](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
