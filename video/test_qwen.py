import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import (
    TorchAoConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

# quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    # quantization_config=quantization_config
)

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "../../Audio-noise-effects/chaplin_speech.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)
