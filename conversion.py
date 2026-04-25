from mlx_lm import convert
from huggingface_hub import HfApi
import os

# https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
# HF_REPO = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
# OUTPUT_PATH = "./models/Qwen3.5-9B-reasoning-4bit"
# REPO_NAME = "Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit"

HF_REPO = "Qwen/Qwen3.5-9B"
OUTPUT_PATH = "./models/Qwen3.5-9B-4bit"
REPO_NAME = "Qwen3.5-9B-MLX-4bit"



README = f"""---
license: apache-2.0
base_model: Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
tags:
  - mlx
  - quantized
  - 4-bit
  - reasoning
---

# {REPO_NAME}

4-bit MLX quantization of {REPO_NAME}

## Conversion
Quantized using `mlx_lm.convert` with 4-bit quantization (q_bits=4, q_group_size=64)

## Usage
```python
from mlx_lm import load, generate

model, tokenizer = load("username/repo_name")
response = generate(model, tokenizer, prompt="Hello!", verbose=True)
```

## Original Model
See {HF_REPO} for full details.

## License
Apache 2.0 (inherited from original)
"""

convert(
    hf_path=HF_REPO,
    mlx_path=OUTPUT_PATH,
    quantize=True,
    q_bits=4,
    q_group_size=64,
)
print(f"Done. MLX model saved to: {OUTPUT_PATH}")

# Upload to HuggingFace
api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/{REPO_NAME}"

print(f"\nUploading to {repo_id} ...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Write README into the model folder so it gets uploaded with the weights
readme_path = os.path.join(OUTPUT_PATH, "README.md")
with open(readme_path, "w") as f:
    f.write(README.replace("{username}", username).replace("{repo_name}", REPO_NAME))

api.upload_folder(
    folder_path=OUTPUT_PATH,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add MLX 4-bit quantized model",
)
print(f"Uploaded: https://huggingface.co/{repo_id}")
