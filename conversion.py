from mlx_lm import convert

# https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
HF_REPO = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
OUTPUT_PATH = "./models/Qwen3.5-9B-reasoning-4bit"

# TODO: push to HF

convert(
    hf_path=HF_REPO,
    mlx_path=OUTPUT_PATH,
    quantize=True,        # 4-bit
    q_bits=4,
    q_group_size=64,
)

print(f"Done. MLX model saved to: {OUTPUT_PATH}")