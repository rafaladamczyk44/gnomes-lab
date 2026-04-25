from dataclasses import dataclass

@dataclass
class Config:
    main_model: str =  'rafal-adamczyk/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit' # 'rafal-adamczyk/Qwen3.5-9B-MLX-4bit'
    small_model: str = 'mlx-community/Qwen3-4B-Instruct-2507-mxfp4'