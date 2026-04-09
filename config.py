from dataclasses import dataclass

@dataclass
class Config:
    main_model: str = './models/Qwen3.5-9B-reasoning-4bit'
    small_model: str = 'mlx-community/Qwen3-4B-Instruct-2507-mxfp4'