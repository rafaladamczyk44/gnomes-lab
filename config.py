from dataclasses import dataclass

@dataclass
class Config:
    main_model: str =  'rafal-adamczyk/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit'
    small_model: str = 'mlx-community/Qwen3-4B-Instruct-2507-mxfp4'
    coder_model: str = 'mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit'

    max_context_tokens: int = 120000 # keep it capped to avoid drawning context
    compact_threshold: int = max_context_tokens * 0.8

    web_search_cache_ttl: int = 600