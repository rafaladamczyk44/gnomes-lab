from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate

def text(result) -> str:
    """Extract plain text from mlx_vlm GenerationResult or a plain string."""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    return str(result)


def generate_response(model, processor, config, prompt: str, system_prompt: str = None, max_tokens: int = 10000, **kwargs) -> str:
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    formatted = apply_chat_template(processor, config, messages, num_images=0)

    return text(generate(model, processor, formatted, max_tokens=max_tokens, verbose=False, **kwargs))