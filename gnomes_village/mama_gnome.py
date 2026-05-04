from mlx_lm import load as mlx_load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-mxfp4
model_repo = 'mlx-community/Qwen3-4B-Instruct-2507-mxfp4'

def summon_mana_gnome():
    model, tokenizer = mlx_load(model_repo)
    return model, tokenizer


def summarize(model, tokenizer, chat_history: str) -> str:

    sys_prompt = """
    You are a context compression assistant. Your job is to turn a chat history into a dense, actionable summary.

    Rules:
    - One to three sentences maximum.
    - Preserve specific facts: file paths, commands, numbers, URLs, outcomes.
    - Preserve user decisions, preferences, and pivots (e.g., "user changed mind from X to Y").
    - Drop noise: pleasantries, repetitive tool outputs, failed attempts that were abandoned.
    - Use precise, terse language. No fluff.

    Examples of good summaries:

    Example 1 (coding / bug fix):
    The user initially asked to refactor tool_registry.py for async dispatch, but pivoted to fixing a JSON parsing bug in papa_gnome.py where tool arguments containing literal newlines broke tool_call_extract(). A regex sanitiser was added and verified working.

    Example 2 (web research + local config):
    Researched Vietnam travel: 45-day visa-free entry for Polish citizens saved to ~/.gnomes/memory/travel.md, Hanoi December weather is dry season at 18-22C. Verified ffmpeg v6.1 is installed locally.

    Example 3 (system maintenance):
    Cleaned up Docker images (freed 12GB), identified 3 large repos in ~/projects and archived two to external drive via tar. User preference noted: backups should go to /Volumes/Backup/ rather than Desktop.
    """

    user_prompt = f"""Summarize the following chat history:
    {chat_history}
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False,
        sampler=make_sampler(temp=0.3, top_p=0.95, min_p=0.05, top_k=20),
        # logits_processors=make_logits_processors(repetition_penalty=1.5),
    )

    return response
