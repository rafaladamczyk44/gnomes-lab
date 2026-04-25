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


def compact_tool_output(model, tokenizer, tool_output: str) -> str:
    sys_prompt = """
    You are a tool output compressor. Your job is to take long tool output and compress it into the minimum number of tokens while keeping every fact the downstream agent needs to act.

    Rules:
    - Preserve all specific facts: numbers, dates, file paths, URLs, error messages, return codes.
    - Preserve code structure if the output is code: keep function signatures, key lines, indentation.
    - Strip boilerplate: HTML tags, ads, navigation, repeated headers, copyright notices, markdown formatting noise.
    - Collapse lists: if a search returns 10 similar results, keep the 2-3 most relevant and note the count.
    - For errors: keep the error type and the key line; drop the stack trace unless it contains unique info.
    - Output ONLY the compressed text. No meta commentary like "Here is the summary:".
    - Be aggressive: cut 80% of tokens if possible without losing actionable information.

    Examples:

    Example 1 — web_search (before):
    [Tool: web_search]
    1. Title: "Python 3.12 Release Notes"
    URL: https://docs.python.org/3.12/whatsnew/...
    Snippet: "The latest version of Python includes improvements to the interpreter, faster startup times, and better error messages. In this article we will explore... Posted on 2023-10-02 by the Python Software Foundation. Read more..."
    2. Title: "What's New In Python 3.12"
    URL: https://...
    Snippet: "Python 3.12 introduces PEP 701 for f-strings, PEP 684 for isolated subinterpreters, and PEP 669 for low-cost debugging. This release also..."
    [5 more results]

    Example 1 — web_search (after):
    Python 3.12 (2023-10-02): PEP 701 f-strings, PEP 684 isolated subinterpreters, PEP 669 low-cost debugging. Faster startup, better error messages.

    Example 2 — read_file (before):
    [Tool: read_file — main.py]
    import os
    import sys
    # This is the main entry point for the application.
    # It handles argument parsing and sets up the logging.
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args()
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        run()
    if __name__ == "__main__":
        main()

    Example 2 — read_file (after):
    [Tool: read_file — main.py]
    import os, sys
    def main(): parses --verbose, sets logging level, calls run()

    Example 3 — bash_exec (before):
    [Tool: bash_exec]
    total 128
    drwxr-xr-x  12 rafal  staff   384 Apr 10 09:23 .
    drwxr-xr-x  45 rafal  staff  1440 Apr 10 09:20 ..
    -rw-r--r--   1 rafal  staff  2048 Apr 10 09:22 README.md
    -rw-r--r--   1 rafal  staff   512 Apr 10 09:21 main.py
    -rw-r--r--   1 rafal  staff  1024 Apr 10 09:21 config.py
    -rw-r--r--   1 rafal  staff  2048 Apr 10 09:21 requirements.txt

    Example 3 — bash_exec (after):
    [Tool: bash_exec]
    Files: README.md (2KB), main.py (512B), config.py (1KB), requirements.txt (2KB). Total 4 files, 128B dir.
    """

    user_prompt = f"""Compress the following tool output:

    {tool_output}
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
        sampler=make_sampler(temp=0.2, top_p=0.95, min_p=0.05, top_k=20),
    )

    return response
