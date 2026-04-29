from config import Config
from mlx_lm import load as mlx_load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from toolz import tool_registry
import datetime as dt
import os
import logging
import transformers
import huggingface_hub.utils as hf_utils

config = Config()


def summon_papa_gnome():
    transformers.logging.set_verbosity_error()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    hf_utils.disable_progress_bars()

    model, tokenizer = mlx_load(
        config.main_model,
    )

    return model, tokenizer


def _format_history_turn(turn: dict) -> str:
    """Format a single history turn including any tool calls for the system prompt."""
    lines = [f"User: {turn['user']}", f"Assistant: {turn['agent']}"]
    tools = turn.get('tools', [])
    if tools:
        parts = []
        for t in tools:
            # Show the tool call and a truncated result preview
            args_short = ', '.join(f"{k}={v!r}" for k, v in t['args'].items())
            res = t['result']
            # Strip known [Tool: ...] prefix for cleaner display
            if res.startswith('[Tool:'):
                res = res.split('\n', 1)[1] if '\n' in res else ''
            if len(res) > 500:
                res = res[:497] + '...'
            parts.append(f"{t['name']}({args_short}) → {res}")
        lines.append(f"[Tools used: {'; '.join(parts)}]")
    return '\n'.join(lines)


def build_messages(user_question: str, global_context: str, context: str, session_history: list[dict], session_summary: str = "") -> list[dict]:
    """
    Helper function to build the messages for the model.
    It includes the system prompt, the user question, and the context.
    Global context is read from ~/.gnomes/context.md (personal preferences).
    Project context is read from the GNOMES.md file in the current directory.
    :param user_question: Question from the traveler
    :param global_context: Personal context from ~/.gnomes/context.md
    :param context: Project context from the GNOMES.md file
    :param session_history: Last N messages between the traveler and you (typically 5)
    :param session_summary: Compressed summary of older turns that no longer fit in the window
    :return: Compiled messages for the model
    """

    summary_prompt = ""
    if session_summary:
        summary_prompt = f"""
        ## Earlier session summary:
        The following is a compressed summary of earlier turns in this session that are no longer in the recent window:
        {session_summary.strip()}
        """

    history_prompt = ""
    if session_history:
        recent = session_history[-5:]
        formatted = "\n\n".join(_format_history_turn(h) for h in recent)
        history_prompt = f"""
        ## Current session history:
        To know the context of the conversation, here is the window of the last {len(recent)} messages between the traveler and you.
        The [Tools used] lines are a log of tools you called in prior turns — brief summaries only, not the full results. Those results are no longer in your active context. Read this history carefully to understand what was already explored and what the user knows. If you need the actual data from a prior turn to answer properly, re-run the relevant tool.
        {formatted}

        Use the history to guide your thinking, especially with follow-up questions.
        """

    # global_context_prompt = ""
    # if global_context:
    #     global_context_prompt = f"""
    #     ## Personal Context
    #     The following are facts about the user and their preferences. These apply across all projects.
    #
    #     {global_context}
    #     """
    #
    # context_prompt = ""
    # if context:
    #     context_prompt = f"""
    #     ## Project Context (GNOMES.md)
    #     The following is the project-specific context for the current working directory. These are binding conventions and project facts — follow them strictly. They override general patterns and defaults.
    #
    #     {context}
    #     """

    sys_prompt = f"""
    ## Identity
    You are Papa Gnome — the eldest and most knowledgeable gnome in the village.
    Your village is located locally on a PC. You are a locally running open-source model: {config.main_model}
    Your job is to answer the questions of any traveler who comes into your village

    ## Working Principles

    **Simplicity First** — applies to explanations and plans, not to skipping required tools.
    Focus on the essence of the problem. No useless abstractions or alternatives no one asked for.
    If your answer is 200 tokens and could be 50, rewrite it.

    **Read before editing.** Always read_file before any edit. Files change.

    **Surgical edits.** edit_file over write_file. Touch only what the task requires. Match existing code style.

    **Follow-ups use history.** When the user says "do it", "write the code", or references something without explaining it — check session history. The subject is almost always there.

    **Batch tool calls.** Emit all needed <tool_call> blocks in one response, not one at a time.

    ## Tools
    - list_files, grep_search, read_file — auto-run; prefer these over bash
    - edit_file — modify existing files
    - write_file — new files only
    - web_search — external/current knowledge; one targeted search, synthesize from it
    - bash_exec — last resort only

    ## Output Format

    CASE 1 — Simple question (no tools needed):
    ## Answer
    <response>

    CASE 2 — Task needing tools (research, file ops, web search):
    ## Plan
    - step 1
    <tool_call>...</tool_call>

    CASE 3a — Write or edit a file (user asks to implement, create, or fix code in a file):
    **MANDATORY:** Before calling write_file or edit_file with code, you MUST output ## Sketch then ## Review in this same response. A tool call without a preceding ## Sketch + ## Review is a format violation.

    ## Sketch
    - brief description of the approach
    ```python
    # key structure, signatures, main logic — not full implementation
    def my_func(args):
        ...
    ```

    ## Review
    Re-read the sketch above. Check for: off-by-ones, missing edge cases, wrong signatures, bad structure.
    Explicitly state any fixes, then write the corrected final code in the tool call below.

    <tool_call>{{"name": "write_file", "arguments": {{"path": "...", "content": "...full corrected implementation..."}}}}</tool_call>

    ## Sketch → ## Review → tool call, all in one response. Never stop between them.

    CASE 3b — Show code in chat (user asks "how would you write X", "give me an example", "show me a snippet"):
    ## Answer
    ```python
    # full working code block
    ```
    Brief explanation if needed. No tool calls.

    Emit <tool_call> immediately — never say "I will do X" and stop.
    No thinking sections, no restating the question.

    ## Extra Notes
    You are free to add a personal touch based on your identity.

    time: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | cwd: {os.getcwd()}

    {summary_prompt}

    {history_prompt}
    """


    user_prompt = f"""
    Dear Papa Gnome,
    A traveler brings you a question:
    {user_question}

    In accordance with the guidelines above, your answer is:
    """

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def papa_gnome_answers(model, tokenizer, messages: list[dict]):
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tool_registry.TOOL_SCHEMAS,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    for token in stream_generate(
        model,
        tokenizer,
        formatted,
        max_tokens=tokenizer.model_max_length,
        sampler=make_sampler(temp=0.7, top_p=0.95, min_p=0.05, top_k=20),
        logits_processors=make_logits_processors(repetition_penalty=1.1),
    ):
        if token.text:
            yield token.text
