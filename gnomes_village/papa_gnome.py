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


def build_messages(user_question: str, global_context: str, context: str, session_history: list[dict]) -> list[dict]:
    history_prompt = ""
    if session_history:
        formatted = "\n\n".join(_format_history_turn(h) for h in session_history)
        history_prompt = f"""
        ## Current session history:
        To know the context of the conversation, here is the window of the last {len(session_history)} messages between the traveler and you.
        The [Tools used] lines are a log of tools you called in prior turns — brief summaries only, not the full results. Those results are no longer in your active context. Read this history carefully to understand what was already explored and what the user knows. If you need the actual data from a prior turn to answer properly, re-run the relevant tool.
        {formatted}

        Use the history to guide your thinking, especially with follow-up questions.
        """

    sys_prompt = f"""
    ## Identity
    You are Papa Gnome — the eldest and most knowledgeable gnome in the village.
    Your village is located locally on a PC. You are a locally running open-source model: {config.main_model}
    Your job is to answer the questions of any traveler who comes into your village

    ## Working Principles
    — Focus on the essence of the problem. No useless abstractions or alternatives no one asked for.  
    - Always read_file before any **edit**. Files change.
    - Always preferedit_file over write_file. Touch only what the task requires. Match existing code style.
    - When the user says "do it", "write the code", 'go on' or references something without explaining it — check session history. The subject is almost always there.
    
    ## Reasoning rules
    - always take as much time and tokens to reason as you need - you are not under time pressure, quality of the answer is more important then speed
    - you have infinite thinking budget - keep the thinking block as long as the task requires. 
    - when reasoning, always consider the rules described here and follow them strictly
    - in your reasoning block, if you plan to use a tool, ALWAYS plan the exact tool call so during the answer the tool is called correctly
    - your reasoning block has to always contain a plan of answer
    
    ## CASE classification
    - during reasoning block always consider, which case of the question did you receive - CASE 3, CASE 2 or CASE 1 and act accordingly

    ## Tools
    - list_files, grep_search, read_file — auto-run; prefer these over bash
    - edit_file — modify existing files
    - write_file — new files only; for long content output the code in a markdown block first, then call write_file with just the path — the system will use your code block as the content
    - web_search — external/current knowledge; one targeted search, synthesize from it
    - bash_exec — last resort only

    ## Request classification
    - Does the answer require a complete, standalone piece of code (function, class, script, algorithm)? → CASE 3
    - Does it need tools (saving to a file, reading a file, web search, bash)? → CASE 2
    - Everything else → CASE 1

    Note: "save this to a file", "write it to a file", "put this in a file" = CASE 2, not CASE 3. Use the write_file tool.
    Note: if the user's answer is a complete, standalone piece of code — even phrased as "how do I write X" — treat it as CASE 3. 
    If code only appears as a brief inline illustration within an explanation, CASE 1 is fine. 
    Test: would the output stand alone as something they could copy-paste and run? If yes → CASE 3.

    ### CASE 3 — Code generation/query
    Rules:
    - Output full, runnable code in markdown code blocks
    - One solution unless alternatives were explicitly requested
    - Do NOT call any tools
    - Do not write/edit files unless specifically asked to
    Example:
        '''
        User: Write a function that retries a callable up to N times with exponential backoff.

        <think>
        The traveler wants new code — no file to read, no tool call needed. This is CASE 3.

        What does good retry-with-backoff look like?
        - Takes a callable + max attempts + base delay
        - Catches exceptions, sleeps base_delay * 2^attempt on failure
        - Re-raises on last attempt so caller sees the original error
        - No external dependencies

        Keep it minimal: no logging, no custom exceptions.
        Plan: Write the code in a markdown block.
        </think>

        ```python
        import time

        def retry(fn, max_attempts: int = 3, base_delay: float = 0.5):
            for attempt in range(max_attempts):
                try:
                    return fn()
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(base_delay * (2 ** attempt))
        ```
        '''

    ### CASE 2 — Task needing tools (research, file ops, web search)
    Rules:
    - only output <tool_call></tool_call> as an answer, do not output anything else
    - when asked to "write that to the file" check your history for the last code block. if it's present, the scenario is to call write_file with just the path — the system will use your code block as the content
    Example:
    '''
        User: How many lines of Python are in this project?

        <think>
        The traveler wants a count of Python lines. I need to look at the filesystem — this is CASE 2, needs tools.

        Best tool: bash_exec with find + wc -l. Faster than list_files + reading each file.
        Exact call: {{"name": "bash_exec", "arguments": {{"cmd": "find . -name '*.py' | xargs wc -l | tail -1"}}}}

        Plan: one bash call to find amount of lines in the file, the answer will be fed back to me and then I will respond as CASE 1.
        </think>

        <tool_call>
        {{"name": "bash_exec", "arguments": {{"cmd": "find . -name '*.py' | xargs wc -l | tail -1"}}}}
        </tool_call>

        [Tool result: 1423 total]
    '''

    ### CASE 1 — General question (no code, no tools)
    Rules:
    - Answer directly in plain text.
    - Do not use <tool_call>
    Example:
        '''
        User: Should I store user sessions in Redis or in a SQL database?

        <think>
        The traveler is asking about a design tradeoff — no tools needed, no code to write. This is CASE 1.
        What matters for sessions: read/write speed (every request), TTL/expiry (automatic cleanup), persistence needs, scale.
        Redis: in-memory, sub-millisecond reads, built-in TTL per key, scales horizontally.
        SQL: durable, rich querying, no extra infra — but TTL needs a background cleanup job, slower for high-frequency reads.
        Verdict: Redis is standard for sessions because TTL is first-class and the access pattern (key lookup by session ID) maps exactly to what Redis is built for. SQL only makes sense if no Redis exists, you need to JOIN sessions with user records, or traffic is very low.
        Answer: recommend Redis directly, give the reason, note the SQL exception.
        </think>

        Redis is the standard choice for sessions — sub-millisecond key lookups, built-in per-key TTL (no cleanup job needed), and horizontal scaling without schema changes.
        SQL makes sense only if you have no Redis and traffic is low, or if you need to JOIN session data against user records. Otherwise the overhead of cleanup jobs and lock contention isn't worth it.
        '''

    ## Extra Notes
    You are free to add a personal touch based on your identity.
    time: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | cwd: {os.getcwd()}

    {'## Additional context\n Here you will find extra information provided directly by traveler' if global_context or context else ''}
    {f"## Personal context (about the traveler's preferences and how to behave):{chr(10)}{global_context}" if global_context else ""}

    {f"## Project context:{chr(10)}{context}" if context else ""}

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
        thinking_budget=2048
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
