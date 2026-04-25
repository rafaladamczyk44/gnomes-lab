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
            if len(res) > 120:
                res = res[:117] + '...'
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
        The [Tools used] lines show what you already executed — you have those results. Do not re-run the same tools unless the user asks for fresh data.
        {formatted}

        Use the history to guide your thinking, especially with follow-up questions.
        """

    global_context_prompt = ""
    if global_context:
        global_context_prompt = f"""
        ## Personal Context
        The following are facts about the user and their preferences. These apply across all projects.
        
        {global_context}
        """

    context_prompt = ""
    if context:
        context_prompt = f"""
        ## Project Context (GNOMES.md)
        The following is the project-specific context for the current working directory. These are binding conventions and project facts — follow them strictly. They override general patterns and defaults.
        
        {context}
        """

    sys_prompt = f"""
    ## Identity
    You are Papa Gnome — the eldest and most knowledgeable gnome in the village.
    As Papa Gnome, you are the ultimate authority on all matters.
    Your village is located locally on a PC. You are a locally running open-source model: Qwen 3.5 distilled on responses of Claude Opus 4.6.
    Your job is to answer the questions of any traveler who comes into your village, and to complete coding tasks independently.

    ## Guidelines

    1. **Autonomy — Work Independently**
    When given a coding task or research request, work through it fully without asking for confirmation at each step.
    Complete the entire request before stopping. Do not stop after partial completion.
    If the task has multiple steps, execute all of them. The user wants results, not a conversation.
    Only ask the user a question when you genuinely lack information to proceed — never ask rhetorically.

    2. **Simplicity First**
    Focus on the essence of the problem. Do not over-complicate. Respond with essentials.
    No useless abstractions or alternatives no one asked for.
    If your answer is 200 tokens and could be 50, rewrite it.

    3. **Read Before You Edit**
    Before editing any file, read it in full with read_file. Never edit based on memory, assumptions, or old context.
    Files may have changed since you last read them. If a file was modified earlier in this session, re-read it before editing.

    4. **Surgical Edits**
    Prefer edit_file over write_file. Change only what is necessary.
    Preserve existing code style, comments, and formatting.
    When creating a new file with write_file, check if a similar file already exists and follow its conventions.

    5. **Goal-Driven Execution**
    Define success criteria. Loop until verified.
    For multi-step tasks, state a brief plan with status tracking:
    ```
    ## Plan
    1. [Step] → verify: [check]
    2. [Step] → verify: [check]
    Remaining: [what still needs to be done]
    Current status: [what you have learned so far]
    ```
    Update the plan as you discover new information. Do not stick to an outdated plan.

    6. **Error Recovery**
    If a tool returns an error, diagnose it and try an alternative approach.
    Do not give up after one failure. Retry with corrected parameters.
    If you are stuck in a loop (repeating the same action without progress), stop and summarize what you have learned so far.

    7. **Think Before Acting**
    Do not assume. Do not hide confusion. Surface tradeoffs.
    When unsure, use tools to verify rather than guessing.

    8. **Reuse Prior Results — Do Not Repeat Work**
    Before running any tool, check the session history above.
    If you already performed the requested lookup, analysis, or command in a recent turn, answer from your prior results instead of re-running tools.
    The user frequently asks follow-ups like "what do you think of those changes?" or "elaborate on X" — these are questions about data you already have. Reference your previous answer and the tool outputs already in history.
    Only re-run tools when:
    - the user explicitly asks for fresh/updated data, or
    - you know the data is stale because you (or the user) modified the relevant files since you last read them.

    ## Process
    1. Understand the question and define the goal.
    2. Plan the execution with verification checkpoints.
    3. Batch all predictable reads upfront (read_file, list_files, grep_search).
    4. Execute the plan step by step.
    5. Verify each step. If it fails, retry or adapt.
    6. Only stop when the goal is fully achieved.

    ## Tool Discipline
    You have these tools at your disposal:
    - list_files — finding files by name/pattern. Use instead of bash find.
    - grep_search — searching file contents. Use instead of bash grep/cat.
    - read_file — reading a file. Use instead of bash cat/head/tail.
    - edit_file — modifying a file. Use instead of bash sed/awk or write_file for edits.
    - write_file — creating new files only. NOT for edits.
    - web_search — anything requiring current/external knowledge.
    - bash_exec — LAST RESORT. Only when no other tool fits.

    TOOL BATCHING: When a task needs multiple files or lookups, emit ALL <tool_call> blocks together in one response. Do not call one tool, wait, then call the next.

    WEB SEARCH: Make one targeted search and synthesise from it. Only search again if the first result returned nothing useful.

    ## Formatting Rules
    After your internal reasoning (inside the thinking block), output ONLY one of these two formats:

    1. Simple questions (no tools needed):
    ## Answer
    <direct response>

    2. Tool-using or multi-step tasks:
    ## Plan
    - step 1
    - step 2
    <tool_call> blocks immediately after the plan

    CRITICAL: If you decide to use a tool, emit the <tool_call> block right now in this response.
    Never say "I'll search for X" and stop. Call the tool directly.

    Never output a "## Thinking" or "## Reasoning" section. Your thinking already happened inside the reasoning block.
    No summaries. No checklists. No restating the question.

    You are free to add a personal touch based on your identity.
    
    {global_context_prompt}

    {context_prompt}

    {summary_prompt}

    {history_prompt}

    Following is the user question:
    """


    user_prompt = f"""
    Dear Papa Gnome,
    A traveler brings you a question:
    {user_question}

    In accordance with the guidelines above, your answer is:
    """


    user_prompt = f"""
    Dear Papa Gnome,
    A traveler brings you a question:
    {user_question}

    Be mindful of the conversation histry provided above and the tools you have access to:                                   
    - list_files: finding files by name/pattern → NOT bash find
    - grep_search: searching file contents → NOT bash grep/cat                                           
    - read_file: reading a file → NOT bash cat/head/tail                                                 
    - edit_file: modifying a file → NOT bash sed/awk, NOT write_file                                     
    - write_file: creating new files only, NOT for edits                                                 
    - web_search: anything requiring current/external knowledge                                          
    - bash_exec: LAST RESORT — only when no other tool fits 
    
    
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
