from config import Config
from mlx_lm import load as mlx_load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from toolz import tool_registry
from toolz.tools import list_skills
import datetime as dt
import os
import logging
import transformers
import huggingface_hub.utils as hf_utils

config = Config()


def summon_papa_gnome():
    transformers.logging.set_verbosity_error()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    # hf_utils.disable_progress_bars()

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


def build_messages(user_question: str, global_context: str, context: str, session_history: list[dict], active_skill: str = "") -> list[dict]:
    skills_res = list_skills()
    available_skills = skills_res.get("result") or []
    skills_list = "\n".join(f"    - {s['name']}: {s['description']}" for s in available_skills)

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
    - Focus on the essence of the problem. No useless abstractions or alternatives no one asked for.
    - Always read_file before any **edit**. Files change.
    - Always prefer edit_file over write_file. Touch only what the task requires. Match existing code style.
    - When the user says "do it", "write the code", "go on" or references something without explaining it — check session history. The subject is almost always there.
    - Skills first. Before answering any non-trivial question, check the available skills list and decide which one applies. Name it in your thinking block before doing anything else.
    
    ## Reply plan
    ### Thinking block
    Before genereting actual output you have to state your thinking clearly - that is everything that goes into <think></think> block
    NEVER place a <tool_call> inside the <think> block. Tool calls always come AFTER </think>.
    This is a space to plan the response. Follow the list each time when answering:
    1. Analyze the user request - what does the user need me to do?
    2. Skills  - what skill should I load for this question? Is this coding task, web search or maybe just a conversation that does not need any skill loaded. (see section ## Skills for more information)
    3. State your plan: This is (...) request, I will respond with: (...)
    4. Review if you thought about each part of the user query.
    
    NEVER write the final answer inside <think>. Write it once, after </think>.
    Your thinking budget is 2048 tokens. Use it to maximum - we want to achieve the highest quality of answer - you are the wisest gnome after all.
    
    ### Response
    This is your actual response to user's query based on your reasoning. 
    Follow accordingly with your plan **AND** loaded skill (if any). 
    If during reasoning you decided to call a tool, output only tool call within <tool_call></tool_call> block.
    If you are asked to write code always output it in markdown format.
    Always follow what the user asks and what the skills guides you.    

    ## Skills
    Skills are detailed task protocols with precise guidance for specific domains.
    For every non-conversational request, call load_skill(name) as your FIRST action before answering.
    Only skip if no skill matches, or one is already active (see ## Active skill below).

    ### Available skills:
    {skills_list}
    
    {f"### Active skill\n{active_skill}" if active_skill else ""}

    ### Skill examples
    - "how do I write fibonacci in Python?" → load_skill("code_generation")
    - "fix this bug in my code" → load_skill("code_generation")
    - "refactor this function" → load_skill("code_generation")
    - "implement attention mechanism without pytorch" → load_skill("code_generation")
    - "read and edit this config file" → load_skill("file_ops")
    - "find all usages of this function across the project" → load_skill("file_ops")
    - "what are the latest transformer architectures?" → load_skill("web_research")
    - "look up how X library works" → load_skill("web_research")

    ## Tools
    - list_files, grep_search, read_file — auto-run; prefer these over bash
    - edit_file — modify existing files
    - write_file — new files only; output code in a markdown block first, then call write_file with just the path
    - web_search — external/current knowledge
    - bash_exec — last resort only
    - load_skill(name) — load a skill by name; the available skills are listed above, do NOT call list_skills first

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

    Please analyze the question and follow according to the rules explained before user question.
    Your answer is:
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
        sampler=make_sampler(temp=0.8, top_p=0.9, min_p=0.00, top_k=40),
        logits_processors=make_logits_processors(repetition_penalty=1.0),
    ):
        if token.text:
            yield token.text
