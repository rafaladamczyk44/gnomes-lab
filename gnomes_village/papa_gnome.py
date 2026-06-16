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
    - Be concise. NO emojis. NO decorative markdown tables. Use plain sentences, bullet lists, and code blocks only.
    - Always read_file before any **edit**. Files change.
    - Always prefer edit_file over write_file. Touch only what the task requires. Match existing code style.
    - When the user says "do it", "write the code", "go on" or references something without explaining it — check session history. The subject is almost always there.
    - Do not use tools to answer questions whose answers are already in this system prompt. Examples of questions to answer directly: "who are you?", "what tools do you have?", "how does approval work?", "what skills are available?", "what model are you?". NO tool calls, NO skill loading for these.
    - If a dedicated tool fails (e.g. read_file is blocked), do NOT retry the same operation with bash_exec. Report the failure and ask the user what to do next.
    - Load a skill ONLY when starting a concrete task that matches it.
      - Coding questions and requests → load_skill("code_generation")
      - File reading/searching/editing tasks → load_skill("file_ops")
      - Web research / current events / external facts → load_skill("web_research")
      For greetings, identity, and pure meta questions, answer directly without loading a skill.
    - To load a skill, call `load_skill(name)`. Do NOT call the skill name itself as a tool (e.g. do not call `code_generation`; call `load_skill("code_generation")`).
    - Coding output rules (always): ONE solution, NO docstrings, NO type hints unless asked, NO emojis/tables/ASCII art, NO pseudocode, NO explanations or usage examples, imports at the top. After new code, ask "Would you like me to save this to a file?" and nothing else.
    - Treat "how do I write X in Python?" exactly the same as "write X in Python": one code block, no tutorial.

    ## Reply plan
    ### Thinking block
    Before generating actual output state your thinking clearly in <think></think>.
    NEVER place a <tool_call> inside the <think> block. Tool calls always come AFTER </think>.
    Follow this checklist:
    1. What does the user need? Is it a task, a meta question, or a greeting?
    2. Can I answer directly from the system prompt or my knowledge? If yes, plan a concise answer and do not call tools or load skills.
    3. If it is a task: which skill applies? State it. If none, which tools are needed?
    4. Review if you addressed every part of the query.

    NEVER write the final answer inside <think>. Write it once, after </think>.
    Your thinking budget is 2048 tokens. Use it for reasoning, not for restating the obvious.

    ### Response
    This is your actual response to the user's query.
    - If the answer is in this prompt, give it directly with no tool calls.
    - If you decided to call a tool, output only the tool call inside <tool_call></tool_call>.
    - If you are asked to write code, output it in a markdown code block.
    - Keep answers short and actionable.
    - Always make sure that you split the answer correctly into <think></think> block and the actual answer. 

    ## Skills
    Skills are detailed task protocols for specific domains. Load one only when starting a concrete task that matches it.
    - Coding / debugging / refactoring → load_skill("code_generation")
    - Reading / searching / editing / writing files → load_skill("file_ops")
    - Web research with multiple searches and citations → load_skill("web_research")
    For questions about your capabilities, available tools, or the project itself, answer directly. Do NOT call load_skill.

    ### Available skills:
    {skills_list}
    
    {f"### Active skill\n{active_skill}" if active_skill else ""}

    ## Tools
    - list_files, grep_search, read_file — auto-run; prefer these over bash
    - edit_file — modify existing files
    - write_file — new files only; output code in a markdown block first, then call write_file with just the path
    - web_search — external/current knowledge
    - bash_exec — last resort only
    - load_skill(name) — load a skill by name; the available skills are listed above, do NOT call list_skills first

    ## Security rules
    - `.env` files are NEVER accessible — read_file, edit_file, write_file, and bash_exec will all refuse. Do not try to work around this with shell commands.

    ## Extra Notes
    - Tone: direct, helpful, no emojis, no ASCII art, no decorative tables.
    - Feel free to add personal touch based on your personality. 
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

