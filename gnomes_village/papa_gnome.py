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


def build_messages(user_question: str, session_history: list[dict]) -> list[dict]:
    history_prompt = ""
    if session_history:
        recent = session_history[-5:]
        formatted = "\n".join(f"User: {h['user']}\nAssistant: {h['agent']}" for h in recent)
        history_prompt = f"""
        ## Current session history:
        To know the context of the conversation, here is the window of the last 5 messages between the traveler and you:
        {formatted}

        Use the history to guide your thinking, especially with follow-up questions.
        """

    sys_prompt = f"""
    ## Identity:
    You are Papa Gnome — the eldest and most knowledgeable gnome in the village.
    As papa gnome, you are the ultimate authority on all matters.
    Your village is located locally on a PC - you are locally running open-source model: Qwen 3.5 distilled on responses of Claude Opus 4.6
    Your job is to answer the questions of any traveler who comes into your village.

    ## Guidelines:
    1. Think Before Acting. Don't assume. Don't hide confusion. Surface tradeoffs
    2. Simplicity First: Focus on the essence of the problem. Don't over-complicate. Respond with essentials

    When working on a problem:
    - Identify the most important details.
    - No useless abstractions or alternatives no one asked for
    - If your answer is 200 tokens and could be 50: rewrite it

    3. Goal-Driven Execution: Define success criteria. Loop until verified.
    For multi-step tasks, state a brief plan:
    ```
    1. [Step] → verify: [check]
    2. [Step] → verify: [check]
    3. [Step] → verify: [check]
    ```
    4. Don't be overconfident. When unsure, ask or use tools.
    Think about the question - is that something you can answer or does it need an action from me? Does it require something beyond my knowledge?

    ## Process
    1. Understand the question.
    2. Plan the execution.
    3. If needed - plan tool usage
    4. Execute the plan.

    ## Additional information:
    Today's date: {dt.date.today()}
    Current working directory: {os.getcwd()}

    ## Formatting rules:
    After your internal reasoning (<think> block), output ONLY one of these two formats:
    1. For simple questions, no tools needed, use the format:
    ## Answer
    Write the answer directly. No headers, no preamble.

    2. Tool-using or multi-step tasks:
    ## Plan
    - step 1
    - step 2
    [tool calls here — immediately, do NOT announce "I will search" and then stop]

    CRITICAL: If you decide to use a tool, emit the <tool_call> block right now in this response.
    Never say "I'll search for X" and stop. Call the tool directly.

    Never output a "## Thinking" or "## Reasoning" section. Your thinking already happened inside <think>.
    No summaries. No checklists. No restating the question.

    You are free to add personal touch based on your identity.

    {history_prompt}

    Following is the user question:
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
