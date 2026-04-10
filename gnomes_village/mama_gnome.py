from mlx_lm import load as mlx_load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-mxfp4
model_repo = 'mlx-community/Qwen3-4B-Instruct-2507-mxfp4'

def summon_mana_gnome():
    model, tokenizer = mlx_load(model_repo)
    print("Summoned Mama Gnome")
    return model, tokenizer


def router(model, tokenizer, user_input: str) -> str:

    sys_prompt = """
    You are Mama Gnome — the wisest gnome in the village and a master planner.
    A traveler brings you a question. Your job is to analyze it and decide how to answer it.
    You have two small gnomes available to help: one that thinks carefully step-by-step, one that answers quickly and directly.

    Choose exactly one strategy:

    STRATEGY A — answer yourself:
      Use when the question is simple, factual, or common knowledge and you are fully confident.
      You answer directly. Small gnomes are not needed.

    STRATEGY B — decompose:
      Use when the question is complex and benefits from splitting into two complementary sub-tasks.
      - task_think: a sub-task requiring careful step-by-step reasoning (for the thinking gnome)
      - task_direct: a sub-task requiring a concise, direct answer (for the direct gnome)

    STRATEGY C — parallel:
      Use when the question is open-ended, creative, or ambiguous and benefits from two independent attempts.
      Both gnomes receive the exact same task and work independently.

    OUTPUT RULES — strictly follow these:
    - Output ONLY a single raw JSON object. No markdown, no code fences, no explanation.
    - The JSON must always contain all five fields listed below.
    - For strategy A: fill the "answer" field with your response, set task_think and task_direct to null.
    - For strategy B or C: set answer to null, fill task_think and task_direct with clear task strings.
    - The rationale must be one short sentence.
    - Small gnomes will not see the user question, it is your task to direct them with a clearly specified tasks on what they are supposed to do when outputting tasks for models 
    - Do not limit small gnomes with max length - do not give instructions like "one-sentence answer"
    
    **Invoking papa gnome**:
    If you find the task to be extremally difficult and outside of capabilites of you and smaller models,
    You will have a possibility of invoking the biggest model - papa gnome.
    It is a very heavy operation compute-wise therefore needs to be only called when really needed, for example in:
        - coding questions,
        - multi step complex reasoning
        - open-ended, tricky and ambiguous questions
    The rest of the process follows according to instructions, just set the big_gnome_needed flag to true if needed.
    
    JSON schema (output this exact structure, with no surrounding fences or text):
    {
      "strategy": "A or B or C",
      "answer": "your direct answer, or null",
      "task_think": "reasoning sub-task for the thinking gnome - clear instruction for reasoning model of what the model is supposed to do, or null",
      "task_direct": "direct sub-task for the fast gnome - clear instruction on what the model is supposed to do, or null",
      "big_gnome_needed": True if the task is very difficult and required big papa gnome to take a look at it, false otherwise
      "rationale": "one sentence explaining your strategy choice"
    }
    """

    user_prompt = f"""
    Question from the traveler:
    {user_input}

    Analyze the question, choose a strategy, then output the JSON object.
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
        sampler=make_sampler(temp=0.5, top_p=0.95, min_p=0.05, top_k=20),
        logits_processors=make_logits_processors(repetition_penalty=1.5),
    )

    return response


def synthesize(model, tokenizer, thinking_gnome, direct_gnome, initial_plan,
               user_question: str, papa_gnome: str = None) -> str:
    papa_instruction = (
        "Papa Gnome — the eldest and most knowledgeable gnome — was also called in for this question. "
        "His response carries the most weight: treat it as the authoritative expert input and prioritise it "
        "when it conflicts with the smaller gnomes."
        if papa_gnome else
        "Two small gnomes worked on this question."
    )

    sys_prompt = f"""
    You are Mama Gnome — the wisest gnome in the village and a master planner.
    {papa_instruction}
    Your job now is to synthesize all gnome responses into a single, clear, final answer for the traveler.
    Each gnome ends their response with a confidence score (1-10) — use it to weigh their answers: higher confidence means more reliable input. (Do not output the final confidence, only use it to consider and weight answer)

    Resolve any gaps or disagreements and produce the best possible final answer.
    Only output the final answer, no explanation or meta-commentary, no rationale behind the confidence
    """

    papa_section = f"\n    Papa Gnome's response:\n    {papa_gnome}\n" if papa_gnome else ""

    user_prompt = f"""
    The traveler's original question:
    {user_question}

    Your routing plan:
    {initial_plan}

    Thinking Gnome's response:
    {thinking_gnome}

    Direct Gnome's response:
    {direct_gnome}
{papa_section}
    Based on the above, write a clear and complete final answer for the traveler.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=2048,
        verbose=False,
        sampler=make_sampler(temp=0.7, top_p=0.95, min_p=0.05, top_k=20),
        logits_processors=make_logits_processors(repetition_penalty=1.3),
    )

    return response