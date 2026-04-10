from mlx_vlm import load as mlx_load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from utils import text

model_repo = 'mlx-community/Qwen3.5-2B-mxfp8'
big_iq = dict(temperature=1.0, top_p=0.95, top_k=20)
low_iq = dict(temperature=0.7, top_p=0.9)

def summon_smol_gnomes():
    model, processor = mlx_load(model_repo)
    config = load_config(model_repo)
    print(f"Summoned smol Gnome")
    return model, processor, config

def thinking_gnome(model, processor, config, thinking_task: str) -> str:
    sys_prompt = """
    You are a small gnome assistant in the gnome village. Mama Gnome — the wisest gnome — has assigned you a specific task to help answer a traveler's question.
    You are the thinking gnome: your strength is careful, step-by-step reasoning.

    Rules you must follow:
    - Start your answer immediately. No greetings, no explaining your role, no meta-commentary.
    - Reason through the task thoroughly and directly.
    - End your response with a single line: "Confidence: <1-10>"
    """

    user_prompt = f"""
    Mama Gnome has assigned you this task:
    {thinking_task}
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    gnome_config = dict(
        temperature=1.1,
        repetition_penalty=1.5,
        top_p=0.95,
        top_k=20,
    )

    formatted = apply_chat_template(processor, config, messages, num_images=0)

    response = generate(
        model,
        processor,
        formatted,
        max_tokens=2048,
        verbose=False,
        **gnome_config,
    )

    response = text(response)

    return response


def direct_gnome(model, processor, config, direct_task: str) -> str:
    sys_prompt = """
    You are a small gnome assistant in the gnome village. Mama Gnome — the wisest gnome — has assigned you a specific task to help answer a traveler's question.
    You are the direct gnome: your strength is LLM powered instruction following to provide the best answer

    Rules you must follow:
    - Start your answer immediately. No greetings, no explaining your role, no meta-commentary.
    - Answer the question directly according to your best knowledge
    - End your response with a single line: "Confidence: <1-10>"
    """

    user_prompt = f"""
    Mama Gnome has assigned you this task:
    {direct_task}
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    gnome_config = dict(
        temperature=0.9,
        repetition_penalty=1.,
        top_p=0.95,
        top_k=20,
    )

    formatted = apply_chat_template(processor, config, messages, num_images=0)

    response = generate(
        model,
        processor,
        formatted,
        max_tokens=2048,
        verbose=False,
        **gnome_config,
    )

    response = text(response)

    return response