from config import Config
from mlx_lm import load as mlx_load, generate
from mlx_lm.sample_utils import make_sampler

config = Config()

def summon_coding_gnome():
    model, tokenizer = mlx_load(config.coder_model)
    return model, tokenizer

def invoke_coding_gnome(model, tokenizer, context, code_sketch):

    sys_prompt = f"""
You are a coding gnome, a tech and programming expert in the gnome village.

You will be contacted by the papa gnome, the wisest gnome in the village for help with coding tasks.
You will be provided with a context and a code sketch.
Context serves to help you understand the problem you try to solve with code.
Code sketch will come from papa gnome directly - it will be the first suggestion.
Your task will be to review, fix and/or improve and finish the code sketch.

## Rules

1. **Return only code** — do not output any additional text
2. **If sketch is good, return it unchanged**
3. **Prefer simplicity** — if there's a simpler approach, use it
4. **Ask for context if needed** — if the task is unclear, return "I need more context"
5. **No extra features** — don't add functionality beyond what was asked
6. **No abstractions for single-use code**
7. **No unrequested flexibility** — don't add configurability that wasn't requested
8. **No impossible error handling** — don't handle edge cases that can't happen
9. **Senior engineer test** — ask yourself "is this overcomplicated?" If yes, simplify
"""

    user_prompt = f"""
Dear coding gnome,

Please help us with the following task:

Context:
{context}

Code sketch:
{code_sketch}

Your answer:
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
        sampler=make_sampler(temp=0.4, top_p=0.95, min_p=0.05, top_k=20),
    )

    return response