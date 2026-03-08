import json
from gnomes_village import mama_gnome, small_gnomes
# https://github.com/QwenLM/Qwen3.5


mama_model, mama_tokenizer = mama_gnome.summon_mana_gnome()

questions = [
    "What is the capital of France?",
    'What is the direction of modern quantum physics?',
    "Write a Fibonacci sequence as python script.",
]

print('Mama Gnome is thinking...')

query = questions[1]

resp = mama_gnome.router(mama_model, mama_tokenizer, user_input=query)
print(resp)

try:
    think_end = resp.find('</think>')
    text = resp[think_end + len('</think>'):].strip() if think_end != -1 else resp
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")
    try:
        mama_answer, _ = json.JSONDecoder().raw_decode(text, start)
    except (json.JSONDecodeError, ValueError):
        end = text.rfind('}') + 1
        json_str = text[start:end].replace('\\"', '"').replace('""', '"')
        mama_answer = json.loads(json_str)
    mama_answer = {k: v.strip('"') if isinstance(v, str) else v for k, v in mama_answer.items()}
except (json.JSONDecodeError, ValueError) as e:
    print("Failed to parse Mama Gnome's response: ", e)
    mama_answer = None

if mama_answer:
    try:
        strategy = mama_answer["strategy"]
    except KeyError:
        print('Cannot extra strategy from the response.')
        strategy = None

    try:
        task_think = mama_answer["task_think"]
    except KeyError:
        print('Cannot extract thinking task from the response.')
        task_think = None

    try:
        task_direct = mama_answer["task_direct"]
    except KeyError:
        print('Cannot extract direct task from the response.')
        task_direct = None

    try:
        rationale = mama_answer["rationale"]
    except KeyError:
        print('Cannot extract rationale from the response.')
        rationale = None

    if strategy == 'A':
        answer = mama_answer.get("answer")
        print(f"Mama Gnome answered directly: {answer}")
    else:
        # small_direct_gnome_model, small_direct_gnome_processor, small_direct_gnome_config = small_gnomes.summon_smol_gnomes()
        small_thinking_gnome_model, small_thinking_gnome_processor, small_thinking_gnome_config = small_gnomes.summon_smol_gnomes()

        print('Smol Gnome is thinking...')

        response_thinking = small_gnomes.thinking_gnome(
            small_thinking_gnome_model,
            small_thinking_gnome_processor,
            small_thinking_gnome_config,
            thinking_task=task_think
        )

        print('=' * 100, '\n')
        print('Smol Gnome answer:')
        print(response_thinking)

        response_direct = None

        print('=' * 100, '\n')
        response_synthesized = mama_gnome.synthesize(
            mama_model,
            mama_tokenizer,
            thinking_gnome=response_thinking,
            direct_gnome=response_direct,
            initial_plan=mama_answer,
            user_question=query
        )

        print('Synthesized answer:')
        print(response_synthesized)