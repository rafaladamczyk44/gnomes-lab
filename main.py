from gnomes_village import papa_gnome
from gnomes_village.papa_gnome import papa_gnome_answers
import re, json
from toolz import tool_registry, tools


def tool_call_extract(text):
    """
    <tool_call>
    {"name": "web_search", "arguments": {"query": "Miami current weather today 2026-04-10", "n": 3}}
    </tool_call>
    """
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not matches:
        return None

    return [json.loads(m.strip()) for m in matches]

model, tokenizer = papa_gnome.summon_papa_gnome()
current_session_history = []
while True:
    query = input("User: ")

    if query == "exit":
        break

    agent_answer = ''
    in_thinking = True  # reasoning model always starts in thinking mode
    pending = ''

    for chunk in papa_gnome_answers(model, tokenizer, query, current_session_history):
        print(chunk, end="", flush=True)

        # split on </think>, save only answer to the history
        if in_thinking:
            pending += chunk
            if '</think>' in pending:
                in_thinking = False
                agent_answer = pending.split('</think>', 1)[1]
                pending = ''
        else:
            agent_answer += chunk

    current_session_history.append(
        {
            'user': query,
            'agent': agent_answer,
        }
    )

    print()

    # check for tool calls
    tool_calls = tool_call_extract(agent_answer)

    # run the tools
    if tool_calls:
        for tool in tool_calls:
            confirm = input(f"\nRun tool '{tool['name']}' with args {tool['arguments']}? [1=allow / 2=skip] ")
            if confirm.strip() != '1':
                print("Skipped.")
                continue
            tool_res = tool_registry.dispatch(tool['name'], tool['arguments'])
            print(tool_registry.format_result(tool_res))


    # clear the answer
    agent_answer = ''

