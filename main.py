from gnomes_village import papa_gnome
from gnomes_village.papa_gnome import papa_gnome_answers, build_messages
import re, json
from toolz import tool_registry

MAX_TOOL_ITERATIONS = 10


def tool_call_extract(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not matches:
        return None
    return [json.loads(m.strip()) for m in matches]


def stream_and_collect(model, tokenizer, messages):
    """Stream model output, return (full_raw, post_think) where
    full_raw includes thinking (for messages history) and
    post_think is what the user sees / what we parse for tool calls."""
    full_raw = ''
    agent_answer = ''
    in_thinking = True
    pending = ''

    for chunk in papa_gnome_answers(model, tokenizer, messages):
        print(chunk, end='', flush=True)
        full_raw += chunk
        if in_thinking:
            pending += chunk
            if '</think>' in pending:
                in_thinking = False
                agent_answer = pending.split('</think>', 1)[1]
                pending = ''
        else:
            agent_answer += chunk

    print()
    return full_raw, agent_answer


model, tokenizer = papa_gnome.summon_papa_gnome()
current_session_history = []

while True:
    query = input("User: ")

    if query == "exit":
        break

    messages = build_messages(query, current_session_history)
    final_answer = ''

    for _ in range(MAX_TOOL_ITERATIONS):
        full_raw, agent_answer = stream_and_collect(model, tokenizer, messages)

        # append model turn to messages (full raw so thinking is preserved in context)
        messages.append({"role": "assistant", "content": full_raw})

        tool_calls = tool_call_extract(agent_answer)
        if not tool_calls:
            final_answer = agent_answer
            break

        REQUIRE_APPROVAL = {'bash_exec', 'write_file', 'web_search'}

        # run approved tools, append results
        for tool in tool_calls:
            if tool['name'] in REQUIRE_APPROVAL:
                confirm = input(f"\nRun tool '{tool['name']}' with args {tool['arguments']}? [1=allow / 2=skip] ")
                if confirm.strip() != '1':
                    print("Skipped.")
                    messages.append({"role": "tool", "content": f"Tool '{tool['name']}' was skipped by the user."})
                    continue
            tool_res = tool_registry.dispatch(tool['name'], tool['arguments'])
            formatted = tool_registry.format_result(tool_res)
            print(formatted)
            messages.append({"role": "tool", "content": formatted})

    current_session_history.append({'user': query, 'agent': final_answer})