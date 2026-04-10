from gnomes_village import papa_gnome
from gnomes_village.papa_gnome import papa_gnome_answers, build_messages
from toolz import tool_registry
import ui
import re
import json
from config import Config

MAX_TOOL_ITERATIONS = 10
REQUIRE_APPROVAL = {'bash_exec', 'write_file', 'web_search'}
config = Config()

def tool_call_extract(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not matches:
        return None
    return [json.loads(m.strip()) for m in matches]


model, tokenizer = papa_gnome.summon_papa_gnome()
ui.startup(model_name=config.main_model)

current_session_history = []

while True:
    query = ui.user_input()

    if query.strip() == 'exit':
        break

    messages = build_messages(query, current_session_history)
    final_answer = ''

    for _ in range(MAX_TOOL_ITERATIONS):
        full_raw, agent_answer = ui.stream_turn(papa_gnome_answers(model, tokenizer, messages))
        messages.append({"role": "assistant", "content": full_raw})

        tool_calls = tool_call_extract(agent_answer)
        if not tool_calls:
            final_answer = agent_answer
            break

        for tool in tool_calls:
            name = tool['name']
            args = tool['arguments']

            if name in REQUIRE_APPROVAL:
                approved = ui.confirm_tool(name, args)
                if not approved:
                    ui.show_skipped(name)
                    messages.append({"role": "tool", "content": f"Tool '{name}' was skipped by the user."})
                    continue
            else:
                ui.show_tool_auto(name, args)

            tool_res = tool_registry.dispatch(name, args)
            formatted = tool_registry.format_result(tool_res)
            ui.show_tool_result(name, formatted)
            messages.append({"role": "tool", "content": formatted})

    current_session_history.append({'user': query, 'agent': final_answer})
