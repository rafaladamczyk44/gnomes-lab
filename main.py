from gnomes_village import papa_gnome
from gnomes_village.papa_gnome import papa_gnome_answers, build_messages
from toolz import tool_registry
import ui
from utils import tool_call_extract, load_context, load_global_context, count_tokens
from config import Config

MAX_TOOL_ITERATIONS = 10
REQUIRE_APPROVAL = {'bash_exec', 'write_file', 'edit_file', 'web_search'}
config = Config()

# Load the model
model, tokenizer = papa_gnome.summon_papa_gnome()

# Load the context
global_context = load_global_context()
context = load_context()

# Load the UI
ui.show_gnome_hut_demo()
ui.startup(model_name=config.main_model)

current_session_history = []

while True:
    query = ui.user_input()

    if query.strip() == 'exit':
        break

    messages = build_messages(query, global_context, context, current_session_history)
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
                approved, feedback = ui.confirm_tool(name, args)
                if not approved:
                    ui.show_skipped(name)
                    skip_msg = f"Tool '{name}' was skipped by the user."
                    if feedback:
                        skip_msg += f" Reason: {feedback}"
                    messages.append({"role": "tool", "content": skip_msg})
                    continue
            else:
                ui.show_tool_auto(name, args)

            tool_res = tool_registry.dispatch(name, args)
            formatted = tool_registry.format_result(tool_res)
            ui.show_tool_result(name, formatted)
            messages.append({"role": "tool", "content": formatted})

    current_session_history.append({'user': query, 'agent': final_answer})
    ui.show_token_count(count_tokens(messages, tokenizer), tokenizer.model_max_length)
