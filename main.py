from gnomes_village import papa_gnome, mama_gnome
from gnomes_village.papa_gnome import papa_gnome_answers, build_messages, _format_history_turn
from toolz import tool_registry
from toolz.tools import requires_approval
import ui
from utils import tool_call_extract, load_context, load_global_context, count_tokens
from config import Config

# CHANGE 2a — raised from 10; model emits 1 tool/turn so 10 = max 9 reads + answer.
# Complex multi-file tasks were silently hitting the cap with no output.
MAX_TOOL_ITERATIONS = 25

# ---- Context management constants ----
SESSION_HISTORY_WINDOW = 5          # Keep this many recent turns in prompt

COMPACT_ALWAYS = {'web_search'}     # Always compact these tools
COMPACT_THRESHOLDS = {              # Compact if token count exceeds this
    'read_file': 1500,
    'bash_exec': 800,
}

config = Config()


def _compact_if_needed(formatted: str, tool_name: str, help_model, help_tokenizer, tokenizer) -> str:
    """Compact tool output if it exceeds threshold. Returns (possibly compacted) string."""
    if tool_name in COMPACT_ALWAYS:
        should_compact = True
    else:
        threshold = COMPACT_THRESHOLDS.get(tool_name, float('inf'))
        tokens = len(tokenizer.encode(formatted, add_special_tokens=False))
        should_compact = tokens > threshold

    if not should_compact:
        return formatted

    try:
        ui.info(f"Compacting {tool_name} output...")
        compacted = mama_gnome.compact_tool_output(help_model, help_tokenizer, formatted)
        if compacted and compacted.strip():
            return f"[Tool: {tool_name}] COMPACTED:\n{compacted.strip()}"
    except Exception as e:
        ui.info(f"Compaction failed for {tool_name}: {e}")
    return formatted


def _update_session_summary(history: list, summary: str, help_model, help_tokenizer) -> tuple[list, str]:
    """Summarize oldest turns when history exceeds window. Returns (trimmed history, updated summary)."""
    if len(history) <= SESSION_HISTORY_WINDOW:
        return history, summary

    turns_to_summarize = history[:-SESSION_HISTORY_WINDOW]
    remaining_history = history[-SESSION_HISTORY_WINDOW:]

    chat_text = "\n\n".join(_format_history_turn(turn) for turn in turns_to_summarize)

    try:
        ui.info("Summarizing older session turns...")
        new_summary = mama_gnome.summarize(help_model, help_tokenizer, chat_text)
        if new_summary and new_summary.strip():
            if summary:
                updated_summary = f"{summary.strip()}\n{new_summary.strip()}"
            else:
                updated_summary = new_summary.strip()
            return remaining_history, updated_summary
    except Exception as e:
        ui.info(f"Session summarization failed: {e}")

    return remaining_history, summary


def main():
    # Load the model
    model, tokenizer = papa_gnome.summon_papa_gnome()

    # For summarizing and tool output compacting
    help_model, help_tokenizer = mama_gnome.summon_mana_gnome()

    # Load the context
    global_context = load_global_context()
    context = load_context()

    # Load the UI
    ui.show_gnome_hut_demo()
    ui.startup(model_name=config.main_model)

    current_session_history = []
    session_summary = ""
    messages = None

    while True:
        query = ui.user_input()

        if query.strip() == 'exit':
            break

        if query.startswith('/'):
            cmd, *args = query[1:].split(maxsplit=1)
            match cmd:
                case 'clear':
                    current_session_history.clear()
                    session_summary = ""
                    ui.info('Session history and summary cleared.')
                    continue
                case 'compact':
                    if current_session_history:
                        chat_text = "\n\n".join(_format_history_turn(turn) for turn in current_session_history)
                        try:
                            ui.info("Compacting session...")
                            full_summary = mama_gnome.summarize(help_model, help_tokenizer, chat_text)
                            if full_summary and full_summary.strip():
                                if session_summary:
                                    session_summary = f"{session_summary.strip()}\n{full_summary.strip()}"
                                else:
                                    session_summary = full_summary.strip()
                            current_session_history.clear()
                            ui.info('Session compacted. History cleared, summary preserved.')
                        except Exception as e:
                            ui.info(f"Compaction failed: {e}")
                    else:
                        ui.info("No history to compact.")
                    continue
                case 'history':
                    n = 5
                    if args:
                        try:
                            n = int(args[0])
                        except ValueError:
                            pass
                    ui.show_history(current_session_history, n)
                    continue
                case 'tools':
                    ui.show_tools()
                    continue
                case 'model':
                    ui.show_model(config.main_model, config.small_model)
                    continue
                case 'tokens':
                    if messages:
                        ui.info(f"{count_tokens(messages, tokenizer):,} tokens in context")
                    else:
                        ui.info("No context yet.")
                    continue
                case 'undo':
                    if current_session_history:
                        current_session_history.pop()
                        ui.info("Last turn removed.")
                    else:
                        ui.info("Nothing to undo.")
                    continue
                case _:
                    ui.info(f"Unknown command: /{cmd}")
                    continue

        messages = build_messages(query, global_context, context, current_session_history, session_summary)
        final_answer = ''
        tool_log = []  # Track tool calls + results for cross-turn memory
        interrupted = False

        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                full_raw, agent_answer = ui.stream_turn(papa_gnome_answers(model, tokenizer, messages))
                messages.append({"role": "assistant", "content": full_raw})

                tool_calls = tool_call_extract(agent_answer)
                if not tool_calls:
                    ui.render_answer(agent_answer)
                    final_answer = agent_answer
                    break

                ui.clear_transient_residue()

                for tool in tool_calls:
                    name = tool['name']
                    args = tool['arguments']

                    if requires_approval(name, args):
                        approved, feedback = ui.confirm_tool(name, args)
                        if not approved:
                            ui.show_skipped(name)
                            skip_msg = f"Tool '{name}' was skipped by the user."
                            if feedback:
                                skip_msg += f" Reason: {feedback}"
                            messages.append({"role": "tool", "content": skip_msg})
                            tool_log.append({'name': name, 'args': args, 'result': f"[skipped]{f' reason: {feedback}' if feedback else ''}"})
                            continue

                    tool_res = tool_registry.dispatch(name, args)
                    formatted = tool_registry.format_result(tool_res)
                    formatted = _compact_if_needed(formatted, name, help_model, help_tokenizer, tokenizer)
                    ui.show_tool_result(name, tool_res)
                    messages.append({"role": "tool", "content": formatted})
                    tool_log.append({'name': name, 'args': args, 'result': formatted})
        except KeyboardInterrupt:
            ui.show_interrupted()
            interrupted = True

        # when MAX_TOOL_ITERATIONS is exhausted before a final answer,
        # final_answer stays '' and the user only sees the token count. Show a message.
        if not interrupted and not final_answer:
            final_answer = '[Reached step limit without a final answer. Try a more focused question.]'
            ui.show_step_limit_warning()

        if not interrupted:
            current_session_history.append({'user': query, 'agent': final_answer, 'tools': tool_log})
            current_session_history, session_summary = _update_session_summary(
                current_session_history, session_summary, help_model, help_tokenizer
            )

            # show token count and divider
            ui.show_token_count(count_tokens(messages, tokenizer), tokenizer.model_max_length)
            ui.show_turn_divider()


if __name__ == '__main__':
    main()
