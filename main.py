from gnomes_village import papa_gnome
from gnomes_village.papa_gnome import papa_gnome_answers, build_messages
from toolz import tool_registry
from toolz.tools import requires_approval, load_skill
import ui
from utils import tool_call_extract, extract_code_block, load_global_context, load_context, count_tokens
from config import Config

MAX_TOOL_ITERATIONS = 25

SESSION_HISTORY_WINDOW = 10

COMPACT_CHAR_LIMITS = {
    'read_file': 15000,
    'bash_exec': 8000,
}

config = Config()


def compact_tool_output(formatted: str, tool_name: str) -> str:
    limit = COMPACT_CHAR_LIMITS.get(tool_name)
    if limit and len(formatted) > limit:
        shown_lines = formatted[:limit].count('\n')
        return formatted[:limit] + f'\n... [truncated — showing ~{shown_lines} lines. Use read_file with offset={shown_lines} to read more]'
    return formatted


def main():
    model, tokenizer = papa_gnome.summon_papa_gnome()

    global_context = load_global_context()
    context = load_context()

    ui.show_gnome_hut_demo()
    ui.startup(model_name=config.main_model)

    current_session_history = []
    messages = None
    last_code_block = None
    active_skill = ""

    while True:
        query = ui.user_input()

        if query.strip() == 'exit':
            break

        if query.startswith('/'):
            cmd, *args = query[1:].split(maxsplit=1)
            match cmd:
                case 'clear':
                    current_session_history.clear()
                    ui.info('History cleared.')
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
                    ui.show_model(config.main_model)
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
                case 'skill':
                    skill_arg = args[0].strip() if args else 'off'
                    if skill_arg == 'off':
                        active_skill = ""
                        ui.info("Skill cleared.")
                    else:
                        res = load_skill(skill_arg)
                        if res['ok']:
                            active_skill = res['result']
                            ui.info(f"Skill loaded: {skill_arg}")
                        else:
                            ui.info(f"Skill not found: {skill_arg}")
                    continue
                case _:
                    ui.info(f"Unknown command: /{cmd}")
                    continue

        messages = build_messages(query, global_context, context, current_session_history, active_skill)
        final_answer = ''
        tool_log = []
        interrupted = False
        parse_retries = 0
        # Per-turn cache: (path, offset, length) → formatted result; prevents redundant reads
        read_cache = {}

        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                full_raw, agent_answer = ui.stream_turn(papa_gnome_answers(model, tokenizer, messages))
                messages.append({"role": "assistant", "content": agent_answer})

                block = extract_code_block(agent_answer)
                if block:
                    last_code_block = block

                tool_calls = tool_call_extract(agent_answer)
                if not tool_calls:
                    if '<tool_call>' in agent_answer and parse_retries < 3:
                        parse_retries += 1
                        ui.info(f"Tool call parse failed — asking Papa Gnome to retry ({parse_retries}/3)...")
                        messages.append({"role": "tool", "content": "[Tool call failed: JSON could not be parsed. Please retry with a valid, complete tool call — ensure the JSON is properly closed and the </tool_call> tag is present.]"})
                        continue
                    ui.render_answer(agent_answer)
                    final_answer = agent_answer
                    break

                parse_retries = 0
                ui.clear_transient_residue()

                for tool in tool_calls:
                    name = tool['name']
                    args = tool['arguments']

                    if ui.DEBUG:
                        ui.show_tool_call(name, args)

                    # write_file: if content missing, pull from last code block seen this session
                    if name == 'write_file' and not args.get('content'):
                        code = extract_code_block(agent_answer) or last_code_block
                        if code:
                            args['content'] = code
                        else:
                            messages.append({"role": "tool", "content": "[write_file failed: no content provided and no code block found in your message. Please output the code in a markdown block first, then call write_file with the path.]"})
                            tool_log.append({'name': name, 'args': args, 'result': '[failed: no content]'})
                            continue

                    # Within-turn read deduplication
                    if name == 'read_file':
                        cache_key = (args.get('path'), args.get('offset'), args.get('length'))
                        if cache_key in read_cache:
                            cached = read_cache[cache_key]
                            note = '[Already read this turn — cached result. Use different offset/length to read another section.]\n'
                            result_msg = note + cached
                            messages.append({"role": "tool", "content": result_msg})
                            preview = result_msg[:500] + '...' if len(result_msg) > 500 else result_msg
                            tool_log.append({'name': name, 'args': args, 'result': preview})
                            continue

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
                    if name == 'load_skill' and tool_res.get('ok'):
                        active_skill = tool_res['result']
                    formatted = tool_registry.format_result(tool_res)
                    formatted = compact_tool_output(formatted, name)
                    ui.show_tool_result(name, tool_res)
                    if ui.DEBUG:
                        ui.show_tool_result_debug(name, tool_res)
                    messages.append({"role": "tool", "content": formatted})

                    if name == 'read_file':
                        cache_key = (args.get('path'), args.get('offset'), args.get('length'))
                        read_cache[cache_key] = formatted

                    preview = formatted[:500] + '...' if len(formatted) > 500 else formatted
                    tool_log.append({'name': name, 'args': args, 'result': preview})

        except KeyboardInterrupt:
            ui.show_interrupted()
            interrupted = True

        if not interrupted and not final_answer:
            final_answer = '[Reached step limit without a final answer. Try a more focused question.]'
            ui.show_step_limit_warning()

        agent_record = final_answer if final_answer else '[interrupted]'
        current_session_history.append({'user': query, 'agent': agent_record, 'tools': tool_log})
        current_session_history = current_session_history[-SESSION_HISTORY_WINDOW:]

        if not interrupted:
            ui.show_token_count(count_tokens(messages, tokenizer), tokenizer.model_max_length)
            ui.show_turn_divider()


if __name__ == '__main__':
    main()
