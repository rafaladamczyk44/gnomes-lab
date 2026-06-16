# Gnomes Lab

A local, lightweight personal assistant running on Apple Silicon. Claude Code-like experience powered by local models — interactive REPL, tool use, agentic loop.

## Quickstart

```bash
uv sync              # install deps
uv run main.py       # launch the REPL
```

Or with an activated venv:

```bash
source .venv/bin/activate && python main.py
```

## Switching the model

Models are configured in `config.py`. Change the path there to swap the primary model.
Default model is the base Qwen3.5-9B 4-bit MLX conversion:

```python
# config.py
main_model = "rafal-adamczyk/Qwen3.5-9B-MLX-4bit"  # Papa Gnome (primary)
```

To convert a HuggingFace model to local MLX:

```bash
python conversion.py
```

## Architecture

```
User message
    └─► Papa Gnome (9B) — reads full context, drives loop
            ├─ trivial? → answer directly
            ├─ needs tools? → <tool_call> → Python executes
            └─ stream final answer
```

There is only one model: Papa Gnome (9B). No secondary model, no context reducer.

## File Structure

```
gnomes-lab/
├── main.py                  # REPL entry point + agentic loop
├── config.py                # Model paths + unused config placeholders
├── conversion.py            # Convert HF model → local MLX quantized
├── ui.py                    # Terminal UI (rich panels)
├── utils.py                 # Tool-call parsing, token counting, context loading
├── skills/                  # Task-specific skill files
│   ├── code_generation.md
│   ├── file_ops.md
│   └── web_research.md
├── gnomes_village/
│   ├── __init__.py
│   └── papa_gnome.py        # Primary agent (9B): build_messages, stream
└── toolz/
    ├── tools.py             # Tool implementations + approval policy
    └── tool_registry.py     # TOOL_SCHEMAS, dispatch(), format_result()
```

## Tools

9 tools available: `list_files`, `grep_search`, `read_file`, `edit_file`, `write_file`, `web_search`, `bash_exec`, `list_skills`, `load_skill`.

Directory changes are handled through `bash_exec` if needed; there is no dedicated `cd` tool.

Destructive tools (`bash_exec`, `write_file`, `edit_file`, `web_search`) require confirmation before running.

## TODO

- [ ] Persistent history (`~/.gnomes/history.jsonl`)
- [ ] Agentic memory (`~/.gnomes/memories.jsonl`)
- [ ] Tool result cache (web_search TTL — `Config.web_search_cache_ttl` is unused)
- [ ] Context-size management (`Config.max_context_tokens` and `Config.compact_threshold` are unused)
- [ ] Session recap on exit/startup
