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

Models are configured in `config.py`. Change the path there to swap the primary or context-reducer model.

```python
# config.py
main_model = "./models/Qwen3.5-9B-reasoning-4bit"   # Papa Gnome (primary)
reducer_model = "mlx-community/Qwen3-4B-Instruct-2507-mxfp4"  # Mama Gnome (context reducer)
```

To convert a HuggingFace model to local MLX 4-bit:

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

**Mama Gnome (4B)** — context reducer, activated when a tool output exceeds ~500 tokens. Not yet wired in.

## File Structure

```
gnomes-lab/
├── main.py                  # REPL entry point + agentic loop
├── config.py                # Model paths
├── conversion.py            # Convert HF model → local MLX 4-bit
├── ui.py                    # Terminal UI (rich panels)
├── PLAN.md                  # Full implementation roadmap
└── gnomes_village/
    ├── papa_gnome.py        # Primary agent (9B): build_messages, stream
    ├── mama_gnome.py        # Context reducer (4B) — not yet wired in
    └── small_gnomes.py      # Unused
└── toolz/
    ├── tools.py             # Tool implementations
    └── tool_registry.py     # TOOL_SCHEMAS, dispatch(), format_result()
```

## Tools

8 tools available: `list_files`, `grep_search`, `read_file`, `edit_file`, `write_file`, `web_search`, `bash_exec`, `cd`.

Destructive tools (`bash_exec`, `write_file`, `edit_file`, `web_search`) require confirmation before running.

## TODO

- [ ] Persistent history (`~/.gnomes/history.jsonl`)
- [ ] Always-on context files (`~/.gnomes/context.md`, `./GNOMES.md`)
- [ ] Agentic memory (`~/.gnomes/memory/`)
- [ ] 4B context reducer for large tool outputs
- [ ] Slash commands (`/clear`, `/history`, `/tools`)
- [ ] Ctrl+C handling during generation
- [ ] Token count indicator per turn
