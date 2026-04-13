# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Gnomes Lab

A local, lightweight personal assistant running entirely on Apple Silicon. The goal is a Claude Code-like experience powered by local models — interactive REPL, tool use (file ops, bash, search), agentic loop, and persistent memory.

See `PLAN.md` for the full implementation roadmap and current status.

---

## Architecture

### Model roles

| Role | Name | Model | Runtime |
|------|------|-------|---------|
| Primary agent | Papa Gnome | `Qwen3.5-9B-reasoning-4bit` (local, distilled) | mlx_lm → Metal/GPU |
| Context reducer | Mama Gnome | `Qwen3-4B-Instruct-2507-mxfp4` | mlx_lm → Metal/GPU |

All models run at **4-bit quantization**. Memory: ~5 GB (9B) + ~2.5 GB (4B) = ~7.5 GB total on 16 GB M2 Pro.

**Papa Gnome (9B)** is the primary — reads every message, drives the agent loop, decides tool calls, generates the final answer.

**Mama Gnome (4B)** is a helper only — activated when a tool returns a large output (> ~500 tokens) to compress it before it enters 9B's context. Not yet wired in.

### Flow

```
User message
    └─► Papa Gnome (9B) — reads full conversation context
            ├─ trivial? → answer directly
            ├─ needs tools? → output <tool_call> → Python executes tools
            │       └─ output > 500 tokens? → Mama Gnome (4B) compresses it  [not yet]
            └─ stream final answer to user
```

---

## Models

**Primary agent (Papa Gnome):**
- Local path: `./models/Qwen3.5-9B-reasoning-4bit`
- Source: `Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2` converted via `conversion.py`
- Loaded via `mlx_lm` — instruct-tuned, reasoning-distilled from Claude Opus 4.6

**Context reducer (Mama Gnome):**
- `mlx-community/Qwen3-4B-Instruct-2507-mxfp4` — loaded via `mlx_lm`

**Note on Qwen3.5 vs Qwen3:**
- Qwen3.5 mlx-community models have vision towers → require `mlx_vlm`
- Qwen3-Instruct models are text-only → use `mlx_lm`
- The distilled 9B (`Jackrong/...`) is text-only → use `mlx_lm`

---

## Files

```
gnomes-lab/
├── main.py                        # Entry point — REPL + agentic loop
├── config.py                      # Config dataclass: model paths
├── conversion.py                  # Convert HF model to local MLX 4-bit
├── PLAN.md                        # Full implementation roadmap + status
└── gnomes_village/
    ├── papa_gnome.py              # PRIMARY agent: build_messages(), papa_gnome_answers()
    ├── mama_gnome.py              # Context reducer (4B) — not yet wired into main flow
    └── small_gnomes.py            # Unused
└── toolz/
    ├── tools.py                   # Tool implementations (bash_exec, read_file, web_search, etc.)
    └── tool_registry.py           # TOOL_SCHEMAS, dispatch(), format_result()
```

---

## Current Implementation Status

**Working:**
- Agentic loop in `main.py`: stream → parse tool calls → confirm → execute → feed results back → repeat
- Native Qwen3 tool calling via `<tool_call>` / `</tool_call>` tags
- Thinking token stripping: `<think>...</think>` shown to user but stripped before history
- Session history: last 5 turns injected into system prompt
- Tool guardrails: `bash_exec`, `write_file`, `edit_file`, `web_search` require confirmation; others run automatically
- 8 tools: `list_files`, `grep_search`, `read_file` (with offset/length), `edit_file`, `write_file`, `web_search`, `bash_exec`, `cd`
- Approval UI: 3 options — Allow / Skip / Skip + feedback (feedback injected into model context)
- Diff view for `edit_file` approval: shows red/green unified diff instead of raw args
- JSON control-character sanitiser in `tool_call_extract` (handles literal newlines in model-generated JSON)
- Tool usage rules in system prompt steering agent away from `bash_exec` for file ops

**Not yet done:**
- Persistent history (`~/.gnomes/history.jsonl`)
- Always-on context files (`~/.gnomes/context.md`, `./GNOMES.md`)
- Agentic memory (`~/.gnomes/memory/`)
- 4B context reducer for large tool outputs
- Slash commands (`/clear`, `/history`, `/tools`)

---

## Tech Stack

- **mlx_lm** — used for both Papa and Mama Gnome
  - `load(path)` returns `(model, tokenizer)`
  - `stream_generate()` returns token iterator
  - Sampling: `make_sampler(temp, top_p, min_p, top_k)` from `mlx_lm.sample_utils`
  - Repetition: `make_logits_processors(repetition_penalty)` from `mlx_lm.sample_utils`
  - Thinking: reasoning model always starts in thinking mode; `apply_chat_template` injects `<think>` into generation prompt automatically

### mlx_lm load pattern
```python
from mlx_lm import load as mlx_load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from config import Config

config = Config()
model, tokenizer = mlx_load(config.main_model)
```

### Tool-call format (native Qwen3 chat template)

Model outputs:
```
<tool_call>
{"name": "read_file", "arguments": {"path": "main.py"}}
</tool_call>
```

Tool result fed back as:
```python
{"role": "tool", "content": "formatted result string"}
```

### Thinking token handling
- `apply_chat_template` with `add_generation_prompt=True` injects `<think>` into the prompt
- Model stream starts already inside thinking — `<think>` never appears in the token stream
- Start `in_thinking = True`, watch for `</think>` to switch
- `full_raw` (with thinking) → stored in `messages` list for model context
- `agent_answer` (post-`</think>`) → stored in session history, parsed for tool calls

### Agent output format (enforced via system prompt)

Simple answer:
```
## Answer
<response>
```

Tool-using turn:
```
## Plan
- step 1
- step 2
<tool_call>...</tool_call>
```

### Agentic loop (main.py)
```
messages = build_messages(query, session_history)
for _ in range(MAX_TOOL_ITERATIONS=10):
    stream + collect → (full_raw, agent_answer)
    messages.append({"role": "assistant", "content": full_raw})
    if no tool_calls in agent_answer → final answer, break
    for each tool_call:
        confirm if in REQUIRE_APPROVAL = {bash_exec, write_file, edit_file, web_search}
        dispatch → format_result → messages.append({"role": "tool", "content": result})
session_history.append({"user": query, "agent": final_answer})
```

---

## Environment

Uses `uv` for dependency management. Python 3.13.

```bash
uv sync                                      # install all dependencies
uv run main.py                               # run via managed venv
source .venv/bin/activate && python main.py  # or activate directly
python conversion.py                         # convert + quantize model to local MLX
```
