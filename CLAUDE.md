# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Gnomes Lab

A local, lightweight personal assistant running entirely on Apple Silicon. The goal is a Claude Code-like experience powered by local models — interactive REPL, tool use (file ops, bash, search), agentic loop, and persistent memory.

See `PLAN.md` for the full implementation roadmap.

---

## Architecture

### Model roles

| Role | Name | Model | Runtime |
|------|------|-------|---------|
| Primary agent | Papa Gnome | `Qwen3.5-9B-reasoning-4bit` (local, distilled) | mlx_lm → Metal/GPU |
| Context reducer | Mama Gnome | `Qwen3-4B-Instruct-2507-mxfp4` | mlx_lm → Metal/GPU |

All models run at **4-bit quantization**. Memory: ~5 GB (9B) + ~2.5 GB (4B) = ~7.5 GB total on 16 GB M2 Pro.

**Papa Gnome (9B)** is the primary — it reads every message, drives the agent loop, decides tool calls, and generates the final answer.

**Mama Gnome (4B)** is a helper only — activated when a tool returns a large output (> ~500 tokens) to compress it before it enters the 9B's context. Never in the routing critical path.

### Flow

```
User message
    └─► Papa Gnome (9B) — reads full conversation context
            ├─ trivial? → answer directly
            ├─ needs tools? → output tool_calls JSON → Python executes tools
            │       └─ output > 500 tokens? → Mama Gnome (4B) compresses it
            ├─ needs memory? → call memory_recall tool → inject relevant facts
            └─ stream final answer to user
```

### Why not route via 4B first?

The tasks worth using a local assistant for are tool-heavy and multi-step — exactly where 9B-primary wins. A 4B routing classifier adds a round-trip to every query (including complex ones that need 9B anyway) with no benefit. Simple tasks where 4B would be faster are tasks you'd handle more quickly yourself.

---

## Models

**Primary agent (Papa Gnome):**
- Local path: `./models/Qwen3.5-9B-reasoning-4bit`
- Source: `Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2` converted via `conversion.py`
- Loaded via `mlx_lm` — instruct-tuned, reasoning-distilled from Claude Opus 4.6
- Use `tokenizer_config={"fix_mistral_regex": True}` when loading (tokenizer quirk)

**Context reducer (Mama Gnome):**
- `mlx-community/Qwen3-4B-Instruct-2507-mxfp4` — loaded via `mlx_lm`
- Text-only, instruct-tuned, reliable JSON output

**Embedding model (for semantic memory — optional):**
- `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` — already in HF cache
- Used for Tier 2 memory search (not yet implemented)

**Note on Qwen3.5 vs Qwen3:**
- Qwen3.5 mlx-community models have vision towers → require `mlx_vlm`
- Qwen3-Instruct models are text-only → use `mlx_lm` (better JSON, instruct-tuned)
- The distilled 9B (`Jackrong/...`) is text-only → use `mlx_lm`

---

## Files

```
gnomes-lab/
├── main.py                        # Entry point — REPL loop (in progress)
├── config.py                      # Config dataclass: model paths, settings
├── conversion.py                  # Convert HF model to local MLX 4-bit
├── test.py                        # Quick model smoke test
├── utils.py                       # Shared helpers (partially outdated)
├── sketch.py                      # Original prototype (not imported)
├── PLAN.md                        # Full implementation roadmap
└── gnomes_village/
    ├── __init__.py
    ├── papa_gnome.py              # PRIMARY agent (9B): load, generate, tool-call loop
    ├── mama_gnome.py              # Context reducer (4B): compress large tool output
    ├── small_gnomes.py            # Legacy small gnome workers (2B, not in main flow)
    ├── tools.py                   # Tool implementations: bash_exec, read_file, etc.
    └── tool_registry.py           # Tool schemas, dispatch, format_result
~/.gnomes/
    ├── context.md                 # Personal always-on context (user-written)
    ├── history.jsonl              # Conversation history (append-only log)
    └── memory/
        ├── INDEX.md               # One-liner per memory file (loaded at startup)
        └── *.md                   # Agent-written memory files
```

---

## Config

```python
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    main_model: str = './models/Qwen3.5-9B-reasoning-4bit'   # Papa Gnome (9B)
    small_model: str = 'mlx-community/Qwen3-4B-Instruct-2507-mxfp4'  # Mama Gnome (4B)
```

Always use `os.path.expanduser()` when passing paths to `mlx_lm.load()`.

---

## Tech Stack

- **mlx_lm** — used for both Papa and Mama Gnome
  - `load(path, tokenizer_config={...})` returns `(model, tokenizer)`
  - `generate()` returns plain `str`
  - `stream_generate()` returns token iterator (for streaming final answers)
  - Sampling: `make_sampler(temp, top_p, min_p, top_k)` from `mlx_lm.sample_utils`
  - Repetition: `make_logits_processors(repetition_penalty)` from `mlx_lm.sample_utils`
  - Thinking: `tokenizer.apply_chat_template(..., enable_thinking=True/False)`

### mlx_lm load pattern
```python
import os
from mlx_lm import load as mlx_load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from config import Config

config = Config()
model, tokenizer = mlx_load(
    os.path.expanduser(config.main_model),
    tokenizer_config={"fix_mistral_regex": True},
)
```

### Tool-call JSON schema (Papa Gnome output)

Papa Gnome outputs one of two structures per turn:

```json
// Needs tools
{"tool_calls": [{"tool": "read_file", "args": {"path": "main.py"}}], "thinking": "..."}

// Final answer
{"answer": "the response text"}
```

`enable_thinking=False` for tool-call turns (structured JSON); `enable_thinking=True` for final answer.

### JSON parsing (tool-call response)
1. Strip everything before/up to `</think>` token if present
2. Try `raw_decode` from first `{`
3. Fallback: extract `{...}` block, clean escape issues

---

## Agent Flow Detail

1. **REPL** reads user input, appends to conversation history
2. **Papa Gnome** generates response — either `tool_calls` JSON or `answer`
3. If `tool_calls`: Python dispatches each tool (parallel via `ThreadPoolExecutor`)
   - If result > 500 tokens: **Mama Gnome** compresses it
   - Compressed/raw result appended to history as `role: tool`
   - Loop back to step 2
4. If `answer`: stream tokens to terminal, append to history, save to `history.jsonl`
5. If agent calls `memory_save`: write to `~/.gnomes/memory/`, update INDEX
6. If agent calls `memory_recall`: grep memory files, inject results into context

Max iterations: 10. User confirmation required before destructive tool calls.

---

## Environment

Uses `uv` for dependency management. Python 3.13.

```bash
uv sync                                      # install all dependencies
uv run main.py                               # run via managed venv
source .venv/bin/activate && python main.py  # or activate directly
python conversion.py                         # convert + quantize model to local MLX
python test.py                               # smoke test the primary model
```
