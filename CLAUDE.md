# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Gnomes Lab

A local, lightweight personal assistant running entirely on Apple Silicon. The goal is a Claude Code-like experience powered by local models — interactive REPL, tool use (file ops, bash, search), agentic loop, and persistent memory.

---

## Architecture

### Model roles

| Role | Name | Model | Runtime |
|------|------|-------|---------|
| Primary agent | Papa Gnome | `rafal-adamczyk/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit` (local, distilled) | mlx_lm → Metal/GPU |

**Papa Gnome (9B)** is the primary — reads every message, drives the agent loop, decides tool calls, generates the final answer.

### Flow

```
User message
    └─► Papa Gnome (9B) — reads full conversation context
            ├─ needs tools? → output <tool_call> → Python executes tools → repeat
            ├─ coding question? → answer with code in markdown blocks
            └─ stream final answer to user
```
---

## Models

**Primary agent (Papa Gnome):**
- Config: `config.py` — `Config.main_model`
- Default: `rafal-adamczyk/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit`
- Loaded via `mlx_lm` — instruct-tuned, reasoning-distilled from Claude Opus 4.6
- Always runs with `enable_thinking=True`

**Note on Qwen3.5 vs Qwen3:**
- Qwen3.5 mlx-community models have vision towers → require `mlx_vlm`
- Qwen3-Instruct models are text-only → use `mlx_lm`
- The distilled 9B (`rafal-adamczyk/...`) is text-only → use `mlx_lm`
- **Never change model names in code** — they may be newer than Claude's knowledge cutoff

---

## Files

```
gnomes-lab/
├── main.py                        # Entry point — REPL + agentic loop + slash commands
├── config.py                      # Config dataclass: model paths
├── conversion.py                  # Convert HF model to local MLX 4-bit
├── ui.py                          # Rich terminal UI: streaming, panels, tool display, approval
├── utils.py                       # tool_call_extract(), count_tokens(), load_context()
├── PLAN.md                        # Full implementation roadmap + status
├── skills/                        # Skill files loaded by Papa Gnome on demand
│   ├── code_generation.md
│   ├── file_ops.md
│   └── web_research.md
└── gnomes_village/
    ├── papa_gnome.py              # PRIMARY agent: build_messages(), papa_gnome_answers()
└── toolz/
    ├── tools.py                   # Tool implementations + approval policy
    └── tool_registry.py           # TOOL_SCHEMAS, dispatch(), format_result()
~/.gnomes/
    ├── context.md                 # personal always-on context
    ├── history.jsonl              # raw transcript
    └── memories.jsonl             # agent-authored insights
```

---

## Skills

Skills live in `skills/*.md`. Papa Gnome loads them on demand via `load_skill(name)`.
The available list is injected into the system prompt on every turn via `list_skills()` in `build_messages()`.

### Skill file structure (mandatory)

```markdown
# Skill: <Human Readable Name>
<One-sentence description — this line is shown to Papa Gnome in the system prompt skill list.>

---

## <Section>
...content...
```

Rules:
- **Line 1** — `# Skill: Name` (title, not shown in skill list)
- **Line 2** — plain sentence description, no markdown. This is what `list_skills()` extracts and what Papa Gnome reads to decide whether to load the skill.
- All subsequent content — detailed protocol, scenarios, examples, rules

### Current skills

| File | Description |
|---|---|
| `code_generation.md` | All code-related tasks: writing, editing, debugging, refactoring |
| `file_ops.md` | File and directory operations: reading, searching, editing, writing |
| `web_research.md` | Web research: multi-query search, cross-reference, cite sources |

### Adding a new skill

1. Create `skills/<name>.md` following the structure above
2. It appears automatically in the system prompt on the next run — no code changes needed

---

## Current Implementation Status

**Working:**
- Agentic loop in `main.py`: stream → parse tool calls → confirm → execute → feed results back → repeat
- Native Qwen3 tool calling via `<tool_call>` / `</tool_call>` tags
- Thinking token stripping: thinking shown to user (in `--verbose` mode), stripped from messages fed back to model
- Session history: last 5 turns injected into system prompt as a text log; tool summaries truncated to 500 chars
- Interrupted turns saved to history so follow-up questions have context
- Tool output compaction: applied to keep context window manageable
  - `read_file` — truncated if > 15000 chars
  - `bash_exec` — truncated if > 8000 chars
- Within-turn `read_file` deduplication: same (path, offset, length) returns cached result
- Tool guardrails: `write_file`, `edit_file`, `web_search` always require confirmation; `bash_exec` requires confirmation only for risky patterns (rm, mv, git push, etc.); others run automatically
- Blocked bash patterns: `rm -rf /`, `sudo`, `mkfs`, `dd if=`, etc.
- 8 tools: `list_files`, `grep_search`, `read_file` (with offset/length), `edit_file`, `write_file`, `web_search`, `bash_exec`, `cd`
- Approval UI: 3 options — Allow / Skip / Skip + feedback (feedback injected into model context)
- Diff view for `edit_file` approval: shows red/green unified diff instead of raw args
- Robust `tool_call_extract` in `utils.py`: two strategies — (1) balanced brace scan ignoring closing tag name, (2) tag-boundary fallback for model line-wrap inside JSON strings; `_escape_control_chars` fixes literal newlines before `json.loads`
- Slash commands: `/clear`, `/history [n]`, `/tools`, `/model`, `/tokens`, `/undo`
- Context loading: `load_global_context()` reads `~/.gnomes/context.md`, `load_context()` reads `GNOMES.md`/`CLAUDE.md`/`AGENTS.md` — both injected into system prompt at startup
- Time and working directory injected into every system prompt

**Not yet done:**
- Persistent history (`~/.gnomes/history.jsonl`)
- Agentic memory (`~/.gnomes/memories.jsonl`)
- Tool result cache (web_search TTL)

**Questions / nice-to-have for later:**
- Workflows (recall mode)
- Session recap on exit/startup
- Memory deduplication

---

## Tech Stack

- **mlx_lm** — used for Papa Gnome
  - `load(path)` returns `(model, tokenizer)`
  - `stream_generate()` returns token iterator
  - Sampling: `make_sampler(temp, top_p, min_p, top_k)` from `mlx_lm.sample_utils`
  - Repetition: `make_logits_processors(repetition_penalty)` from `mlx_lm.sample_utils`

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
- `apply_chat_template` with `add_generation_prompt=True` and `enable_thinking=True` injects `<think>` into the prompt
- Model stream starts already inside thinking — `<think>` never appears in the token stream
- `stream_turn()` in `ui.py` splits on `</think>`: everything before → `thinking_content`, everything after → `agent_answer`
- `full_raw` = entire stream including thinking (used only for verbose display)
- `agent_answer` (post-`</think>`) → appended to `messages` as assistant turn, parsed for tool calls, stored in session history
- Think blocks are NOT fed back into the model context — only the final answer is

### Agentic loop (main.py)
```
messages = build_messages(query, global_context, context, session_history)
for _ in range(MAX_TOOL_ITERATIONS=25):
    stream + collect → (full_raw, agent_answer)
    messages.append({"role": "assistant", "content": agent_answer})   # no think blocks
    if no tool_calls in agent_answer → render answer, break
    for each tool_call:
        confirm if requires_approval(name, args)  # policy in tools.py
        dispatch → format_result → compact_if_needed → messages.append({"role": "tool", "content": result})
        tool_log.append({name, args, result})
session_history.append({"user": query, "agent": final_answer, "tools": tool_log})
if len(session_history) > 5 → oldest turn dropped from window (history is lossy by design)
```

### Tool approval policy (`tools.py`)
```python
REQUIRE_APPROVAL = {"write_file", "edit_file", "web_search"}   # always prompt
# bash_exec: prompt only if command matches _RISKY_BASH_PATTERNS (rm, mv, git push, etc.)
# All other tools: auto-run without prompt
```

---

## Environment

Uses `uv` for dependency management. Python 3.13.

```bash
uv sync                                      # install all dependencies
uv run main.py                               # run via managed venv
source .venv/bin/activate && python main.py  # or activate directly
python conversion.py                         # convert + quantize model to local MLX
uv tool install .                            # install global `gnomes` command
uv tool upgrade gnomes-lab .                 # reinstall after code changes
```

**Development workflow:** `uv tool install` creates a snapshot of the code. After editing files, run `uv tool upgrade gnomes-lab .` to update the installed version. For day-to-day development, use `uv run main.py` which reads live source.