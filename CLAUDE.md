# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Gnomes Lab

A local, lightweight personal assistant running entirely on Apple Silicon. The goal is a Claude Code-like experience powered by local models — interactive REPL, tool use (file ops, bash, search), agentic loop, and persistent memory.

---

## Architecture

### Model roles

| Role | Name | Model | Runtime |
|------|------|-------|---------|
| Primary agent | Papa Gnome | `rafal-adamczyk/Qwen3.5-9B-MLX-4bit` (base, 4-bit) | mlx_lm → Metal/GPU |

**Papa Gnome (9B)** is the primary — reads every message, drives the agent loop, decides tool calls, generates the final answer.

There is no secondary/small model and no context-reducer model. Everything runs through Papa Gnome.

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
- Current: `rafal-adamczyk/Qwen3.5-9B-MLX-4bit` (base Qwen3.5-9B, 4-bit quantized)
- Loaded via `mlx_lm` — always runs with `enable_thinking=True` and `thinking_budget=2048`
- **Never change model names in code** — they may be newer than Claude's knowledge cutoff

**Note on Qwen3.5 model variants:**
- Base `Qwen3.5-9B-MLX-4bit` uses XML-style tool calls: `<function=name><parameter=key>value</parameter></function>`
- Distilled variant (`Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit`) uses JSON tool calls: `{"name": ..., "arguments": {...}}`
- `utils.py` `tool_call_extract()` handles both formats automatically
- 8-bit models (~18GB) exceed 16GB M2 Pro unified memory — use 4-bit only
- Qwen3.5 mlx-community models have vision towers → require `mlx_vlm`; text-only variants use `mlx_lm`

---

## Files

```
gnomes-lab/
├── main.py                        # Entry point — REPL + agentic loop + slash commands
├── config.py                      # Config dataclass: model paths + unused placeholder fields
├── conversion.py                  # Convert HF model to local MLX quantized
├── ui.py                          # Rich terminal UI: streaming, panels, tool display, approval
├── utils.py                       # tool_call_extract(), count_tokens(), load_context()
├── skills/                        # Skill files loaded by Papa Gnome on demand
│   ├── code_generation.md
│   ├── file_ops.md
│   └── web_research.md
├── gnomes_village/
│   ├── __init__.py
│   └── papa_gnome.py              # PRIMARY agent: build_messages(), papa_gnome_answers()
└── toolz/
    ├── tools.py                   # Tool implementations + approval policy + list_skills/load_skill
    └── tool_registry.py           # TOOL_SCHEMAS, dispatch(), format_result()
~/.gnomes/
    ├── context.md                 # personal always-on context
    ├── history.jsonl              # raw transcript (planned, not implemented)
    └── memories.jsonl             # agent-authored insights (planned, not implemented)
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
- Dual tool-call format support in `tool_call_extract()`: JSON (`{"name":...}`) and XML (`<function=name><parameter=k>v</parameter></function>`)
- Thinking streamed live token-by-token in a transient panel (persistent in `--verbose` mode)
- Thinking split: `</think>` as primary split; `<tool_call>` as fallback if model emits tool call before closing think block
- `</think>` and `</thinking>` variants both stripped from thinking display
- Session history: last 10 turns injected into system prompt as a text log; tool summaries truncated to 500 chars
- Interrupted turns saved to history so follow-up questions have context
- Tool output compaction: `read_file` > 15000 chars and `bash_exec` > 8000 chars truncated
- Within-turn `read_file` deduplication: same (path, offset, length) returns cached result
- Tool guardrails: `write_file`, `edit_file`, `web_search` always require confirmation; `bash_exec` requires confirmation only for risky patterns; others auto-run
- Blocked bash patterns: `rm -rf /`, `sudo`, `mkfs`, `dd if=`, etc.
- 9 tools: `list_files`, `grep_search`, `read_file`, `edit_file`, `write_file`, `web_search`, `bash_exec`, `list_skills`, `load_skill`
- Skills system: `skills/*.md` files auto-discovered; list injected into system prompt every turn; loaded skill injected as `## Active skill` section; `/skill [name|off]` slash command
- `load_skill` indicator shows skill name: `⚙ load_skill (code_generation)`
- Approval UI: 3 options — Allow / Skip / Skip + feedback
- Diff view for `edit_file` approval
- Slash commands: `/clear`, `/history [n]`, `/tools`, `/model`, `/tokens`, `/undo`, `/skill [name|off]`
- Context loading: `load_global_context()` reads `~/.gnomes/context.md`, `load_context()` reads `GNOMES.md`/`CLAUDE.md`/`AGENTS.md`
- Time and working directory injected into every system prompt

**Not yet done:**
- Persistent history (`~/.gnomes/history.jsonl`)
- Agentic memory (`~/.gnomes/memories.jsonl`)
- Tool result cache (web_search TTL — `web_search_cache_ttl` exists in `Config` but is unused)
- Context-size management (`max_context_tokens` and `compact_threshold` exist in `Config` but are unused)

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

### Chat template (current)

`papa_gnome_answers()` builds the prompt like this:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    tools=tool_registry.TOOL_SCHEMAS,
    add_generation_prompt=True,
    enable_thinking=True,
    thinking_budget=2048
)
```

Key facts:
- `enable_thinking=True` injects `<think>` into the prompt.
- `thinking_budget=2048` is passed explicitly.
- `tools=tool_registry.TOOL_SCHEMAS` registers the compact tool schemas with the chat template.
- The model stream starts already inside thinking — `<think>` never appears in the token stream.

### Tool-call formats

**JSON format** (distilled model — `Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-MLX-4bit`):
```
<tool_call>
{"name": "read_file", "arguments": {"path": "main.py"}}
</tool_call>
```

**XML format** (base model — `Qwen3.5-9B-MLX-4bit`):
```
<tool_call>
<function=read_file>
<parameter=path>
main.py
</parameter>
</function>
</tool_call>
```

Both are parsed by `tool_call_extract()` in `utils.py`. Tool results fed back as:
```python
{"role": "tool", "content": "formatted result string"}
```
The chat template wraps these in `<tool_response>` automatically.

### Thinking token handling
- `apply_chat_template` with `enable_thinking=True` and `thinking_budget=2048` injects `<think>` into the prompt
- Model stream starts already inside thinking — `<think>` never appears in the token stream
- `stream_turn()` in `ui.py` primary split: `</think>` → `thinking_content` / `agent_answer`
- Fallback split: if `<tool_call>` appears before `</think>` (base model behaviour), split there instead
- Thinking streamed live into a transient panel; disappears when answer starts (stays in `--verbose`)
- `agent_answer` (post-split) → appended to `messages` as assistant turn, parsed for tool calls, stored in session history
- Think blocks NOT fed back to the model — only the final answer is

### Agentic loop (`main.py`)

```python
SESSION_HISTORY_WINDOW = 10
MAX_TOOL_ITERATIONS = 25
active_skill: str = ""   # session-level; set when model calls load_skill()

messages = build_messages(query, global_context, context, current_session_history, active_skill)
for _ in range(MAX_TOOL_ITERATIONS):
    full_raw, agent_answer = ui.stream_turn(papa_gnome_answers(model, tokenizer, messages))
    messages.append({"role": "assistant", "content": agent_answer})   # no think blocks
    if not tool_call_extract(agent_answer):
        ui.render_answer(agent_answer)
        final_answer = agent_answer
        break
    for tool in tool_call_extract(agent_answer):
        name = tool['name']
        args = tool['arguments']
        if name == 'write_file' and not args.get('content'):
            args['content'] = extract_code_block(agent_answer) or last_code_block
        if requires_approval(name, args):
            approved, feedback = ui.confirm_tool(name, args)
            if not approved:
                # inject skip message + continue
        tool_res = tool_registry.dispatch(name, args)
        if name == 'load_skill' and tool_res.get('ok'):
            active_skill = tool_res['result']
        formatted = tool_registry.format_result(tool_res)
        formatted = compact_tool_output(formatted, name)
        messages.append({"role": "tool", "content": formatted})
        tool_log.append({'name': name, 'args': args, 'result': preview})
current_session_history.append({'user': query, 'agent': final_answer, 'tools': tool_log})
current_session_history = current_session_history[-SESSION_HISTORY_WINDOW:]
```

Notes:
- `active_skill` is session-level and survives across turns.
- `last_code_block` is a session-level cache used to backfill `write_file.content` when the model emits a markdown code block followed by a path-only `write_file` call.
- Within a single turn, `read_file` results are cached by `(path, offset, length)` to prevent redundant reads.

### Tool approval policy (`tools.py`)
```python
REQUIRE_APPROVAL = {"write_file", "edit_file", "web_search"}   # always prompt
# bash_exec: prompt only if command matches _RISKY_BASH_PATTERNS (rm, mv, git push, etc.)
# All other tools: auto-run without prompt
```

### Bash guardrails (`tools.py`)
- Blocked outright: `rm -rf /`, `sudo`, `mkfs`, `dd if=`, redirects to `/dev/sd*`, `chmod -R 777 /`
- Require approval: `rm`, `mv`, `cp`, `chmod`, `chown`, git mutations, output redirection to files, docker destructive ops, package installs, running `.py` scripts, `curl -o`, `wget -O`
- Safe commands (e.g. `ls`, `pwd`, `git status`, `git diff`) run without approval

---

## Environment

Uses `uv` for dependency management. Python 3.13.

```bash
uv sync                                      # install all dependencies
uv run main.py                               # run via managed venv
source .venv/bin/activate && python main.py  # or activate directly
python conversion.py                         # convert + quantize model to local MLX
```

### Testing / debugging the REPL

When testing or inspecting Papa Gnome's behavior, run with `--verbose` (or `-v`) so the thinking panel persists and you can see the model's reasoning:

```bash
uv run main.py --verbose
```

This is especially useful when checking why the model chose a particular tool, whether it loaded a skill, or how it parsed a tool call. Without `--verbose`, the thinking panel is transient and disappears once the answer starts.

For day-to-day development, use `uv run main.py` — it reads live source.
