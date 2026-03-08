# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Gnomes Lab

A local, lightweight **mixture-of-agents** system. Designed to run entirely on Apple Silicon with efficient GPU/CPU split. Eventual goal: a Docker image exposable as a terminal or API (similar to Claude Code).

### Agent Architecture

| Role | Name | Model | Runtime |
|------|------|-------|---------|
| Router + Verifier | Mama Gnome | Qwen3-4B-Instruct-2507 (mxfp4) | mlx_lm → Metal/GPU |
| Worker agents (2–3 parallel) | Small Gnome | Qwen3.5-2B | llama-cpp-python → CPU |
| Fallback re-thinker (optional) | Big Gnome | Qwen3.5-9B (8bit) | mlx_lm → Metal/GPU |

**Flow:**
1. Mama Gnome routes the question → outputs JSON plan (strategy A/B/C)
2. Small Gnomes execute sub-tasks in parallel
3. Mama Gnome synthesizes small gnome responses (thinking mode ON) → returns final answer + confidence 0–10
4. If confidence < 7 → Big Gnome re-evaluates

### Compute Strategy

- **mlx_lm** handles Metal GPU ops (4B router/verifier, 9B fallback)
- **llama-cpp-python** handles CPU ops (2B agents via GGUF)
- Apple Silicon has **unified memory** — CPU and GPU share the same physical RAM pool; no separate VRAM
- Keep `n_gpu_layers=0` in llama-cpp to ensure agents stay on CPU and don't compete with MLX

### ⚠️ Model upgrade reminder
**Check periodically:** Has `mlx-community/Qwen3.5-4B-Instruct-*` been released on HuggingFace?
- Collection to check: https://huggingface.co/collections/mlx-community/qwen-35
- When available, swap `model_repo` in `gnomes_village/mama_gnome.py` — one-line change
- Qwen3.5 Instruct would replace `Qwen3-4B-Instruct-2507-mxfp4` as the router/verifier


## Files

```
gnomes-lab/
├── main.py                        # Entry point: loads Mama Gnome, runs a query
├── utils.py                       # Shared helpers (partially outdated, see below)
├── sketch.py                      # Original prototype / architecture inspiration (not imported)
└── gnomes_village/                # Agent package
    ├── __init__.py
    ├── mama_gnome.py              # Router + Verifier (4B mlx_lm): summon_mana_gnome(), router(), synthesize() (TODO)
    └── small_gnomes.py            # Small Gnomes (2B): summon_smol_gnomes(), thinking_gnome(), direct_gnome() (TODO)
```

### Key functions

| File | Function | Purpose |
|------|----------|---------|\
| `gnomes_village/mama_gnome.py` | `summon_mana_gnome()` | Loads 4B mlx_lm model; returns `(model, tokenizer)` |
| `gnomes_village/mama_gnome.py` | `router(model, tokenizer, user_input)` | Plans strategy A/B/C, returns JSON string |
| `gnomes_village/mama_gnome.py` | `synthesize()` | TODO: digest small gnome responses, return final answer + confidence |
| `gnomes_village/small_gnomes.py` | `summon_smol_gnomes()` | Loads 2B model; returns `(model, processor, config)` |
| `gnomes_village/small_gnomes.py` | `thinking_gnome(...)` | Runs reasoning sub-task |
| `gnomes_village/small_gnomes.py` | `direct_gnome(...)` | TODO: runs concise sub-task |

### Routing strategies (mama_gnome)

- **A — self-answer**: Mama Gnome answers directly (`answer` field filled, task fields null)
- **B — decompose**: splits into `task_think` (reasoning gnome) + `task_direct` (concise gnome)
- **C — parallel**: both small gnomes get the exact same problem independently

### Mama Gnome — two modes on the same model

The same loaded `(model, tokenizer)` is used for both calls:
- **Routing**: `enable_thinking=False` → fast, structured JSON output
- **Synthesis**: `enable_thinking=True` → thinking mode on, digests all gnome responses, produces final answer + confidence score

Synthesis prompt should include: original question, routing JSON, each gnome's labeled response, then ask for final answer + confidence 0–10.

Synthesis JSON schema:
```json
{
  "answer": "final synthesized answer",
  "confidence": 8,
  "gaps": "anything gnomes missed or disagreed on, or null"
}
```


## Models

**Router/Verifier (current):**
- `mlx-community/Qwen3-4B-Instruct-2507-mxfp4` — Qwen3 Instruct, text-only, loaded via `mlx_lm`

**Small Gnomes:**
- MLX: `mlx-community/Qwen3.5-2B-8bit` (via mlx_vlm — has vision tower)
- GGUF (preferred for CPU): `unsloth/Qwen3.5-2B-GGUF`, recommended `Q4_K_M` (~1.28 GB)

**Big Gnome (fallback):**
- `mlx-community/Qwen3.5-9B-8bit` (via mlx_vlm — has vision tower)

**Note on Qwen3.5 vs Qwen3:**
- Qwen3.5 mlx-community models have vision towers → must use `mlx_vlm`, not `mlx_lm`
- Qwen3-Instruct models are text-only → use `mlx_lm` (better JSON, instruct-tuned)
- No Qwen3.5 Instruct MLX version exists yet — watch the mlx-community collection


## Tech Stack

- **mlx_lm** — used for Mama Gnome (text-only Qwen3 Instruct)
  - `load()` returns `(model, tokenizer)` — no config needed
  - `generate()` returns plain `str`
  - Sampling: `make_sampler(temp, top_p, min_p, top_k)` from `mlx_lm.sample_utils`
  - Repetition: `make_logits_processors(repetition_penalty)` from `mlx_lm.sample_utils`
  - Thinking: `tokenizer.apply_chat_template(..., enable_thinking=True/False)`
- **mlx_vlm** — used for Small Gnomes and Big Gnome (Qwen3.5 with vision towers)
  - `load()` returns `(model, processor)`; also need `load_config()`
  - `generate()` returns `GenerationResult` — wrap with `text()` helper from `utils.py`
- **llama-cpp-python** — GGUF inference on CPU for Small Gnomes
  - Keep `n_gpu_layers=0`

### mlx_lm usage pattern (Mama Gnome)
```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load('mlx-community/Qwen3-4B-Instruct-2507-mxfp4')

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
response = generate(
    model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False,
    sampler=make_sampler(temp=0.5, top_p=0.95, min_p=0.05, top_k=20),
    logits_processors=make_logits_processors(repetition_penalty=1.5),
)
# response is a plain str
```

### JSON parsing (main.py)
Emergency fallback chain for mama gnome JSON output:
1. Strip everything before/up to `</think>` token
2. Try `raw_decode` from first `{`
3. Fallback: extract `{...}` block, `replace('\\"', '"').replace('""', '"')`
4. Strip leading/trailing `"` from all string values


## Agent flow detail

1. **Mama Gnome routes** — `enable_thinking=False`, outputs JSON plan (strategy A/B/C)
2. **Small Gnomes run in parallel** via `concurrent.futures.ThreadPoolExecutor` — separate instances (not thread-safe to share one)
3. **Mama Gnome synthesizes** — `enable_thinking=True`, receives original question + routing JSON + all gnome responses labeled, outputs final answer + confidence 0–10
4. **Big Gnome** — loaded on demand only if `confidence < 7` (saves memory)

### Thinking mode
- `enable_thinking=True/False` in `tokenizer.apply_chat_template()` — works on Qwen3 Instruct
- For Qwen3.5 via mlx_vlm: `/no_think` suffix on user message (chat template may not support `enable_thinking`)
- **2B model warning**: prone to infinite thinking loops — use `presence_penalty=1.5` and `temperature=1.0, top_p=0.95, top_k=20`


## Environment

Uses `uv` for dependency management. Python 3.13.

```bash
uv sync            # install all dependencies
uv add <pkg>       # add a dependency
uv run main.py     # run via managed venv
source .venv/bin/activate && python main.py  # or activate directly
```
