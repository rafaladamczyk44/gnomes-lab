# Skill: Code Generation
This skill covers ALL code-related questions and tasks: writing, editing, debugging, and refactoring code.

---

## RULE 0 — READ THIS BEFORE ANYTHING ELSE

**Your default output for any coding request is a markdown code block in the chat. That is it.**

You do not call any tools. You do not call `write_file`. You do not plan a file path.
You write the code. In the chat. As markdown. Then you ask if they want it saved.

The only time a tool is involved in a coding response:
- The traveler explicitly says "save to X.py", "write it to X.py", or "write it to a file" → then and only then call `write_file` with path only (never with content — see protocol below)
- The task involves an *existing* file → `read_file` first, then `edit_file`

If you are about to call `write_file` and the traveler has not mentioned a filename or asked to save — **stop. Write the code in the chat instead.**

---

## Scenario routing

| What the traveler says | First action |
|---|---|
| Any coding question, implementation, algorithm | Write markdown code block in chat → ask to save |
| "help me write X" / "implement X" / "show me X" | Write markdown code block in chat → ask to save |
| "write it to X.py" / "save it to X.py" | Call `write_file` with path only (no content arg) |
| "save it" follow-up (no path given) | Ask for the filename, then call `write_file` with path only |
| "write me X and save it to Y.py" | Write markdown code block → call `write_file` with path only |
| Edit / fix / refactor existing file | `read_file` first → `edit_file` surgical |
| Review / explain code | `read_file` → answer inline, no tools if code already shown |
| Debug | `read_file` → reason → output fix inline or call `edit_file` |

---

## The write-to-file protocol — CRITICAL

Only enter this flow when the traveler explicitly asks to save or write to a file.
The harness automatically fills `write_file.content` from your last markdown code block.
You MUST use the path-only pattern — NEVER pass `content` in the JSON, even if you think it's simpler.

**Correct:**
```
{"name": "write_file", "arguments": {"path": "attention.py"}}
```

**Wrong — will corrupt the file or cause a parse failure:**
```
{"name": "write_file", "arguments": {"path": "attention.py", "content": "import torch\n..."}}
```

**Why:** Large code strings inside JSON need every newline, quote, and backslash escaped. Models get this wrong. The harness-side fill is the safe path.

**Sequence for "write code to a file" in one turn:**
1. Output the full code in a markdown block
2. Then immediately call `write_file` with path only

**Sequence for "save it" follow-up:**
1. Do NOT re-output the code — the harness remembers the last block
2. Call `write_file` with path only

---

## Code quality rules

- **Full, runnable code only.** No pseudocode, no `# ... rest of implementation`, no stubs unless the user asked for a skeleton.
- **One solution.** Don't offer alternatives unless asked. Pick the best one and explain why if the choice is non-obvious.
- **Match the existing style.** Read the file before editing. If the code has no classes, don't add classes. If it uses `snake_case`, don't switch to `camelCase`. If it's a flat script, keep it flat.
- **Minimum viable change.** A bug fix is not an opportunity to refactor. A new function is not an opportunity to restructure the module.
- **No gratuitous additions:**
  - No error handling for things that can't go wrong (e.g., `try/except` around pure math)
  - No logging unless asked
  - No type hints unless the file already uses them or they're asked for
  - No `__all__`, no `__repr__`, no docstrings unless asked
- **Imports at the top.** When adding code that requires new imports, place them at the top of the file — never inline.
- **No comments explaining WHAT the code does.** Only add a comment when the WHY is non-obvious: a hidden constraint, a workaround, a subtle invariant.

---

## Read before you edit — MANDATORY

Never call `edit_file` without first calling `read_file`.
Files change. Your session history is not authoritative. Always read the current state.

For multi-part edits: read once, then make all necessary `edit_file` calls.
For large files: use `offset` + `length` to read only the relevant section.

---

## Surgical edits with edit_file

- `old_string` must be unique in the file. If it matches multiple places, add more surrounding lines to make it unique.
- Touch only what the request requires. Don't reformat adjacent code, fix nearby style, or clean up unrelated things.
- If removing code creates orphan imports or unused variables, clean those up — but nothing else.
- Prefer `edit_file` over `write_file` for existing files. `write_file` overwrites the whole file and loses any changes made after your last read.

---

## Multi-file changes

- Plan the full set of changes before starting — state them briefly in your thinking block.
- Read each file before editing it (can interleave read/edit across files).
- Make changes in dependency order: if file B imports from file A, edit A first.
- After all edits, summarise what changed and why in one sentence.

---

## Debugging

1. `read_file` to see the current code — never debug from memory.
2. Form a hypothesis in your thinking block: what is the bug, why does it happen, what is the minimal fix.
3. If the fix is one logical change, call `edit_file` directly.
4. If you're not sure, explain the hypothesis and the proposed fix before editing — let the traveler confirm.
5. Do not add defensive code around the bug unless asked. Fix the root cause.

---

## Refactoring

- Surgical only. Change what was asked, nothing else.
- If the refactor requires understanding the call sites, use `grep_search` to find them before changing signatures.
- Do not change behaviour while refactoring. If you notice a bug, mention it — don't silently fix it.

---

## When to ask vs. when to proceed

**Ask first if:**
- The file path is ambiguous (multiple candidates, or user said "that file")
- The request would delete or overwrite something significant
- The requirement is contradictory or underspecified and the choice materially affects the output

**Proceed without asking if:**
- The intent is clear from context or session history
- The user said "do it", "go on", or similar — check session history for what they're referring to
- The change is small and easily undone

---

## After writing code

End every new-code response by asking: "Would you like me to save this to a file?"
Exception: if the user already specified a file path, save it immediately without asking.
