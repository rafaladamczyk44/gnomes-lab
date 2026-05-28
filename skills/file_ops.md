# Skill: File Operations
This skill covers all file and directory operations: reading, searching, editing, and writing files.

---

## Tool selection — pick the right tool first

| Goal | Tool |
|---|---|
| Know what files exist | `list_files` with a glob pattern |
| Find where something is defined or used | `grep_search` |
| Read a file's content | `read_file` |
| Modify part of an existing file | `edit_file` |
| Create a new file or fully replace one | `write_file` |
| Move, rename, delete, or complex shell ops | `bash_exec` (last resort) |

Never use `bash_exec` for reading or editing when a dedicated tool exists.

---

## list_files

Use glob patterns — be specific to avoid noise.

```
{"name": "list_files", "arguments": {"pattern": "gnomes_village/**/*.py"}}
{"name": "list_files", "arguments": {"pattern": "*.md"}}
{"name": "list_files", "arguments": {"pattern": "toolz/tool*.py"}}
```

Excluded automatically: `.git`, `.venv`, `__pycache__`, `node_modules`, `.mypy_cache`, gitignored paths.

---

## grep_search

Use for finding where a symbol is defined, imported, or called.

```
{"name": "grep_search", "arguments": {"pattern": "def build_messages", "path": "."}}
{"name": "grep_search", "arguments": {"pattern": "load_skill", "path": "main.py"}}
{"name": "grep_search", "arguments": {"pattern": "REQUIRE_APPROVAL", "path": "toolz/"}}
```

- Returns file, line number, and matched text — use this to navigate before reading
- Cap results at 50 shown; if you need more, narrow the pattern or path
- For rename/refactor: grep for all call sites before changing a signature

---

## read_file

Always read before editing. Session history is not authoritative — files change.

**Full file:**
```
{"name": "read_file", "arguments": {"path": "main.py"}}
```

**Specific section (large files):**
```
{"name": "read_file", "arguments": {"path": "main.py", "offset": 80, "length": 40}}
```

- `offset` is 1-based line number
- Use grep_search first to find the line number, then read_file with offset+length
- Within a turn, the same (path, offset, length) is cached — re-reading returns the cached result

**Restrictions:**
- `.env` files: blocked
- Gitignored files: blocked

---

## edit_file

The default tool for modifying existing files. Always prefer over write_file.

```
{
  "name": "edit_file",
  "arguments": {
    "path": "main.py",
    "old_string": "    active_skill = \"\"\n    ui.info(\"Skill cleared.\")",
    "new_string": "    active_skill = \"\"\n    ui.info(\"Skill cleared.\")\n    continue"
  }
}
```

**Rules:**
- `old_string` must match **exactly** — whitespace, indentation, and all
- `old_string` must be **unique** in the file. If it appears more than once, add surrounding lines until it's unique
- Make one logical change per call — multiple `edit_file` calls in sequence are fine
- Do not reformat or touch adjacent code
- If removing code leaves orphan imports or unused variables, clean those up

**Common failure:** copying `old_string` from memory instead of a fresh `read_file` — the file may have changed. Always read first.

---

## write_file

For new files only, or when you need to replace the entire content.

**With code block (the safe pattern for code files):**
1. Output the full content in a markdown code block
2. Call `write_file` with path only — the harness fills content from your last code block

```
{"name": "write_file", "arguments": {"path": "utils/retry.py"}}
```

**Direct content (for short non-code files like configs, text):**
```
{"name": "write_file", "arguments": {"path": ".gitignore", "content": ".venv\n__pycache__\n.env\n"}}
```

**Never** use `write_file` to edit a file you've already read this turn — use `edit_file` instead. `write_file` overwrites everything, discarding any concurrent changes.

**Restrictions:** `.env` files blocked.

---

## Multi-file operations

1. Plan all changes in your thinking block before starting any tool calls
2. Read each file before editing — even if you read it earlier in a prior turn
3. Edit in dependency order: if B imports from A, edit A first
4. After all edits, state what changed and why in one sentence

---

## Navigating large files

Strategy for a file you haven't read yet:
1. `grep_search` to find the relevant symbol and its line number
2. `read_file` with `offset` + `length` around that line
3. `edit_file` on the specific section

Avoid reading the whole file when you only need 20 lines of it.
