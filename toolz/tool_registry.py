from toolz.tools import bash_exec, read_file, write_file, edit_file, list_files, grep_search, web_search, fetch_url, list_skills, load_skill

# Compact schema fed into the model's system prompt
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern (e.g. '**/*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match files"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a regex pattern in files under a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use offset+length to read a specific line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from"},
                    "length": {"type": "integer", "description": "Number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing an exact string with new content. Use this for file modification (code files, correcting text, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Exact string to find (must be unique in the file)"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a new file or overwrite a file with new content. Use when specifically asked",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Full file content as a string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "n": {"type": "integer", "description": "Number of results to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract readable text from a web page URL. Use after web_search when a snippet is not enough. Only https:// URLs are allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full https:// URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default 15000)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all available skills by name and description.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load a task-specific protocol into the session. Once loaded, it stays active for all subsequent turns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name (e.g. 'code_generation', 'web_research')"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Run a shell command and return stdout/stderr. Use ONLY when **no other** tool fits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["cmd"],
            },
        },
    },
]

_DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
    "grep_search": grep_search,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "bash_exec": bash_exec,
    "list_skills": list_skills,
    "load_skill": load_skill,
}


def dispatch(tool_name: str, args: dict) -> dict:
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"tool": tool_name, "ok": False, "result": None, "error": f"Unknown tool: {tool_name}"}

    return fn(**args)


def format_result(result: dict) -> str:
    """Format a tool result dict into a readable string for the model context."""
    if not result["ok"]:
        return f"[Tool: {result['tool']}] ERROR: {result['error']}"

    tool = result["tool"]
    r = result["result"]

    if tool == "bash_exec":
        out = r["stdout"].strip()
        err = r["stderr"].strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"stderr: {err}")
        if not parts:
            parts.append(f"(exit code {r['exit_code']})")
        return f"[Tool: bash_exec]\n" + "\n".join(parts)

    if tool == "read_file":
        return f"[Tool: read_file — {r['path']} ({r['lines']} lines total)]\n{r['content']}"

    if tool == "write_file":
        return f"[Tool: write_file] Written to {r['path']}"

    if tool == "edit_file":
        return f"[Tool: edit_file] Edited {r['path']}"

    if tool == "list_files":
        files = "\n".join(r["matches"]) if r["matches"] else "(no matches)"
        return f"[Tool: list_files] {r['count']} matches:\n{files}"

    if tool == "grep_search":
        if not r["matches"]:
            return f"[Tool: grep_search] No matches found"

        lines = [f"{m['file']}:{m['line']}: {m['text']}" for m in r["matches"][:50]]
        suffix = f"\n... [{r['count'] - 50} more]" if r["count"] > 50 else ""
        return f"[Tool: grep_search] {r['count']} matches:\n" + "\n".join(lines) + suffix

    if tool == "web_search":
        lines = []
        for i, res in enumerate(r):
            title = res.get("title") or "Untitled"
            url = res.get("url") or ""
            snippet = res.get("content", "")
            lines.append(f"{i+1}. {title}\n   URL: {url}\n   {snippet}")
        return f"[Tool: web_search]\n" + "\n\n".join(lines)

    if tool == "fetch_url":
        title = r.get("title") or "Untitled"
        content = r.get("content", "")
        header = f"[Tool: fetch_url — {title}\n{r['url']} ({r['chars']} chars)]"
        return f"{header}\n{content}"

    if tool == "list_skills":
        if not r:
            return "[Tool: list_skills] No skills available."
        lines = [f"- {s['name']}: {s['description']}" for s in r]
        return "[Tool: list_skills]\n" + "\n".join(lines)

    if tool == "load_skill":
        return f"[Skill loaded: {result['skill_name']} — active for this session]"

    return f"[Tool: {tool}] {r}"


