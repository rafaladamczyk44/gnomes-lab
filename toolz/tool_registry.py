from toolz.tools import bash_exec, read_file, write_file, list_files, grep_search, web_search

# Compact schema fed into the model's system prompt
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Run a shell command and return stdout/stderr. Use for git, python, file ops, etc.",
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
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a file with new content.",
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
]

_DISPATCH = {
    "bash_exec": bash_exec,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "grep_search": grep_search,
    "web_search": web_search,
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
        content = r["content"]
        if len(content) > 4000:
            content = content[:4000] + f"\n... [truncated, {r['lines']} lines total]"
        return f"[Tool: read_file — {r['path']}]\n{content}"

    if tool == "write_file":
        return f"[Tool: write_file] Written to {r['path']}"

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
        lines = [f"{i+1}. {res}" for i, res in enumerate(r)]
        return f"[Tool: web_search]\n" + "\n".join(lines)

    return f"[Tool: {tool}] {r}"


