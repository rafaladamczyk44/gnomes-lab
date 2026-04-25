import glob
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import os
from tavily import TavilyClient
load_dotenv('.env')


# ---- Approval policy ----
# These tools ALWAYS require approval regardless of arguments.
REQUIRE_APPROVAL = {"write_file", "edit_file", "web_search"}

# bash_exec is special: auto-run by default, but requires approval if the
# command matches any risky pattern (state-changing operations).
_RISKY_BASH_PATTERNS = [
    r"\brm\b",  # remove files
    r"\bmv\b",  # move files
    r"\bcp\b",  # copy files
    r"\bchmod\b",
    r"\bchown\b",  # permission changes
    r"\bgit\s+(push|commit|merge|rebase|reset|checkout|cherry-pick|stash|tag)\b",
    r"(>|>>)\s*(?!/dev/(?:null|stdout|stderr)\b)\S+",  # output redirection to actual files (not /dev/null)
    r"\bdocker\s+(system\s+prune|rm|stop|kill|restart)\b",
    r"\b(pip|npm|yarn|pnpm)\s+install\b",
    r"\bpython[23]?\s+\S+\.py\b",  # running python scripts
    r"\bcurl\s+.*\s+-o\b",
    r"\bwget\s+.*\s+-O\b",
]


# Patterns blocked in bash_exec for safety
_BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\b",
    r"mkfs\b",
    r"dd\s+if=",
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/",
]


def is_risky_bash_exec(cmd: str) -> bool:
    """Return True if a bash command modifies state and requires approval."""
    for pattern in _RISKY_BASH_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False


def requires_approval(name: str, args: dict) -> bool:
    """Determine whether a tool call requires user approval."""
    if name in REQUIRE_APPROVAL:
        return True
    if name == "bash_exec":
        return is_risky_bash_exec(args.get("cmd", ""))
    return False


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# CHANGE 3b — removed module-level TavilyClient(TAVILY_API_KEY) here.
# It crashed startup when TAVILY_API_KEY was missing even if web_search was never used.
# Client is now created lazily inside web_search().


def bash_exec(cmd: str, timeout: int = 30) -> dict:
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, cmd):
            return {"tool": "bash_exec", "ok": False, "result": None,
                    "error": f"Blocked: command matches unsafe pattern '{pattern}'"}
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "tool": "bash_exec",
            "ok": proc.returncode == 0,
            "result": {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode},
            "error": proc.stderr if proc.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {"tool": "bash_exec", "ok": False, "result": None, "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"tool": "bash_exec", "ok": False, "result": None, "error": str(e)}


def read_file(path: str, offset: int = None, length: int = None) -> dict:
    try:
        p = Path(path).expanduser()
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        total_lines = len(lines)
        if offset is not None or length is not None:
            start = (offset or 1) - 1
            end = start + length if length is not None else total_lines
            lines = lines[start:end]
            content = "\n".join(lines)
        return {
            "tool": "read_file",
            "ok": True,
            "result": {"content": content, "lines": total_lines, "path": str(p)},
            "error": None,
        }
    except Exception as e:
        return {"tool": "read_file", "ok": False, "result": None, "error": str(e)}


def write_file(path: str, content: str) -> dict:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {
            "tool": "write_file",
            "ok": True,
            "result": {"path": str(p)},
            "error": None,
        }
    except Exception as e:
        return {"tool": "write_file", "ok": False, "result": None, "error": str(e)}


def list_files(pattern: str) -> dict:
    try:
        matches = glob.glob(pattern, recursive=True)
        return {
            "tool": "list_files",
            "ok": True,
            "result": {"matches": sorted(matches), "count": len(matches)},
            "error": None,
        }
    except Exception as e:
        return {"tool": "list_files", "ok": False, "result": None, "error": str(e)}


def grep_search(pattern: str, path: str) -> dict:
    try:
        proc = subprocess.run(
            ["grep", "-rn", "--include=*", pattern, path],
            capture_output=True, text=True, timeout=15
        )
        matches = []
        for line in proc.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                matches.append({"file": parts[0], "line": parts[1], "text": parts[2]})
        return {
            "tool": "grep_search",
            "ok": True,
            "result": {"matches": matches, "count": len(matches)},
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"tool": "grep_search", "ok": False, "result": None, "error": "Timed out"}
    except Exception as e:
        return {"tool": "grep_search", "ok": False, "result": None, "error": str(e)}




# Track current working directory across tool calls
current_dir: str = os.getcwd()


# TODO: register?
def cd(path: str) -> dict:
    """Change current working directory."""
    # CHANGE 3a — `global current_dir` is required; without it, `current_dir = resolved_path`
    # is a local assignment and the module-level variable is never updated.
    global current_dir
    try:
        resolved_path = os.path.abspath(os.path.expanduser(path))
        resolved_path = os.path.normpath(resolved_path)
        if not os.path.exists(resolved_path):
            return {"tool": "cd", "ok": False, "result": None, "error": f"No such directory: {path}"}
        if not os.path.isdir(resolved_path):
            return {"tool": "cd", "ok": False, "result": None, "error": f"Not a directory: {path}"}

        previous_dir = current_dir       # save before overwrite
        os.chdir(resolved_path)
        current_dir = resolved_path      # now correctly updates the module-level var

        return {
            "tool": "cd",
            "ok": True,
            # CHANGE 3a — was `current_dir` (new path after reassignment); now `previous_dir`
            "result": {"path": resolved_path, "previous_dir": previous_dir},
            "error": None,
        }
    except Exception as e:
        return {"tool": "cd", "ok": False, "result": None, "error": str(e)}


def web_search(query: str, n: int = 5) -> dict:
    # TODO: plug in a search backend (e.g. duckduckgo_search, SearXNG, Brave API)
    # CHANGE 3b — lazy init: create TavilyClient here instead of at module import.
    # Returns a clean error if the API key is missing rather than crashing startup.
    if not TAVILY_API_KEY:
        return {"tool": "web_search", "ok": False, "result": None,
                "error": "TAVILY_API_KEY not set in environment"}
    try:
        client = TavilyClient(TAVILY_API_KEY)
        response = client.search(query=query, maxResults=3)
        results = [res['content'] for res in response['results']]
        return {"tool": "web_search", "ok": True, "result": results, "error": None}
    except Exception as e:
        return {"tool": "web_search", "ok": False, "result": None, "error": str(e)}


def edit_file(path: str, old_string: str, new_string: str) -> dict:
    try:
        p = Path(path).expanduser()
        content = p.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_string)
        if count == 0:
            return {"tool": "edit_file", "ok": False, "result": None, "error": "old_string not found in file"}
        if count > 1:
            return {"tool": "edit_file", "ok": False, "result": None, "error": f"old_string matches {count} locations — make it more specific"}
        updated = content.replace(old_string, new_string, 1)
        p.write_text(updated, encoding="utf-8")
        return {"tool": "edit_file", "ok": True, "result": {"path": str(p)}, "error": None}
    except Exception as e:
        return {"tool": "edit_file", "ok": False, "result": None, "error": str(e)}


