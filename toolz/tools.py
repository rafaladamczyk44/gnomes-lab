import glob
import re
import subprocess
from pathlib import Path

# Patterns blocked in bash_exec for safety
_BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\b",
    r"mkfs\b",
    r"dd\s+if=",
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/",
]


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


def read_file(path: str) -> dict:
    try:
        p = Path(path).expanduser()
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        return {
            "tool": "read_file",
            "ok": True,
            "result": {"content": content, "lines": len(lines), "path": str(p)},
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


def web_search(query: str, n: int = 5) -> dict:
    # TODO: plug in a search backend (e.g. duckduckgo_search, SearXNG, Brave API)
    return {
        "tool": "web_search",
        "ok": False,
        "result": None,
        "error": "web_search not yet implemented — no backend configured",
    }
