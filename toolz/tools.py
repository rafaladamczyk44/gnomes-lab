import glob
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from config import Config

load_dotenv('.env')

config = Config()


# ---- Approval policy ----
# These tools ALWAYS require approval regardless of arguments.
REQUIRE_APPROVAL = {"write_file", "edit_file", "web_search", "fetch_url"}

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
    r"(?:^|\s|/)\.env(?:\s|$)",  # reading .env files via shell is blocked
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
        if name in ("web_search", "fetch_url") and config.auto_web_permission:
            return False
        return True
    if name == "bash_exec":
        return is_risky_bash_exec(args.get("cmd", ""))
    return False


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# CHANGE 3b — removed module-level TavilyClient(TAVILY_API_KEY) here.
# It crashed startup when TAVILY_API_KEY was missing even if web_search was never used.
# Client is now created lazily inside web_search().


_EXCLUDE_DIRS = {'.git', '.venv', 'node_modules', '__pycache__', '.mypy_cache'}


def _is_env_file(path: str) -> bool:
    return os.path.basename(str(path)) == '.env'


def _has_excluded_component(path: str) -> bool:
    return any(part in _EXCLUDE_DIRS for part in Path(path).parts)


def _gitignored_set(paths: list) -> set:
    """Batch-check paths against .gitignore. Returns set of ignored paths; empty set on any error."""
    try:
        result = subprocess.run(
            ['git', 'check-ignore', '--stdin'],
            input='\n'.join(paths),
            capture_output=True, text=True, timeout=10
        )
        return set(result.stdout.splitlines())
    except Exception:
        return set()


def _is_gitignored(path: str) -> bool:
    try:
        result = subprocess.run(
            ['git', 'check-ignore', '-q', path],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


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
    if _is_env_file(path):
        return {"tool": "read_file", "ok": False, "result": None, "error": "Reading .env files is not permitted"}
    if _is_gitignored(path):
        return {"tool": "read_file", "ok": False, "result": None, "error": f"Reading gitignored files is not permitted: {path}"}
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
    if _is_env_file(path):
        return {"tool": "write_file", "ok": False, "result": None, "error": "Writing .env files is not permitted"}
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
        matches = [m for m in matches if not _has_excluded_component(m)]
        if matches:
            ignored = _gitignored_set(matches)
            matches = [m for m in matches if m not in ignored]
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
            ["grep", "-rn",
             "--exclude-dir=.git", "--exclude-dir=.venv", "--exclude-dir=node_modules",
             "--exclude-dir=__pycache__", "--exclude-dir=.mypy_cache",
             "--binary-files=without-match",
             pattern, path],
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
    # CHANGE 3b — lazy init: create TavilyClient here instead of at module import.
    # Returns a clean error if the API key is missing rather than crashing startup.
    if not TAVILY_API_KEY:
        return {"tool": "web_search", "ok": False, "result": None,
                "error": "TAVILY_API_KEY not set in environment"}
    try:
        client = TavilyClient(TAVILY_API_KEY)
        response = client.search(query=query, maxResults=min(n, 10))
        results = [
            {"title": res.get("title", ""), "url": res.get("url", ""), "content": res.get("content", "")}
            for res in response.get("results", [])
        ]
        return {"tool": "web_search", "ok": True, "result": results, "error": None}
    except Exception as e:
        return {"tool": "web_search", "ok": False, "result": None, "error": str(e)}


def fetch_url(url: str, max_chars: int = 15000) -> dict:
    """Fetch and extract readable text from a URL. Only HTTPS is allowed for security."""
    if not url.startswith("https://"):
        return {"tool": "fetch_url", "ok": False, "result": None,
                "error": "Only https:// URLs are allowed. Plain http:// is not permitted for security."}

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        title = ""

        if "text/html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title:
                title = soup.title.get_text(strip=True)

            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                             "form", "noscript", "iframe", "svg", "canvas", "video", "audio"]):
                tag.decompose()

            texts = []
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                                      "p", "li", "pre", "code", "td", "th",
                                      "article", "section", "main", "div"]):
                text = tag.get_text(strip=True)
                if text:
                    texts.append(text)

            # Deduplicate adjacent identical lines (common in noisy HTML)
            deduped = []
            prev = None
            for line in texts:
                if line != prev:
                    deduped.append(line)
                    prev = line
            content = "\n\n".join(deduped)
        else:
            content = resp.text

        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
            truncated = True

        return {
            "tool": "fetch_url",
            "ok": True,
            "result": {
                "url": url,
                "title": title,
                "content": content,
                "chars": len(content),
                "truncated": truncated,
            },
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {"tool": "fetch_url", "ok": False, "result": None, "error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"tool": "fetch_url", "ok": False, "result": None, "error": f"Request failed: {e}"}
    except Exception as e:
        return {"tool": "fetch_url", "ok": False, "result": None, "error": str(e)}


SKILLS_DIR = os.path.join(os.path.dirname(__file__), '..', 'skills')


def list_skills() -> dict:
    try:
        skills = []
        for path in sorted(Path(SKILLS_DIR).glob('*.md')):
            lines = path.read_text(encoding='utf-8').splitlines()
            # Skip title line (starts with #) and separators (---), read first content line as description
            desc = next(
                (l.strip() for l in lines if l.strip() and not l.startswith('#') and l.strip() != '---'),
                path.stem
            )
            skills.append({'name': path.stem, 'description': desc})
        return {'tool': 'list_skills', 'ok': True, 'result': skills, 'error': None}
    except Exception as e:
        return {'tool': 'list_skills', 'ok': False, 'result': None, 'error': str(e)}


def load_skill(name: str) -> dict:
    path = Path(SKILLS_DIR) / f'{name}.md'
    if not path.exists():
        return {'tool': 'load_skill', 'ok': False, 'result': None, 'skill_name': name,
                'error': f'Skill not found: {name}'}
    content = path.read_text(encoding='utf-8')
    return {'tool': 'load_skill', 'ok': True, 'result': content, 'skill_name': name, 'error': None}


def edit_file(path: str, old_string: str, new_string: str) -> dict:
    if _is_env_file(path):
        return {"tool": "edit_file", "ok": False, "result": None, "error": "Editing .env files is not permitted"}
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
