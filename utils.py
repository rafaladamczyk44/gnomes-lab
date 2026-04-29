import re
import json
import os


def _escape_control_chars(s):
    """Escape literal control characters inside JSON string values."""
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ord(ch) < 0x20:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(f'\\u{ord(ch):04x}')
        else:
            result.append(ch)
    return ''.join(result)


def _try_parse(raw: str) -> dict | None:
    """Apply control-char escaping then json.loads. Returns parsed dict or None."""
    cleaned = _escape_control_chars(raw.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def tool_call_extract(text):
    import sys
    calls = []
    for m in re.finditer(r'<tool_call>\s*', text):
        rest = text[m.end():]
        if not rest.startswith('{'):
            print(f'[tool_call_extract] WARNING: <tool_call> not followed by {{. Got: {rest[:80]!r}', file=sys.stderr)
            continue

        # Strategy 1: balanced brace scan — handles wrong/missing closing tags.
        depth = 0
        in_string = False
        escape_next = False
        end = -1
        for i, ch in enumerate(rest):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end != -1:
            result = _try_parse(rest[:end])
            if result is not None:
                calls.append(result)
                continue
            # Balanced scan found braces but JSON still invalid — fall through to tag fallback.
            print(f'[tool_call_extract] balanced extraction got invalid JSON, trying tag fallback', file=sys.stderr)

        # Strategy 2: fall back to content between <tool_call> and the next closing tag.
        # The model line-wraps inside string values, embedding literal newlines that make
        # the balanced scan fail. _escape_control_chars fixes those before json.loads.
        tag_end = re.search(r'</[\w_]+>', rest)
        if tag_end:
            result = _try_parse(rest[:tag_end.start()])
            if result is not None:
                calls.append(result)
                continue

        print(f'[tool_call_extract] WARNING: all strategies failed. Snippet: {rest[:120]!r}', file=sys.stderr)

    return calls if calls else None


def count_tokens(messages, tokenizer) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_global_context() -> str:
    context_file = os.path.expanduser('~/.gnomes/context.md')
    if not os.path.exists(context_file):
        return ""
    try:
        with open(context_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def load_context() -> str:
    cwd = os.getcwd()
    for filename in ('GNOMES.md', 'CLAUDE.md', 'AGENTS.md'):
        context_file = os.path.join(cwd, filename)
        if os.path.exists(context_file):
            try:
                with open(context_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except (OSError, UnicodeDecodeError):
                continue
    return ""


