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


def tool_call_extract(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)

    if not matches:
        return None

    return [json.loads(_escape_control_chars(m.strip())) for m in matches]


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
    context_file = os.path.join(os.getcwd(), 'GNOMES.md')
    if not os.path.exists(context_file):
        return ""

    try:
        with open(context_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return ""