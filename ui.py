import os
import sys
import re
import difflib
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner
from rich.table import Table

console = Console()

def show_gnome_hut_demo():
    pass  # replaced by startup()

VERBOSE = '--verbose' in sys.argv or '-v' in sys.argv

_STRIP_HEADERS = re.compile(r'^(##\s*(Answer|Plan)\s*\n?|---\s*\n?)', re.MULTILINE)
_STRIP_TOOL_CALLS = re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL)
# CHANGE 1a — Qwen3 sometimes emits a second <think>…</think> block mid-response
# (inline re-reasoning before a tool call). stream_turn only splits on the first
# </think>, so subsequent blocks and stray closing tags land in agent_answer and
# render verbatim in the panel. Strip both patterns before display.
_STRIP_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)
_STRIP_THINK_CLOSE = re.compile(r'</think>')


def _strip_model_headers(text: str) -> str:
    text = _STRIP_THINK_BLOCK.sub('', text)   # CHANGE 1a — remove inline <think> blocks
    text = _STRIP_THINK_CLOSE.sub('', text)   # CHANGE 1a — remove stray </think> tags
    text = _STRIP_TOOL_CALLS.sub('', text)
    text = _STRIP_HEADERS.sub('', text)
    return text.lstrip('\n').rstrip()


def stream_turn(generator):
    """
    Stream a model turn with rich UI.
    Phase 1: spinner while thinking.
    Phase 2: live-stream the answer into a transient panel.
    Returns (full_raw, agent_answer).
    """
    full_raw = ''
    pending = ''
    thinking_content = ''
    agent_answer = ''

    # Phase 1 — thinking: spinner with live token count.
    think_tok = 0
    spinner = Spinner('dots', text='[dim]Thinking…[/dim]')
    with Live(
        spinner,
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        for chunk in generator:
            full_raw += chunk
            pending += chunk
            thinking_content += chunk
            think_tok += 1
            spinner.text = f'[dim]Thinking… ({think_tok} tok)[/dim]'
            if '</think>' in pending:
                agent_answer = pending.split('</think>', 1)[1]
                break

    if VERBOSE and thinking_content.strip():
        console.print(Panel(
            thinking_content.strip(),
            title='[dim]Thinking[/dim]',
            border_style='dim',
            padding=(0, 1),
        ))

    # Phase 2 — stream into a transient panel.
    with Live(
        _answer_panel(Text('…', style='dim')),
        console=console,
        refresh_per_second=20,
        transient=True,
    ) as live:
        for chunk in generator:
            full_raw += chunk
            agent_answer += chunk
            visible = _strip_model_headers(agent_answer)
            live.update(_answer_panel(visible if visible else Text('…', style='dim')))

    return full_raw, agent_answer


def clear_transient_residue():
    """Erase the leftover transient answer panel before showing tool indicators."""
    console.print()



def _answer_panel(content):
    """Build the Papa Gnome answer panel. Accepts str or Text."""
    if isinstance(content, str):
        content = Text(content)
    return Panel(
        content,
        title='[bold green]Papa Gnome[/bold green]',
        border_style='green',
        padding=(0, 1),
        width=console.width,
    )


def render_answer(agent_answer: str):
    """Re-print the final answer as a persistent panel after the transient stream."""
    visible = _strip_model_headers(agent_answer)
    if visible:
        console.print(_answer_panel(visible))


def _result_hint(name: str, res: dict) -> str:
    """One-word summary of a tool result for compact display."""
    if not res.get('ok'):
        return 'err'
    r = res.get('result') or {}
    if name == 'read_file':
        return f"{r.get('lines', '?')}L"
    if name == 'list_files':
        return f"{r.get('count', '?')} files"
    if name == 'grep_search':
        return f"{r.get('count', '?')} hits"
    if name == 'bash_exec':
        return f"exit {r.get('exit_code', '?')}"
    if name == 'web_search':
        results = r if isinstance(r, list) else []
        return f"{len(results)} results"
    if name in ('edit_file', 'write_file'):
        return 'saved'
    return 'ok'


def show_tool_summary(tool_log: list):
    """Compact one-liner replacing separate per-tool messages.
    Shows all tools executed in one model turn with brief result hints.
    e.g.  ⚙  read_file (66L)  ·  list_files (14 files)  (2 tools)
    """
    parts = []
    for name, res in tool_log:
        hint = _result_hint(name, res)
        hint_style = 'dim' if res.get('ok') else 'red'
        parts.append(f'[dim]{name}[/dim] [{hint_style}]({hint})[/{hint_style}]')
    joined = '  [dim]·[/dim]  '.join(parts)
    n = len(tool_log)
    suffix = f'  [dim]({n} tools)[/dim]' if n > 1 else ''
    console.print(f'  [dim]⚙[/dim]  {joined}{suffix}')


def _render_edit_diff(args) -> Text:
    path = args.get('path', '')
    old = args.get('old_string', '')
    new = args.get('new_string', '')
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, lineterm=''))
    t = Text()
    t.append(f'  {path}\n', style='bold')
    for line in diff[2:]:  # skip --- +++ header lines
        if line.startswith('+'):
            t.append(line + '\n', style='green')
        elif line.startswith('-'):
            t.append(line + '\n', style='red')
        elif line.startswith('@@'):
            t.append(line + '\n', style='cyan dim')
        else:
            t.append(line + '\n', style='dim')
    return t


def confirm_tool(name, args):
    """Yellow warning panel + prompt. Returns True if approved."""
    if name == 'edit_file':
        content = _render_edit_diff(args)
    else:
        args_str = ', '.join(f'[bold]{k}[/bold]={repr(v)}' for k, v in args.items())
        content = f'  {name}({args_str})'
    console.print(Panel(
        content,
        title='[yellow bold]⚠  Approval required[/yellow bold]',
        border_style='yellow',
        padding=(0, 1),
    ))
    # CHANGE 1d — same EOF guard as user_input(); default to skip (safe) on EOF
    try:
        choice = console.input('  [yellow][[1] Allow  [2] Skip  [3] Skip + feedback][/yellow] › ').strip()
    except EOFError:
        return False, None
    if choice == '1':
        return True, None
    if choice == '3':
        feedback = console.input('  [yellow]Feedback › [/yellow]').strip()
        return False, feedback
    return False, None


def show_tool_result(name, result):
    """Compact one-liner shown immediately after a tool executes."""
    hint = _result_hint(name, result)
    hint_style = 'dim' if result.get('ok') else 'red'
    console.print(f'  [dim]⚙[/dim]  [dim]{name}[/dim] [{hint_style}]({hint})[/{hint_style}]')


def show_skipped(name):
    console.print(f'  [dim]✗  {name} skipped[/dim]')


def show_step_limit_warning():
    """Amber panel shown when MAX_TOOL_ITERATIONS exhausts before a final answer."""
    console.print(Panel(
        '  Reached the step limit without a final answer.\n'
        '  Try breaking the task into smaller, more focused questions.',
        title='[yellow bold]⚠  Step limit[/yellow bold]',
        border_style='yellow',
        padding=(0, 1),
    ))


def show_turn_divider():
    """Dim horizontal rule between conversation turns."""
    console.rule(style='dim')


def user_input():
    # CHANGE 1c — EOFError is raised when stdin closes (Ctrl+D, piped input
    # exhausted, terminal killed). Without this, the app crashes with a traceback.
    try:
        return console.input('\n[bold cyan]You[/bold cyan] › ')
    except EOFError:
        return 'exit'


def show_token_count(used, max_tokens):
    """Coloured bar: green → yellow → red as context fills."""
    pct = used / max_tokens
    bar_width = 18
    filled = round(bar_width * pct)
    empty = bar_width - filled
    color = 'red' if pct >= 0.85 else 'yellow' if pct >= 0.6 else 'green'
    bar = f'[{color}]{"█" * filled}[/{color}][dim]{"░" * empty}[/dim]'
    console.print(f'  {bar} [dim]{used/1000:.1f}k / {max_tokens/1000:.0f}k ctx[/dim]')


def startup(model_name: str):
    """Claude Code-style header: gnome face on the left, info stacked on the right."""
    short_model = model_name.split('/')[-1] if '/' in model_name else model_name
    cwd = os.getcwd().replace(os.path.expanduser('~'), '~')

    # Gnome face built line-by-line to preserve leading whitespace in table cells.
    face = Text()
    face.append("       /\\ \n", style="bold green")
    face.append("      /  \\ \n", style="bold green")
    face.append("     / ^^ \\ \n", style="bold green")
    face.append("    | o  o | \n", style="green")
    face.append("    |  \\/  | \n", style="green")
    face.append("     \\ __ / \n", style="green")
    face.append("     /~~~~\\ \n", style="green")
    face.append("    / ~~~~ \\ \n", style="green")
    face.append("   /~~~~~~~~\\", style="green")

    info = Text.assemble(
        ("Gnomes Lab\n", "bold green"),
        (f"{short_model}\n", "dim"),
        (f"{cwd}\n", "dim"),
        ('type ', "dim"),
        ('"exit"', "cyan"),
        (' to quit', "dim"),
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(vertical='middle')
    grid.add_column(vertical='middle')
    grid.add_row(face, info)

    console.print()
    console.print(grid)
    console.print()


def info(message: str):
    """Dim status line for slash command feedback."""
    console.print(f'  [dim]ℹ {message}[/dim]')