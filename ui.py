import sys
import re
import difflib
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner

console = Console()

# --- Gnome Hut welcome ---
def show_gnome_hut_demo():
    console.print()
    console.print('  [bold green]🏡 Gnome Hut[/bold green]')
    console.print()
    console.print('  Welcome to the Gnome Village!')
    console.print()
    console.print('  ⛺ The Hut ⛺')
    console.print()
    console.print('  [green]   /\\   [/green]')
    console.print('  [green]  /  \\  [/green]')
    console.print('  [green] | O O| [/green]')
    console.print('  [green] |  U | [/green]')
    console.print('  [green]  \\__/  [/green]')
    console.print()

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
    Phase 2: live-stream the answer into a panel.
    Returns (full_raw, agent_answer).
    """
    full_raw = ''
    pending = ''
    thinking_content = ''
    agent_answer = ''

    # Phase 1 — thinking: show spinner, buffer until </think>
    with Live(
        Spinner('dots', text='[dim]Thinking...[/dim]'),
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        for chunk in generator:
            full_raw += chunk
            pending += chunk
            thinking_content += chunk
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

    # Phase 2 — answer: live-stream remaining tokens into a panel
    def _answer_panel(text):
        return Panel(
            Text(text),
            title='[bold green]Papa Gnome[/bold green]',
            border_style='green',
            padding=(0, 1),
            width=console.width,
        )

    with Live(
        _answer_panel(''),
        console=console,
        refresh_per_second=20,
    ) as live:
        for chunk in generator:
            full_raw += chunk
            agent_answer += chunk
            # CHANGE 1b — when the model puts its plan inside <think> and emits
            # only a <tool_call>, stripping leaves empty string → blank panel.
            # Show a dim placeholder so the user knows something is happening.
            visible = _strip_model_headers(agent_answer)
            live.update(_answer_panel(visible if visible else '[dim]…[/dim]'))

    return full_raw, agent_answer


def show_tool_auto(name, args):
    """Dim one-liner for tools that run without approval."""
    args_str = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
    console.print(f'  [dim]⚙  {name}({args_str})[/dim]')


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
    """Dim panel showing tool output — only in verbose mode."""
    if not VERBOSE:
        return
    display = result if len(result) < 1500 else result[:1500] + '\n[dim]... truncated[/dim]'
    console.print(Panel(
        display,
        title=f'[dim]↳ {name}[/dim]',
        border_style='dim',
        padding=(0, 1),
    ))


def show_skipped(name):
    console.print(f'  [dim]✗  {name} skipped[/dim]')


def user_input():
    # CHANGE 1c — EOFError is raised when stdin closes (Ctrl+D, piped input
    # exhausted, terminal killed). Without this, the app crashes with a traceback.
    try:
        return console.input('\n[bold cyan]You[/bold cyan] › ')
    except EOFError:
        return 'exit'


def show_token_count(used, max_tokens):
    pct = used / max_tokens * 100
    console.print(f'  [dim][{used/1000:.1f}k / {max_tokens/1000:.0f}k tokens — {pct:.0f}%][/dim]')


def startup(model_name):
    console.print(f'  [bold green]Gnomes Village[/bold green]')
    console.print(f'  [dim]model: {model_name}[/dim]')
    console.print(f'  [dim]type "exit" to quit{" · --verbose for thinking" if not VERBOSE else " · thinking visible"}[/dim]')
    console.print()
