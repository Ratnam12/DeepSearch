import asyncio
import time
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from backend.agent import run_agent

THEME = Theme({
    "tool":    "bold cyan",
    "result":  "dim white",
    "answer":  "bold white",
    "meta":    "dim yellow",
    "success": "bold green",
    "error":   "bold red",
    "step":    "bold magenta",
})

console = Console(theme=THEME, highlight=False)

QUESTION = "How to become a senior AI engineer in 2026 From Scratch?"


def _tool_label(name: str) -> str:
    icons = {
        "web_search":       "🌐 web_search",
        "scrape_and_index": "🕷️  scrape_and_index",
        "retrieve_chunks":  "🗄️  retrieve_chunks",
    }
    return icons.get(name, f"🔧 {name}")


def _short_args(args: str) -> str:
    import json
    try:
        d = json.loads(args)
        return "  ".join(f"[meta]{k}[/meta]=[cyan]{v}[/cyan]" for k, v in d.items())
    except Exception:
        return args


def _format_result(name: str, content: str) -> str:
    return content


async def run():
    console.print()
    console.print(Rule("[bold white]DeepSearch Agent[/bold white]", style="white"))
    console.print(
        Panel(
            f"[bold white]{QUESTION}[/bold white]",
            title="[meta]Question[/meta]",
            border_style="white",
            padding=(0, 2),
        )
    )
    console.print()

    step = 0
    answer_parts: list[str] = []
    start = time.time()

    async for e in run_agent(QUESTION):
        if e["type"] == "tool_call":
            step += 1
            elapsed = f"{time.time() - start:.1f}s"
            console.print(
                f"[step]Step {step}[/step]  [tool]{_tool_label(e['name'])}[/tool]"
                f"  [meta]({elapsed})[/meta]"
            )
            console.print(f"       {_short_args(e['args'])}")

        elif e["type"] == "tool_result":
            formatted = _format_result(e["name"], e["content"])
            console.print(
                Panel(
                    Text.from_markup(formatted),
                    title=f"[meta]result · {e['name']}[/meta]",
                    border_style="dim",
                    padding=(0, 2),
                )
            )
            console.print()

        elif e["type"] == "text":
            answer_parts.append(e["content"])

    elapsed_total = f"{time.time() - start:.1f}s"
    full_answer = "".join(answer_parts)

    console.print(Rule("[bold green]Answer[/bold green]", style="green"))
    console.print(Markdown(full_answer))
    console.print()
    console.print(
        f"[meta]Completed in {elapsed_total}  ·  {step} tool calls  ·  "
        f"{datetime.now().strftime('%H:%M:%S')}[/meta]"
    )
    console.print(Rule(style="dim"))


asyncio.run(run())
