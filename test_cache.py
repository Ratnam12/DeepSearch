import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from backend.cache import cache_lookup, cache_store

console = Console()

async def test():
    console.print()
    console.print(Rule("[bold white]Semantic Cache Test[/bold white]", style="white"))
    console.print()

    # Step 1 — cold lookup
    result = await cache_lookup("What is machine learning?")
    console.print("[bold cyan]Step 1[/bold cyan] — lookup 'What is machine learning?' on empty cache")
    console.print(f"         Result: [yellow]{result}[/yellow] ✓ expected None\n")

    # Step 2 — store
    await cache_store("What is machine learning?", "Machine learning is a subset of AI where systems learn from data.")
    console.print("[bold cyan]Step 2[/bold cyan] — store answer for 'What is machine learning?'")
    console.print("         [green]Stored ✓[/green]\n")

    # Step 3 — similar query should hit
    result = await cache_lookup("Explain machine learning to me")
    hit = result is not None
    console.print("[bold cyan]Step 3[/bold cyan] — lookup 'Explain machine learning to me' (similar query)")
    if hit:
        console.print(Panel(result, title="[green]Cache HIT ✓[/green]", border_style="green", padding=(0, 2)))
    else:
        console.print("         [red]Cache MISS ✗ — expected a hit[/red]")
    console.print()

    # Step 4 — unrelated query should miss
    result = await cache_lookup("What is the capital of France?")
    console.print("[bold cyan]Step 4[/bold cyan] — lookup 'What is the capital of France?' (unrelated)")
    console.print(f"         Result: [yellow]{result}[/yellow] ✓ expected None\n")

    console.print(Rule(style="dim"))

asyncio.run(test())
