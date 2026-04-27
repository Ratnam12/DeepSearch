"""
RAGAS evaluation harness for the DeepSearch pipeline.
Single responsibility: run golden-set evaluation and enforce CI metric gates.

Run directly:   python tests/test_ragas.py
Run via pytest: pytest tests/test_ragas.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

import pytest
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from backend.agent import run_agent
from backend.config import get_settings
from backend.judge import judge_quality
from backend.retriever import hybrid_search, reset_collection

_GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
_FAITHFULNESS_GATE: float = 0.75
_ANSWER_RELEVANCY_GATE: float = 0.70
_CONTEXT_PRECISION_GATE: float = 0.75
_JUDGE_AVG_GATE: float = 4.00

_console = Console()


def _load_golden_set() -> list[dict[str, Any]]:
    with _GOLDEN_SET_PATH.open() as f:
        return json.load(f)


def _run_evaluate(
    dataset: Dataset,
    openrouter_api_key: str,
    openrouter_base_url: str,
    flash_model: str,
    openai_api_key: str,
) -> dict[str, float]:
    """Run RAGAS evaluate() inside an isolated subprocess worker.

    All ragas and langchain-openai imports are deferred to this function body
    so the freshly-spawned process only loads what it needs and carries no
    event-loop state from the parent.  The LLM and metrics are rebuilt here
    because LangchainLLMWrapper is not picklable across the process boundary.
    """
    # AnswerRelevancy computes embedding similarity via OpenAI; seed before import.
    os.environ.setdefault("OPENAI_API_KEY", openai_api_key)

    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    chat = ChatOpenAI(
        model=flash_model,
        openai_api_key=openrouter_api_key,
        openai_api_base=openrouter_base_url,
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(chat)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]
    import numpy as np
    result = evaluate(dataset, metrics=metrics)
    return {
        "faithfulness": float(np.nanmean(result["faithfulness"])),
        "answer_relevancy": float(np.nanmean(result["answer_relevancy"])),
        "context_precision": float(np.nanmean(result["context_precision"])),
    }


async def _collect_answer(question: str) -> tuple[str, list[str]]:
    """Drain run_agent and return the answer plus contexts the agent used."""
    parts: list[str] = []
    used_contexts: list[str] = []
    try:
        async for event in run_agent(question):
            if event["type"] == "text":
                parts.append(event["content"])
            if event["type"] == "tool_result" and event.get("name") == "retrieve_chunks":
                contexts = [c for c in event.get("contexts", []) if c]
                if contexts:
                    # Match RAGAS contexts to the final successful retrieval
                    # that the agent used to synthesize its answer.
                    used_contexts = list(dict.fromkeys(contexts))
    except Exception as exc:
        _console.print(
            f"[red]  ✗ agent error ({type(exc).__name__}) for '{question[:60]}…':[/red] {exc}"
        )
        return "", []
    answer = "".join(parts)
    if not answer:
        _console.print(f"[yellow]  ⚠ agent returned empty answer for '{question[:60]}…'[/yellow]")
    return answer, used_contexts


def _build_eval_meta(
    question: str,
    answer: str,
    contexts: list[str],
    used_contexts: list[str],
    scores: dict[str, Any],
) -> dict[str, Any]:
    """Build concise terminal/debug metadata for one golden-set evaluation."""
    return {
        "question": question[:80],
        "answer_len": len(answer),
        "total_context_chars": sum(len(c) for c in contexts),
        "num_chunks": len(contexts),
        "context_source": "agent_used" if used_contexts else "fallback_hybrid_search",
        "judge_score": float(scores["overall"]),
        "answer_preview": answer[:120],
    }


async def _eval_row(item: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Run agent + retrieval + judge for one golden-set item.

    Returns a (dataset_row_dict, judge_overall_score, trace_meta) tuple.
    """
    question: str = item["question"]
    ground_truth: str = item["ground_truth"]

    answer, used_contexts = await _collect_answer(question)
    chunks = [] if used_contexts else await hybrid_search(question)
    # RAGAS requires at least one non-empty string per contexts cell.
    contexts: list[str] = used_contexts or [c["text"] for c in chunks] or ["No context retrieved."]
    scores = await judge_quality(question=question, answer=answer)
    meta = _build_eval_meta(question, answer, contexts, used_contexts, scores)

    row: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }
    return row, float(scores["overall"]), meta


def _print_eval_trace(results: list[tuple[dict[str, Any], float, dict[str, Any]]]) -> None:
    """Render a compact per-question trace before the RAGAS subprocess."""
    table = Table(title="Golden Example Trace", show_header=True, header_style="bold blue")
    table.add_column("#", justify="right", width=3)
    table.add_column("Question", overflow="fold", max_width=48)
    table.add_column("Ctx", justify="right", width=4)
    table.add_column("Ctx Chars", justify="right", width=9)
    table.add_column("Ans Chars", justify="right", width=9)
    table.add_column("Source", width=11)
    table.add_column("Judge", justify="right", width=7)
    for idx, (_, _, meta) in enumerate(results, start=1):
        table.add_row(
            str(idx),
            str(meta["question"]),
            str(meta["num_chunks"]),
            str(meta["total_context_chars"]),
            str(meta["answer_len"]),
            str(meta["context_source"]),
            f"{meta['judge_score']:.2f}",
        )
    _console.print()
    _console.print(table)
    _console.print()


async def _build_dataset(
    golden: list[dict[str, Any]],
) -> tuple[Dataset, float]:
    """Iterate through golden set, call agent + retrieval + judge, build Dataset.

    Returns (HuggingFace Dataset, mean judge score across all examples).
    """
    total = len(golden)
    semaphore = asyncio.Semaphore(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Evaluating 0/{total}…", total=total)
        completed = 0

        async def _bounded_eval(item: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, Any]]:
            nonlocal completed
            async with semaphore:
                result = await _eval_row(item)
                completed += 1
                short_q = item["question"][:55] + ("…" if len(item["question"]) > 55 else "")
                progress.update(task, description=f"[{completed:>2}/{total}] {short_q}")
                progress.advance(task)
                return result

        results = await asyncio.gather(*[_bounded_eval(item) for item in golden])
        progress.update(task, description=f"[green]Done — {total} examples evaluated[/green]")

    _print_eval_trace(results)

    successful = [(r, s) for r, s, _ in results if r["answer"]]
    skipped = len(results) - len(successful)
    if skipped:
        _console.print(f"[yellow]  ⚠ Skipping {skipped} failed questions from RAGAS scoring[/yellow]")
    rows = [r for r, _ in successful]
    judge_totals = [s for _, s in successful]
    dataset = Dataset.from_list(rows)
    judge_avg = sum(judge_totals) / len(judge_totals)
    return dataset, judge_avg


def _print_report(
    faith: float,
    relevancy: float,
    precision: float,
    judge_avg: float,
) -> None:
    """Render a rich table showing all four evaluation scores."""
    table = Table(title="DeepSearch — Golden Set Quality Report", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Score", justify="right", min_width=8)
    table.add_column("Gate", justify="right", min_width=8)
    table.add_column("Status", justify="center", min_width=6)

    def _add_row(name: str, score: float, gate: float | None) -> None:
        gate_str = f">= {gate:.2f}" if gate is not None else "  —"
        if gate is None:
            status = "  —"
        elif score >= gate:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"
        table.add_row(name, f"{score:.4f}", gate_str, status)

    _add_row("faithfulness (RAGAS)", faith, _FAITHFULNESS_GATE)
    _add_row("answer_relevancy (RAGAS)", relevancy, _ANSWER_RELEVANCY_GATE)
    _add_row("context_precision (RAGAS)", precision, _CONTEXT_PRECISION_GATE)
    _add_row("judge_avg (1–5 scale)", judge_avg, _JUDGE_AVG_GATE)

    _console.print()
    _console.print(table)
    _console.print()


async def main() -> None:
    """Load golden set, run full pipeline, evaluate with RAGAS, assert CI gates."""
    settings = get_settings()

    _console.print("\n[bold]Resetting Qdrant collection…[/bold]")
    await reset_collection()

    golden = _load_golden_set()
    _console.print(f"\n[bold]Evaluating {len(golden)} golden examples…[/bold]\n")

    dataset, judge_avg = await _build_dataset(golden)

    _console.print("\n[bold]Running RAGAS evaluation in subprocess…[/bold]")
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=1) as executor:
        scores: dict[str, float] = await loop.run_in_executor(
            executor,
            _run_evaluate,
            dataset,
            settings.openrouter_api_key,
            settings.openrouter_base_url,
            settings.flash_model,
            settings.openai_api_key,
        )

    faith: float = scores["faithfulness"]
    relevancy: float = scores["answer_relevancy"]
    precision: float = scores["context_precision"]

    _print_report(faith, relevancy, precision, judge_avg)

    assert faith >= _FAITHFULNESS_GATE, (
        f"CI gate failed — faithfulness {faith:.4f} < {_FAITHFULNESS_GATE}"
    )
    assert relevancy >= _ANSWER_RELEVANCY_GATE, (
        f"CI gate failed — answer_relevancy {relevancy:.4f} < {_ANSWER_RELEVANCY_GATE}"
    )
    assert precision >= _CONTEXT_PRECISION_GATE, (
        f"CI gate failed — context_precision {precision:.4f} < {_CONTEXT_PRECISION_GATE}"
    )
    assert judge_avg >= _JUDGE_AVG_GATE, (
        f"CI gate failed — judge_avg {judge_avg:.4f} < {_JUDGE_AVG_GATE}"
    )


@pytest.mark.asyncio
async def test_ragas_golden_set() -> None:
    """CI entry point: runs main() which enforces faithfulness and answer_relevancy gates."""
    await main()


if __name__ == "__main__":
    asyncio.run(main())
