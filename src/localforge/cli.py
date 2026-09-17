from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from localforge.catalog import load_catalog, recommendations
from localforge.hardware import detect_hardware
from localforge.orchestrator import run as run_orchestrator

app = typer.Typer(
    name="localforge",
    help="A frontier model orchestrates open-weight models running locally on your machine.",
)
console = Console()


@app.command()
def scan() -> None:
    """Detect this machine's hardware."""
    hw = detect_hardware()
    console.print(f"OS: {hw.os} ({hw.arch})")
    console.print(f"CPU cores: {hw.cpu_cores}")
    console.print(f"RAM: {hw.ram_gb} GB")
    if hw.gpus:
        for gpu in hw.gpus:
            console.print(f"GPU: {gpu.name} — {gpu.vram_gb} GB VRAM ({gpu.backend})")
    else:
        console.print("GPU: none detected (CPU-only)")


@app.command()
def models() -> None:
    """Show the best-fitting local model per task type for this machine."""
    hw = detect_hardware()
    recs = recommendations(hw)

    table = Table(title="Recommended local models for this machine")
    table.add_column("Modality")
    table.add_column("Model")
    table.add_column("Runtime")
    table.add_column("Quality tier")

    for modality, entry in recs.items():
        if entry is None:
            table.add_row(modality, "[red]none fit this hardware[/red]", "-", "-")
        else:
            table.add_row(modality, entry.name, entry.runtime, str(entry.quality_tier))

    console.print(table)


@app.command()
def catalog() -> None:
    """List every model in the catalog, regardless of hardware fit."""
    table = Table(title="Full model catalog")
    table.add_column("Model")
    table.add_column("Modality")
    table.add_column("Runtime")
    table.add_column("Min VRAM (GB)")
    table.add_column("Min RAM (GB)")
    table.add_column("Quality tier")

    for entry in load_catalog():
        table.add_row(
            entry.name, entry.modality, entry.runtime,
            str(entry.min_vram_gb), str(entry.min_ram_gb), str(entry.quality_tier),
        )
    console.print(table)


@app.command()
def run(
    task: str = typer.Argument(..., help="What you want built, e.g. \"Build a todo REST API with docs\""),
    frontier_model: str = typer.Option(
        "claude-opus-5",
        "--model",
        "-m",
        help="Frontier model to orchestrate with (any LiteLLM model string, e.g. claude-opus-5, gpt-5).",
    ),
) -> None:
    """Run a task: the frontier model plans it and delegates subtasks to local models."""
    try:
        with console.status(f"[bold green]Orchestrating with {frontier_model}..."):
            result = run_orchestrator(task, frontier_model)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary: show a clean message, not a traceback
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None
    console.print(result)


if __name__ == "__main__":
    app()
