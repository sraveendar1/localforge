from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time

import typer
from rich.console import Console
from rich.table import Table

from localforge import config
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import load_catalog, recommendations
from localforge.hardware import detect_hardware
from localforge.orchestrator import run as run_orchestrator

FRONTIER_PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
FRONTIER_API_KEY_ENV_VARS = list(FRONTIER_PROVIDERS.values()) + ["AWS_ACCESS_KEY_ID"]  # Bedrock

app = typer.Typer(
    name="localforge",
    help="A frontier model orchestrates open-weight models running locally on your machine.",
)
console = Console()


@app.callback()
def _load_saved_config() -> None:
    config.load()


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
def setup() -> None:
    """One-time interactive setup: installs Ollama, pulls recommended models,
    and saves your frontier model API key so future runs just work.
    """
    console.print("[bold]localforge setup[/bold]\n")

    # 1. Ollama
    if shutil.which("ollama") is None:
        if platform.system() == "Darwin" and shutil.which("brew"):
            if typer.confirm("Ollama isn't installed. Install it now via Homebrew?", default=True):
                subprocess.run(["brew", "install", "ollama"], check=True)
        else:
            console.print(
                "[yellow]Ollama isn't installed.[/yellow] Install it from "
                "https://ollama.com, then re-run `localforge setup`."
            )
            raise typer.Exit(code=1)

    ollama = OllamaBackend()
    if not ollama.is_running():
        console.print("Starting Ollama...")
        if platform.system() == "Darwin" and shutil.which("brew"):
            subprocess.run(["brew", "services", "start", "ollama"], check=False)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for _ in range(10):
            if ollama.is_running():
                break
            time.sleep(1)
        else:
            console.print("[red]Could not confirm Ollama started.[/red] Start it manually and re-run setup.")
            raise typer.Exit(code=1)
    console.print("[green]✓[/green] Ollama is installed and running\n")

    # 2. Pull recommended local models for this hardware
    hw = detect_hardware()
    recs = recommendations(hw)
    to_pull = {e.name for e in recs.values() if e is not None and e.runtime == "ollama"}
    if to_pull and typer.confirm(f"Pull recommended local models ({', '.join(sorted(to_pull))})?", default=True):
        for model_name in sorted(to_pull):
            console.print(f"Pulling {model_name} (this can take a while)...")
            ollama.ensure_available(model_name)
    console.print("[green]✓[/green] Local models ready\n")

    # 3. Frontier model API key
    existing = [v for v in FRONTIER_API_KEY_ENV_VARS if os.environ.get(v)]
    if existing:
        console.print(f"[green]✓[/green] Frontier API key already set: {', '.join(existing)}\n")
    else:
        provider = typer.prompt(
            f"Which frontier model provider will you orchestrate with? ({'/'.join(FRONTIER_PROVIDERS)})",
            default="anthropic",
        ).strip().lower()
        env_var = FRONTIER_PROVIDERS.get(provider)
        if env_var is None:
            console.print(f"[red]Unknown provider {provider!r}.[/red] Skipping — set an API key manually later.")
        else:
            api_key = typer.prompt(f"Paste your {env_var}", hide_input=True)
            config.save({env_var: api_key})
            os.environ[env_var] = api_key
            console.print(f"[green]✓[/green] Saved {env_var} to {config.CONFIG_FILE}\n")

    console.print("[bold green]Setup complete.[/bold green] Try: localforge run \"...\"")


@app.command()
def doctor() -> None:
    """Check that everything localforge needs is installed and reachable."""
    ok = True

    if shutil.which("ollama") is not None:
        console.print("[green]✓[/green] Ollama is installed")
    else:
        ok = False
        console.print("[red]✗[/red] Ollama is not installed — get it from https://ollama.com")

    if OllamaBackend().is_running():
        console.print("[green]✓[/green] Ollama is running")
    else:
        ok = False
        console.print("[red]✗[/red] Ollama is not running — start it (e.g. `ollama serve` or `brew services start ollama`)")

    found_keys = [var for var in FRONTIER_API_KEY_ENV_VARS if os.environ.get(var)]
    if found_keys:
        console.print(f"[green]✓[/green] Frontier model API key found: {', '.join(found_keys)}")
    else:
        ok = False
        console.print(
            "[red]✗[/red] No frontier model API key set — export one of: "
            + ", ".join(FRONTIER_API_KEY_ENV_VARS)
        )

    hw = detect_hardware()
    recs = recommendations(hw)
    missing = [modality for modality, entry in recs.items() if entry is None]
    if not missing:
        console.print("[green]✓[/green] A local model fits every known modality")
    else:
        console.print(f"[yellow]![/yellow] No fitting model for: {', '.join(missing)} (hardware too limited)")

    if ok:
        console.print("\n[bold green]Ready to go.[/bold green] Try: localforge run \"...\"")
    else:
        console.print("\n[bold red]Fix the items above before running `localforge run`.[/bold red]")
        raise typer.Exit(code=1)


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
