from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from localforge import config
from localforge.advisor import recommend_models
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import load_catalog, recommendations
from localforge.config import FRONTIER_API_KEY_ENV_VARS, FRONTIER_PROVIDERS
from localforge.hardware import detect_hardware
from localforge.orchestrator import run as run_orchestrator

app = typer.Typer(
    name="localforge",
    help="A frontier model orchestrates open-weight models running locally on your machine.",
)
console = Console()


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    config.load()
    if ctx.invoked_subcommand is None:
        _print_getting_started()
        raise typer.Exit()


def _print_getting_started() -> None:
    ready = any(os.environ.get(v) for v in FRONTIER_API_KEY_ENV_VARS)
    if ready:
        body = (
            "[bold]You're set up.[/bold] Try:\n\n"
            '  [cyan]localforge run "Build a todo REST API with docs"[/cyan]\n\n'
            "Other commands: [bold]scan[/bold] · [bold]models[/bold] · [bold]doctor[/bold] · [bold]wizard[/bold]"
        )
    else:
        body = (
            "[bold]Get started in one step:[/bold]\n\n"
            "  [cyan]localforge setup[/cyan]   (or [cyan]localforge wizard[/cyan] for a terminal UI)\n\n"
            "That installs Ollama, has a frontier model pick local models for your\n"
            "hardware, and saves your API key — then you're ready for:\n\n"
            '  [cyan]localforge run "Build a todo REST API with docs"[/cyan]'
        )
    console.print(Panel(body, title="localforge", expand=False))


@app.command()
def scan() -> None:
    """Detect this machine's hardware."""
    hw = detect_hardware()
    console.print(f"OS: {hw.os} ({hw.arch})")
    console.print(f"CPU cores: {hw.cpu_cores}")
    console.print(f"RAM: {hw.ram_gb} GB")
    console.print(f"Free disk: {hw.free_disk_gb} GB")
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
    table.add_column("Disk (GB)")
    table.add_column("Quality tier")

    for entry in load_catalog():
        table.add_row(
            entry.name, entry.modality, entry.runtime,
            str(entry.min_vram_gb), str(entry.min_ram_gb), str(entry.disk_gb), str(entry.quality_tier),
        )
    console.print(table)


@app.command()
def wizard() -> None:
    """Launch the interactive terminal getting-started wizard (same steps as
    `setup`, but as a navigable screen-by-screen UI).
    """
    from localforge.tui import LocalforgeWizard

    LocalforgeWizard().run()


@app.command()
def setup() -> None:
    """One-time interactive setup: installs Ollama, pulls recommended models,
    and saves your frontier model API key so future runs just work.
    """
    console.print("[bold]localforge setup[/bold]\n")
    console.print("This installs everything localforge needs automatically; the only")
    console.print("thing you'll need to provide is a frontier model API key.\n")

    # 1. Ollama
    if shutil.which("ollama") is None:
        if platform.system() == "Darwin" and shutil.which("brew"):
            console.print("Installing Ollama via Homebrew...")
            subprocess.run(["brew", "install", "ollama"], check=True)
        else:
            console.print(
                "[yellow]Ollama isn't installed and can't be auto-installed on this OS.[/yellow] "
                "Install it from https://ollama.com, then re-run `localforge setup`."
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

    # 2. Frontier model API key -- needed now, before model selection, since
    # the frontier model itself picks which local models to download.
    existing = [v for v in FRONTIER_API_KEY_ENV_VARS if os.environ.get(v)]
    frontier_model: str | None = None
    if existing:
        console.print(f"[green]✓[/green] Frontier API key already set: {', '.join(existing)}\n")
        frontier_model = next((config.frontier_model_for_env_var(v) for v in existing), None)
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
            frontier_model = config.FRONTIER_DEFAULT_MODELS.get(provider)
            console.print(f"[green]✓[/green] Saved {env_var} to {config.CONFIG_FILE}\n")

    # 3. Hardware scan, then let the frontier model pick which local models
    # to download (constrained to catalog entries that already fit this
    # machine's RAM/VRAM/disk space -- see advisor.recommend_models).
    hw = detect_hardware()
    console.print(
        f"Hardware: {hw.ram_gb}GB RAM, {hw.total_vram_gb}GB VRAM, "
        f"{hw.free_disk_gb}GB free disk\n"
    )
    if frontier_model:
        console.print(f"Asking {frontier_model} to pick the best local models for this machine...")
        recs = recommend_models(hw, frontier_model)
    else:
        console.print("[yellow]No usable frontier model id — falling back to the built-in heuristic.[/yellow]")
        recs = recommendations(hw)

    to_pull = {e.name for e in recs.values() if e is not None and e.runtime == "ollama"}
    for model_name in sorted(to_pull):
        console.print(f"Pulling {model_name} (this can take a while, only happens once)...")
        ollama.ensure_available(model_name)
    console.print("[green]✓[/green] Local models ready\n")

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
