from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
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


@app.command()
def help(ctx: typer.Context) -> None:
    """Show this help message (same as --help)."""
    console.print(ctx.parent.get_help())


def _usage_bar(local_tokens: int, frontier_tokens: int, width: int = 40) -> str:
    """A Claude-Code-style horizontal bar showing the local/frontier token
    split for one run, e.g. "██████████████░░░░░░ 70% local / 30% frontier".
    """
    total = local_tokens + frontier_tokens
    if total == 0:
        return f"[dim]{'░' * width}[/dim] no tokens used"
    local_width = round(width * local_tokens / total)
    frontier_width = width - local_width
    bar = f"[green]{'█' * local_width}[/green][yellow]{'█' * frontier_width}[/yellow]"
    pct_local = round(100 * local_tokens / total)
    return f"{bar}  {pct_local}% local / {100 - pct_local}% frontier"


def _print_getting_started() -> None:
    ready = bool(os.environ.get(config.FRONTIER_MODEL_ENV_VAR))
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


def _installed_ollama_models() -> list[dict]:
    ollama = OllamaBackend()
    if not ollama.is_running():
        console.print("[red]Ollama is not running.[/red] Start it, then retry.")
        raise typer.Exit(code=1)
    return ollama.list_installed()


@app.command()
def installed() -> None:
    """List local models actually pulled via Ollama (not just the catalog)."""
    models_on_disk = _installed_ollama_models()
    if not models_on_disk:
        console.print("No local models installed yet. Run `localforge setup` or `localforge wizard`.")
        return

    table = Table(title="Installed local models")
    table.add_column("Name")
    table.add_column("Size (GB)")
    table.add_column("Modified")
    total_bytes = 0
    for m in models_on_disk:
        total_bytes += m.get("size", 0)
        table.add_row(m["name"], f"{m.get('size', 0) / (1024**3):.2f}", str(m.get("modified_at", ""))[:19])
    console.print(table)
    console.print(f"\nTotal: {total_bytes / (1024**3):.2f} GB across {len(models_on_disk)} model(s)")


@app.command()
def delete(
    models_arg: list[str] = typer.Argument(
        None, help="Model name(s) to delete, e.g. qwen2.5-coder:14b. Omit to choose interactively."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete locally installed model(s) to free disk space."""
    models_on_disk = _installed_ollama_models()
    if not models_on_disk:
        console.print("No local models installed.")
        return
    names = [m["name"] for m in models_on_disk]
    sizes = {m["name"]: m.get("size", 0) for m in models_on_disk}

    queue: list[str] = []
    if models_arg:
        for requested in models_arg:
            if requested in names:
                queue.append(requested)
            else:
                console.print(f"[yellow]Skipping — not installed: {requested}[/yellow]")
    else:
        table = Table(title="Installed local models")
        table.add_column("#")
        table.add_column("Name")
        table.add_column("Size (GB)")
        for i, name in enumerate(names, start=1):
            table.add_row(str(i), name, f"{sizes[name] / (1024**3):.2f}")
        console.print(table)

        selection = typer.prompt("Which to delete? (numbers or names, comma-separated, or 'all')")
        parts = names if selection.strip().lower() == "all" else [p.strip() for p in selection.split(",")]
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= len(names):
                queue.append(names[int(part) - 1])
            elif part in names:
                queue.append(part)
            else:
                console.print(f"[yellow]Skipping unknown selection: {part!r}[/yellow]")

    queue = sorted(set(queue))
    if not queue:
        console.print("Nothing queued for deletion.")
        return

    freed = sum(sizes[name] for name in queue)
    console.print(f"\nQueued for deletion: {', '.join(queue)}")
    console.print(f"This will free {freed / (1024**3):.2f} GB. You'll need to re-pull any of these to use them again.\n")

    if not yes and not typer.confirm("Proceed?", default=False):
        console.print("Cancelled — nothing was deleted.")
        return

    ollama = OllamaBackend()
    for name in queue:
        try:
            ollama.delete(name)
            console.print(f"[green]✓[/green] Deleted {name}")
        except Exception as exc:  # noqa: BLE001 - one failed delete shouldn't abort the rest of the queue
            console.print(f"[red]✗[/red] Failed to delete {name}: {exc}")


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


def _prompt_for_model(provider: str) -> str:
    """Ask which specific model to use within `provider` (e.g. Opus vs
    Sonnet), rather than silently defaulting to one. Always offers a free-
    text "Other" escape hatch so any LiteLLM-supported model id can be used,
    not just the curated list.
    """
    choices = config.FRONTIER_MODEL_CHOICES.get(provider, [])
    if not choices:
        return typer.prompt("Enter the exact frontier model id").strip()

    console.print("\nAvailable models:")
    for i, model_id in enumerate(choices, start=1):
        console.print(f"  {i}) {model_id}")
    other_idx = len(choices) + 1
    console.print(f"  {other_idx}) Other (type a model id)")

    raw = typer.prompt("Choose a model", default="1").strip()
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
        if idx == other_idx:
            return typer.prompt("Enter the exact model id").strip()
    return raw  # let them type the model id directly instead of a number


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

    # 2. Frontier model provider -- always asked explicitly, even if a key
    # for some provider already happens to be sitting in the environment.
    # We never silently guess which one the user wants localforge to use.
    provider = typer.prompt(
        f"Which frontier model provider should localforge use? ({'/'.join(FRONTIER_PROVIDERS)})",
        default="anthropic",
    ).strip().lower()
    env_var = FRONTIER_PROVIDERS.get(provider)
    frontier_model: str | None = None
    if env_var is None:
        console.print(f"[red]Unknown provider {provider!r}.[/red] Skipping — configure a frontier model manually later.")
    else:
        frontier_model = _prompt_for_model(provider)
        if os.environ.get(env_var):
            console.print(f"[green]✓[/green] Using existing {env_var} from your environment.\n")
            config.save({config.FRONTIER_MODEL_ENV_VAR: frontier_model})
        else:
            api_key = typer.prompt(f"Paste your {env_var}", hide_input=True)
            config.save({env_var: api_key, config.FRONTIER_MODEL_ENV_VAR: frontier_model})
            os.environ[env_var] = api_key
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
    failed: list[str] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for model_name in sorted(to_pull):
            task_id = progress.add_task(model_name, total=None)

            def _on_progress(event: dict, task_id=task_id, model_name=model_name) -> None:
                total = event.get("total")
                completed = event.get("completed")
                if total and completed is not None:
                    progress.update(task_id, total=total, completed=completed)
                else:
                    progress.update(task_id, description=f"{model_name}: {event.get('status', '')}")

            try:
                ollama.ensure_available(model_name, on_progress=_on_progress)
                progress.update(task_id, description=f"{model_name} (done)", completed=progress.tasks[task_id].total or 1)
            except Exception as exc:  # noqa: BLE001 - one failed pull shouldn't abort the rest of setup
                failed.append(model_name)
                progress.update(task_id, description=f"[red]{model_name} (failed: {exc})[/red]")

    if failed:
        console.print(
            f"\n[yellow]![/yellow] {len(failed)} model(s) failed to pull: {', '.join(failed)}. "
            "Re-run `localforge setup` to retry, or pull manually with `ollama pull <name>`.\n"
        )
    else:
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
    chosen_model = os.environ.get(config.FRONTIER_MODEL_ENV_VAR)
    if found_keys and chosen_model:
        console.print(f"[green]✓[/green] Frontier model configured: {chosen_model} (via {', '.join(found_keys)})")
    elif found_keys:
        console.print(
            f"[yellow]![/yellow] API key(s) found ({', '.join(found_keys)}) but no frontier model "
            "chosen — run `localforge setup` to pick one explicitly."
        )
        ok = False
    else:
        ok = False
        console.print(
            "[red]✗[/red] No frontier model configured — run `localforge setup`, "
            "or export one of: " + ", ".join(FRONTIER_API_KEY_ENV_VARS)
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
        None,
        "--model",
        "-m",
        help="Frontier model to orchestrate with (any LiteLLM model string, e.g. claude-opus-5, gpt-5). "
        "Defaults to whatever `localforge setup` saved, or claude-opus-5 if setup was never run.",
    ),
) -> None:
    """Run a task: the frontier model plans it and delegates subtasks to local models."""
    frontier_model = frontier_model or os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or "claude-opus-5"

    def _on_delegate(modality: str, entry) -> None:
        console.print(f"  → delegating [bold]{modality}[/bold] to [cyan]{entry.name}[/cyan] (local, via {entry.runtime})")

    try:
        with console.status(f"[bold green]Orchestrating with {frontier_model}..."):
            result = run_orchestrator(task, frontier_model, on_delegate=_on_delegate)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary: show a clean message, not a traceback
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    console.print(result.answer)

    stats = result.stats
    usage_lines = [
        _usage_bar(stats.local_tokens_generated, stats.frontier_total_tokens),
        "",
        f"[green]■[/green] Local models: {stats.local_tokens_generated} tokens — "
        "never sent to or billed by the frontier API",
        f"[yellow]■[/yellow] Frontier ({frontier_model}): {stats.frontier_prompt_tokens} in + "
        f"{stats.frontier_completion_tokens} out = {stats.frontier_total_tokens} tokens"
        + (f" (${stats.frontier_cost_usd:.4f})" if stats.frontier_cost_usd else ""),
    ]
    console.print(Panel("\n".join(usage_lines), title="Usage"))


def _ollama_installed_via_brew() -> bool:
    if shutil.which("brew") is None:
        return False
    return subprocess.run(
        ["brew", "list", "--formula", "ollama"], capture_output=True, check=False
    ).returncode == 0


@app.command()
def uninstall(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the main confirmation prompt."),
    purge_ollama: bool = typer.Option(
        False,
        "--purge-ollama",
        help="Also uninstall Ollama itself (if installed via Homebrew) and delete ~/.ollama entirely, "
        "not just the models localforge pulled. Off by default since Ollama may be used by other tools.",
    ),
) -> None:
    """Remove everything localforge manages: installed models, saved config,
    optionally Ollama itself, and finally the localforge CLI tool.
    """
    ollama = OllamaBackend()
    models_on_disk = ollama.list_installed() if ollama.is_running() else []
    total_model_bytes = sum(m.get("size", 0) for m in models_on_disk)
    ollama_via_brew = _ollama_installed_via_brew()

    lines = ["[bold red]This will remove:[/bold red]"]
    if models_on_disk:
        lines.append(f"  - {len(models_on_disk)} local model(s) — {total_model_bytes / (1024**3):.2f} GB")
    if config.CONFIG_FILE.exists():
        lines.append(f"  - Saved config/API key at {config.CONFIG_FILE}")
    if purge_ollama:
        if ollama_via_brew:
            lines.append("  - Ollama itself (via Homebrew) and its entire ~/.ollama data directory")
        else:
            lines.append("  - ~/.ollama data directory (Ollama wasn't installed via Homebrew, so the app itself is left alone)")
    lines.append("  - The localforge CLI tool")
    lines.append("\n[bold]This cannot be undone.[/bold]")
    console.print(Panel("\n".join(lines), title="Uninstall localforge"))

    if not yes and not typer.confirm("\nContinue?", default=False):
        console.print("Cancelled — nothing was removed.")
        raise typer.Exit()

    if not purge_ollama and not yes:
        purge_ollama = typer.confirm(
            "\nAlso uninstall Ollama itself and delete ~/.ollama? "
            "Skip this if you use Ollama for anything besides localforge.",
            default=False,
        )

    for m in models_on_disk:
        try:
            ollama.delete(m["name"])
            console.print(f"[green]✓[/green] Deleted model {m['name']}")
        except Exception as exc:  # noqa: BLE001 - one failed delete shouldn't abort the rest of uninstall
            console.print(f"[red]✗[/red] Failed to delete {m['name']}: {exc}")

    if purge_ollama:
        if ollama_via_brew:
            subprocess.run(["brew", "uninstall", "ollama"], check=False)
        ollama_data_dir = Path.home() / ".ollama"
        if ollama_data_dir.exists():
            shutil.rmtree(ollama_data_dir, ignore_errors=True)
        console.print("[green]✓[/green] Removed Ollama and its data")

    if config.CONFIG_DIR.exists():
        shutil.rmtree(config.CONFIG_DIR, ignore_errors=True)
        console.print(f"[green]✓[/green] Removed {config.CONFIG_DIR}")

    console.print("\nUninstalling the localforge CLI tool itself...")
    subprocess.run(["uv", "tool", "uninstall", "localforge"], check=False)
    console.print("[bold green]Done.[/bold green] localforge has been fully removed.")


if __name__ == "__main__":
    app()
