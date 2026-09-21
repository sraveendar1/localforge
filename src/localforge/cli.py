from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from localforge import cli_transport, config, theme
from localforge.advisor import recommend_models
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import load_catalog, recommendations
from localforge.config import FRONTIER_API_KEY_ENV_VARS, FRONTIER_PROVIDERS
from localforge.hardware import detect_hardware
from localforge.orchestrator import OrchestrationError
from localforge.orchestrator import run as run_orchestrator

app = typer.Typer(
    name="localforge",
    help="A frontier model orchestrates open-weight models running locally on your machine.",
)
console = Console()


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    config.load()
    console.push_theme(theme.get_theme(os.environ.get(config.THEME_ENV_VAR, theme.DEFAULT_THEME)))
    if ctx.invoked_subcommand is None:
        _print_getting_started()
        raise typer.Exit()


@app.command(name="theme")
def theme_command(
    name: str = typer.Argument(None, help=f"Theme to switch to: {', '.join(theme.THEMES)}. Omit to show the current theme."),
) -> None:
    """Show or change the CLI color theme (matrix, dark, light)."""
    current = os.environ.get(config.THEME_ENV_VAR, theme.DEFAULT_THEME)
    if name is None:
        console.print(f"Current theme: [accent]{current}[/accent]")
        console.print(f"Available: {', '.join(theme.THEMES)}")
        console.print("Switch with: localforge theme <name>")
        return

    if name not in theme.THEMES:
        console.print(f"[error]Unknown theme {name!r}.[/error] Available: {', '.join(theme.THEMES)}")
        raise typer.Exit(code=1)

    config.save({config.THEME_ENV_VAR: name})
    console.push_theme(theme.get_theme(name))
    console.print(f"[success]✓[/success] Theme set to [accent]{name}[/accent].")


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
    bar = f"[success]{'█' * local_width}[/success][warning]{'█' * frontier_width}[/warning]"
    pct_local = round(100 * local_tokens / total)
    return f"{bar}  {pct_local}% local / {100 - pct_local}% frontier"


def _print_getting_started() -> None:
    ready = bool(os.environ.get(config.FRONTIER_MODEL_ENV_VAR))
    if ready:
        body = (
            "[bold]You're set up.[/bold] Try:\n\n"
            '  [accent]localforge run "Build a todo REST API with docs"[/accent]\n\n'
            "Other commands: [bold]scan[/bold] · [bold]models[/bold] · [bold]doctor[/bold] · [bold]wizard[/bold]"
        )
    else:
        body = (
            "[bold]Get started in one step:[/bold]\n\n"
            "  [accent]localforge setup[/accent]   (or [accent]localforge wizard[/accent] for a terminal UI)\n\n"
            "That installs Ollama, has a frontier model pick local models for your\n"
            "hardware, and saves your API key — then you're ready for:\n\n"
            '  [accent]localforge run "Build a todo REST API with docs"[/accent]'
        )
    console.print(Panel(body, title="localforge", expand=False, border_style="panel.border"))


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
            table.add_row(modality, "[error]none fit this hardware[/error]", "-", "-")
        else:
            table.add_row(modality, entry.name, entry.runtime, str(entry.quality_tier))

    console.print(table)


def _installed_ollama_models() -> list[dict]:
    ollama = OllamaBackend()
    if not ollama.is_running():
        console.print("[error]Ollama is not running.[/error] Start it, then retry.")
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
                console.print(f"[warning]Skipping — not installed: {requested}[/warning]")
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
                console.print(f"[warning]Skipping unknown selection: {part!r}[/warning]")

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
            console.print(f"[success]✓[/success] Deleted {name}")
        except Exception as exc:  # noqa: BLE001 - one failed delete shouldn't abort the rest of the queue
            console.print(f"[error]✗[/error] Failed to delete {name}: {exc}")


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


def _drain_buffered_input() -> None:
    """Discard keystrokes already sitting in the terminal input buffer before
    an interactive prompt. Without this, an extra Enter pressed earlier (e.g.
    at install.sh's "Press Enter to continue") gets consumed by the *next*
    prompt, silently accepting it -- which is how provider selection used to
    get skipped. No-op when stdin isn't a real terminal.
    """
    try:
        import termios

        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 - best-effort only; never break a prompt over this
        pass


def _prompt_for_provider() -> str:
    """Ask which frontier provider to use as an explicit numbered choice.

    Deliberately has **no default**: an empty answer re-asks rather than
    silently selecting one. A pre-filled default here meant a single stray
    keystroke picked Anthropic without the user ever making a choice.
    """
    providers = list(FRONTIER_PROVIDERS)
    labels = {
        "anthropic": "Anthropic  (Claude — needs an API key)",
        "openai": "OpenAI     (GPT — needs an API key)",
        "gemini": "Gemini     (Google — needs an API key)",
        "local": "Local      (open-weight model via Ollama — no API key, fully self-hosted)",
    }

    console.print("\nWhich frontier model should orchestrate your tasks?")
    for i, name in enumerate(providers, start=1):
        console.print(f"  {i}) {labels.get(name, name)}")

    _drain_buffered_input()
    while True:
        raw = typer.prompt("Choose a provider (number or name)").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(providers):
            return providers[int(raw) - 1]
        if raw in FRONTIER_PROVIDERS:
            return raw
        console.print(f"[warning]'{raw}' isn't one of the options — pick 1-{len(providers)} or a name.[/warning]")


def _prompt_for_auth_method(provider: str) -> str:
    """API key vs the provider's own logged-in CLI. Numbered, no default.

    CLI login reuses whatever subscription that account has instead of
    separate per-token API charges. Availability is *probed*, never assumed:
    if the CLI isn't installed, that option says so and explains how to get
    it rather than being silently offered and then failing.
    """
    spec = config.FRONTIER_CLI_AUTH.get(provider)
    if spec is None:
        return config.AUTH_API_KEY  # no CLI path for this provider

    cli_ready = cli_transport.available(provider)
    cli_label = f"CLI login   (uses your `{spec['command']}` subscription — no per-token API cost)"
    if not cli_ready:
        cli_label += f"\n     ⚠ `{spec['command']}` not installed — {spec['install_hint']}"
    elif not spec.get("verified", False):
        cli_label += f"\n     ⚠ untested for {provider} — verify it works before relying on it"

    console.print("\nHow should localforge authenticate to this provider?")
    console.print("  1) API key     (pay-per-token, billed separately)")
    console.print(f"  2) {cli_label}")

    _drain_buffered_input()
    while True:
        raw = typer.prompt("Choose an auth method (1-2)").strip().lower()
        if raw in {"1", "api", "api_key", "key"}:
            return config.AUTH_API_KEY
        if raw in {"2", "cli", "cli_login", "login"}:
            if not cli_ready:
                console.print(f"[warning]{cli_transport.requirements_message(provider)}[/warning]")
                console.print("[warning]Install and log in first, or choose 1 for an API key.[/warning]")
                continue
            # Verify the login *now*, before spending the user's time on the
            # model menu and before writing a config that can't actually run.
            # "Check again" re-probes right here rather than bouncing back to
            # the auth menu -- the user just logged in, don't make them re-pick.
            while True:
                console.print(f"Checking that `{spec['command']}` is logged in...")
                if cli_transport.logged_in(provider):
                    console.print(f"[success]✓[/success] `{spec['command']}` is logged in.")
                    return config.AUTH_CLI_LOGIN
                console.print(
                    f"[warning]![/warning] `{spec['command']}` is installed but not logged in — {spec['login_hint']}."
                )
                console.print("  1) I've logged in now — check again")
                console.print("  2) Use an API key instead")
                while True:
                    retry = typer.prompt("Choose (1-2)").strip()
                    if retry in {"1", "2"}:
                        break
                    console.print("[warning]Pick 1 or 2.[/warning]")
                if retry == "2":
                    return config.AUTH_API_KEY
                # "1" -> loop re-probes without re-asking the auth question
        console.print("[warning]Pick 1 (API key) or 2 (CLI login).[/warning]")


def _prompt_for_model(provider: str) -> str:
    """Ask which specific model to use within `provider` (e.g. Opus vs
    Sonnet), rather than silently defaulting to one. Always offers a free-
    text "Other" escape hatch so any LiteLLM-supported model id can be used,
    not just the curated list.
    """
    choices = config.FRONTIER_MODEL_CHOICES.get(provider, [])
    if not choices:
        _drain_buffered_input()
        return _prompt_nonempty("Enter the exact frontier model id")

    console.print("\nWhich model should it use?")
    for i, model_id in enumerate(choices, start=1):
        console.print(f"  {i}) {model_id}")
    other_idx = len(choices) + 1
    console.print(f"  {other_idx}) Other (type a model id)")
    if provider == "local":
        console.print("     (any Ollama model works, but must be prefixed \"ollama/\", e.g. ollama/llama3.1:8b)")

    # No default: an empty answer re-asks rather than silently picking one.
    _drain_buffered_input()
    while True:
        raw = typer.prompt(f"Choose a model (1-{other_idx})").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            if idx == other_idx:
                return _prompt_nonempty("Enter the exact model id")
        elif raw:
            return raw  # let them type the model id directly instead of a number
        console.print(f"[warning]Pick a number from 1 to {other_idx}, or type a model id.[/warning]")


def _prompt_nonempty(message: str) -> str:
    """Prompt until a non-empty answer is given -- never silently accept ''."""
    while True:
        value = typer.prompt(message).strip()
        if value:
            return value
        console.print("[warning]That can't be empty.[/warning]")


def _prompt_nonempty_hidden(message: str) -> str:
    """Same, for secrets: masked input, and an empty paste re-asks rather
    than saving an empty key that would fail confusingly much later.
    """
    while True:
        value = typer.prompt(message, hide_input=True).strip()
        if value:
            return value
        console.print("[warning]That can't be empty — paste the key.[/warning]")


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
                "[warning]Ollama isn't installed and can't be auto-installed on this OS.[/warning] "
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
            console.print("[error]Could not confirm Ollama started.[/error] Start it manually and re-run setup.")
            raise typer.Exit(code=1)
    console.print("[success]✓[/success] Ollama is installed and running\n")

    # 2. Frontier model provider -- always asked explicitly, even if a key
    # for some provider already happens to be sitting in the environment.
    # We never silently guess which one the user wants localforge to use.
    provider = _prompt_for_provider()

    frontier_model: str | None = None
    auth_method = config.AUTH_API_KEY
    if provider not in FRONTIER_PROVIDERS:
        console.print(f"[error]Unknown provider {provider!r}.[/error] Skipping — configure a frontier model manually later.")
    else:
        env_var = FRONTIER_PROVIDERS[provider]
        auth_method = config.AUTH_API_KEY if env_var is None else _prompt_for_auth_method(provider)
        frontier_model = _prompt_for_model(provider)
        if env_var is None:
            # An open-weight model as the orchestrator itself: no API key
            # needed, but if it's served via Ollama, make sure it's pulled.
            console.print(
                f"[success]✓[/success] Using {frontier_model} as the frontier orchestrator "
                "(self-hosted, no API key needed).\n"
            )
            config.save({config.FRONTIER_MODEL_ENV_VAR: frontier_model})
            if frontier_model.startswith("ollama/"):
                orchestrator_model_name = frontier_model.removeprefix("ollama/")
                console.print(f"Pulling {orchestrator_model_name} for orchestration (this can take a while)...")
                try:
                    ollama.ensure_available(orchestrator_model_name)
                    console.print(f"[success]✓[/success] {orchestrator_model_name} ready\n")
                except Exception as exc:  # noqa: BLE001 - reported, doesn't abort the rest of setup
                    console.print(f"[error]Failed to pull {orchestrator_model_name}: {exc}[/error]\n")
        elif auth_method == config.AUTH_CLI_LOGIN:
            # Orchestrate through the provider's own logged-in CLI: nothing
            # to store here, the CLI holds its own credentials (for Claude
            # Code, in the OS keychain -- never in localforge's config).
            spec = config.FRONTIER_CLI_AUTH[provider]
            config.save(
                {
                    config.FRONTIER_MODEL_ENV_VAR: frontier_model,
                    config.AUTH_METHOD_ENV_VAR: config.AUTH_CLI_LOGIN,
                    config.FRONTIER_PROVIDER_ENV_VAR: provider,
                }
            )
            # Login was already verified in _prompt_for_auth_method(), which
            # won't return AUTH_CLI_LOGIN unless the CLI answered a probe.
            console.print(
                f"[success]✓[/success] Using your `{spec['command']}` login — "
                "drawn from that subscription, no per-token API charges.\n"
            )
        elif os.environ.get(env_var):
            console.print(f"[success]✓[/success] Using existing {env_var} from your environment.\n")
            config.save(
                {
                    config.FRONTIER_MODEL_ENV_VAR: frontier_model,
                    config.AUTH_METHOD_ENV_VAR: config.AUTH_API_KEY,
                    config.FRONTIER_PROVIDER_ENV_VAR: provider,
                }
            )
        else:
            console_url = config.FRONTIER_CONSOLE_URLS.get(provider)
            if console_url:
                console.print(f"Opening {console_url} in your browser to create an API key...")
                webbrowser.open(console_url)
            api_key = _prompt_nonempty_hidden(f"Paste your {env_var}")
            config.save(
                {
                    env_var: api_key,
                    config.FRONTIER_MODEL_ENV_VAR: frontier_model,
                    config.AUTH_METHOD_ENV_VAR: config.AUTH_API_KEY,
                    config.FRONTIER_PROVIDER_ENV_VAR: provider,
                }
            )
            os.environ[env_var] = api_key
            console.print(f"[success]✓[/success] Saved {env_var} to {config.CONFIG_FILE}\n")

    # 3. Hardware scan, then let the frontier model pick which local models
    # to download (constrained to catalog entries that already fit this
    # machine's RAM/VRAM/disk space -- see advisor.recommend_models).
    hw = detect_hardware()
    console.print(
        f"Hardware: {hw.ram_gb}GB RAM, {hw.total_vram_gb}GB VRAM, "
        f"{hw.free_disk_gb}GB free disk\n"
    )
    if frontier_model:
        # Route the advisor the same way the orchestrator will be routed --
        # under CLI login there's no API key, so a litellm call here would
        # fail and silently degrade to the heuristic.
        advisor_cli = provider if auth_method == config.AUTH_CLI_LOGIN else None
        via = f"your `{config.FRONTIER_CLI_AUTH[provider]['command']}` login" if advisor_cli else frontier_model
        console.print(f"Asking {via} to pick the best local models for this machine...")
        recs = recommend_models(hw, frontier_model, cli_provider=advisor_cli)
    else:
        console.print("[warning]No usable frontier model id — falling back to the built-in heuristic.[/warning]")
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
                progress.update(task_id, description=f"[error]{model_name} (failed: {exc})[/error]")

    if failed:
        console.print(
            f"\n[warning]![/warning] {len(failed)} model(s) failed to pull: {', '.join(failed)}. "
            "Re-run `localforge setup` to retry, or pull manually with `ollama pull <name>`.\n"
        )
    else:
        console.print("[success]✓[/success] Local models ready\n")

    console.print("[bold success]Setup complete.[/bold success] Try: localforge run \"...\"")


@app.command()
def doctor() -> None:
    """Check that everything localforge needs is installed and reachable."""
    ok = True

    if shutil.which("ollama") is not None:
        console.print("[success]✓[/success] Ollama is installed")
    else:
        ok = False
        console.print("[error]✗[/error] Ollama is not installed — get it from https://ollama.com")

    if OllamaBackend().is_running():
        console.print("[success]✓[/success] Ollama is running")
    else:
        ok = False
        console.print("[error]✗[/error] Ollama is not running — start it (e.g. `ollama serve` or `brew services start ollama`)")

    found_keys = [var for var in FRONTIER_API_KEY_ENV_VARS if os.environ.get(var)]
    chosen_model = os.environ.get(config.FRONTIER_MODEL_ENV_VAR)
    is_local_frontier = bool(chosen_model) and chosen_model.startswith("ollama/")

    # CLI login: no API key by design -- check the CLI instead.
    if os.environ.get(config.AUTH_METHOD_ENV_VAR) == config.AUTH_CLI_LOGIN:
        cli_provider = os.environ.get(config.FRONTIER_PROVIDER_ENV_VAR, "")
        spec = config.FRONTIER_CLI_AUTH.get(cli_provider)
        if spec is None:
            ok = False
            console.print(f"[error]✗[/error] CLI login configured for unknown provider {cli_provider!r} — re-run `localforge setup`")
        elif not cli_transport.available(cli_provider):
            ok = False
            console.print(f"[error]✗[/error] {cli_transport.requirements_message(cli_provider)}")
        elif cli_transport.logged_in(cli_provider):
            console.print(
                f"[success]✓[/success] Frontier via `{spec['command']}` login "
                f"({cli_provider} subscription — no API key, no per-token billing)"
            )
        else:
            ok = False
            console.print(
                f"[error]✗[/error] `{spec['command']}` is installed but not logged in — {spec['login_hint']}"
            )
    elif chosen_model and (found_keys or is_local_frontier):
        via = "self-hosted, no API key needed" if is_local_frontier else f"via {', '.join(found_keys)}"
        console.print(f"[success]✓[/success] Frontier model configured: {chosen_model} ({via})")
    elif found_keys:
        console.print(
            f"[warning]![/warning] API key(s) found ({', '.join(found_keys)}) but no frontier model "
            "chosen — run `localforge setup` to pick one explicitly."
        )
        ok = False
    else:
        ok = False
        console.print(
            "[error]✗[/error] No frontier model configured — run `localforge setup`, "
            "export one of: " + ", ".join(FRONTIER_API_KEY_ENV_VARS) + ", or pick an open-weight "
            "model as the orchestrator (`localforge setup`, provider \"local\")"
        )

    hw = detect_hardware()
    recs = recommendations(hw)
    missing = [modality for modality, entry in recs.items() if entry is None]
    if not missing:
        console.print("[success]✓[/success] A local model fits every known modality")
    else:
        console.print(f"[warning]![/warning] No fitting model for: {', '.join(missing)} (hardware too limited)")

    if ok:
        console.print("\n[bold success]Ready to go.[/bold success] Try: localforge run \"...\"")
    else:
        console.print("\n[bold error]Fix the items above before running `localforge run`.[/bold error]")
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

    # An explicit --model always means "use the API path for this model";
    # only a saved cli_login choice routes through the provider's CLI.
    cli_provider = None
    if os.environ.get(config.AUTH_METHOD_ENV_VAR) == config.AUTH_CLI_LOGIN:
        cli_provider = os.environ.get(config.FRONTIER_PROVIDER_ENV_VAR)
        if cli_provider and not cli_transport.available(cli_provider):
            console.print(f"[error]Error:[/error] {cli_transport.requirements_message(cli_provider)}")
            raise typer.Exit(code=1)
        if cli_provider:
            spec = config.FRONTIER_CLI_AUTH[cli_provider]
            console.print(f"[dim]Orchestrating via your `{spec['command']}` login (subscription, not API billing)[/dim]")

    def _on_delegate(modality: str, entry) -> None:
        console.print(f"  → delegating [bold]{modality}[/bold] to [accent]{entry.name}[/accent] (local, via {entry.runtime})")

    try:
        with console.status(f"[bold success]Orchestrating with {frontier_model}..."):
            result = run_orchestrator(task, frontier_model, on_delegate=_on_delegate, cli_provider=cli_provider)
    except cli_transport.CLINotAvailableError as exc:
        # The provider CLI failed mid-run -- most often a session that expired
        # since setup. Say how to fix it, in shell terms.
        console.print(f"[bold error]Error:[/bold error] {exc}")
        if cli_provider:
            spec = config.FRONTIER_CLI_AUTH[cli_provider]
            console.print(
                f"[warning]Your `{spec['command']}` session may have expired — {spec['login_hint']}, "
                "then try again. Or run `localforge setup` to switch to an API key.[/warning]"
            )
        raise typer.Exit(code=1) from None
    except OrchestrationError as exc:
        # Even a non-convergent run spent real frontier tokens/cost and local
        # compute along the way -- show that before reporting the failure.
        console.print(f"[bold error]Error:[/bold error] {exc}")
        _print_usage_panel(exc.stats, frontier_model)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary: show a clean message, not a traceback
        console.print(f"[bold error]Error:[/bold error] {exc}")
        raise typer.Exit(code=1) from None

    console.print(result.answer)
    _print_usage_panel(result.stats, frontier_model)


def _frontier_cost_note(stats) -> str:
    """Never present subscription usage as money separately billed: under CLI
    login the figure is what pay-per-token *would* have cost, and it came out
    of the account's subscription instead.
    """
    if not stats.frontier_cost_usd:
        return ""
    if stats.frontier_via_subscription:
        return f" (~${stats.frontier_cost_usd:.4f} of subscription usage, not billed separately)"
    return f" (${stats.frontier_cost_usd:.4f})"


def _print_usage_panel(stats, frontier_model: str) -> None:
    usage_lines = [
        _usage_bar(stats.local_tokens_generated, stats.frontier_total_tokens),
        "",
        f"[success]■[/success] Local models: {stats.local_tokens_generated} tokens — "
        "never sent to or billed by the frontier API",
        f"[warning]■[/warning] Frontier ({frontier_model}): {stats.frontier_prompt_tokens} in + "
        f"{stats.frontier_completion_tokens} out = {stats.frontier_total_tokens} tokens"
        + _frontier_cost_note(stats),
    ]
    console.print(Panel("\n".join(usage_lines), title="Usage", border_style="panel.border"))


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

    lines = ["[bold error]This will remove:[/bold error]"]
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
    console.print(Panel("\n".join(lines), title="Uninstall localforge", border_style="error"))

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
            console.print(f"[success]✓[/success] Deleted model {m['name']}")
        except Exception as exc:  # noqa: BLE001 - one failed delete shouldn't abort the rest of uninstall
            console.print(f"[error]✗[/error] Failed to delete {m['name']}: {exc}")

    if purge_ollama:
        if ollama_via_brew:
            subprocess.run(["brew", "uninstall", "ollama"], check=False)
        ollama_data_dir = Path.home() / ".ollama"
        if ollama_data_dir.exists():
            shutil.rmtree(ollama_data_dir, ignore_errors=True)
        console.print("[success]✓[/success] Removed Ollama and its data")

    if config.CONFIG_DIR.exists():
        shutil.rmtree(config.CONFIG_DIR, ignore_errors=True)
        console.print(f"[success]✓[/success] Removed {config.CONFIG_DIR}")

    console.print("\nUninstalling the localforge CLI tool itself...")
    subprocess.run(["uv", "tool", "uninstall", "localforge"], check=False)
    console.print("[bold success]Done.[/bold success] localforge has been fully removed.")


if __name__ == "__main__":
    app()
