from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import httpx
import litellm
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from localforge import brief, cli_transport, config, local_transport, memory, repl, theme, trust, upgrades, usage_store
from localforge.advisor import recommend_models
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import NoFittingModelError, best_match, load_catalog, recommendations
from localforge.config import FRONTIER_API_KEY_ENV_VARS, FRONTIER_PROVIDERS
from localforge.hardware import detect_hardware
from localforge.orchestrator import Conversation, OrchestrationError, RunStats, TaskCancelled
from localforge.orchestrator import run as run_orchestrator
from localforge.tools import ActivityHooks, Dispatcher
from localforge.background import TaskRunner
from localforge import workspace as workspace_module
from localforge.scratchpad import Scratchpad
from localforge.workspace import Workspace

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
        # A real interactive terminal gets the Claude-Code-style session
        # (banner + slash commands); a pipe/script/non-tty invocation (e.g.
        # existing CI usage, or `localforge | cat`) keeps the old
        # print-and-exit behavior so nothing that scripts against a bare
        # `localforge` call starts waiting on stdin forever.
        if sys.stdin.isatty() and sys.stdout.isatty():
            runner = TaskRunner(lambda task: _run_in_background(task))
            runner.question_handler = explain_to_user
            _session.runner = runner
            try:
                repl.run_repl(
                    app,
                    console,
                    on_start=_start_session,
                    on_exit=_save_memory_at_exit,
                    history_file=config.CONFIG_DIR / "history",
                    runner=runner,
                )
            finally:
                runner.shutdown()
                _session.runner = None
        else:
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


START_SESSION_HINT = "Type [accent]localforge[/accent] to start a session."


def _print_getting_started() -> None:
    """Shown when bare `localforge` runs without a real terminal (pipe, script,
    installer). The next step it points to is the interactive session --
    just typing `localforge` -- not a one-shot `localforge run`.
    """
    ready = bool(os.environ.get(config.FRONTIER_MODEL_ENV_VAR))
    if ready:
        body = (
            "[bold]You're set up.[/bold] Start a session:\n\n"
            "  [accent]localforge[/accent]\n\n"
            "Then just type what you want built, or [accent]/help[/accent] for commands.\n"
            'One-off without a session: [accent]localforge run "Build a todo REST API with docs"[/accent]'
        )
    else:
        body = (
            "[bold]Get started:[/bold]\n\n"
            "  [accent]localforge[/accent]\n\n"
            "The first session walks you through it: installing Ollama, picking a model\n"
            "for your hardware, and how you want to reach a frontier model (or staying\n"
            "fully local). To do that part now instead: [accent]localforge setup[/accent]."
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
    """Show the best-fitting local model per task type for this machine,
    preferring what's already installed over a fresh download."""
    hw = detect_hardware()
    installed = _installed_model_names(OllamaBackend())
    recs = recommendations(hw, installed=installed)

    table = Table(title="Recommended local models for this machine")
    table.add_column("Modality")
    table.add_column("Model")
    table.add_column("Runtime")
    table.add_column("Quality tier")
    table.add_column("Installed")

    for modality, entry in recs.items():
        if entry is None:
            table.add_row(modality, "[error]none fit this hardware[/error]", "-", "-", "-")
        else:
            on_disk = "[success]yes[/success]" if entry.name in installed else "no"
            table.add_row(modality, entry.name, entry.runtime, str(entry.quality_tier), on_disk)

    console.print(table)


def _installed_model_names(ollama: OllamaBackend) -> set[str]:
    """Exact Ollama tags currently on disk, or an empty set if Ollama can't
    be reached -- in which case we simply fall back to recommending from
    the catalog, never block setup on it.
    """
    try:
        return {m["name"] for m in ollama.list_installed()}
    except Exception:  # noqa: BLE001 - best-effort; setup continues without it
        return set()


def _print_model_plan(recs: dict, installed: set[str], hw) -> None:
    """Before pulling anything, say exactly what will be reused and what
    will be downloaded, and when an installed model is being reused over a
    higher-tier one that also fits, say that too -- so the user can judge
    whether the upgrade is worth the download rather than it being decided
    silently either way.
    """
    catalog = load_catalog()
    reuse, download, notes = [], [], []
    # One model can cover several task types (on a smaller machine it's the
    # same model for all text work): list it once, with every task it covers.
    tasks: dict[str, list[str]] = {}
    for modality, entry in recs.items():
        if entry is not None and entry.runtime == "ollama":
            tasks.setdefault(entry.name, []).append(modality)
    for modality, entry in recs.items():
        if entry is None or entry.runtime != "ollama":
            continue
        covers = ", ".join(tasks[entry.name])
        if entry.name in installed:
            if tasks[entry.name][0] == modality:
                reuse.append(f"{entry.name} ({covers})")
            try:
                ideal = best_match(modality, hw, catalog)  # ignoring what's installed
            except NoFittingModelError:
                continue
            if ideal.name != entry.name and ideal.quality_tier > entry.quality_tier:
                notes.append(
                    f"{modality}: {ideal.name} (higher tier, ~{ideal.disk_gb:g} GB) also fits — "
                    "/upgrade swaps it in and removes the old one"
                )
        elif tasks[entry.name][0] == modality:
            download.append(f"{entry.name} ({covers}, ~{entry.disk_gb:g} GB)")

    if reuse:
        console.print(f"[success]✓[/success] Reusing already-installed: {', '.join(reuse)}")
    if download:
        console.print(f"[warning]↓[/warning] Will download: {', '.join(download)}")
    if not download:
        console.print("[success]✓[/success] Nothing to download — every task type is covered by what you have.")
    for note in notes:
        console.print(f"  [dim]note — {note}[/dim]")
    console.print()


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
        console.print("No local models installed yet. Run `localforge setup` to get some.")
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
            upgrades.unmark(name)
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
    choices = _provider_models(provider)
    labels = {}
    if provider == "local":
        installed = _installed_model_names(OllamaBackend())
        choices = local_transport.orchestrator_choices(detect_hardware(), installed)
        labels = {c: "already downloaded" for c in choices if local_transport.model_name(c) in installed}
    if not choices:
        _drain_buffered_input()
        return _prompt_nonempty("Enter the exact frontier model id")

    console.print("\nWhich model should it use?")
    for i, model_id in enumerate(choices, start=1):
        note = f"  ({labels[model_id]})" if model_id in labels else ""
        console.print(f"  {i}) {model_id}{note}")
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
    console.print("This gets the machine ready: Ollama, local models that fit your hardware,")
    console.print("and how you want to reach a frontier model — or stay fully local.\n")

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
            # Overwrite the auth settings too: config.save() merges, so a
            # cli_login left from an earlier setup would otherwise keep
            # routing every turn through that provider's CLI.
            config.save(
                {
                    config.FRONTIER_MODEL_ENV_VAR: frontier_model,
                    config.AUTH_METHOD_ENV_VAR: config.AUTH_LOCAL,
                    config.FRONTIER_PROVIDER_ENV_VAR: "local",
                }
            )
            if frontier_model.startswith("ollama/"):
                orchestrator_model_name = frontier_model.removeprefix("ollama/")
                console.print(f"Pulling {orchestrator_model_name} for orchestration (this can take a while)...")
                try:
                    ollama.ensure_available(orchestrator_model_name)
                    upgrades.mark_managed(orchestrator_model_name)
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

    # Check what's already on disk *before* deciding what to download, so a
    # re-run of setup reuses suitable models instead of pulling new ones.
    installed = _installed_model_names(ollama)
    if installed:
        console.print(f"Already installed: {', '.join(sorted(installed))}\n")

    if frontier_model:
        # Route the advisor the same way the orchestrator will be routed --
        # under CLI login there's no API key, so a litellm call here would
        # fail and silently degrade to the heuristic.
        advisor_cli = provider if auth_method == config.AUTH_CLI_LOGIN else None
        via = f"your `{config.FRONTIER_CLI_AUTH[provider]['command']}` login" if advisor_cli else frontier_model
        console.print(f"Asking {via} to pick the best local models for this machine...")
        recs = recommend_models(hw, frontier_model, cli_provider=advisor_cli, installed=installed)
    else:
        console.print("[warning]No usable frontier model id — falling back to the built-in heuristic.[/warning]")
        recs = recommendations(hw, installed=installed)

    _print_model_plan(recs, installed, hw)
    to_pull = {e.name for e in recs.values() if e is not None and e.runtime == "ollama" and e.name not in installed}
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
                upgrades.mark_managed(model_name)
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

    console.print(f"[bold success]Setup complete.[/bold success] {START_SESSION_HINT}", highlight=False)


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

    if is_local_frontier:
        # An open-weight orchestrator needs no key or login, whatever other
        # auth settings are lying around -- only that Ollama has the model.
        name = local_transport.model_name(chosen_model)
        on_disk = _installed_model_names(OllamaBackend())
        if name in on_disk or f"{name}:latest" in on_disk:
            console.print(f"[success]✓[/success] Orchestrator: {name} (local, via Ollama — no account needed)")
        else:
            ok = False
            console.print(f"[error]✗[/error] Orchestrator {name} is not downloaded — run `ollama pull {name}`")
    # CLI login: no API key by design -- check the CLI instead.
    elif os.environ.get(config.AUTH_METHOD_ENV_VAR) == config.AUTH_CLI_LOGIN:
        cli_provider = os.environ.get(config.FRONTIER_PROVIDER_ENV_VAR, "")
        spec = config.FRONTIER_CLI_AUTH.get(cli_provider)
        if spec is None:
            ok = False
            console.print(f"[error]✗[/error] CLI login configured for unknown provider {cli_provider!r} — re-run `localforge setup`")
        elif not cli_transport.available(cli_provider):
            ok = False
            console.print(f"[error]✗[/error] {cli_transport.requirements_message(cli_provider)}")
        else:
            works, why = cli_transport.probe(cli_provider)
            if works:
                console.print(
                    f"[success]✓[/success] Frontier via `{spec['command']}` login "
                    f"({cli_provider} subscription — no API key, no per-token billing)"
                )
            else:
                ok = False
                console.print(f"[error]✗[/error] `{spec['command']}` didn't answer: {escape(why)}")
                console.print(f"  [warning]{escape(_cli_failure_advice(cli_provider, why))}[/warning]")
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
        console.print(f"\n[bold success]Ready to go.[/bold success] {START_SESSION_HINT}", highlight=False)
    else:
        console.print("\n[bold error]Fix the items above, then type localforge to start a session.[/bold error]")
        raise typer.Exit(code=1)


def _is_local_model(model: str) -> bool:
    return model.startswith(local_transport.PREFIXES)


def _cli_provider_for(frontier_model: str, explicit: bool) -> str | None:
    """Which provider CLI to orchestrate through, if any: only with a saved
    CLI-login choice, only for a model from that same provider, and never
    for an open-weight model (a stale `cli_login` once made a local
    orchestrator call `claude` anyway). An explicit `--model claude-sonnet-5`
    with a saved Claude login still uses the login; `--model gpt-5` doesn't.
    """
    if _is_local_model(frontier_model):
        return None
    if os.environ.get(config.AUTH_METHOD_ENV_VAR) != config.AUTH_CLI_LOGIN:
        return None
    saved = os.environ.get(config.FRONTIER_PROVIDER_ENV_VAR) or None
    if explicit and _cloud_provider(frontier_model) != saved:
        return None
    return saved


def _orchestrator_label(frontier_model: str, cli_provider: str | None) -> str:
    if _is_local_model(frontier_model):
        label = f"{local_transport.model_name(frontier_model)} (local, via Ollama — no account, no billing)"
        size = local_transport.parameter_billions(frontier_model)
        if size is not None and size < local_transport.MIN_RELIABLE_BILLIONS:
            label += (
                f". Heads-up: a {size:g}B model is often unreliable at planning; "
                f"a {local_transport.MIN_RELIABLE_BILLIONS}B+ model or a cloud model works much better"
            )
        return label
    if cli_provider:
        return f"{frontier_model} via your `{config.FRONTIER_CLI_AUTH[cli_provider]['command']}` login (subscription)"
    return f"{frontier_model} (API key)"


_LIMIT_MARKERS = ("limit", "quota", "rate limit", "rate-limit", "credit balance", "billing", "exceeded", "spend")


def _cli_failure_advice(cli_provider: str, error: str) -> str:
    spec = config.FRONTIER_CLI_AUTH[cli_provider]
    lowered = error.lower()
    if any(marker in lowered for marker in _LIMIT_MARKERS):
        return (
            f"Your {cli_provider} account hit a usage limit. Wait for it to reset, or switch the orchestrator "
            "to a local model with /model (e.g. /model ollama/qwen2.5:7b) or `localforge setup`."
        )
    if any(marker in lowered for marker in ("log in", "login", "logged", "auth", "api key", "credential", "unauthorized", "401", "expired")):
        return (
            f"`{spec['command']}` isn't signed in (or the sign-in expired) — {spec['login_hint']}, then try again. "
            "Or switch the orchestrator with /model, or run `localforge setup` to use an API key."
        )
    return f"`{spec['command']}` failed. Check it works on its own (`{spec['command']} -p hi`), or switch with /model."


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
    show_usage: bool = typer.Option(
        False,
        "--usage",
        help="Print token usage after the task. Inside a session, use /usage instead.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Approve every file change and command without asking (inside a session: /auto).",
    ),
) -> None:
    """Run a task in the current folder: the frontier model investigates and plans,
    local models write the code, and you approve each change."""
    explicit_model = frontier_model
    frontier_model = config.litellm_model_id(frontier_model or os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or "claude-opus-5")
    cli_provider = _cli_provider_for(frontier_model, explicit=bool(explicit_model))
    if cli_provider:
        if not cli_transport.available(cli_provider):
            console.print(f"[error]Error:[/error] {cli_transport.requirements_message(cli_provider)}")
            raise typer.Exit(code=1)
    if not _session.announced:
        console.print(f"[dim]Orchestrator: {escape(_orchestrator_label(frontier_model, cli_provider))} · /model to change[/dim]")
        _session.announced = True

    if yes:
        _session.auto_approve = True
    folder = Path.cwd().resolve()
    if not trust.is_trusted(folder):
        if sys.stdin.isatty():
            if not _ask_trust(folder):
                console.print("Not trusted — nothing was run.")
                raise typer.Exit(code=1)
        elif not yes:
            console.print(
                f"[error]{escape(str(folder))} isn't a trusted folder.[/error] Run `localforge` here once "
                "interactively to trust it, or pass --yes to allow this run."
            )
            raise typer.Exit(code=1)
    activity = _session.make_activity(frontier_model)
    scratch = _session.scratchpad_for(folder)
    workspace = Workspace(
        folder,
        approver=activity.approve,
        scratch=scratch.root,
        # A local orchestrator reads into a much smaller context window.
        max_read_chars=LOCAL_ORCHESTRATOR_READ_CHARS if _is_local_model(frontier_model) else workspace_module.MAX_READ_CHARS,
    )
    conversation = _session.conversation_for(workspace.root)
    try:
        try:
            result = run_orchestrator(
                task,
                frontier_model,
                cli_provider=cli_provider,
                hooks=activity.hooks(),
                conversation=conversation,
                workspace=workspace,
            )
        finally:
            activity.close()
            if not _session.interactive:
                _session.drop_scratchpad()  # a one-off run's scratch work ends with it
    except cli_transport.CLINotAvailableError as exc:
        # The provider CLI failed mid-run. Say why in plain terms and what to
        # do -- a usage limit is not an expired login, and advising
        # `claude login` for one sent people in circles.
        if exc.stats is not None:
            # Real frontier/local work happened before the failure -- record
            # it like any other non-clean stop, so /usage still accounts for
            # it and the next session offers to resume (see
            # orchestrator._remember_if_unfinished).
            _session_usage.append((frontier_model, exc.stats))
            _record_usage(frontier_model, exc.stats)
        console.print(f"[bold error]Error:[/bold error] {escape(str(exc))}")
        if cli_provider:
            console.print(f"[warning]{escape(_cli_failure_advice(cli_provider, str(exc)))}[/warning]")
        raise typer.Exit(code=1) from None
    except local_transport.LocalOrchestratorError as exc:
        console.print(f"[bold error]Error:[/bold error] {escape(str(exc))}")
        raise typer.Exit(code=1) from None
    except TaskCancelled as exc:
        # Ctrl+C mid-task: only the task stops, not the session. Changes the
        # user already approved stay; the conversation notes it was stopped.
        _session_usage.append((frontier_model, exc.stats))
        _record_usage(frontier_model, exc.stats)
        console.print("\n[warning]Stopped.[/warning] Changes you already approved are kept. Type your next message.")
        raise typer.Exit(code=130) from None
    except OrchestrationError as exc:
        # Even a non-convergent run spent real frontier tokens/cost and local
        # compute along the way -- record it so /usage still accounts for it.
        console.print(f"[bold error]Error:[/bold error] {exc}")
        _session_usage.append((frontier_model, exc.stats))
        _record_usage(frontier_model, exc.stats)
        if show_usage:
            _print_usage_panel(exc.stats, frontier_model)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary: show a clean message, not a traceback
        console.print(f"[bold error]Error:[/bold error] {exc}")
        raise typer.Exit(code=1) from None

    if not activity.reply_streamed:
        # Not streamed (a CLI without streaming, or an unparseable reply that
        # became the answer as-is): show it now.
        console.print()
        console.print(Markdown(result.answer or "(no answer)"))
    _session_usage.append((frontier_model, result.stats))
    _record_usage(frontier_model, result.stats)
    if show_usage:
        _print_usage_panel(result.stats, frontier_model)


@app.command()
def serve(
    stdio: bool = typer.Option(
        False, "--stdio", help="Speak the JSON-lines protocol over stdin/stdout (used by the desktop app)."
    ),
    frontier_model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Frontier model to orchestrate with. Defaults to whatever `localforge setup` saved, or claude-opus-5.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Start with auto-approve on (deletions still ask)."),
) -> None:
    """Run localforge as the backend for the desktop app, in the current folder."""
    import json

    from localforge.serve import serve_stdio

    if not stdio:
        console.print("[error]Only --stdio is supported for now.[/error] Run `localforge serve --stdio`.")
        raise typer.Exit(code=1)

    # stdout is the protocol stream, so startup failures are reported as a
    # protocol error event the app can show, not as Rich console output.
    def fail(message: str) -> None:
        sys.stdout.write(json.dumps({"type": "error", "message": message, "fatal": True}) + "\n")
        sys.stdout.flush()
        raise typer.Exit(code=1)

    explicit_model = frontier_model
    frontier_model = frontier_model or os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or "claude-opus-5"
    cli_provider = _cli_provider_for(frontier_model, explicit=bool(explicit_model))
    if cli_provider and not cli_transport.available(cli_provider):
        fail(cli_transport.requirements_message(cli_provider))
    folder = Path.cwd().resolve()
    if not trust.is_trusted(folder):
        fail(f"{folder} isn't a trusted folder. Run `localforge` there once interactively to trust it.")
    serve_stdio(folder, frontier_model, cli_provider, auto_approve=yes)


LIMIT_POLL_SECONDS = 5.0
# What one read_file may return when an open-weight model is orchestrating.
LOCAL_ORCHESTRATOR_READ_CHARS = 12_000


def _human_duration(seconds: float) -> str:
    """"2h 14m" / "45s" -- for countdowns the user reads at a glance."""
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


class _LiveActivity:
    """Shows a run as it happens: a spinner only while the frontier model is
    thinking, and each local model's output streamed as it's generated, so
    delegated work is visible instead of hidden behind one long spinner.
    """

    INDENT = "    "

    def __init__(self, frontier_model: str):
        self.frontier_model = frontier_model
        self._status = None
        self._pull_shown: dict[str, int] = {}
        self._at_line_start = True
        self._first_token_at: float | None = None
        self._live: Live | None = None
        self._answer = ""
        # True once the current reply's answer has been streamed to screen,
        # so run() doesn't print it a second time.
        self.reply_streamed = False
        self.limit_waits = 0

    # Waiting out a usage limit (see on_limit).
    MAX_LIMIT_WAIT_SECONDS = 6 * 3600
    MAX_LIMIT_WAITS = 3

    def hooks(self) -> ActivityHooks:
        return ActivityHooks(
            on_frontier=self._on_frontier,
            on_delegate=self._on_delegate,
            on_token=self._on_token,
            on_done=self._on_done,
            on_pull=self._on_pull,
            on_tool=self._on_tool,
            on_tool_result=self._on_tool_result,
            on_todos=self._on_todos,
            on_answer_text=self._on_answer_text,
            on_limit=self.on_limit,
        )

    def _stop_spinner(self) -> None:
        """Clear whatever is live on screen -- the spinner, or an answer being
        streamed -- before anything else prints. Every hook calls this first."""
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self._live is not None:
            self._live.update(Markdown(self._answer), refresh=True)
            self._live.stop()
            self._live = None

    def _on_answer_text(self, text: str) -> None:
        """The orchestrator's answer, rendered as markdown while it's written
        (like Claude Code), instead of appearing all at once at the end."""
        if self._live is None:
            self._stop_spinner()
            console.print()
            self._answer = ""
            self._live = Live(Markdown(""), console=console, refresh_per_second=12, vertical_overflow="visible")
            self._live.start()
            self.reply_streamed = True
        self._answer += text
        self._live.update(Markdown(self._answer))

    def _on_frontier(self, round_number: int) -> None:
        self._stop_spinner()
        self.reply_streamed = False  # a new reply starts
        doing = "thinking about the plan" if round_number == 1 else "thinking about the results"
        self._status = console.status(f"[bold success]Forging with {self.frontier_model}… ({doing})")
        self._status.start()

    def _on_delegate(self, modality: str, entry) -> None:
        self._stop_spinner()
        console.print(f"  → delegating [bold]{modality}[/bold] to [accent]{entry.name}[/accent] (local, via {entry.runtime})")
        self._at_line_start = True
        self._first_token_at = None
        # Nothing streams while the model loads into memory, which can take
        # tens of seconds for a big model; keep that visible too.
        self._status = console.status(f"[dim]Forging with {entry.name}… (loading the model)[/dim]")
        self._status.start()

    def _on_token(self, chunk: str) -> None:
        if self._first_token_at is None:
            self._stop_spinner()
            self._first_token_at = time.monotonic()
        # Indent every line so streamed output reads as nested under its
        # "delegating" line; markup/highlight off so code isn't mangled.
        text = chunk.replace("\n", "\n" + self.INDENT)
        if self._at_line_start:
            text = self.INDENT + text
        self._at_line_start = chunk.endswith("\n")
        if self._at_line_start:
            text = text[: -len(self.INDENT)]
        console.print(text, end="", style="dim", markup=False, highlight=False, soft_wrap=True)

    def _on_done(self, modality: str, entry, tokens: int, seconds: float) -> None:
        self._stop_spinner()
        if not self._at_line_start:
            console.print()
        # Rate from the first token on: the wall time also covers loading the
        # model into memory, which would make a big model look absurdly slow.
        generating = time.monotonic() - self._first_token_at if self._first_token_at else 0
        rate = f", {tokens / generating:.0f} tok/s" if generating > 0 and tokens else ""
        console.print(f"  [success]✓[/success] {entry.name} finished {modality}: {tokens} tokens in {seconds:.1f}s{rate}")
        self._at_line_start = True

    TOOL_LABELS = {
        "read_file": "Read",
        "list_files": "List",
        "search": "Search",
        "edit_file": "Edit",
        "run_command": "Bash",
        "make_dir": "Mkdir",
        "move_path": "Move",
        "delete_path": "Delete",
        "web_search": "Web search",
        "fetch_url": "Fetch",
        "compact": "Memory",
        "retry": "Retry",
        "checkpoint": "Checkpoint",
        "check": "Check",
        "remember": "Remember",
        "forget": "Forget",
    }

    def _on_tool(self, tool_name: str, summary: str) -> None:
        self._stop_spinner()
        label = self.TOOL_LABELS.get(tool_name, tool_name)
        # escape(): paths, queries and commands can contain [brackets] Rich would eat as markup
        console.print(f"  [accent]●[/accent] [bold]{label}[/bold] {escape(summary)}", highlight=False)
        if tool_name in ("web_search", "fetch_url", "run_command", "compact"):
            self._status = console.status("[dim]working...[/dim]")
            self._status.start()

    def _on_tool_result(self, tool_name: str, result: str) -> None:
        self._stop_spinner()
        lines = result.strip().splitlines() or [""]
        if tool_name == "read_file":
            shown = [lines[0]]
        elif tool_name == "run_command":
            shown = lines[:1] + lines[-6:] if len(lines) > 7 else lines
        elif tool_name in ("list_files", "search"):
            shown = [f"{len(lines)} result line(s)" if not lines[0].startswith("No ") else lines[0]]
        else:
            shown = lines[:1]
        for i, line in enumerate(shown):
            prefix = "    ⎿ " if i == 0 else "      "
            console.print(f"[dim]{prefix}{escape(line[:160])}[/dim]", highlight=False)

    def _on_todos(self, todos: list[dict]) -> None:
        self._stop_spinner()
        marks = {"completed": "[success]☒[/success]", "in_progress": "[warning]◐[/warning]"}
        console.print("  [accent]●[/accent] [bold]Plan[/bold]")
        for t in todos:
            mark = marks.get(t.get("status"), "☐")
            text = escape(str(t.get("content", "")))
            if t.get("status") == "completed":
                text = f"[dim strike]{text}[/dim strike]"
            console.print(f"      {mark} {text}", highlight=False)

    def _note_change(self, kind: str, allowed: bool) -> None:
        if allowed and kind in ("write", "delete"):
            _session.files_changed += 1

    def approve(self, kind: str, title: str, detail: str) -> bool:
        """Claude-Code-style permission prompt: show the diff or command, then
        yes / no / always (for this kind, this session). Without a terminal
        to ask on, the change is refused and the orchestrator is told so.
        """
        self._stop_spinner()
        if _session.auto_approve or kind in _session.always_allow:
            self._print_change(kind, title, detail)
            self._note_change(kind, True)
            return True
        self._print_change(kind, title, detail)
        if not sys.stdin.isatty():
            console.print("[warning]  No terminal to ask on — declined. Use --yes to approve changes non-interactively.[/warning]")
            return False
        _drain_buffered_input()
        what = {"write": "file changes", "delete": "deletions", "command": "commands", "download": "model downloads"}.get(kind, kind)
        while True:
            # (y)es not [y]es: Rich reads [y] as a style tag and prints nothing
            answer = console.input(f"  Allow? [bold](y)[/bold]es / [bold](n)[/bold]o / [bold](a)[/bold]lways allow {what} this session: ").strip().lower()
            if answer in ("y", "yes"):
                self._note_change(kind, True)
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                _session.always_allow.add(kind)
                self._note_change(kind, True)
                return True

    def _print_change(self, kind: str, title: str, detail: str) -> None:
        if kind == "command":
            body = f"[bold]$ {escape(detail)}[/bold]"
        elif kind == "download":
            body = f"[warning]↓[/warning] {escape(detail)}"
        elif kind == "delete":
            body = f"[error]{escape(detail)}[/error]"
        else:
            styled = []
            for line in detail.splitlines():
                if line.startswith(("+++", "---")):
                    styled.append(f"[bold]{escape(line)}[/bold]")
                elif line.startswith("+"):
                    styled.append(f"[success]{escape(line)}[/success]")
                elif line.startswith("-"):
                    styled.append(f"[error]{escape(line)}[/error]")
                elif line.startswith("@@"):
                    styled.append(f"[accent]{escape(line)}[/accent]")
                else:
                    styled.append(f"[dim]{escape(line)}[/dim]")
            body = "\n".join(styled) or "[dim](empty file)[/dim]"
        console.print(Panel(body, title=escape(title), title_align="left", expand=True, border_style="panel.border"))

    def _on_pull(self, model_name: str, event: dict) -> None:
        # A download should be rare now (installed models are preferred), but
        # never silent: report it in 10% steps.
        self._stop_spinner()
        total, completed = event.get("total"), event.get("completed")
        if not total or completed is None:
            return
        pct = int(100 * completed / total) // 10 * 10
        if pct > self._pull_shown.get(model_name, -1):
            self._pull_shown[model_name] = pct
            console.print(f"    [warning]↓[/warning] downloading {model_name} ({total / 1e9:.1f} GB): {pct}%")

    def on_limit(self, exc):
        """The account hit its usage limit. The work so far is fine, so wait
        for the reset and carry on rather than throwing the task away. While
        waiting, /model (or `localforge model` in another window) switches
        the orchestrator and the task continues straight away."""
        self._stop_spinner()
        self.limit_waits += 1
        who = exc.provider or self.frontier_model
        reset_at = getattr(exc, "reset_at", None)
        seconds = (reset_at - datetime.now(reset_at.tzinfo)).total_seconds() if reset_at else 0
        if reset_at is None or seconds > self.MAX_LIMIT_WAIT_SECONDS or self.limit_waits > self.MAX_LIMIT_WAITS:
            console.print(
                f"[bold error]{escape(who)} hit its usage limit[/bold error] and "
                + ("didn't say when it resets." if reset_at is None else f"it doesn't reset until {reset_at:%d %b %H:%M}.")
                + " The work so far is kept — switch with /model (e.g. /model ollama/qwen2.5:7b) and say continue, "
                "or come back after the reset.",
                highlight=False,
            )
            return None
        console.print(
            f"[warning]⏸ {escape(who)} hit its usage limit.[/warning] Waiting until "
            f"{reset_at:%H:%M} ({_human_duration(seconds)}) and then carrying on. "
            "/model switches the orchestrator to continue now; /stop gives up.",
            highlight=False,
        )
        switched = self._sleep_until(reset_at)
        if switched is not None:
            return switched
        console.print(f"[success]▶ {escape(who)} should be available again — continuing.[/success]", highlight=False)
        return "retry"

    def _sleep_until(self, reset_at):
        """Wait, checking every few seconds whether the user switched the
        orchestrator (then the task continues on that one instead)."""
        started_with = os.environ.get(config.FRONTIER_MODEL_ENV_VAR)
        while True:
            remaining = (reset_at - datetime.now(reset_at.tzinfo)).total_seconds()
            if remaining <= 0:
                return None
            self._limit_tick(remaining)
            time.sleep(min(LIMIT_POLL_SECONDS, max(remaining, 0.1)))
            current = os.environ.get(config.FRONTIER_MODEL_ENV_VAR)
            if current and current != started_with:
                console.print(f"[success]▶ switching to {escape(current)} and carrying on.[/success]", highlight=False)
                return ("switch", current, _cli_provider_for(current, explicit=False))

    def _limit_tick(self, remaining: float) -> None:
        """Foreground: a spinner with the countdown (the background display
        puts it in the status bar instead)."""
        if self._status is None:
            self._status = console.status("")
            self._status.start()
        self._status.update(f"[warning]waiting for the usage limit to reset — {_human_duration(remaining)} left[/warning]")

    def close(self) -> None:
        self._stop_spinner()


class _BackgroundActivity(_LiveActivity):
    """Hooks for a task running on the background worker (see background.py).
    Same events as _LiveActivity, but compressed: no spinner, no Live, no
    token-by-token output -- the status bar shows what's in progress, each
    finished step prints one line, and the answer prints when it's done.
    Permission prompts are handed to the main thread, which owns the keyboard."""

    def __init__(self, frontier_model: str, runner):
        super().__init__(frontier_model)
        self.runner = runner
        runner.state.orchestrator = frontier_model
        # Effort only applies to a provider CLI; worth showing, since it's
        # what the orchestrator's thinking costs.
        provider = _cli_provider_for(frontier_model, explicit=False)
        effort = os.environ.get(config.ORCHESTRATOR_EFFORT_ENV_VAR) or config.DEFAULT_ORCHESTRATOR_EFFORT
        self._effort = f" with {effort} effort" if provider and config.FRONTIER_CLI_AUTH.get(provider, {}).get("effort_flag") else ""

    def hooks(self) -> ActivityHooks:
        hooks = super().hooks()
        hooks.on_answer_text = self._on_answer_text
        hooks.poll_notes = self._poll_notes
        return hooks

    def _step(self, line: str) -> None:
        self.runner.state.steps.append(line)
        self.runner.note_event()

    def _stop_spinner(self) -> None:  # nothing live to stop in background mode
        return

    def _on_frontier(self, round_number: int) -> None:
        self.runner.check_cancel()
        self.runner.note_event()
        state = self.runner.state
        state.local_model = state.downloading = ""
        state.answer_words = 0
        thinking = "thinking" + self._effort
        state.phase = f"{thinking} about the plan" if round_number == 1 else f"{thinking} about the results (step {round_number})"
        self.reply_streamed = False

    def _poll_notes(self) -> list[str]:
        notes = self.runner.take_notes()
        for note in notes:
            console.print(f"  [accent]●[/accent] [bold]Your note[/bold] passed to {escape(self.frontier_model)}: {escape(note)}", highlight=False)
        return notes

    def _on_delegate(self, modality: str, entry) -> None:
        self.runner.check_cancel()
        self.runner.clear_preview()
        state = self.runner.state
        state.local_model, state.local_what = entry.name, f"working on {modality}"
        state.local_tokens, state.local_started = 0, time.monotonic()
        console.print(f"  → {modality} → [accent]{entry.name}[/accent] (local)", highlight=False)
        self._step(f"→ {modality} delegated to {entry.name}")

    def _on_token(self, chunk: str) -> None:
        self.runner.check_cancel()
        state = self.runner.state
        state.local_tokens += 1
        state.local_total += 1
        self.runner.note_output(chunk)
        if _session.stream_output:  # /stream on: the full firehose, as the foreground display does
            super()._on_token(chunk)

    def _on_done(self, modality: str, entry, tokens: int, seconds: float) -> None:
        state = self.runner.state
        state.local_model = ""
        self.runner.clear_preview()
        console.print(f"  [success]✓[/success] {entry.name} finished {modality}: {tokens} tokens in {seconds:.1f}s", highlight=False)
        self._step(f"✓ {entry.name} finished {modality} ({tokens} tokens)")

    def _on_pull(self, model_name: str, event: dict) -> None:
        total, completed = event.get("total"), event.get("completed")
        if total and completed is not None:
            self.runner.state.downloading = f"downloading {model_name}: {int(100 * completed / total)}%"

    def _on_tool(self, tool_name: str, summary: str) -> None:
        self.runner.check_cancel()
        label = self.TOOL_LABELS.get(tool_name, tool_name)
        self.runner.state.phase = f"{label.lower()} {summary}"[:80]
        console.print(f"  [accent]●[/accent] [bold]{label}[/bold] {escape(summary)}", highlight=False)
        self._step(f"● {label} {summary}"[:100])

    def _on_tool_result(self, tool_name: str, result: str) -> None:
        first = (result.strip().splitlines() or [""])[0]
        console.print(f"[dim]    ⎿ {escape(first[:140])}[/dim]", highlight=False)

    def _on_todos(self, todos: list[dict]) -> None:
        self.runner.state.todos = todos
        done = sum(t.get("status") == "completed" for t in todos)
        current = next((t.get("content", "") for t in todos if t.get("status") == "in_progress"), "")
        console.print(
            f"  [accent]●[/accent] [bold]Plan[/bold] {done}/{len(todos)} done" + (f" — now: {escape(current)}" if current else ""),
            highlight=False,
        )

    def _on_answer_text(self, text: str) -> None:
        # Collected, not streamed: run() prints it formatted once it's done.
        self._answer += text
        state = self.runner.state
        state.answer_words = len(self._answer.split())
        state.answer_chars = len(self._answer)
        self.runner.note_event()

    def _limit_tick(self, remaining: float) -> None:
        self.runner.check_cancel()
        self.runner.state.waiting_for = f"usage limit resets in {_human_duration(remaining)}"

    def _sleep_until(self, reset_at):
        try:
            return super()._sleep_until(reset_at)
        finally:
            self.runner.state.waiting_for = ""

    def approve(self, kind: str, title: str, detail: str) -> bool:
        if _session.auto_approve or kind in _session.always_allow:
            self._print_change(kind, title, detail)
            self._note_change(kind, True)
            return True
        self._print_change(kind, title, detail)
        answer = self.runner.ask(kind, title, detail)
        if answer.always:
            _session.always_allow.add(kind)
        self._note_change(kind, answer.allowed)
        return answer.allowed

    def close(self) -> None:
        return


class _SessionState:
    """What an interactive session remembers between messages. The REPL runs
    every command in-process, so module-level state is the session."""

    def __init__(self) -> None:
        self.conversation: Conversation | None = None
        self.root: Path | None = None
        self.scratch: Scratchpad | None = None
        self.auto_approve = False
        self.always_allow: set[str] = set()
        self.stream_output = False  # /stream on: print local output in full as well
        self.id = uuid.uuid4().hex[:12]  # this session, for the usage history
        self.files_changed = 0  # approved writes/deletes, so the exit hook knows if the brief is stale
        self.announced = False
        self.interactive = False  # True inside the REPL; a one-off run cleans up after itself
        self.runner = None  # background.TaskRunner when the session runs tasks in the background

    def make_activity(self, frontier_model: str):
        """The display for a task: compressed and approval-by-handoff on the
        background worker, the classic live one everywhere else."""
        runner = self.runner
        if runner is not None and threading.current_thread() is runner.thread:
            return _BackgroundActivity(frontier_model, runner)
        return _LiveActivity(frontier_model)

    def conversation_for(self, root: Path) -> Conversation:
        if self.conversation is None or self.root != root:
            self.root = root
            self.conversation = Conversation(memory=memory.load(root), facts=memory.facts_for_prompt(root))
            notes = []
            if self.conversation.memory:
                notes.append(f"the last session's summary ({len(self.conversation.memory.split())} words)")
            if facts := memory.list_facts(root):
                notes.append(f"{len(facts)} remembered fact{'s' if len(facts) != 1 else ''}")
            if notes:
                console.print(f"[dim]Resuming with {' and '.join(notes)} — /memory to see, /clear to start fresh.[/dim]")
        return self.conversation

    def scratchpad_for(self, root: Path) -> Scratchpad:
        if self.scratch is None:
            self.scratch = Scratchpad(root)
            self.scratch.ensure()
        return self.scratch

    def drop_scratchpad(self) -> None:
        if self.scratch is not None:
            self.scratch.remove()
            self.scratch = None


_session = _SessionState()


@app.command()
def clear(
    forget: bool = typer.Option(False, "--forget", help="Also delete this folder's saved session memory."),
) -> None:
    """Start a fresh conversation (the saved session memory is kept unless --forget)."""
    _session.conversation = Conversation()
    _session.root = Path.cwd().resolve()
    if forget:
        memory.forget(Path.cwd())
        console.print("[success]✓[/success] Conversation cleared and this folder's saved memory deleted.")
    else:
        console.print("[success]✓[/success] Conversation cleared. (Saved memory for this folder is kept; /clear --forget deletes it.)")


@app.command()
def compact() -> None:
    """Have a local model condense the conversation so far into session memory."""
    conversation = _session.conversation
    if conversation is None or len(conversation.messages) < 3:
        console.print("Nothing to compact yet.")
        return
    activity = _LiveActivity("local model")
    dispatcher = _memory_dispatcher(activity.hooks())  # never `or None`: an empty set means nothing installed, not unknown
    try:
        done = memory.compact(conversation, dispatcher, activity.hooks(), keep_recent_turns=0, root=_session.root or Path.cwd())
    finally:
        activity.close()
    if done and conversation.memory:
        console.print(Panel(Markdown(conversation.memory), title="Session memory", border_style="panel.border"))


_PROVIDER_PREFIXES = {"claude": "anthropic", "gpt": "openai", "o1": "openai", "o3": "openai", "o4": "openai", "gemini": "gemini"}


def _cloud_provider(model: str) -> str | None:
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    return None


def _model_choices() -> list[tuple[str, str, dict]]:
    """(model id, label, config to save) for every orchestrator usable here:
    each model already in Ollama, plus cloud models whose key or CLI exists.
    """
    choices = []
    for name in sorted(_installed_model_names(OllamaBackend())):
        choices.append(
            (
                f"ollama/{name}",
                f"ollama/{name}  (local — no account, no billing)",
                {config.AUTH_METHOD_ENV_VAR: config.AUTH_LOCAL, config.FRONTIER_PROVIDER_ENV_VAR: "local"},
            )
        )
    for provider in config.FRONTIER_MODEL_CHOICES:
        if provider == "local":
            continue
        env_var = FRONTIER_PROVIDERS.get(provider)
        spec = config.FRONTIER_CLI_AUTH.get(provider)
        if env_var and os.environ.get(env_var):
            auth, how = config.AUTH_API_KEY, f"API key {env_var}"
        elif spec and cli_transport.available(provider):
            auth, how = config.AUTH_CLI_LOGIN, f"your `{spec['command']}` login"
        else:
            continue
        for model in _provider_models(provider) if auth == config.AUTH_API_KEY else config.FRONTIER_MODEL_CHOICES[provider]:
            choices.append(
                (model, f"{model}  ({how})", {config.AUTH_METHOD_ENV_VAR: auth, config.FRONTIER_PROVIDER_ENV_VAR: provider})
            )
    return choices


_GEMINI_SKIP = ("tts", "image", "embedding", "live", "audio", "transcribe", "computer-use", "customtools", "translate", "aqa")


_gemini_cache: dict[str, tuple[str, ...]] = {}


def _gemini_models(api_key: str) -> tuple[str, ...]:
    if api_key not in _gemini_cache:
        found = _fetch_gemini_models(api_key)
        if not found:
            return ()  # not cached: offline now doesn't mean offline for the whole session
        _gemini_cache[api_key] = found
    return _gemini_cache[api_key]


def _fetch_gemini_models(api_key: str) -> tuple[str, ...]:
    """The text models this Google AI Studio key can use, newest and most
    capable first, straight from Google's model list -- so the menu is
    current without localforge guessing at model ids. Empty on any failure
    (the curated list is used instead)."""
    try:
        resp = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"pageSize": 1000},
            headers={"x-goog-api-key": api_key},  # a header, so the key never lands in a URL or log
            timeout=5.0,
        )
        resp.raise_for_status()
        models = resp.json().get("models") or []
    except Exception:  # noqa: BLE001 - a nicer menu, never a reason to fail
        return ()
    names = []
    for m in models:
        name = str(m.get("name") or "").removeprefix("models/")
        if (
            name.startswith("gemini-")
            and "generateContent" in (m.get("supportedGenerationMethods") or [])
            and not any(skip in name for skip in _GEMINI_SKIP)
        ):
            names.append(name)

    def rank(name: str):
        version = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        tier = 0 if "-pro" in name else 2 if "lite" in name else 1
        return (-float(version.group(1)) if version else 0.0, "preview" in name or "exp" in name, tier, name)

    return tuple(f"gemini/{n}" for n in sorted(set(names), key=rank)[:10])


def _provider_models(provider: str) -> list[str]:
    """Model ids to offer for `provider`: Google's live list when an AI
    Studio key is set, else the curated list."""
    curated = list(config.FRONTIER_MODEL_CHOICES.get(provider, []))
    if provider == "gemini" and (key := os.environ.get(FRONTIER_PROVIDERS["gemini"])):
        return list(_gemini_models(key)) or curated
    return curated


def _set_orchestrator(model: str, auth: dict) -> None:
    config.save({config.FRONTIER_MODEL_ENV_VAR: model, **auth})
    _session.announced = False
    console.print(f"[success]✓[/success] Orchestrator is now {escape(_orchestrator_label(model, _cli_provider_for(model, explicit=False)))}")


def _ask_trust(folder: Path) -> bool:
    """Claude-Code-style trust prompt for a folder not trusted before.
    Numbered, no default. Returns whether the folder is (now) trusted."""
    if trust.is_trusted(folder):
        return True
    home_note = (
        "\n[warning]This is your home folder, so everything in it would be in reach. "
        "Usually you want a project folder instead.[/warning]"
        if trust.looks_like_home(folder)
        else ""
    )
    console.print(
        Panel(
            f"[bold]Do you trust the files in this folder?[/bold]\n\n  {escape(str(folder))}\n\n"
            "localforge will be able to read, create, change, move and delete files here and run commands in it. "
            "Each change and command still asks you first (unless you turn on /auto)."
            f"{home_note}",
            title="Folder trust",
            border_style="panel.border",
        )
    )
    console.print("  1) Yes, trust this folder\n  2) No, exit")
    _drain_buffered_input()
    while True:
        try:
            answer = console.input("Choose 1-2: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        if answer == "1":
            console.print(f"[success]✓[/success] Trusted {escape(str(folder))}\n")
            return trust.apply_choice(folder, "yes")
        if answer == "2":
            return trust.apply_choice(folder, "no")
        if answer:  # a question rather than a choice
            console.print(
                "Trusting a folder lets localforge read, create, change, move and delete files in it and run commands "
                "there, on your behalf. It's asked once per folder because everything localforge does happens inside "
                "it. Each individual change and command still asks you separately (unless you turn on /auto), and "
                "nothing outside this folder is touched. Say 2 to leave without trusting it.",
                highlight=False,
            )
            continue
        console.print("[warning]Type 1 or 2.[/warning]")


def _run_in_background(task: str) -> None:
    """The worker thread's job: the same `run` command a foreground task uses."""
    try:
        app(["run", task], standalone_mode=False)
    except typer.Exit:
        pass
    except Exception as exc:  # noqa: BLE001 - reported, then the queue carries on
        console.print(f"[error]Error:[/error] {escape(str(exc))}")
    finally:
        console.print()


def _start_session() -> bool:
    """Before the first prompt: folder trust, then the orchestrator choice."""
    _session.interactive = True
    if not _ask_trust(Path.cwd().resolve()):
        console.print("Not trusted — exiting. cd into a folder you trust and run localforge there.")
        return False
    if not _choose_orchestrator_at_start():
        return False
    _offer_upgrades_at_start()
    _offer_brief_update_at_start(Path.cwd().resolve())
    _offer_open_work_at_start(Path.cwd().resolve())
    return True


def _offer_open_work_at_start(folder: Path) -> None:
    """A task that stopped before it was done (Ctrl+C, a usage ceiling, an
    error the orchestrator couldn't work around) is saved rather than lost
    to a summary; offer to pick it back up (reported: "some check/loop to
    ensure the task is completed"). Kept until it's resumed and finishes, or
    the user says to drop it -- not silently forgotten either way."""
    if not sys.stdin.isatty():
        return
    items = memory.open_work(folder)
    if not items:
        return
    entry = items[-1]
    more = f" (and {len(items) - 1} older unfinished task(s); /memory to see them)" if len(items) > 1 else ""
    console.print(f"\n[bold]Unfinished from last session:[/bold]{more}")
    console.print(escape(memory.describe_open_work(entry)))
    console.print("  1) Continue it now\n  2) Not now (you'll be asked again next time)\n  3) Discard it")
    answer = _ask_number("Choose", 3)
    if answer == 1:
        resume = memory.RESUME_PREFIX + entry["task"] + "\n\n" + memory.describe_open_work(entry)
        try:
            app(["run", resume], standalone_mode=False)
        except typer.Exit:
            pass
    elif answer == 3:
        memory.clear_open_work(folder, entry["task"])
        console.print("[dim]Discarded.[/dim]")
    console.print()


def _offer_brief_update_at_start(folder: Path) -> None:
    """A drafted AGENTS.md update from the end of the last session (it
    changed files, so the project's scope may have moved). It used to wait,
    with one dim line at exit, until the user happened to run /init, and the
    brief drifted. Offer it now; approving still goes through the diff."""
    if not brief.pending_path(folder).is_file() or not sys.stdin.isatty():
        return
    console.print(
        f"\n[bold]{brief.BRIEF_FILE} may be out of date.[/bold] Last session changed files, and a local model drafted an "
        "update: what's still true kept, what changed added."
    )
    console.print("  1) Review it now (you'll see the diff)\n  2) Later (/init when you're ready)")
    if _ask_number("Choose", 2) == 1:
        try:
            app(["init"], standalone_mode=False)
        except typer.Exit:
            pass
    console.print()


# --- upgrading installed models --------------------------------------------------


def _upgrade_plan() -> tuple[upgrades.Plan, OllamaBackend] | None:
    """What an upgrade would do here, or None if Ollama can't be asked."""
    ollama = OllamaBackend()
    try:
        on_disk = {m["name"]: int(m.get("size") or 0) for m in ollama.list_installed()}
    except Exception:  # noqa: BLE001 - no Ollama, nothing to upgrade
        return None
    return upgrades.plan(detect_hardware(), on_disk, keep=upgrades.local_orchestrator()), ollama


def _print_upgrade_plan(plan: upgrades.Plan) -> None:
    for line in plan.summary():
        console.print(f"  • {escape(line)}", highlight=False)
    if plan.download_gb or plan.freed_gb:
        console.print(f"  [dim]download ~{plan.download_gb:g} GB · frees ~{plan.freed_gb:g} GB afterwards[/dim]")


def _offer_upgrades_at_start() -> None:
    """At session start: if a better model fits this machine, or models
    localforge installed are no longer used, upgrade -- asking the first
    time, then as the user chose ("always" runs it in the background)."""
    setting = os.environ.get(upgrades.AUTO_UPGRADE_ENV_VAR, "")
    if setting == "never" or not sys.stdin.isatty():
        return
    found = _upgrade_plan()
    if found is None:
        return
    plan, ollama = found
    if plan.blocked:
        console.print(f"[dim]A better local model fits this machine, but the upgrade {escape(plan.blocked)}.[/dim]")
    if plan.empty:
        return
    if setting == "always":
        _start_background_upgrade(plan, ollama)
        return
    console.print("\n[bold]Better local models fit this machine:[/bold]")
    _print_upgrade_plan(plan)
    console.print(
        "  1) Upgrade now, in the background\n  2) Always upgrade automatically from now on\n"
        "  3) Not now\n  4) Never ask (/upgrade still works)"
    )
    answer = _ask_number("Choose", 4)
    if answer in (1, 2):
        if answer == 2:
            config.save({upgrades.AUTO_UPGRADE_ENV_VAR: "always"})
        _start_background_upgrade(plan, ollama)
    elif answer == 4:
        config.save({upgrades.AUTO_UPGRADE_ENV_VAR: "never"})
    console.print()


_upgrade_lock = threading.Lock()
UPGRADE_IDLE_POLL_SECONDS = 5.0


def _start_background_upgrade(plan: upgrades.Plan, ollama: OllamaBackend) -> threading.Thread | None:
    """Download in the background while the session carries on; the new
    model is used from the next task on (a model on disk wins). Old models
    are removed only once no task is running, since one might be using them."""
    if not _upgrade_lock.acquire(blocking=False):
        return None  # one upgrade at a time
    names = ", ".join(u.new.name for u in plan.upgrades)
    if names:
        console.print(f"[dim]↓ Upgrading in the background: {escape(names)} (~{plan.download_gb:g} GB). Keep working.[/dim]")

    shown: dict[str, int] = {}

    def milestones(name: str, event: dict) -> None:
        # A big download takes a while: a line every 25% so it's never silent.
        total, done = event.get("total"), event.get("completed")
        if total and done is not None:
            quarter = int(done * 4 / total)
            if 0 < quarter < 4 and quarter > shown.get(name, 0):
                shown[name] = quarter
                console.print(f"[dim]↓ {escape(name)}: {quarter * 25}% of ~{total / 1e9:.1f} GB[/dim]")

    def work() -> None:
        try:
            _apply_upgrade(plan, ollama, wait_for_idle=True, on_progress=milestones)
        finally:
            _upgrade_lock.release()

    thread = threading.Thread(target=work, name="localforge-upgrade", daemon=True)
    thread.start()
    return thread


def _apply_upgrade(plan: upgrades.Plan, ollama: OllamaBackend, wait_for_idle: bool = False, on_progress=None) -> bool:
    """Download every new model, then remove the ones no longer used. Old
    models stay if any download fails -- they're still what's in use."""
    for upgrade in plan.upgrades:
        name = upgrade.new.name
        try:
            ollama.ensure_available(name, on_progress=(lambda event, name=name: on_progress(name, event)) if on_progress else None)
        except Exception as exc:  # noqa: BLE001 - reported; the old model stays in use
            console.print(f"[warning]![/warning] Upgrade stopped: couldn't download {escape(name)} ({escape(str(exc))}). Nothing was removed.")
            return False
        upgrades.mark_managed(name)
        console.print(
            f"[success]✓[/success] {escape(', '.join(upgrade.modalities))} now uses {escape(name)} (was {escape(upgrade.old)}) "
            "from the next task. Memory and the conversation carry over; they're kept per project, not per model."
        )
    if plan.remove and wait_for_idle:
        while _session.runner is not None and _session.runner.busy:
            time.sleep(UPGRADE_IDLE_POLL_SECONDS)  # a running task may still be using an old model
    removed = []
    # Checked now, not when the plan was made: /model may have made one of
    # these the orchestrator during a long download.
    keep = upgrades.local_orchestrator()
    for name in plan.remove:
        if name in keep:
            continue
        try:
            ollama.delete(name)
        except Exception as exc:  # noqa: BLE001 - one failed delete doesn't stop the rest
            console.print(f"[warning]![/warning] Couldn't remove {escape(name)}: {escape(str(exc))}")
            continue
        upgrades.unmark(name)
        removed.append(name)
    if removed:
        console.print(f"[success]✓[/success] Removed {escape(', '.join(removed))}, freeing ~{plan.freed_gb:g} GB.")
    return True


@app.command()
def upgrade(yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before downloading/removing.")) -> None:
    """Upgrade installed local models to better ones that fit this machine, and remove old ones."""
    found = _upgrade_plan()
    if found is None:
        console.print("[error]Ollama isn't reachable.[/error] Start it, then retry.")
        raise typer.Exit(code=1)
    plan, ollama = found
    if plan.blocked:
        console.print(f"[warning]![/warning] A better model fits, but the upgrade {escape(plan.blocked)}.")
    if plan.empty and not plan.unused_own:
        console.print("[success]✓[/success] Your local models are already the best fit for this machine.")
        return
    if not plan.empty:
        console.print("[bold]Upgrade plan:[/bold]")
        _print_upgrade_plan(plan)
        if yes or (sys.stdin.isatty() and typer.confirm("Go ahead?", default=False)):
            with Progress(
                TextColumn("[progress.description]{task.description}"), BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), DownloadColumn(), console=console,
            ) as progress:
                tasks: dict[str, object] = {}

                def on_progress(name: str, event: dict) -> None:
                    task_id = tasks.setdefault(name, progress.add_task(name, total=None))
                    if event.get("total") and event.get("completed") is not None:
                        progress.update(task_id, total=event["total"], completed=event["completed"])

                _apply_upgrade(plan, ollama, on_progress=on_progress)
        else:
            console.print("Nothing changed.")
    if plan.unused_own:
        # Not installed by localforge: only ever removed when asked, one prompt, never with --yes.
        console.print(
            f"\nThese were installed outside localforge and localforge no longer uses them: {escape(', '.join(plan.unused_own))}"
        )
        if not yes and sys.stdin.isatty() and typer.confirm("Remove them too?", default=False):
            _apply_upgrade(upgrades.Plan(remove=plan.unused_own), ollama)


def _ask_number(prompt: str, count: int) -> int | None:
    """A 1..count choice, re-asked until valid; no default. None if the user
    backs out with Ctrl+C / Ctrl+D."""
    _drain_buffered_input()
    while True:
        try:
            answer = console.input(f"{prompt} (1-{count}): ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        if answer.isdigit() and 1 <= int(answer) <= count:
            return int(answer)
        console.print(f"[warning]Type a number from 1 to {count}.[/warning]")


def _current_is_usable(current: str, choices: list[tuple[str, str, dict]]) -> bool:
    """The saved orchestrator can still run here: it's among the choices
    (a downloaded local model, a cloud model with a key/login), or it's a
    custom cloud model id whose provider is set up."""
    if not current:
        return False
    if any(model_id == current for model_id, _, _ in choices):
        return True
    if _is_local_model(current):
        return False  # not downloaded (any more)
    provider = _cloud_provider(current)
    env_var = FRONTIER_PROVIDERS.get(provider) if provider else None
    return bool(
        (env_var and os.environ.get(env_var))
        or (provider and provider in config.FRONTIER_CLI_AUTH and cli_transport.available(provider))
    )


def _first_run_setup() -> bool:
    """Nothing to orchestrate with yet -- the install deliberately leaves
    this until now, so it happens where the user can see it. Offers to run
    setup here rather than telling them to go and do it themselves."""
    console.print(
        "[bold]Nothing to work with yet.[/bold] Setup installs Ollama if you don't have it, has a model "
        "picked for this machine's hardware, and saves how you want to reach a frontier model (or lets "
        "you skip that and stay fully local)."
    )
    console.print("  1) Set it up now\n  2) Not now")
    answer = _ask_number("Choose", 2)
    if answer is None:
        return False
    if answer != 1:
        console.print("[dim]No problem — run /setup when you're ready, or /model if you pull a model yourself.[/dim]\n")
        return True
    try:
        app(["setup"], standalone_mode=False)
    except typer.Exit:
        pass
    except Exception as exc:  # noqa: BLE001 - setup reports its own problems; the session goes on
        console.print(f"[error]Setup didn't finish:[/error] {escape(str(exc))}")
    console.print()
    if _model_choices():
        return _choose_orchestrator_at_start()  # now there's something to pick
    return True


def _choose_orchestrator_at_start() -> bool:
    """Every new session confirms the orchestrator. With a usable one from
    last time, it's a short "keep it, or choose another?"; the full list
    only comes up when asked for, or when there's nothing usable to keep.
    Numbered, no default. Returns False if the user backs out (Ctrl+C / Ctrl+D),
    which ends the session."""
    current = os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or ""
    choices = _model_choices()

    if _current_is_usable(current, choices):
        label = _orchestrator_label(current, _cli_provider_for(current, explicit=False))
        console.print(f"[bold]Orchestrator from last time:[/bold] {escape(label)}")
        console.print("  1) Keep using it\n  2) Choose a different model")
        answer = _ask_number("Choose", 2)
        if answer is None:
            return False
        if answer == 1:
            _session.announced = True  # just confirmed; don't repeat it on the first task
            _session.conversation_for(Path.cwd().resolve())
            console.print()
            return True
    elif current:
        console.print(f"[warning]{escape(current)} from last time isn't available here any more.[/warning]")

    if not choices:
        return _first_run_setup()
    console.print("[bold]Which model should orchestrate this session?[/bold]")
    for i, (model_id, label, _) in enumerate(choices, 1):
        last = "  [dim](last used)[/dim]" if model_id == current else ""
        console.print(f"  {i}) {escape(label)}{last}", highlight=False)
    answer = _ask_number("Choose", len(choices))
    if answer is None:
        return False
    model_id, _, auth = choices[answer - 1]
    _set_orchestrator(model_id, auth)
    _session.conversation_for(Path.cwd().resolve())
    console.print()
    return True


@app.command(name="model")
def model_command(
    model: str = typer.Argument(None, help="Model id, e.g. ollama/qwen2.5:7b or claude-opus-5. Omit to pick from a list."),
) -> None:
    """Show or switch the orchestrator model (a local Ollama model or a cloud one)."""
    current = os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or "claude-opus-5"
    choices = _model_choices()

    if model:
        model = config.litellm_model_id(model)
        for model_id, _, auth in choices:
            if model_id == model:
                _set_orchestrator(model_id, auth)
                return
        if _is_local_model(model):
            console.print(
                f"[error]{escape(local_transport.model_name(model))} isn't downloaded.[/error] "
                f"Run `ollama pull {escape(local_transport.model_name(model))}` first, then /model {escape(model)}."
            )
        elif _cloud_provider(model):
            provider = _cloud_provider(model)
            console.print(
                f"[error]No API key or CLI login for {provider} on this machine.[/error] "
                "Run /setup to add one, or pick a local model with /model."
            )
        else:
            console.print(f"[error]Unknown model {escape(model)!r}.[/error] Run /model to see what's available.")
        raise typer.Exit(code=1)

    console.print(f"Current orchestrator: [accent]{escape(_orchestrator_label(current, _cli_provider_for(current, explicit=False)))}[/accent]")
    if not choices:
        console.print("Nothing else is available: no models in Ollama and no cloud key or login. Run /setup.")
        return
    for i, (_, label, _) in enumerate(choices, 1):
        console.print(f"  {i}) {escape(label)}", highlight=False)
    if not sys.stdin.isatty():
        console.print("Switch with: /model <id>")
        return
    _drain_buffered_input()
    answer = console.input("Pick a number, or press Enter to keep the current one: ").strip()
    if not answer:
        return
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        model_id, _, auth = choices[int(answer) - 1]
        _set_orchestrator(model_id, auth)
    else:
        console.print("[warning]Not a number from the list — nothing changed.[/warning]")


def _memory_dispatcher(hooks=None) -> Dispatcher:
    return Dispatcher(detect_hardware(), installed=_installed_model_names(OllamaBackend()), hooks=hooks)


def _save_memory_at_exit() -> None:
    """At the end of a session: the local memory keeper saves what's worth
    remembering and a summary of where things stand (so the next session
    here picks up from it), and this session's scratchpad is deleted."""
    try:
        conversation = _session.conversation
        if conversation is None or not any(m.get("role") == "user" for m in conversation.messages[1:]):
            return
        root = _session.root or Path.cwd()
        activity = _LiveActivity("local model")
        dispatcher = _memory_dispatcher(activity.hooks())
        if memory.keeper(dispatcher) is None:
            console.print("[dim]No local model to keep memory, so this session wasn't saved to memory.[/dim]")
            return
        try:
            memory.extract_facts(conversation, dispatcher, root, activity.hooks())  # before compact trims the turns
            _draft_brief_update(root, dispatcher, activity)
            memory.compact(conversation, dispatcher, activity.hooks(), keep_recent_turns=0, root=root)
        finally:
            activity.close()
    finally:
        _session.drop_scratchpad()


def _draft_brief_update(root: Path, dispatcher, activity) -> None:
    """After a session that changed files, have the keeper draft an updated
    brief and leave it pending. Never written to the project here: the user
    reviews it with /init, like any other change."""
    current = brief.existing_brief(root)
    keeper = memory.keeper(dispatcher)
    if not current or not _session.files_changed or keeper is None:
        return
    workspace = Workspace(root, scratch=_session.scratch.root if _session.scratch else None)
    try:
        context = brief.gather_context(workspace, memory.load(root), memory.facts_for_prompt(root))
        text = brief.draft(keeper, context, current, context_limit=dispatcher.context_limit(keeper))
    except Exception:  # noqa: BLE001 - an optional nicety at exit
        return
    if text and text.strip() != current.strip():
        brief.save_pending(root, text)
        console.print(f"[dim]{keeper.name} drafted an updated {brief.BRIEF_FILE}; you'll be offered it next session (or /init).[/dim]")


@app.command(name="memory")
def memory_command(
    action: str = typer.Argument(None, help="Omit to show this folder's memory; `clear` deletes all of it; `forget` one."),
    name: str = typer.Argument(None, help="With `forget`: the memory's name."),
) -> None:
    """Show this folder's memory (remembered facts + last session summary), or clear it."""
    root = Path.cwd()
    if action == "clear":
        memory.forget(root)
        if _session.conversation is not None:
            _session.conversation.memory = _session.conversation.facts = ""
        console.print("[success]✓[/success] Everything remembered for this folder is deleted.")
        return
    if action == "forget":
        if not name:
            console.print("[error]Which one?[/error] /memory forget <name> (names are listed by /memory).")
            raise typer.Exit(code=1)
        if memory.forget_fact(root, name):
            console.print(f"[success]✓[/success] Forgot {escape(name)}.")
        else:
            console.print(f"[warning]No memory named {escape(name)!r}.[/warning]")
        return
    if action:
        console.print(f"[error]Unknown action {escape(action)!r}.[/error] Use /memory, /memory forget <name>, or /memory clear.")
        raise typer.Exit(code=1)

    keeper = memory.keeper(_memory_dispatcher())
    who = f"kept by {keeper.name} (local)" if keeper else "no local model available to keep it"
    facts, summary = memory.list_facts(root), memory.load(root)
    if not facts and not summary:
        console.print(f"Nothing remembered for this folder yet ({who}). Memory is saved when you /exit.")
        return
    if facts:
        table = Table(title=f"Remembered for {escape(root.name)}", title_justify="left")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Memory")
        for f in facts:
            table.add_row(escape(f["name"]), escape(f.get("type", "")), escape(f.get("description", "")))
        console.print(table)
    if summary:
        console.print(Panel(Markdown(summary), title="Last session", border_style="panel.border"))
    console.print(f"[dim]{escape(who)} · stored in {escape(str(memory.project_dir(root)))}[/dim]")


@app.command()
def scratch(
    action: str = typer.Argument(None, help="Omit to list the scratchpad; `clear` to empty it."),
) -> None:
    """Show this session's scratchpad (temporary drafts, deleted when the session ends)."""
    pad = _session.scratchpad_for(Path.cwd().resolve())
    if action == "clear":
        pad.clear()
        console.print("[success]✓[/success] Scratchpad emptied.")
        return
    if action:
        console.print(f"[error]Unknown action {escape(action)!r}.[/error] Use /scratch or /scratch clear.")
        raise typer.Exit(code=1)
    files = pad.files()
    console.print(f"[dim]{escape(str(pad.root))} — private to this session, deleted when it ends[/dim]")
    if not files:
        console.print("The scratchpad is empty.")
    for f in files:
        console.print(f"  scratchpad/{escape(str(f.relative_to(pad.root)))}  [dim]({f.stat().st_size:,} bytes)[/dim]", highlight=False)


@app.command()
def summary() -> None:
    """What's happening right now: the task, which model is doing what, the plan, the queue."""
    runner = _session.runner
    if runner is None:
        console.print("Nothing is running in the background (tasks run in the foreground here).")
        return
    for line in runner.summary():
        console.print(escape(line), highlight=False)


@app.command(name="queue")
def queue_command(
    action: str = typer.Argument(None, help="Omit to list queued tasks; `clear` to drop them."),
) -> None:
    """Tasks waiting behind the current one."""
    runner = _session.runner
    if runner is None or not runner.queue:
        console.print("The queue is empty. Type a task while one is running to queue it.")
        return
    if action == "clear":
        runner.queue.clear()
        console.print("[success]✓[/success] Queue cleared.")
        return
    for i, task in enumerate(runner.queue, 1):
        console.print(f"  {i}. {escape(task)}", highlight=False)


@app.command()
def stop() -> None:
    """Stop the running task (queued tasks then continue)."""
    runner = _session.runner
    if runner is None or not runner.cancel():
        console.print("Nothing is running.")
        return
    console.print("[warning]Stopping…[/warning] it will stop at its next step.")


@app.command()
def tell(note: list[str] = typer.Argument(None, help="A note for the running task.")) -> None:
    """Add a note to the running task; the orchestrator sees it at its next step."""
    text = " ".join(note or []).strip()
    runner = _session.runner
    if not text:
        console.print("[error]What should I tell it?[/error] /tell <note>")
        raise typer.Exit(code=1)
    if runner is None or not runner.tell(text):
        console.print("Nothing is running — just type it as a new task.")
        return
    console.print("[success]✓[/success] It'll see that at its next step.")


WHAT_IT_MEANS = {
    "write": (
        "A local model wrote this file and localforge is asking before saving it. The diff above is the whole change, "
        "nothing else is touched. Say no and the file stays exactly as it is; the orchestrator is told you declined "
        "and can try something else."
    ),
    "delete": (
        "This would remove the listed file(s) from your project for good — localforge has no undo, so if they aren't "
        "in git they're gone. Say no and nothing is removed."
    ),
    "command": (
        "This runs that exact command in your project folder, with your permissions. A shell command can do anything "
        "you can do (including reaching outside the folder), which is why every one is shown first. Say no and it "
        "isn't run."
    ),
    "download": (
        "No installed model fits this kind of work, so this would download one from Ollama — a real download of "
        "roughly the size shown, onto your disk. Say no and the orchestrator has to manage with what's installed."
    ),
}


def _explain_request(approval) -> list[str]:
    """Why localforge is asking, in plain terms -- from what we know, so it's
    instant and costs nothing."""
    if approval is None:
        return ["Nothing is waiting for an answer right now."]
    lines = [f"[bold]{escape(approval.title)}[/bold]", WHAT_IT_MEANS.get(approval.kind, "localforge needs your go-ahead for this.")]
    runner = _session.runner
    if runner is not None and runner.busy:
        state = runner.state
        lines.append(f"It came up while working on: {escape(state.task)}")
        if state.steps:
            lines.append("Just before this: " + escape("; ".join(list(state.steps)[-3:])))
    lines.append("[dim]Answer with (y)es, (n)o, or (a)lways for this kind — or ask another question.[/dim]")
    return lines


def _answer_with_local_model(question: str, approval) -> tuple[str, str] | None:
    """A short answer from a local model (free, and it's idle while the task
    waits). Skipped when no local model is installed."""
    dispatcher = _memory_dispatcher()
    entry = memory.keeper(dispatcher)
    if entry is None or approval is None:
        return None
    state = _session.runner.state if _session.runner is not None else None
    prompt = (
        "You are helping someone decide whether to approve a change a coding assistant wants to make. "
        "Answer their question in two or three sentences, plainly, using only the details below. "
        "Don't invent anything; if the details don't answer it, say so.\n\n"
        f"Request: {approval.title} ({approval.kind})\n"
        f"Details:\n{(approval.detail or '')[:2000]}\n"
        f"Task being worked on: {getattr(state, 'task', 'unknown')}\n\n"
        f"Their question: {question}"
    )
    try:
        from localforge.backends import BACKENDS

        result = BACKENDS[entry.runtime].generate(entry.name, prompt, context_limit=dispatcher.context_limit(entry))
    except Exception:  # noqa: BLE001 - an explanation is never worth an error
        return None
    text = str(result.get("content") or "").strip()
    return (text, entry.name) if text else None


def explain_to_user(question: str, approval) -> None:
    """Handles a question typed at a permission prompt."""
    for line in _explain_request(approval):
        console.print(line, highlight=False)
    if question.strip().lower().rstrip("?") not in ("why", "why this", "what", "explain", ""):
        if answer := _answer_with_local_model(question, approval):
            text, model = answer
            console.print(Panel(Markdown(text), title=f"Answer — from {escape(model)} (local)", border_style="panel.border"))


@app.command()
def init(
    refresh: bool = typer.Option(False, "--refresh", help="Rewrite the brief from scratch instead of updating it."),
) -> None:
    """Write (or update) AGENTS.md: what this project is, for every future session.

    A local model reads the project and what localforge remembers, and drafts
    it; you see the diff and approve it like any other change.
    """
    folder = Path.cwd().resolve()
    if not trust.is_trusted(folder) and not _ask_trust(folder):
        console.print("Not trusted — nothing written.")
        raise typer.Exit(code=1)

    activity = _session.make_activity("local model")
    dispatcher = _memory_dispatcher(activity.hooks())
    keeper = memory.keeper(dispatcher)
    if keeper is None:
        console.print(
            "[error]No local model is available to write the brief.[/error] "
            "Install one (e.g. `ollama pull qwen2.5:7b`) and run /init again."
        )
        raise typer.Exit(code=1)

    workspace = Workspace(folder, approver=activity.approve, scratch=_session.scratchpad_for(folder).root)
    current = "" if refresh else brief.existing_brief(folder)
    pending = "" if refresh else brief.take_pending(folder)
    if pending:
        console.print("[dim]Using the update drafted at the end of the last session.[/dim]")
        text = pending
    else:
        console.print(f"[dim]Reading the project with {escape(keeper.name)} (local)…[/dim]")
        context = brief.gather_context(workspace, memory.load(folder), memory.facts_for_prompt(folder))
        try:
            text = brief.draft(keeper, context, current, context_limit=dispatcher.context_limit(keeper))
        except Exception as exc:  # noqa: BLE001 - reported plainly, nothing written
            console.print(f"[error]Couldn't write the brief:[/error] {escape(str(exc))}")
            raise typer.Exit(code=1) from None
        finally:
            activity.close()
    if not text:
        console.print(f"[warning]{escape(keeper.name)} returned nothing usable. Try /init --refresh.[/warning]")
        raise typer.Exit(code=1)

    result = workspace.write_file(brief.BRIEF_FILE, text)
    console.print(result.splitlines()[0])
    if result.startswith(("Created ", "Updated ")):
        if _session.conversation is not None:
            _session.conversation.brief = brief.brief_for_prompt(folder)
        console.print(f"[dim]Every session in this folder now starts with {brief.BRIEF_FILE}. /init again to refresh it.[/dim]")
        if (folder / brief.LEGACY_BRIEF).is_file():
            # The brief's old name: its content is in AGENTS.md now. Removing it
            # goes through the normal delete approval, like any other change.
            console.print(f"[dim]{brief.LEGACY_BRIEF} is the old name for this file, and AGENTS.md replaces it.[/dim]")
            console.print(workspace.delete_path(brief.LEGACY_BRIEF).splitlines()[0])


@app.command()
def why() -> None:
    """Explain what localforge is asking you to approve, and why."""
    approval = _session.runner.approval if _session.runner is not None else None
    for line in _explain_request(approval):
        console.print(line, highlight=False)


@app.command()
def stream(
    state: str = typer.Argument(None, help="on or off; omit to toggle."),
) -> None:
    """Print local models' output in full as they write (off by default: the
    status line shows the last few lines instead)."""
    _session.stream_output = state == "on" if state in ("on", "off") else not _session.stream_output
    if _session.stream_output:
        console.print("[warning]Streaming ON[/warning]: every line a local model writes is printed. /stream off to stop.")
    else:
        console.print("[success]Streaming off[/success]: the status line shows the last few lines as they're written.")


@app.command()
def auto(
    state: str = typer.Argument(None, help="on or off; omit to toggle."),
) -> None:
    """Toggle auto-approval of file changes and commands for this session."""
    if state in ("on", "off"):
        _session.auto_approve = state == "on"
    else:
        _session.auto_approve = not _session.auto_approve
    if _session.auto_approve:
        console.print("[warning]Auto-approve ON[/warning]: file changes and commands run without asking. /auto off to stop.")
    else:
        console.print("[success]Auto-approve off[/success]: you'll be asked before each change or command.")


def _record_usage(frontier_model: str, stats) -> None:
    """Keep this task in the project's usage history, so /usage can still
    show it after the session ends. Never worth failing a task over."""
    try:
        usage_store.record(_session.root or Path.cwd().resolve(), _session.id, frontier_model, stats)
    except OSError:
        pass


# (frontier model, stats) for every task run in this process. Usage is only
# shown on request, so inside a session this is what /usage reads; a one-off
# `localforge run` process has nothing to show later (use `run --usage`).
_session_usage: list[tuple[str, RunStats]] = []


@app.command()
def usage() -> None:
    """Token usage: the last task, this session, the previous session, and this project's total."""
    root = _session.root or Path.cwd().resolve()
    if _session_usage:
        last_model, last_stats = _session_usage[-1]
        _print_usage_panel(last_stats, last_model, title="Usage — last task")
        if len(_session_usage) > 1:
            _print_session_usage_panel(_session_usage)
    else:
        console.print(
            "No tasks run in this session yet. For a one-off run, use "
            '[accent]localforge run --usage "..."[/accent].',
            highlight=False,
        )

    previous, total = usage_store.previous_session(root, _session.id), usage_store.all_time(root)
    if not total.tasks:
        return
    table = Table(title=f"Usage history for {escape(root.name)}", title_justify="left")
    for column in ("Span", "Tasks", "Local tokens", "Frontier tokens", "Cost"):
        table.add_column(column, justify="right" if column != "Span" else "left")
    if previous and previous.tasks:
        table.add_row("Previous session", *_usage_row(previous))
    table.add_row("This project, all time", *_usage_row(total))
    console.print(table)
    if total.local_tokens_generated and (saved := _savings_line(total.local_tokens_generated, total.models[0] if total.models else "")):
        console.print("All time: " + saved, highlight=False)


def _usage_row(totals) -> list[str]:
    cost = []
    if totals.frontier_cost_usd:
        cost.append(f"${totals.frontier_cost_usd:.2f}")
    if totals.subscription_cost_usd:
        cost.append(f"~${totals.subscription_cost_usd:.2f} of subscription")
    share = totals.local_tokens_generated * 100 // max(totals.local_tokens_generated + totals.frontier_total_tokens, 1)
    return [
        str(totals.tasks),
        f"{totals.local_tokens_generated:,} ({share}%)",
        f"{totals.frontier_total_tokens:,}",
        " + ".join(cost) or "-",
    ]


def _print_session_usage_panel(entries: list[tuple[str, RunStats]]) -> None:
    local = sum(s.local_tokens_generated for _, s in entries)
    prompt = sum(s.frontier_prompt_tokens for _, s in entries)
    completion = sum(s.frontier_completion_tokens for _, s in entries)
    models = sorted({m for m, _ in entries})
    # Tasks can mix transports (a --model run bills the API even when the
    # saved choice is CLI login), so keep billed and subscription cost apart.
    billed = sum(s.frontier_cost_usd for _, s in entries if not s.frontier_via_subscription)
    subscription = sum(s.frontier_cost_usd for _, s in entries if s.frontier_via_subscription)
    notes = []
    if billed:
        notes.append(f"${billed:.4f}")
    if subscription:
        notes.append(f"~${subscription:.4f} of subscription usage, not billed separately")
    cost = f" ({'; '.join(notes)})" if notes else ""
    lines = [
        *_work_split_lines(local, prompt, completion, ", ".join(models)),
        f"  [dim]{prompt + completion} frontier tokens in all{cost}[/dim]",
    ]
    if len(models) == 1 and (saved := _savings_line(local, models[0])):
        lines.append(saved)
    console.print(
        Panel("\n".join(lines), title=f"Usage — session ({len(entries)} tasks)", border_style="panel.border")
    )


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


def _savings_line(local_tokens: int, frontier_model: str) -> str:
    """What the local models' output would have cost if the orchestrator
    had written it, at its own output price (LiteLLM's price table). Empty
    when there's nothing to price (a local orchestrator, an unknown model)."""
    if not local_tokens or _is_local_model(frontier_model):
        return ""
    try:
        price = litellm.model_cost.get(frontier_model, {}).get("output_cost_per_token")
    except Exception:  # noqa: BLE001 - an estimate is optional
        price = None
    if not price:
        return ""
    return f"[success]≈ ${local_tokens * price:.4f} saved[/success]: what {local_tokens} tokens of local output would have cost from {frontier_model}"


def _print_usage_panel(stats, frontier_model: str, title: str = "Usage") -> None:
    usage_lines = [
        *_work_split_lines(
            stats.local_tokens_generated, stats.frontier_prompt_tokens, stats.frontier_completion_tokens, frontier_model
        ),
        f"  [dim]{stats.frontier_total_tokens} frontier tokens in all{_frontier_cost_note(stats)}[/dim]",
    ]
    if saved := _savings_line(stats.local_tokens_generated, frontier_model):
        usage_lines.append(saved)
    console.print(Panel("\n".join(usage_lines), title=title, border_style="panel.border"))


def _work_split_lines(local: int, frontier_in: int, frontier_out: int, frontier_label: str) -> list[str]:
    """Who did the work. The bar compares what each side *wrote*: local
    output against the orchestrator's own output (plans, instructions,
    answers). Comparing local output with everything the orchestrator
    *read* made every task look frontier-heavy, since each step re-reads
    the whole conversation. What it read is shown on its own line: that's
    where most frontier cost goes."""
    return [
        _usage_bar(local, frontier_out),
        "",
        f"[success]■[/success] Written by local models: {local} tokens — never sent to or billed by the frontier API",
        f"[warning]■[/warning] Written by the orchestrator ({frontier_label}): {frontier_out} tokens",
        f"  Orchestrator read: {frontier_in} tokens (its instructions, the conversation, file reads)",
    ]


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
    lines.append(
        "\n[dim]Kept: each project's own .localforge/ folder (its memory and usage) and AGENTS.md. "
        "Delete a project's .localforge/ yourself to remove its memory.[/dim]"
    )
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
