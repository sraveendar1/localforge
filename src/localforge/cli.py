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
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from localforge import cli_transport, config, memory, repl, theme
from localforge.advisor import recommend_models
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import NoFittingModelError, best_match, load_catalog, recommendations
from localforge.config import FRONTIER_API_KEY_ENV_VARS, FRONTIER_PROVIDERS
from localforge.hardware import detect_hardware
from localforge.orchestrator import Conversation, OrchestrationError, RunStats
from localforge.orchestrator import run as run_orchestrator
from localforge.tools import ActivityHooks, Dispatcher
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
            repl.run_repl(app, console)
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
            "[bold]Get started in one step:[/bold]\n\n"
            "  [accent]localforge setup[/accent]   (or [accent]localforge wizard[/accent] for a terminal UI)\n\n"
            "That installs Ollama, has a frontier model pick local models for your\n"
            "hardware, and saves your API key. Then type [accent]localforge[/accent] to start a session."
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
    for modality, entry in recs.items():
        if entry is None or entry.runtime != "ollama":
            continue
        if entry.name in installed:
            reuse.append(f"{entry.name} ({modality})")
            try:
                ideal = best_match(modality, hw, catalog)  # ignoring what's installed
            except NoFittingModelError:
                continue
            if ideal.name != entry.name and ideal.quality_tier > entry.quality_tier:
                notes.append(
                    f"{modality}: {ideal.name} (higher tier, ~{ideal.disk_gb:g} GB) also fits — "
                    f"`ollama pull {ideal.name}` if you want the upgrade"
                )
        else:
            download.append(f"{entry.name} ({modality}, ~{entry.disk_gb:g} GB)")

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
        console.print(f"\n[bold success]Ready to go.[/bold success] {START_SESSION_HINT}", highlight=False)
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
    frontier_model = frontier_model or os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or "claude-opus-5"

    # An explicit --model always means "use the API path for this model";
    # only a saved cli_login choice routes through the provider's CLI.
    cli_provider = None
    if os.environ.get(config.AUTH_METHOD_ENV_VAR) == config.AUTH_CLI_LOGIN:
        cli_provider = os.environ.get(config.FRONTIER_PROVIDER_ENV_VAR)
        if cli_provider and not cli_transport.available(cli_provider):
            console.print(f"[error]Error:[/error] {cli_transport.requirements_message(cli_provider)}")
            raise typer.Exit(code=1)
        if cli_provider and not _session.announced:
            spec = config.FRONTIER_CLI_AUTH[cli_provider]
            console.print(f"[dim]Orchestrating via your `{spec['command']}` login (subscription, not API billing)[/dim]")
            _session.announced = True

    if yes:
        _session.auto_approve = True
    activity = _LiveActivity(frontier_model)
    workspace = Workspace(Path.cwd(), approver=activity.approve)
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
        # compute along the way -- record it so /usage still accounts for it.
        console.print(f"[bold error]Error:[/bold error] {exc}")
        _session_usage.append((frontier_model, exc.stats))
        if show_usage:
            _print_usage_panel(exc.stats, frontier_model)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary: show a clean message, not a traceback
        console.print(f"[bold error]Error:[/bold error] {exc}")
        raise typer.Exit(code=1) from None

    console.print()
    console.print(Markdown(result.answer or "(no answer)"))
    _session_usage.append((frontier_model, result.stats))
    if show_usage:
        _print_usage_panel(result.stats, frontier_model)


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
        )

    def _stop_spinner(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _on_frontier(self, round_number: int) -> None:
        self._stop_spinner()
        doing = "planning" if round_number == 1 else "reviewing results"
        self._status = console.status(f"[bold success]{self.frontier_model} is {doing}...")
        self._status.start()

    def _on_delegate(self, modality: str, entry) -> None:
        self._stop_spinner()
        console.print(f"  → delegating [bold]{modality}[/bold] to [accent]{entry.name}[/accent] (local, via {entry.runtime})")
        self._at_line_start = True
        self._first_token_at = None
        # Nothing streams while the model loads into memory, which can take
        # tens of seconds for a big model; keep that visible too.
        self._status = console.status(f"[dim]{entry.name} is loading / thinking...[/dim]")
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
        "web_search": "Web search",
        "fetch_url": "Fetch",
        "compact": "Memory",
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

    def approve(self, kind: str, title: str, detail: str) -> bool:
        """Claude-Code-style permission prompt: show the diff or command, then
        yes / no / always (for this kind, this session). Without a terminal
        to ask on, the change is refused and the orchestrator is told so.
        """
        self._stop_spinner()
        if _session.auto_approve or kind in _session.always_allow:
            self._print_change(kind, title, detail)
            return True
        self._print_change(kind, title, detail)
        if not sys.stdin.isatty():
            console.print("[warning]  No terminal to ask on — declined. Use --yes to approve changes non-interactively.[/warning]")
            return False
        _drain_buffered_input()
        what = "file changes" if kind == "write" else "commands"
        while True:
            # (y)es not [y]es: Rich reads [y] as a style tag and prints nothing
            answer = console.input(f"  Allow? [bold](y)[/bold]es / [bold](n)[/bold]o / [bold](a)[/bold]lways allow {what} this session: ").strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                _session.always_allow.add(kind)
                return True

    def _print_change(self, kind: str, title: str, detail: str) -> None:
        if kind == "command":
            body = f"[bold]$ {escape(detail)}[/bold]"
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

    def close(self) -> None:
        self._stop_spinner()


class _SessionState:
    """What an interactive session remembers between messages. The REPL runs
    every command in-process, so module-level state is the session."""

    def __init__(self) -> None:
        self.conversation: Conversation | None = None
        self.root: Path | None = None
        self.auto_approve = False
        self.always_allow: set[str] = set()
        self.announced = False

    def conversation_for(self, root: Path) -> Conversation:
        if self.conversation is None or self.root != root:
            self.root = root
            self.conversation = Conversation(memory=memory.load(root))
            if self.conversation.memory:
                console.print("[dim]Resuming with this folder's session memory (/clear to start fresh).[/dim]")
        return self.conversation


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
    dispatcher = Dispatcher(detect_hardware(), installed=_installed_model_names(OllamaBackend()) or None, hooks=activity.hooks())
    try:
        done = memory.compact(conversation, dispatcher, activity.hooks(), keep_recent_turns=0, root=_session.root or Path.cwd())
    finally:
        activity.close()
    if done and conversation.memory:
        console.print(Panel(Markdown(conversation.memory), title="Session memory", border_style="panel.border"))


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


# (frontier model, stats) for every task run in this process. Usage is only
# shown on request, so inside a session this is what /usage reads; a one-off
# `localforge run` process has nothing to show later (use `run --usage`).
_session_usage: list[tuple[str, RunStats]] = []


@app.command()
def usage() -> None:
    """Show token usage for the last task and this session so far."""
    if not _session_usage:
        console.print(
            "No tasks run in this session yet. For a one-off run, use "
            '[accent]localforge run --usage "..."[/accent].',
            highlight=False,
        )
        return

    last_model, last_stats = _session_usage[-1]
    _print_usage_panel(last_stats, last_model, title="Usage — last task")
    if len(_session_usage) > 1:
        _print_session_usage_panel(_session_usage)


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
        _usage_bar(local, prompt + completion),
        "",
        f"[success]■[/success] Local models: {local} tokens",
        f"[warning]■[/warning] Frontier ({', '.join(models)}): {prompt} in + {completion} out = "
        f"{prompt + completion} tokens" + cost,
    ]
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


def _print_usage_panel(stats, frontier_model: str, title: str = "Usage") -> None:
    usage_lines = [
        _usage_bar(stats.local_tokens_generated, stats.frontier_total_tokens),
        "",
        f"[success]■[/success] Local models: {stats.local_tokens_generated} tokens — "
        "never sent to or billed by the frontier API",
        f"[warning]■[/warning] Frontier ({frontier_model}): {stats.frontier_prompt_tokens} in + "
        f"{stats.frontier_completion_tokens} out = {stats.frontier_total_tokens} tokens"
        + _frontier_cost_note(stats),
    ]
    console.print(Panel("\n".join(usage_lines), title=title, border_style="panel.border"))


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
