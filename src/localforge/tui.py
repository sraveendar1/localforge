"""A terminal getting-started wizard: the same setup flow as `localforge
setup`, but as an interactive screen-by-screen UI instead of a sequence of
Y/N prompts. Launched via `localforge wizard`.
"""

from __future__ import annotations

import os
import re
import shutil
import webbrowser

from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Log, RadioButton, RadioSet, Select, Static

from localforge import config, local_transport
from localforge.advisor import recommend_models
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import ModelEntry, recommendations
from localforge.config import FRONTIER_PROVIDERS
from localforge.hardware import detect_hardware


def _safe_id(name: str) -> str:
    return "cb-" + re.sub(r"[^a-zA-Z0-9_-]", "_", name)


class WelcomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(
                "\n[bold]Welcome to localforge[/bold]\n\n"
                "This wizard will:\n"
                "  1. Check that Ollama is installed and running\n"
                "  2. Save your frontier model API key\n"
                "  3. Have the frontier model pick and pull local models "
                "that fit your hardware\n\n"
                "Press [bold]Get Started[/bold] to begin.\n",
                id="welcome-text",
            ),
            Button("Get Started", id="start", variant="primary"),
            id="welcome-body",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.app.push_screen(OllamaScreen())


class OllamaScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(Static("Checking Ollama...", id="ollama-status"), id="ollama-body")
        yield Footer()

    def on_mount(self) -> None:
        self._check()

    @work(thread=True)
    def _check(self) -> None:
        installed = shutil.which("ollama") is not None
        running = OllamaBackend().is_running() if installed else False
        self.app.call_from_thread(self._show_result, installed, running)

    def _show_result(self, installed: bool, running: bool) -> None:
        status = self.query_one("#ollama-status", Static)
        body = self.query_one("#ollama-body", Vertical)
        if installed and running:
            status.update("[green]✓ Ollama is installed and running.[/green]")
            body.mount(Button("Next", id="next", variant="primary"))
        elif installed and not running:
            status.update(
                "[yellow]Ollama is installed but not running.[/yellow]\n"
                "Start it (e.g. `ollama serve` or `brew services start ollama`), then retry."
            )
            body.mount(Button("Retry", id="retry"))
        else:
            status.update(
                "[red]Ollama is not installed.[/red]\n"
                "Install it from https://ollama.com, then retry."
            )
            body.mount(Button("Retry", id="retry"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            # Always ask which frontier model to use explicitly -- never
            # silently pick one just because some API key happens to be
            # sitting in the environment.
            self.app.push_screen(ApiKeyScreen())
        elif event.button.id == "retry":
            self.app.pop_screen()
            self.app.push_screen(OllamaScreen())


class ApiKeyScreen(Screen):
    """Collected before model selection: the frontier model needs an API key
    to be the one picking which local models to download.
    """

    OTHER = "__other__"

    def __init__(self) -> None:
        super().__init__()
        self._opened_console_for: set[str] = set()

    def compose(self) -> ComposeResult:
        first_provider = next(iter(FRONTIER_PROVIDERS))
        initial_choices = config.FRONTIER_MODEL_CHOICES.get(first_provider, [])
        initial_options = [(m, m) for m in initial_choices] + [("Other (type a model id)", self.OTHER)]

        yield Header()
        yield Vertical(
            Static("\nWhich frontier model provider should localforge use?\n"),
            RadioSet(*(RadioButton(name.capitalize(), id=f"radio-{name}") for name in FRONTIER_PROVIDERS)),
            Static("\nModel:\n"),
            Select(initial_options, id="model-select", allow_blank=False),
            Input(placeholder="exact model id, e.g. claude-opus-5", id="custom-model-input"),
            Static("", id="api-key-hint"),
            Input(placeholder="sk-...", password=True, id="api-key-input"),
            Button("Save & Continue", id="save", variant="primary"),
            Static("", id="api-key-status"),
            id="apikey-body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#custom-model-input", Input).display = False
        first_provider = next(iter(FRONTIER_PROVIDERS))
        self.query_one(f"#radio-{first_provider}", RadioButton).value = True

    def _populate_models(self, provider: str) -> None:
        choices = config.FRONTIER_MODEL_CHOICES.get(provider, [])
        if provider == "local":
            # models this machine can actually run, downloaded ones first
            try:
                installed = {m["name"] for m in OllamaBackend().list_installed()}
            except Exception:  # noqa: BLE001 - Ollama may not be up yet on this screen
                installed = set()
            choices = local_transport.orchestrator_choices(detect_hardware(), installed)
        select = self.query_one("#model-select", Select)
        select.set_options([(model_id, model_id) for model_id in choices] + [("Other (type a model id)", self.OTHER)])
        select.value = choices[0] if choices else self.OTHER
        self.query_one("#custom-model-input", Input).display = select.value == self.OTHER

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        provider = event.pressed.id.removeprefix("radio-")
        env_var = FRONTIER_PROVIDERS[provider]
        hint = self.query_one("#api-key-hint", Static)
        key_input = self.query_one("#api-key-input", Input)
        # Clear anything typed for the previously selected provider -- otherwise
        # a key typed for e.g. Anthropic stays in the (masked) field and would
        # be saved under the newly picked provider's env var on Save.
        key_input.value = ""
        if env_var is None:
            hint.update("[green]Open-weight model, self-hosted via Ollama — no API key needed.[/green]")
            key_input.display = False
        else:
            key_input.display = True
            if os.environ.get(env_var):
                hint.update(f"[green]{env_var} already set — leave the field below empty to reuse it.[/green]")
            else:
                console_url = config.FRONTIER_CONSOLE_URLS.get(provider)
                if console_url:
                    hint.update(f"Paste your {env_var} (opening {console_url} in your browser):")
                    if provider not in self._opened_console_for:
                        webbrowser.open(console_url)
                        self._opened_console_for.add(provider)
                else:
                    hint.update(f"Paste your {env_var}:")
        self._populate_models(provider)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select":
            self.query_one("#custom-model-input", Input).display = event.value == self.OTHER

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save":
            return
        radio_set = self.query_one(RadioSet)
        status = self.query_one("#api-key-status", Static)
        if radio_set.pressed_button is None:
            status.update("[red]Pick a provider first.[/red]")
            return
        provider = radio_set.pressed_button.id.removeprefix("radio-")
        env_var = FRONTIER_PROVIDERS[provider]
        api_key = self.query_one("#api-key-input", Input).value.strip()

        model_choice = self.query_one("#model-select", Select).value
        if model_choice == self.OTHER:
            frontier_model = self.query_one("#custom-model-input", Input).value.strip()
            if not frontier_model:
                status.update("[red]Enter a model id.[/red]")
                return
        else:
            frontier_model = model_choice

        # Always write the auth method and provider too: config.save() merges,
        # so a cli_login left from an earlier setup would otherwise keep
        # routing turns through that provider's CLI.
        if env_var is None:
            auth = {config.AUTH_METHOD_ENV_VAR: config.AUTH_LOCAL, config.FRONTIER_PROVIDER_ENV_VAR: "local"}
        else:
            auth = {config.AUTH_METHOD_ENV_VAR: config.AUTH_API_KEY, config.FRONTIER_PROVIDER_ENV_VAR: provider}
        if env_var is None:
            config.save({config.FRONTIER_MODEL_ENV_VAR: frontier_model, **auth})
        elif api_key:
            config.save({env_var: api_key, config.FRONTIER_MODEL_ENV_VAR: frontier_model, **auth})
            os.environ[env_var] = api_key
        elif os.environ.get(env_var):
            config.save({config.FRONTIER_MODEL_ENV_VAR: frontier_model, **auth})
        else:
            status.update(f"[red]No {env_var} found — enter an API key.[/red]")
            return
        self.app.push_screen(ModelsScreen(frontier_model))


class ModelsScreen(Screen):
    """Asks the frontier model to pick the best local model per modality for
    this machine (falling back to the deterministic heuristic if that call
    fails or `frontier_model` is unavailable), then pulls them via Ollama.
    """

    def __init__(self, frontier_model: str | None) -> None:
        super().__init__()
        self.frontier_model = frontier_model
        self.to_pull: set[str] = set()
        self.installed: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        status = "Asking the frontier model to pick local models..." if self.frontier_model else "Matching local models to your hardware..."
        yield VerticalScroll(Static(status, id="models-status"), id="models-body")
        yield Footer()

    def on_mount(self) -> None:
        self._recommend()

    @work(thread=True)
    def _recommend(self) -> None:
        hw = detect_hardware()
        # What's already on disk, so a re-run reuses suitable models rather
        # than downloading new ones (same rule as `localforge setup`).
        try:
            self.installed = {m["name"] for m in OllamaBackend().list_installed()}
        except Exception:  # noqa: BLE001 - best-effort; fall back to catalog-only
            self.installed = set()

        if self.frontier_model:
            # Under CLI login there's no API key, so route the advisor through
            # the provider's CLI rather than letting a litellm call fail.
            cli_provider = None
            if os.environ.get(config.AUTH_METHOD_ENV_VAR) == config.AUTH_CLI_LOGIN:
                cli_provider = os.environ.get(config.FRONTIER_PROVIDER_ENV_VAR)
            recs = recommend_models(hw, self.frontier_model, cli_provider=cli_provider, installed=self.installed)
        else:
            recs = recommendations(hw, installed=self.installed)
        self.app.call_from_thread(self._show_recommendations, recs)

    def _show_recommendations(self, recs: dict[str, ModelEntry | None]) -> None:
        # Only offer to pull what isn't already installed.
        self.to_pull = {
            e.name for e in recs.values() if e is not None and e.runtime == "ollama" and e.name not in self.installed
        }

        table = Table(title="Recommended for this machine")
        table.add_column("Modality")
        table.add_column("Model")
        for modality, entry in recs.items():
            if entry is None:
                table.add_row(modality, "[red]none fit[/red]")
            elif entry.name in self.installed:
                table.add_row(modality, f"{entry.name} [green](already installed)[/green]")
            else:
                table.add_row(modality, f"{entry.name} [yellow](will download ~{entry.disk_gb:g} GB)[/yellow]")

        body = self.query_one("#models-body", VerticalScroll)
        self.query_one("#models-status", Static).update(table)
        for name in sorted(self.to_pull):
            body.mount(Checkbox(name, value=True, id=_safe_id(name)))
        body.mount(Button("Pull selected", id="pull", variant="primary"))
        body.mount(Log(id="pull-log", highlight=True))
        body.mount(Button("Skip / Next", id="next"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pull":
            selected = [
                name for name in sorted(self.to_pull) if self.query_one(f"#{_safe_id(name)}", Checkbox).value
            ]
            self._pull(selected)
        elif event.button.id == "next":
            self.app.push_screen(DoneScreen())

    @work(thread=True)
    def _pull(self, models: list[str]) -> None:
        log = self.query_one("#pull-log", Log)
        backend = OllamaBackend()
        for name in models:
            self.app.call_from_thread(log.write_line, f"Pulling {name}...")
            last_reported = -1

            def _on_progress(event: dict, name=name) -> None:
                nonlocal last_reported
                total, completed = event.get("total"), event.get("completed")
                if total and completed is not None:
                    pct = int(completed * 100 / total)
                    if pct >= last_reported + 10 or pct == 100:  # throttle to avoid flooding the log
                        last_reported = pct
                        self.app.call_from_thread(log.write_line, f"  {name}: {pct}%")
                else:
                    status = event.get("status", "")
                    if status:
                        self.app.call_from_thread(log.write_line, f"  {name}: {status}")

            try:
                backend.ensure_available(name, on_progress=_on_progress)
                self.app.call_from_thread(log.write_line, f"  done: {name}")
            except Exception as exc:  # noqa: BLE001 - shown in the log, not fatal to the wizard
                self.app.call_from_thread(log.write_line, f"  failed: {name}: {exc}")
        self.app.call_from_thread(log.write_line, "All done.")


class DoneScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(
                "\n[bold green]Setup complete.[/bold green]\n\n"
                "Quit, then type [bold]localforge[/bold] in your terminal to start a session.\n"
                "Inside it, just type what you want built, or /help for commands.\n",
            ),
            Button("Quit", id="quit", variant="primary"),
            id="done-body",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()


class LocalforgeWizard(App):
    TITLE = "localforge — getting started"
    CSS = """
    #welcome-body, #ollama-body, #apikey-body, #done-body { padding: 2 4; }
    #models-body { padding: 1 2; }
    Log { height: 10; border: solid $accent; margin-top: 1; }
    """

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())


def main() -> None:
    LocalforgeWizard().run()


if __name__ == "__main__":
    main()
