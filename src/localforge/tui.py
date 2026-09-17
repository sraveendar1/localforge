"""A terminal getting-started wizard: the same setup flow as `localforge
setup`, but as an interactive screen-by-screen UI instead of a sequence of
Y/N prompts. Launched via `localforge wizard`.
"""

from __future__ import annotations

import re
import shutil


def _safe_id(name: str) -> str:
    return "cb-" + re.sub(r"[^a-zA-Z0-9_-]", "_", name)

from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Log, RadioButton, RadioSet, Static

from localforge import config
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import recommendations
from localforge.config import FRONTIER_PROVIDERS
from localforge.hardware import detect_hardware


class WelcomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(
                "\n[bold]Welcome to localforge[/bold]\n\n"
                "This wizard will:\n"
                "  1. Check that Ollama is installed and running\n"
                "  2. Recommend and pull local models that fit your hardware\n"
                "  3. Save your frontier model API key\n\n"
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
            self.app.push_screen(ModelsScreen())
        elif event.button.id == "retry":
            self.app.pop_screen()
            self.app.push_screen(OllamaScreen())


class ModelsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        hw = detect_hardware()
        recs = recommendations(hw)
        self.to_pull = {e.name for e in recs.values() if e is not None and e.runtime == "ollama"}

        table = Table(title="Recommended for this machine")
        table.add_column("Modality")
        table.add_column("Model")
        for modality, entry in recs.items():
            table.add_row(modality, entry.name if entry else "[red]none fit[/red]")

        widgets = [Static(table, id="models-table")]
        for name in sorted(self.to_pull):
            widgets.append(Checkbox(name, value=True, id=_safe_id(name)))
        widgets.append(Button("Pull selected", id="pull", variant="primary"))
        widgets.append(Log(id="pull-log", highlight=True))
        widgets.append(Button("Skip / Next", id="next"))
        yield VerticalScroll(*widgets, id="models-body")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pull":
            selected = [
                name for name in sorted(self.to_pull) if self.query_one(f"#{_safe_id(name)}", Checkbox).value
            ]
            self._pull(selected)
        elif event.button.id == "next":
            self.app.push_screen(ApiKeyScreen())

    @work(thread=True)
    def _pull(self, models: list[str]) -> None:
        log = self.query_one("#pull-log", Log)
        backend = OllamaBackend()
        for name in models:
            self.app.call_from_thread(log.write_line, f"Pulling {name}...")
            try:
                backend.ensure_available(name)
                self.app.call_from_thread(log.write_line, f"  done: {name}")
            except Exception as exc:  # noqa: BLE001 - shown in the log, not fatal to the wizard
                self.app.call_from_thread(log.write_line, f"  failed: {name}: {exc}")
        self.app.call_from_thread(log.write_line, "All done.")


class ApiKeyScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("\nWhich frontier model provider will you orchestrate with?\n"),
            RadioSet(*(RadioButton(name.capitalize(), id=f"radio-{name}") for name in FRONTIER_PROVIDERS)),
            Static("\nPaste your API key:\n"),
            Input(placeholder="sk-...", password=True, id="api-key-input"),
            Button("Save & Finish", id="save", variant="primary"),
            Static("", id="api-key-status"),
            id="apikey-body",
        )
        yield Footer()

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
        if not api_key:
            status.update("[red]Enter an API key.[/red]")
            return
        config.save({env_var: api_key})
        self.app.push_screen(DoneScreen())


class DoneScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static(
                "\n[bold green]Setup complete.[/bold green]\n\n"
                'Try: [bold]localforge run "Build a todo REST API with docs"[/bold]\n',
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
