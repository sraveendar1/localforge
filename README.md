# localforge

A frontier model (Claude, GPT, or any provider) orchestrates open-weight
models running **locally** on your machine. `localforge` profiles your
hardware, picks open-weight models that will actually run well on it, pulls
them via [Ollama](https://ollama.com), and hands them subtasks — coding,
documentation, and (future) image/video generation — while a frontier model
of your choice does the planning and delegation.

No vendor lock-in: the orchestrator is called through
[LiteLLM](https://github.com/BerriAI/litellm), so swapping `claude-opus-5`
for `gpt-5` or a self-hosted model is a one-line config change.

## How it works

1. **`localforge scan`** — detects OS, CPU, RAM, and GPU/VRAM.
2. **`localforge models`** — matches your hardware against a curated catalog
   (`src/localforge/catalog_data.yaml`) to recommend the best local model per
   task type.
3. **`localforge run "<task>"`** — the frontier model plans the task, calls
   tools like `delegate_coding_task` / `delegate_docs_task`, and those calls
   are dispatched to the matched local model running under Ollama. Results
   flow back into the frontier model's context until it produces a final
   answer.

## Install

From a local checkout:

```bash
./install.sh
```

This one command: installs [uv](https://docs.astral.sh/uv/) if you don't
have it, installs `localforge` as a standalone CLI tool onto your `PATH`
(no virtualenv to activate, no `pip` to manage — uv even fetches a matching
Python for you), and then runs the interactive setup wizard, which:

- installs and starts [Ollama](https://ollama.com) if it isn't already,
- pulls the local models that best fit *your* hardware (per `localforge models`),
- prompts for your frontier model API key (Anthropic/OpenAI/Gemini) and saves
  it to `~/.config/localforge/config.env` so you only enter it once.

After that, `localforge` just works in any terminal, no setup steps repeated.

To upgrade after pulling new code: `uv tool install --reinstall .`, or just
re-run `./install.sh`.

You can also run pieces individually:
- `localforge wizard` — the same setup flow as a navigable terminal UI
  (screens, checkboxes, a live pull log) instead of `y/n` prompts.
- `localforge setup` — the plain-prompt version of the same flow (useful
  over a dumb terminal or in scripts).
- `localforge doctor` — checks everything's still in place without changing
  anything.

### Developing locally

```bash
uv sync --extra dev     # creates .venv and installs everything, incl. test deps
uv run pytest -q
uv run localforge scan
```

## Usage

```bash
localforge wizard                                # one-time interactive setup (terminal UI)
localforge setup                                 # one-time interactive setup (plain prompts)
localforge doctor                                # is everything set up correctly?
localforge scan                                  # what hardware do I have?
localforge models                                # what will run well on it?
localforge run "Build a todo REST API with docs" # do the thing
localforge run "..." --model gpt-5               # use a different frontier model
```

## Architecture

- `hardware.py` — hardware detection (RAM, CPU, GPU/VRAM via nvidia-smi or
  Apple Silicon unified memory).
- `catalog.py` / `catalog_data.yaml` — static catalog of open-weight models
  tagged by modality (`coding`, `docs`, `general`, and reserved `image` /
  `video` entries) and hardware requirements; `best_match()` picks the
  highest-quality model that fits.
- `backends/` — one module per serving runtime. `ollama.py` is implemented;
  `comfyui.py` is a stub reserved for image/video generation, since those
  are job-based (submit → poll → fetch file) rather than a single
  request/response like text.
- `tools.py` — turns catalog + backends into tool schemas the frontier model
  can call, and dispatches each call to the right local model.
- `orchestrator.py` — the plan → delegate → collect loop, built directly on
  LiteLLM rather than a multi-agent framework, so the delegation logic stays
  simple, provider-agnostic, and easy to step through.
- `config.py` — persists setup choices (API keys) to
  `~/.config/localforge/config.env`, loaded automatically on every CLI
  invocation without overriding variables already set in the shell.
- `tui.py` — the `localforge wizard` terminal UI ([Textual](https://textual.textualize.io/)):
  the same setup steps as `setup`, as navigable screens instead of prompts.
- `cli.py` — the `localforge` command-line entry point (Typer), including
  the `setup`/`wizard` onboarding flows and `doctor` diagnostic.
- `install.sh` — the one-command bootstrap: installs uv, installs the CLI
  tool, runs `localforge setup`.

Adding a new modality (e.g. wiring up real image generation) means
implementing `backends/comfyui.py`'s two methods and adding a tool entry in
`tools.py` — the orchestrator loop itself does not change.

## Running tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
