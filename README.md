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

## Quickstart

**One-liner (no clone needed):**

```bash
curl -fsSL https://raw.githubusercontent.com/sanjayraveendar/localforge/main/install.sh | bash
```

**Or, from a clone:**

```bash
git clone https://github.com/sanjayraveendar/localforge.git
cd localforge
./install.sh
```

Either way, this single command installs everything — Ollama, the local
models that fit your machine, and the `localforge` CLI itself — with no
prompts to click through. The only thing it will ask you for is a frontier
model API key (Anthropic, OpenAI, or Gemini — paste one when asked). Once it
finishes:

```bash
localforge run "Build a todo REST API with docs"
```

## How it works

1. **`localforge scan`** — detects OS, CPU, RAM, GPU/VRAM, and free disk space.
2. **`localforge models`** — matches your hardware against a curated catalog
   (`src/localforge/catalog_data.yaml`) to recommend the best local model per
   task type, using the deterministic highest-quality-tier heuristic.
3. **`localforge run "<task>"`** — the frontier model plans the task, calls
   tools like `delegate_coding_task` / `delegate_docs_task`, and those calls
   are dispatched to the matched local model running under Ollama. Results
   flow back into the frontier model's context until it produces a final
   answer.

### Choosing a frontier model

`setup`/`wizard` always ask explicitly which frontier provider
(Anthropic/OpenAI/Gemini) to use — localforge never silently guesses this
from whatever API key happens to already be in your environment (if you
have multiple keys set for unrelated tools, that ambiguity gets surfaced,
not resolved for you). If a key for the provider you pick is already
present, you're offered the option to reuse it instead of re-entering it.
The choice is saved to `LOCALFORGE_FRONTIER_MODEL` in
`~/.config/localforge/config.env`, which `localforge run` uses by default
(overridable per-run with `--model`).

### Hardware-aware model selection

During `setup`/`wizard`, model selection isn't purely rule-based: the
catalog is first filtered down to only the models that actually fit this
machine's RAM, VRAM, *and* free disk space (`catalog.candidates()`), and
then the frontier model itself is asked to pick the best one per modality
from that filtered list (`advisor.recommend_models()`), weighing quality
against how much headroom each choice leaves. The frontier model can only
choose from models that already passed the hardware/disk check — it can't
invent one we have no backend for — and if the call fails for any reason
(no network, bad key, malformed response), it falls back to the same
deterministic highest-quality-tier pick `localforge models` uses on its own.

## What `./install.sh` actually does

1. If it's not already sitting inside a checkout of this repo (e.g. you ran
   the curl one-liner), clones it into `~/.local/share/localforge/src`.
2. On macOS, installs [Homebrew](https://brew.sh) if you don't have it —
   needed so Ollama can be auto-installed in the next step. (This can
   prompt for your password once, since Homebrew needs elevated
   permissions to set up its directories on a first-time install — that's
   normal and only happens the first time Homebrew itself is installed.)
3. Installs [uv](https://docs.astral.sh/uv/) if you don't have it.
4. Installs `localforge` as a standalone CLI tool onto your `PATH` (no
   virtualenv to activate, no `pip` to manage — uv even fetches a matching
   Python for you).
5. Asks which frontier model provider to use and for its API key (saving
   both to `~/.config/localforge/config.env` so you only enter it once),
   then installs and starts [Ollama](https://ollama.com) if it isn't
   already, and has the frontier model itself pick which local models to
   pull — see "Hardware-aware model selection" below.

After that, `localforge` just works in any terminal — no repeated setup, no
manual model downloads, no re-exporting API keys. Just running `localforge`
with no arguments always shows a short "what to do next" panel — one set of
instructions if you haven't set an API key yet, another (pointing straight
at `localforge run`) once you have.

To upgrade after a new release: just re-run the one-liner or `./install.sh`
— it pulls the latest source and reinstalls.

You can also run pieces individually:
- `localforge wizard` — the same setup flow as a navigable terminal UI
  (screens, checkboxes, a live pull log) instead of console output.
- `localforge setup` — the automated, non-interactive-except-for-the-API-key
  flow that `install.sh` calls; useful to re-run on its own.
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
  Apple Silicon unified memory, and free disk space).
- `catalog.py` / `catalog_data.yaml` — static catalog of open-weight models
  tagged by modality (`coding`, `docs`, `general`, and reserved `image` /
  `video` entries), hardware requirements, and approximate download size;
  `candidates()` filters to models that fit RAM/VRAM/disk, `best_match()`
  picks the highest-quality one deterministically.
- `advisor.py` — lets the frontier model pick among `candidates()` for each
  modality (via a tool call constrained with a JSON-schema `enum`, so it
  can't hallucinate a model outside the catalog), falling back to
  `best_match()` on any failure.
- `backends/` — one module per serving runtime. `ollama.py` is implemented,
  including a real progress callback fed by Ollama's streaming pull
  response (used to drive an actual download progress bar in `setup` and a
  live percentage in `wizard`, not just a spinner). `comfyui.py` is a stub
  reserved for image/video generation, since those are job-based (submit →
  poll → fetch file) rather than a single request/response like text.
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
