# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`localforge` is a hardware-aware orchestration tool: it profiles the user's machine, matches that hardware against a curated catalog of open-weight models, pulls/serves the best-fitting ones locally via Ollama, and lets a frontier model of the user's choice (Claude, GPT, or any LiteLLM-supported provider) act as the orchestrator — planning a task and delegating subtasks (coding, docs, and reserved future modalities like image/video generation) to those local models. The design goal is **no vendor lock-in on the frontier model side**, and cost/hardware efficiency by keeping the actual grunt work local.

## Commands

Packaged and dependency-managed with [uv](https://docs.astral.sh/uv/) (`uv.lock` is committed; `.python-version` pins 3.12 — uv fetches that Python automatically if it's not present, so a pre-installed Python is not required).

```bash
# Setup for development
uv sync --extra dev

# Run tests
uv run pytest -q
uv run pytest tests/test_catalog.py -q          # single file
uv run pytest tests/test_catalog.py::test_best_match_picks_highest_tier_that_fits -q  # single test

# CLI during development
uv run localforge scan
uv run localforge doctor    # checks Ollama + frontier API key + hardware fit

# Install as a standalone end-user tool (no venv activation needed afterward)
./install.sh                 # bootstraps uv, installs the tool, runs the setup wizard
uv tool install --reinstall . # re-run manually after changing dependencies/entry points
localforge scan
localforge wizard      # same setup flow as a Textual terminal UI (screens, checkboxes, live log)
localforge setup       # plain-prompt version: Ollama + API key, then the frontier model picks models to pull
localforge doctor      # checks Ollama installed/running + frontier API key set + hardware fit
localforge models      # best-fit local model per modality for this machine
localforge catalog     # full model catalog, regardless of fit
localforge run "<task>" [--model gpt-5]   # run the orchestration loop
```

Requires [Ollama](https://ollama.com) running locally for actual model execution, and an API key for whichever frontier model is used (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.). `localforge setup` handles both interactively and saves the key to `~/.config/localforge/config.env` (loaded automatically thereafter via `config.py`); `localforge doctor` checks all of this without changing anything. Running bare `localforge` (no subcommand) prints a short getting-started panel via the `invoke_without_command=True` callback in `cli.py` — the message branches on whether a `FRONTIER_API_KEY_ENV_VARS` entry is already set, so don't remove that check when touching the callback.

## Architecture

The pipeline is: **hardware detection → catalog matching → tool dispatch → orchestration loop**, each in its own module so a piece can change without touching the others.

- `hardware.py` — detects OS/CPU/RAM/GPU/free disk space. GPU detection is NVIDIA via `nvidia-smi` first, falling back to Apple Silicon unified-memory heuristics (`arm64` + `Darwin`); no GPU means CPU-only. Free disk space is measured at `Path.home()` (where Ollama pulls actually land), via `shutil.disk_usage`.
- `catalog.py` + `catalog_data.yaml` — a static, hand-curated list of models tagged by `modality` (`coding`, `docs`, `general`, plus reserved `image`/`video`), `runtime` (which backend serves it), hardware minimums, and `disk_gb` (approximate download size). `candidates()` is a pure function returning every entry whose RAM/VRAM/disk minimums are met by the given `HardwareProfile`; `best_match()` picks the highest `quality_tier` among them. This deterministic heuristic was chosen deliberately over a benchmark-driven approach (download-and-measure candidate models) for speed and predictability, and it's also the fallback `advisor.py` uses when the frontier-model-driven pick fails.
- `advisor.py` — `recommend_models(hardware, frontier_model, catalog=None)` lets the frontier model choose among `catalog.candidates()` per modality via a tool call whose JSON-schema parameters use an `enum` of only the fitting model names — this makes it structurally impossible for the frontier model to hallucinate a model outside the catalog. Any failure (network error, bad key, malformed/missing tool-call args) falls back to `catalog.best_match()` per modality; this fallback is unconditional and silent (no error surfaced to the user) by design, tested in `tests/test_advisor.py` since there's no real API key in tests. `setup`/`wizard` call this *after* the API key step, not before, since the frontier model needs a key to be called at all.
- `backends/` — one module per serving runtime, behind the `Backend` protocol (`ensure_available`, `generate`) in `base.py`. `ollama.py` is implemented (REST calls to `localhost:11434`). `comfyui.py` is an intentional stub: image/video generation is job-based (submit → poll → fetch a file) rather than single-shot request/response like text, and returns `BackendResult(type="file", ...)` instead of `type="text"` — the orchestrator and tool layer already branch on this.
- `tools.py` — the `Dispatcher` class is the glue: given a tool name the frontier model called, it maps tool → modality → `catalog.best_match()` → backend, resolves the model once per modality per run (cached in `_resolved_models`), and runs it. `TASK_MODALITIES` is the single source of truth mapping a modality to its tool name/description; adding a new delegable modality means adding an entry here plus a backend, not touching the orchestrator loop.
- `orchestrator.py` — the actual plan→delegate→collect loop, built directly on `litellm.completion()` rather than a multi-agent framework (LangGraph/AutoGen/CrewAI were considered; rejected for v1 because the project's real complexity is in the catalog-matching/dispatch logic, not conversation/turn-taking, and those frameworks' native "agents talking to agents" abstraction doesn't fit job-based image/video subagents any better than a plain tool function does). The loop is intentionally simple and inspectable: send messages + tool schemas to the frontier model, execute whatever tool calls come back via `Dispatcher`, feed results back as `role: tool` messages, repeat until the frontier model stops calling tools or `MAX_ROUNDS` is hit.
- `cli.py` — Typer entry point (`localforge` console script defined in `pyproject.toml`). The `run` command wraps the orchestrator call in a try/except to print a clean error instead of a raw traceback (LiteLLM raises rich exception types like `AuthenticationError` for missing API keys — surface `exc` directly, that's already a good user-facing message). `run`'s `--model` option defaults to `None` and is resolved inside the function body (`frontier_model or os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or "claude-opus-5"`) rather than as the Typer default, since a static default is fixed at decorator/import time and can't see what `config.load()` puts into the environment at call time. `doctor` is the first-run diagnostic — it distinguishes "no key at all", "key(s) found but no `LOCALFORGE_FRONTIER_MODEL` chosen yet" (ambiguous — surfaced as a warning, not silently resolved), and "configured" — extend `FRONTIER_PROVIDERS`/`FRONTIER_API_KEY_ENV_VARS`/`FRONTIER_DEFAULT_MODELS`/`FRONTIER_MODEL_ENV_VAR` in `config.py` when adding a new provider (both `cli.py` and `tui.py` import from there, don't redefine locally). `setup` **always** asks which provider to use explicitly via `typer.prompt`, even if a key already exists in the environment — this is deliberate: don't reintroduce silently picking a provider from whichever key happens to be set, since a user could have multiple unrelated keys exported. If a key for the chosen provider already exists, `setup` reuses it instead of re-prompting, but the provider choice itself is never skipped. Step order: Ollama check → provider/API key → hardware scan → `advisor.recommend_models()` → pull.
- `tui.py` — the `localforge wizard` Textual UI, same step order/principle as `setup`: `WelcomeScreen` → `OllamaScreen` (backgrounds the install/running check via `@work(thread=True)`, updates widgets through `self.app.call_from_thread` since Textual widgets aren't thread-safe to touch directly from a worker) → `ApiKeyScreen` (`RadioSet` + masked `Input`; **always shown**, never skipped based on environment state — `on_radio_set_changed` shows a "leave empty to reuse" hint when the selected provider's key is already set, and `on_button_pressed` accepts an empty input in that case instead of requiring re-entry) → `ModelsScreen(frontier_model)` (backgrounds `advisor.recommend_models()`, then shows checkboxes + a `Log` widget streaming pull progress) → `DoneScreen`. Widget `id`s can't contain `:` or `.` (Textual's `check_identifiers` rejects them), which model names like `qwen2.5-coder:14b` do — see `_safe_id()` for the sanitizing helper; don't build widget ids directly from model/provider names elsewhere without it. Tested headlessly via Textual's `App.run_test(size=(120, 60))` pilot (a small default virtual terminal makes off-screen widgets fail `pilot.click()` with `OutOfBounds` — size up if that happens) rather than a real terminal; when testing config-saving flows this way, always set `LOCALFORGE_CONFIG_DIR` to a temp dir first so it doesn't write into the real user's `~/.config/localforge/`.

### Extending to a new modality (e.g. real image generation)

1. Implement `backends/comfyui.py`'s `ensure_available`/`generate` (or add a new backend module) and register it in `backends/__init__.py`'s `BACKENDS` dict.
2. Add catalog entries in `catalog_data.yaml` with the right `modality`/`runtime`/hardware minimums.
3. Add a `TASK_MODALITIES` entry in `tools.py`.

Nothing in `orchestrator.py` needs to change — this is the intended seam.

## Conventions

- Pydantic models (`HardwareProfile`, `GPU`, `ModelEntry`) for anything crossing a module boundary; plain dicts/`TypedDict` (`BackendResult`) for backend I/O.
- Tests mock hardware via `HardwareProfile(...)` construction and an inline test catalog (see `tests/test_catalog.py`) rather than patching `detect_hardware()` — keep `catalog.best_match()` pure and test it that way.
- No multi-agent framework dependency by design (see `orchestrator.py` rationale above) — don't reach for LangGraph/AutoGen/CrewAI without revisiting that decision first.
