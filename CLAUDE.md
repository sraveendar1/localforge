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
localforge setup       # plain-prompt version: installs Ollama, pulls recommended models, saves API key
localforge doctor      # checks Ollama installed/running + frontier API key set + hardware fit
localforge models      # best-fit local model per modality for this machine
localforge catalog     # full model catalog, regardless of fit
localforge run "<task>" [--model gpt-5]   # run the orchestration loop
```

Requires [Ollama](https://ollama.com) running locally for actual model execution, and an API key for whichever frontier model is used (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.). `localforge setup` handles both interactively and saves the key to `~/.config/localforge/config.env` (loaded automatically thereafter via `config.py`); `localforge doctor` checks all of this without changing anything.

## Architecture

The pipeline is: **hardware detection → catalog matching → tool dispatch → orchestration loop**, each in its own module so a piece can change without touching the others.

- `hardware.py` — detects OS/CPU/RAM/GPU. GPU detection is NVIDIA via `nvidia-smi` first, falling back to Apple Silicon unified-memory heuristics (`arm64` + `Darwin`); no GPU means CPU-only.
- `catalog.py` + `catalog_data.yaml` — a static, hand-curated list of models tagged by `modality` (`coding`, `docs`, `general`, plus reserved `image`/`video`), `runtime` (which backend serves it), and hardware minimums. `best_match()` is a pure function: highest `quality_tier` entry whose minimums are met by the given `HardwareProfile`. This was chosen deliberately over a benchmark-driven approach (download-and-measure candidate models) for speed and predictability — see git history / project discussion for the tradeoff reasoning if extending this.
- `backends/` — one module per serving runtime, behind the `Backend` protocol (`ensure_available`, `generate`) in `base.py`. `ollama.py` is implemented (REST calls to `localhost:11434`). `comfyui.py` is an intentional stub: image/video generation is job-based (submit → poll → fetch a file) rather than single-shot request/response like text, and returns `BackendResult(type="file", ...)` instead of `type="text"` — the orchestrator and tool layer already branch on this.
- `tools.py` — the `Dispatcher` class is the glue: given a tool name the frontier model called, it maps tool → modality → `catalog.best_match()` → backend, resolves the model once per modality per run (cached in `_resolved_models`), and runs it. `TASK_MODALITIES` is the single source of truth mapping a modality to its tool name/description; adding a new delegable modality means adding an entry here plus a backend, not touching the orchestrator loop.
- `orchestrator.py` — the actual plan→delegate→collect loop, built directly on `litellm.completion()` rather than a multi-agent framework (LangGraph/AutoGen/CrewAI were considered; rejected for v1 because the project's real complexity is in the catalog-matching/dispatch logic, not conversation/turn-taking, and those frameworks' native "agents talking to agents" abstraction doesn't fit job-based image/video subagents any better than a plain tool function does). The loop is intentionally simple and inspectable: send messages + tool schemas to the frontier model, execute whatever tool calls come back via `Dispatcher`, feed results back as `role: tool` messages, repeat until the frontier model stops calling tools or `MAX_ROUNDS` is hit.
- `cli.py` — Typer entry point (`localforge` console script defined in `pyproject.toml`). The `run` command wraps the orchestrator call in a try/except to print a clean error instead of a raw traceback (LiteLLM raises rich exception types like `AuthenticationError` for missing API keys — surface `exc` directly, that's already a good user-facing message). `doctor` is the first-run diagnostic (Ollama installed/running, frontier API key present, hardware fits every modality) — extend `FRONTIER_PROVIDERS`/`FRONTIER_API_KEY_ENV_VARS` in `config.py` when adding support for a new provider (both `cli.py` and `tui.py` import from there, don't redefine locally).
- `tui.py` — the `localforge wizard` Textual UI: `WelcomeScreen` → `OllamaScreen` (backgrounds the install/running check via `@work(thread=True)`, updates widgets through `self.app.call_from_thread` since Textual widgets aren't thread-safe to touch directly from a worker) → `ModelsScreen` (checkboxes + a `Log` widget streaming pull progress) → `ApiKeyScreen` (`RadioSet` + masked `Input`, writes via `config.save`) → `DoneScreen`. Widget `id`s can't contain `:` or `.` (Textual's `check_identifiers` rejects them), which model names like `qwen2.5-coder:14b` do — see `_safe_id()` for the sanitizing helper; don't build widget ids directly from model/provider names elsewhere without it. Tested headlessly via Textual's `App.run_test()` pilot (see conversation history for the smoke-test pattern) rather than a real terminal; when testing config-saving flows this way, always set `LOCALFORGE_CONFIG_DIR` to a temp dir first so it doesn't write into the real user's `~/.config/localforge/`.

### Extending to a new modality (e.g. real image generation)

1. Implement `backends/comfyui.py`'s `ensure_available`/`generate` (or add a new backend module) and register it in `backends/__init__.py`'s `BACKENDS` dict.
2. Add catalog entries in `catalog_data.yaml` with the right `modality`/`runtime`/hardware minimums.
3. Add a `TASK_MODALITIES` entry in `tools.py`.

Nothing in `orchestrator.py` needs to change — this is the intended seam.

## Conventions

- Pydantic models (`HardwareProfile`, `GPU`, `ModelEntry`) for anything crossing a module boundary; plain dicts/`TypedDict` (`BackendResult`) for backend I/O.
- Tests mock hardware via `HardwareProfile(...)` construction and an inline test catalog (see `tests/test_catalog.py`) rather than patching `detect_hardware()` — keep `catalog.best_match()` pure and test it that way.
- No multi-agent framework dependency by design (see `orchestrator.py` rationale above) — don't reach for LangGraph/AutoGen/CrewAI without revisiting that decision first.
