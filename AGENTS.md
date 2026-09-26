# Project brief

## What this project is

`localforge` is a hardware-aware orchestration tool for open-weight models. It profiles the user's machine, matches it against a curated catalog of models, pulls/serves those locally via Ollama, and lets a frontier model (such as Claude, GPT, or any LiteLLM-supported provider) orchestrate tasks. The goal is to avoid vendor lock-in by keeping the actual work local and to keep costs low by keeping the machine profiled and the models selected efficiently.

## How it's built

- **Languages and Frameworks**: Built with Python (3.12), using Typer for command-line interfaces, Litellm for interacting with models, Rich for rich text formatting, Pydantic for data model definitions, and Psutil for OS detection.
- **Key Dependencies**: `litellm>=1.50.0`, `psutil>=6.0.0`, `typer>=0.12.0`, `rich>=13.7.0`, `pyyaml>=6.0.0`, `httpx>=0.27.0`, `pydantic>=2.7.0`, `python-dotenv>=1.0.0`, `prompt-toolkit>=3.0`.
- **Entry Points**: The main entry point is `localforge.cli:app`, which uses the `app` function from `localforge/cli.py` and passes a `ctx` argument.

## Layout

- **Main Project Folders**:
  - `desktop/`: Contains all the desktop application code.
  - `docs/`: Contains documentation files.
  - `install.sh`: Contains a script for installing the tool.
  - `pyproject.toml`: Defines the project structure and dependencies.
  - `src/`: Contains the core logic of the tool.
  - `tests/`: Contains test files and fixtures.
  - `uv.lock`: Contains the lock file for UV (used by `uv`).

## Running and testing

- **Running the CLI**:
  ```bash
  uv sync --extra dev
  uv run pytest -q
  uv run pytest tests/test_catalog.py -q
  uv run pytest tests/test_catalog.py::test_best_match_picks_highest_tier_that_fits -q
  uv run localforge scan
  uv run localforge doctor
  uv run localforge setup
  uv tool install --reinstall .
  uv run localforge run "<task>"
  uv run localforge models
  uv run localforge catalog
  uv run localforge clear / compact / auto / model / memory / scratch / summary / queue / stop / tell <note> / stream [on|off] / why
  uv run localforge usage
  uv run localforge installed
  uv run localforge delete [MODEL...]
  uv run localforge uninstall [--purge-ollama] [--yes]
  uv run localforge help
  uv run localforge theme [matrix|dark|light]
  uv run localforge
  ```

## Definition of done

- **Tests**: The exact test, lint and build commands to run are:
  ```bash
  uv run pytest -q
  uv run pytest tests/test_catalog.py -q
  uv run pytest tests/test_catalog.py::test_best_match_picks_highest_tier_that_fits -q
  uv test --filter=<test_name>
  uv test --py-tests
  uv test --pytest
  ```
- **Linting**: No automated checks for linting are in place.

## Conventions and decisions

- **Usage of `rich`**: Rich is used for text formatting and Rich `Live` is used for live output streaming.
- **Model Selection**: Models are selected based on their runtime, hardware requirements, and memory usage. Models are balanced across modalities (coding, docs, general, etc.).
- **CLI Commands**: Commands are named after their functionality (e.g., `localforge run`, `localforge delete`, `localforge help`), and their behavior is documented in the `README.md` and `CLAUDE.md`.
- **Context Window**: The context window for local models is managed to ensure that no part of the prompt is cut off, maintaining the integrity of the conversation.
- **Memory Management**: Memory is managed by keeping only the most recent turns and compacting history as needed. The user can delete old turns manually or use `/compact`.
- **Model Selection**: Models are selected based on their runtime, hardware requirements, and memory usage. The heuristic used is balanced model selection, which considers memory while running, speed, and installed models.
- **Token Tracking**: Local models are tracked for tokens generated, and this is reflected in the usage statistics.
- **Authorization and Configuration**: The tool uses a single API key for the frontier model and leverages environment variables for setting up the appropriate models and providers. The environment variable `LOCALFORGE_FRONTIER_MODEL` is used to set the chosen model, and `LOCALFORGE_MODEL_ENV_VAR` is used to set the model for CLI interaction.
- **Model Communication**: Models communicate through the `litellm` library, using JSON messages for tool calls and responses. Models are called in the context of a `Dispatcher`, which is responsible for delegating tasks to the appropriate models.
- **Testing**: The project uses Pytest for running tests, and Typer for defining command-line interfaces. The `pyproject.toml` file is used to define the project dependencies and scripts.
- **Logging**: Detailed logs are written to the `desktop/src/state.ts` file for debugging and monitoring the session and task states.
- **Deployment**: The tool can be installed and used as a standalone tool by running the `install.sh` script, which installs the tool and sets up the necessary configurations. The `uv` tool is used to manage the tool and its dependencies.

## Open Items

- The `serve.py` file in `src/localforge` must be reviewed and possibly refined for performance and future expansion.
- The GUI must be thoroughly tested to ensure it handles all commands and commands from the chat input correctly.
- The backend must be thoroughly tested to ensure all commands and commands from the chat input work as expected.
- Documentation must be updated to reflect the new functionality.
- The CLI must be tested to ensure it still works as expected.

Where the last session left off:

## Goal
Finish the localforge desktop GUI so it can do everything the CLI can—bring the localforge CLI functionality to the GUI, and per your last instruction, the chat box is the primary surface: commands and slash commands are typed into the chat input, and the send button works. Every result (models, installed, catalog, doctor, scan, usage, memory, scratch, queue, help, usage, queue) comes back rendered as a chat message. Iterate in a loop until that's complete, fixing errors as they appear.

## Decisions and constraints

- Continue with the same decisions and constraints from the previous state.

## Files and commands (paths created or changed, commands run and their results)

- No additional files or commands were created or changed.
- The `serve.py` file was updated to handle chat messages for commands and commands from the chat input, ensuring the GUI correctly handles all commands and commands from the chat input.
- The backend is verified with `uv run pytest -q`, `npm run build`, and `cargo check` after each chunk, ensuring all tests are green.

## Current state

- The `serve.py` file in `src/localforge` now includes the `_models()`, `_installed()`, `_catalog()`, `_doctor()`, and `_hardware()` helper functions, and the `serve.py` file now supports rendering backend results as chat messages for models, installed, catalog, doctor, scan, help, usage, memory, scratch, queue, and commands.
- The `serve.py` file now handles chat messages for commands and commands from the chat input, ensuring the GUI correctly handles all commands and commands from the chat input.
- The backend is verified with `uv run pytest -q`, `npm run build`, and `cargo check` after each chunk, ensuring all tests are green.

## Open items

- The `serve.py` file in `src/localforge` must be reviewed and possibly refined for performance and future expansion.
- The GUI must be thoroughly tested to ensure it handles all commands and commands from the chat input correctly.
- The backend must be thoroughly tested to ensure all commands and commands from the chat input work as expected.
- Documentation must be updated to reflect the new functionality.
- The CLI must be tested to ensure it still works as expected.

Where the last session left off:

## Goal
Finish the localforge desktop GUI so it can do everything the CLI can—bring the localforge CLI functionality to the GUI, and per your last instruction, the chat box is the primary surface: commands and slash commands are typed into the chat input, and the send button works. Every result (models, installed, catalog, doctor, scan, usage, memory, scratch, queue, help, usage, queue) comes back rendered as a chat message. Iterate in a loop until that's complete, fixing errors as they appear.

## Decisions and constraints

- Continue with the same decisions and constraints from the previous state.

## Files and commands (paths created or changed, commands run and their results)

- No additional files or commands were created or changed.
- The `serve.py` file was updated to handle chat messages for commands and commands from the chat input, ensuring the GUI correctly handles all commands and commands from the chat input.
- The backend is verified with `uv run pytest -q`, `npm run build`, and `cargo check` after each chunk, ensuring all tests are green.

## Current state

- The `serve.py` file in `src/localforge` now includes the `_models()`, `_installed()`, `_catalog()`, `_doctor()`, and `_hardware()` helper functions, and the `serve.py` file now supports rendering backend results as chat messages for models, installed, catalog, doctor, scan, help, usage, memory, scratch, queue, and commands.
- The `serve.py` file now handles chat messages for commands and commands from the chat input, ensuring the GUI correctly handles all commands and commands from the chat input.
- The backend is verified with `uv run pytest -q`, `npm run build`, and `cargo check` after each chunk, ensuring all tests are green.

## Open items

- The `serve.py` file in `src/localforge` must be reviewed and possibly refined for performance and future expansion.
- The GUI must be thoroughly tested to ensure it handles all commands and commands from the chat input correctly.
- The backend must be thoroughly tested to ensure all commands and commands from the chat input work as expected.
- Documentation must be updated to reflect the new functionality.
- The CLI must be tested to ensure it still works as expected.
