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

## Interactive session

Run `localforge` with no arguments in a real terminal and you get a
Claude-Code-style session instead of a one-shot command:

```
 _                 _  __
| |               | |/ _|
| | ___   ___ __ _| | |_ ___  _ __ __ _  ___
| |/ _ \ / __/ _` | |  _/ _ \| '__/ _` |/ _ \
| | (_) | (_| (_| | | || (_) | | | (_| |  __/
|_|\___/ \___\__,_|_|_| \___/|_|  \__, |\___|
                                   __/ |
                                  |___/

Type a task to build it, or /help for commands. /exit to leave.

localforge>
```

Type a task directly and it runs (shorthand for `/run <task>`), or use a
slash command for anything else — `/setup`, `/doctor`, `/scan`, `/theme
dark`, `/delete <names>`, `/help` for the full list, `/exit` to leave. Every
slash command reuses the exact same code as its `localforge <command>`
equivalent — there's no second implementation to drift out of sync.

Piped/scripted invocations (`localforge | cat`, CI, no real terminal
attached) skip the session and keep the old print-and-exit behavior, so
nothing that already scripts against a bare `localforge` call starts
waiting on stdin.

## Quickstart

**Before you run it — what `install.sh` puts on your machine:**
- [Homebrew](https://brew.sh) on macOS, if you don't already have it
- [Ollama](https://ollama.com), if you don't already have it
- One or more real open-weight models matched to your hardware (an actual
  multi-GB download — sizes are shown before each pull)
- The `localforge` CLI tool itself

The script prints this same list and waits for you to press Enter before
touching anything (falls through automatically if there's no interactive
terminal to wait on).

**One-liner (no clone needed):**

```bash
curl -fsSL https://raw.githubusercontent.com/sraveendar1/localforge/main/install.sh | bash
```

**Or, from a clone:**

```bash
git clone https://github.com/sraveendar1/localforge.git
cd localforge
./install.sh
```

Either way, after that one confirmation, everything else runs with no
further prompts to click through except one: a frontier model API key
(Anthropic, OpenAI, or Gemini — paste one when asked). Once it
finishes:

```bash
localforge run "Build a todo REST API with docs"
```

**About the one-liner:** `curl ... | bash` fetches and immediately executes
whatever is currently on this repo's `main` branch — the standard pattern
used by installers like `rustup`/`nvm`, but it does mean you're trusting
the script's content at the moment you run it. If you'd rather see exactly
what will run first, download and read it before executing:

```bash
curl -fsSL https://raw.githubusercontent.com/sraveendar1/localforge/main/install.sh -o install.sh
less install.sh   # inspect it
bash install.sh
```

The flags matter: `-f` fails silently instead of piping an HTML error page
into `bash` if something's wrong, `-sS` keeps it quiet but still shows real
errors, `-L` follows redirects. This URL form
(`raw.githubusercontent.com/<user>/<repo>/<branch>/<path>`) only resolves
once the file exists on that branch in a **public** repo — a private repo's
raw URL returns 404 to an unauthenticated request, so the one-liner won't
work there without extra auth setup.

## How it works

1. **`localforge scan`** — detects OS, CPU, RAM, GPU/VRAM, and free disk space.
2. **`localforge models`** — matches your hardware against a curated catalog
   (`src/localforge/catalog_data.yaml`) to recommend the best local model per
   task type — preferring a model you already have installed over a
   fresh download, and marking which ones are installed.
3. **`localforge run "<task>"`** — the frontier model plans the task, calls
   tools like `delegate_coding_task` / `delegate_docs_task`, and those calls
   are dispatched to the matched local model running under Ollama. Results
   flow back into the frontier model's context until it produces a final
   answer. Every delegation is printed as it happens, e.g.:
   ```
   → delegating coding to qwen2.5-coder:14b (local, via ollama)
   → delegating docs to mistral-nemo:12b (local, via ollama)
   ```
   so you always know which local model is doing the actual work for a
   given task, not just that "something local" is running.

### Catching bad local output

Local models occasionally return something unusable: empty output, an
outright refusal ("I'm sorry, but..."), or something absurdly short for
what was asked. This is caught at two levels:

1. **A cheap heuristic tripwire in the dispatcher.** Every delegated result
   is checked for the obvious failure modes above before it's ever shown to
   the frontier model. If flagged, `localforge` automatically retries once
   — with a different model if this hardware fits more than one for that
   modality, or the same model with reinforced instructions ("your previous
   attempt was rejected: ...") otherwise. You'll see both delegation
   attempts printed live. If the retry is still bad, the result is passed
   through wrapped in an explicit `[WARNING: ...]` tag rather than silently
   accepted.
2. **The frontier model is explicitly instructed not to trust delegated
   results at face value** — to check them against what it asked for,
   treat `[WARNING: ...]`-tagged results with extra scrutiny, and delegate
   a subtask again with clearer instructions rather than passing a bad
   result through to its final answer.

This is a heuristic floor, not a correctness checker — it can't tell if
generated code actually *works*, only that it isn't obviously empty,
refused, or truncated. Every retry attempt still costs local compute, so
the Usage panel's local-token count reflects retries too, not just
whichever attempt ultimately got used.

### Memory management

Each `localforge run` is stateless — it starts a fresh conversation with no
memory of any previous invocation. *Within* one run, though, a long task
can accumulate a lot of tool-result content (e.g. several rounds of
generated code); left unchecked, that would grow the context sent to the
frontier model — and the cost of every round — without bound. Once more
than `KEEP_RECENT_TOOL_RESULTS` (currently 4) tool results have
accumulated, older ones are collapsed in place to a short placeholder
(`[superseded: earlier result, N chars -- no longer kept in full in
context]`); the assistant's own record of having made that call is never
touched, only the old result content. This bounds both context size and
per-round cost on long tasks without needing an extra summarization LLM
call.

### Usage metrics

Every `localforge run` ends with a Usage panel — a Claude-Code-style
horizontal bar showing the local/frontier split at a glance, plus the exact
numbers below it:

```
╭─────────────────────────────── Usage ────────────────────────────────╮
│ ████████████████████████████████████░░░  92% local / 8% frontier    │
│                                                                       │
│ ■ Local models: 4200 tokens — never sent to or billed by the         │
│   frontier API                                                       │
│ ■ Frontier (claude-opus-5): 280 in + 70 out = 350 tokens ($0.0200)   │
╰────────────────────────────────────────────────────────────────────────╯
```

(green segment/marker = local, yellow = frontier)

The frontier numbers are real: token counts come from the API response's
own `usage` field, and the dollar cost is computed by LiteLLM's own pricing
table (`litellm.completion_cost`) for whichever model you actually used —
not a guess. The "local models" figure comes from Ollama's own `eval_count`
for each generation, summed across every subtask delegated during the run;
it's the clearest proxy for "how much work happened without touching the
frontier API at all" (and therefore without frontier tokens/cost for it),
even though we don't attach a speculative dollar figure to what it "would
have cost" on the frontier side, since that depends on a model/pricing
assumption we can't verify.

### Managing disk space

`localforge models`/`catalog` describe the *catalog* (what could be
pulled); `localforge installed` shows what's *actually on disk* right now,
via Ollama, with sizes and a total. `localforge delete` lets you free space:
run it with no arguments to see a numbered list, pick one or more
(comma-separated, or `all`), review exactly what will be freed, and confirm
once before anything is deleted — or pass model name(s) directly
(`localforge delete qwen2.5-coder:14b --yes`) to skip the interactive list
for scripting. A failed delete in a batch doesn't abort the rest of the
queue.

### Uninstalling

`localforge uninstall` removes what localforge manages: every locally
pulled model, the saved config (`~/.config/localforge`), and finally the
`localforge` CLI tool itself. It always shows an itemized list of exactly
what it's about to remove before doing anything, and asks for confirmation
(`--yes` skips that prompt, for scripting). Ollama itself is **left alone
by default** — only its models are deleted, not the app — since Ollama
might be something you use for other things besides localforge. Pass
`--purge-ollama` to also uninstall Ollama (via Homebrew, if that's how it
was installed) and wipe `~/.ollama` entirely, for a fully clean slate; this
is asked about separately even without the flag, so it's never bundled
into a single "yes to everything."

### Choosing a frontier model

`setup`/`wizard` always ask explicitly which frontier provider
(Anthropic/OpenAI/Gemini, **or `local`**) to use — localforge never
silently guesses this from whatever API key happens to already be in your
environment (if you have multiple keys set for unrelated tools, that
ambiguity gets surfaced, not resolved for you). If a key for the provider
you pick is already present, you're offered the option to reuse it instead
of re-entering it. The choice is saved to `LOCALFORGE_FRONTIER_MODEL` in
`~/.config/localforge/config.env`, which `localforge run` uses by default
(overridable per-run with `--model`).

### API key vs. CLI login

For any provider that needs credentials, setup asks **how** to authenticate:

```
How should localforge authenticate to this provider?
  1) API key     (pay-per-token, billed separately)
  2) CLI login   (uses your `claude` subscription — no per-token API cost)
```

**CLI login** orchestrates by shelling out to that provider's own
already-logged-in CLI (`claude`, `codex`, `gemini`) in headless mode, so the
work draws on whatever subscription that account has instead of separate
per-token API charges. localforge never sees or stores those credentials —
the CLI holds its own (Claude Code keeps its token in the OS keychain), so
nothing extra lands in `config.env`.

Availability is **probed, not assumed**: if the CLI isn't installed, that
option tells you so and links the install page rather than failing later.
Only the Anthropic path is verified end to end; the OpenAI and Gemini entries
are best-effort and flagged as untested until you install those CLIs.

Two caveats worth knowing before you depend on it:
- Anthropic's Agent SDK docs state third-party developers shouldn't offer
  claude.ai login *in their own products* and should use API keys instead.
  Invoking a CLI you installed and logged into yourself, on your own
  machine, is a different thing from localforge offering you a login — but
  it's your account and your call.
- The billing arrangement for CLI/subscription usage has changed before and
  may change again. Re-check before building anything durable on it.

The usage panel is explicit about which mode you're in: under CLI login the
dollar figure is labelled *"of subscription usage, not billed separately"*,
because it's what pay-per-token *would* have cost — not money charged on top.

**Getting an API key.** None of Anthropic, OpenAI, or Google expose a
public OAuth/browser-login flow for third-party CLI tools to authenticate
on your behalf (unlike, say, `gh auth login`'s device flow for GitHub) —
so this is not a real "sign in" step, just a shortcut to the right page.
When you pick a provider that needs a key and don't already have one set,
`setup`/`wizard` automatically opens your browser straight to that
provider's API key page (Anthropic's Console, OpenAI's Platform dashboard,
or Google AI Studio) so you don't have to go find it, then you paste the
key in as usual.

**The frontier/orchestrator model doesn't have to be a proprietary API at
all.** Picking provider `local` lets an open-weight model served by Ollama
be the orchestrator itself — no API key, no per-token cost, fully
self-hosted. Model ids for this provider use LiteLLM's `ollama/<model>`
convention (e.g. `ollama/llama3.1:70b`); `setup` pulls it via Ollama just
like any delegated model. Trade-off to know: the orchestrator role plans
the task and makes tool-calling decisions, which most *small* open-weight
models handle unreliably — the curated choices here favor larger models
for that reason, and you can always point `--model` at anything else
LiteLLM/Ollama supports via the "Other" option.

### Reusing what's already installed

Before downloading anything, `setup` (and the wizard) checks which models
Ollama already has on disk. If an installed model fits your hardware for a
task type, it's reused instead of pulling a new one — re-running setup
won't re-download models you already have. Setup prints the plan first:

```
✓ Reusing already-installed: qwen2.5-coder:7b (coding), mistral-nemo:12b (docs), qwen2.5:3b (general)
✓ Nothing to download — every task type is covered by what you have.
  note — coding: qwen2.5-coder:14b (higher tier, ~9 GB) also fits — `ollama pull qwen2.5-coder:14b` if you want the upgrade
```

When a better model would fit, it tells you rather than silently
downloading it (or silently not) — you decide whether the upgrade is worth
the download. An installed model still has to fit your hardware to be
reused; being on disk isn't enough. `localforge models` shows the same
picks with an **Installed** column.

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
localforge help                                  # list every command (same as --help)
localforge wizard                                # one-time interactive setup (terminal UI)
localforge setup                                 # one-time interactive setup (plain prompts)
localforge doctor                                # is everything set up correctly?
localforge scan                                  # what hardware do I have?
localforge models                                # what will run well on it?
localforge run "Build a todo REST API with docs" # do the thing
localforge run "..." --model gpt-5               # use a different frontier model
localforge installed                             # what local models are actually on disk, and how big
localforge delete                                # pick installed model(s) to delete, review, then confirm
localforge delete qwen2.5-coder:14b --yes        # delete a specific model without prompting
localforge uninstall                             # remove models, config, and the CLI tool (asks first)
localforge uninstall --purge-ollama              # also uninstall Ollama itself and wipe ~/.ollama
localforge theme                                 # show current theme + available options
localforge theme dark                            # switch theme (matrix / dark / light)
```

### Themes

`localforge theme` switches the CLI's color scheme between three standard
options — `matrix` (bright green, hacker-terminal aesthetic, the default),
`dark` (a calmer cyan/green scheme for typical dark terminals), and `light`
(deeper/darker color tones that stay readable on a light terminal
background). Every message in the CLI uses semantic style names
(success/error/warning/accent) rather than hardcoded colors, so switching
themes actually changes what you see everywhere, not just in one place.
The choice is saved to `LOCALFORGE_THEME` in `~/.config/localforge/config.env`
and applies immediately to the running command as well as every future one.

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
- `config.py` — persists setup choices (API keys, chosen frontier model,
  theme) to `~/.config/localforge/config.env`, loaded automatically on
  every CLI invocation without overriding variables already set in the
  shell. `FRONTIER_PROVIDERS["local"]` maps to `None` (no API key) for the
  open-weight-orchestrator option.
- `theme.py` — the three `rich.theme.Theme` definitions (`matrix`/`dark`/`light`)
  behind `localforge theme`, keyed by semantic style names
  (`success`/`error`/`warning`/`accent`) that `cli.py` uses everywhere
  instead of literal color words.
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
