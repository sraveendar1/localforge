# localforge

An orchestrator model — a paid one (Claude, GPT, or any provider), or an
open-weighted model itself — directs open-weight models running **locally**
on your machine. `localforge` profiles your hardware, picks open-weight
models that will actually run well on it, pulls them via
[Ollama](https://ollama.com), and hands them subtasks — coding,
documentation, and (future) image/video generation — while the orchestrator
of your choice does the planning and delegation.

No vendor lock-in: the orchestrator is called through
[LiteLLM](https://github.com/BerriAI/litellm), so swapping `claude-opus-5`
for `gpt-5` or a self-hosted model is a one-line config change.

Use it from a terminal (`localforge`, see [Interactive session](#interactive-session)
below) or as a native desktop app (see [Desktop app (GUI)](#desktop-app-gui))
— both drive the exact same engine, so nothing behaves differently between them.

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

It works in the folder you start it in, like Claude Code. The orchestrator
reads your code (`read_file`, `list_files`, `search`), runs commands
(`git clone`, tests, installs), keeps a visible plan, and hands the actual
writing to your local models, which write whole files. Before any file is
saved you see a diff, and before any command runs you see it, and you answer
**(y)es / (n)o / (a)lways this session**. (`/auto on` or `run --yes` skips
the prompts.) The session remembers the conversation. When it gets long, a
local model condenses the older turns into a session-memory note, which is
also saved so your next session in the same folder picks up where you left
off (`/compact` to do it now, `/clear` to start fresh).

**Starting a session:** the first time in a folder, localforge asks whether
you trust it (like Claude Code). Then it confirms the orchestrator. If you used
one last time, it asks whether to keep it or choose a different one. The list
(any model in Ollama, or Claude/GPT/Gemini if you've set one up) appears only
when you want to change. There's no default, so you always choose. Type `/` to see every command with
a description, and press Tab to complete. Arrow keys, mid-line editing and
Up for history all work.

**Watching it work, while you keep working:** a task runs in the background
and your prompt stays live, like Claude Code. A status bar at the bottom shows
who is doing what, for example `qwen2.5-coder:7b working on coding · 312
tokens · 41 tok/s`, and each finished step prints one line instead of
flooding the screen with code. While it works you can:
- `/summary`: the task, which model is doing what, the plan, recent steps
  and the queue (instant, no model call)
- ask a question ("what does the Dispatcher class do?", "why is main.py
  structured this way") and it's answered on the side, from what's already
  cached — the project brief, remembered facts, the last session's summary
  — by a local model, without waiting behind the running task. Anything
  that reads as an actual task (not a question) still queues as normal.
- type another task: it's queued and runs next (see it in `/summary`, `/queue clear` to drop it)
- `/tell <note>`: add something to the task that's running now
- `/stop` or Ctrl+C: stop the current task (the session stays open)
- `/usage`, `/memory`, `/scratch` and the other read-only commands

When a change needs your approval, the diff prints above and the prompt
itself asks: (y)es / (n)o / (a)lways. The answer is printed, formatted, when
the task is done. (A one-off `localforge run "..."` still streams in the
foreground.)

**Files:** in a trusted folder it can read, create, change, move and delete
files and run commands. Every change asks first: (y)es / (n)o / (a)lways this
session. Deletions always ask separately, and `/auto` turns prompts off.

**Scratchpad:** each session gets a private scratch folder outside your
project (so it never shows up in git), where local models draft pseudo-code
and code before anything touches a real file. Moving a draft into the project
shows you the diff first. It's deleted when the session ends (`/scratch` shows
what's in it).

**`localforge goals`** (`/init` still works, as an alias): like Claude Code's
`/init`, this writes a project brief — `AGENTS.md` (the file other coding
agents read too): what the project is, how it's built, how to run and test
it, and the decisions worth keeping. The local models get its stack,
commands and conventions with every task, so the code they write fits the
project. A **local** model writes it, from the project itself plus what
localforge remembers of your sessions, and you approve the diff like any
other change. The first task in a project with no `AGENTS.md` yet offers to
draft it right then, seeded with that task's own description, rather than
waiting for you to remember `/goals` later. Every later session starts with
it, so the orchestrator doesn't have to rediscover your project each time.
After a session that changed files, the local model drafts an update, and
the next session offers it to you as a diff (or `/goals` reviews it;
`/goals --refresh` rewrites from scratch); mid-session, once enough files
have changed, you're offered the same review without waiting for the next
session. An existing `AGENTS.md` is updated in place; a `CLAUDE.md` is read
when there's no `AGENTS.md`, and never written. It also has a "Definition of
done": the exact commands that check a change, which localforge uses before
a task is allowed to finish.

**Memory:** localforge keeps memory per project, in the project's own
`.localforge/` folder, so it moves with the folder. Git ignores that folder by
default (it carries its own `.gitignore`; delete that to share memory with
your team). It holds remembered facts (your preferences, decisions, pointers),
a summary of where the last session left off, and your usage history. Tell it "remember that…" and it
saves a fact. At `/exit`, a local model saves what's worth keeping, so the
next session in that folder picks up from there. `/memory` shows it, and
`/memory forget <name>` or `/memory clear` remove it.

**When something goes wrong:** if a local model crashes, stalls or starts
repeating itself, localforge stops it and tries again, with another installed
model if there is one. The orchestrator is told what failed and works around
it (retry, simpler instructions, a different approach). Only after a few
different attempts does it stop and tell you what's blocking. A brief
connection hiccup with Claude is retried once. Ctrl+C stops the current task
without ending the session.

**Where the money goes:** the orchestrator (a paid model, unless you're running
fully open-weighted) is told it's the expensive one: it plans, directs and
checks, and the open-weighted models write. It doesn't paste files or code
into its instructions: it names `context_files`, and localforge hands those
files straight to the open-weighted model. What an open-weighted model writes
comes back to the orchestrator as a short summary, while you still see the
full diff. Claude also orchestrates at medium thinking effort
(`LOCALFORGE_ORCHESTRATOR_EFFORT` to change). `/usage` shows the last task, this session,
the previous session and this project's all-time totals — how much of the
work the open-weighted models did, and an estimate of what that would have
cost from a paid model. The totals are kept per project, so they survive
closing the session.

**Always visibly alive:** while a task runs, the bottom line shows a hammer
and anvil working away, the model doing the work, how long it's been, a
running token count, and, if a step goes quiet, how long for:

```
localforge> add a health endpoint
  → coding → qwen2.5-coder:7b (open-weighted model)
          ⠴ Forging with qwen2.5-coder:7b… (5s · ↓ 62 tokens · working on coding, 12 tok/s) │ /summary · /stop
  │     app = FastAPI()
  │     @app.get("/health")
localforge>
```

The last few lines the local model is writing show under it, so you can watch
the code take shape without it scrolling your session away (`/stream on`
prints everything instead).

**Asking why:** at any permission prompt you can type a question instead of
y/n/a — "why is this needed?" — and localforge explains what the request
does, what happens if you say no, and what it came up during; a local model
answers anything more specific. The prompt stays waiting, so asking never
approves anything. `/why` does the same on demand, and the folder-trust
question can be asked about too.

**If the paid model runs out mid-task:** the task doesn't die. localforge
reads the reset time from the provider's message, pauses with a countdown in
the status bar, and then carries on exactly where it stopped: nothing you've
already done is lost and you don't have to retype the task. While it's
paused you can `/model` to switch to a local model and continue right away,
or `/stop` to give up. If the provider doesn't say when the limit resets, or
it's hours away, localforge says so instead of waiting. Either way, if the
task ends before it's done — you gave up, or the limit couldn't be waited
out — it's saved, not lost: the next session you open in that folder offers
to pick it back up right where it stopped, and `/usage` still counts the
tokens that task already spent.

**Fully local, no account:** pick "local" in `localforge setup`, or type
`/model` in a session and choose a model you already have in Ollama (e.g.
`/model ollama/qwen2.5:7b`). The orchestrator then runs on your machine too,
with no Claude/OpenAI/Gemini account and no billing, even if you set one up
before. Models under ~7B work but plan unreliably, and localforge says so.
If a cloud account hits its usage limit mid-session, `/model` is the way out.

**Web access:** open-weighted models have no internet access. The orchestrator does any research (`web_search`, `fetch_url`, built into localforge so it works the same with an API key or CLI login) and passes what it found into each subtask's instructions. When you orchestrate through a CLI login, that CLI's own tools (shell, file edits, web, connected apps) are switched off, and it runs in an empty scratch folder, so it can plan and delegate but can't touch your files. `fetch_url` refuses localhost and private-network addresses.

While a task runs you see which local model each subtask goes to, and that model's output streams in live as it's generated, followed by its token count and speed.

Type a task directly and it runs (shorthand for `/run <task>`), or use a
slash command for anything else — `/usage`, `/setup`, `/doctor` (also shows
detected hardware), `/theme dark`, `/delete <names>`, `/help` for the full
list, `/exit` to leave. Every slash command reuses the exact same code as
its `localforge <command>` equivalent — there's no second implementation to
drift out of sync. (`localforge scan` — hardware detection alone, no config
checks — is still there as a plain CLI command, just left off the
interactive `/help` menu since `/doctor` already covers it.)

Piped/scripted invocations (`localforge | cat`, CI, no real terminal
attached) skip the session and keep the old print-and-exit behavior, so
nothing that already scripts against a bare `localforge` call starts
waiting on stdin.

## Quickstart

**Before you run it — what `install.sh` puts on your machine:**
- [uv](https://docs.astral.sh/uv/), if you don't already have it
- The `localforge` CLI tool itself

Models, Ollama and Homebrew are not installed here: the first `localforge`
run offers those, so nothing multi-GB downloads before you've seen the tool.

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

That installs the CLI and nothing else. The first time you run `localforge`
it offers to do the rest — install Ollama, pick a model for your hardware,
and set up how you reach a paid model (or stay fully open-weighted) — where you
can see it and say no. (`./install.sh --setup` does that during install
instead, and `localforge setup` can be run any time.)

Then start a session and type what you want built:

```bash
localforge
localforge> Build a todo REST API with docs
```

(For a one-off without a session: `localforge run "Build a todo REST API with docs"`.)

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

## Desktop app (GUI)

`desktop/` is a native desktop app (React + [Tauri](https://tauri.app)) for
everything above, for anyone who'd rather click than type. It's a real GUI,
not a wrapper around the terminal: it drives the exact same session engine
as the interactive CLI — it launches `localforge serve --stdio` as a child
process and speaks a JSON-lines protocol over its stdin/stdout — so nothing
about the orchestrator, delegation, or approvals is reimplemented for it.

### Installing the desktop app

**The easy way:** grab the installer for your OS from the
[Releases page](https://github.com/sraveendar1/localforge/releases) —
a `.dmg` for macOS (Apple Silicon) or an `.AppImage`/`.deb` for Linux.
No Rust, Node, Python, or `uv` needed on your machine; the whole CLI is
bundled inside as a standalone binary (see "Packaging a standalone app"
below). Open the `.dmg` and drag **LocalForge Desktop** into
`/Applications`, or make the `.AppImage` executable and run it. Windows
isn't published yet — see the note in `.github/workflows/release.yml`.
This is also what makes `/desktop`/`localforge desktop` (below) work: it
looks for the app installed exactly this way.

**macOS: "LocalForge Desktop is damaged and can't be opened."** The app isn't
signed with a paid Apple Developer ID yet, and macOS blocks anything downloaded
that isn't. The file isn't corrupt. Pick whichever applies:

- **Builds from v0.1.4 on** are ad-hoc signed (free), so macOS shows the usual
  *"can't be opened because Apple cannot check it for malicious software"*
  message instead. Open **System Settings → Privacy & Security**, scroll to
  *"LocalForge Desktop" was blocked*, and click **Open Anyway**. (On macOS 15
  the older right-click → Open shortcut no longer works.)
- **Any build**, one command in Terminal after dragging it to Applications:
  `xattr -cr "/Applications/LocalForge Desktop.app"`, then open it normally.
  If it complains before you can even copy it, run
  `xattr -cr ~/Downloads/*.dmg` first.

Removing the warning entirely takes an Apple Developer ID and notarization
(a paid Apple account, plus signing secrets in the release workflow).

**The app opens but nothing happens (no trust prompt, no setup screen).** The
part of the app that does the work is a separate program bundled inside it. If
that program can't start, the app now says so ("localforge didn't start") and
shows what it printed, with **Try again** and **Copy details**; send those
details. v0.1.3 had exactly this problem on macOS: signing the app also signed
that program with the "hardened runtime", which stops it loading its own
libraries. v0.1.4 turns that off, and the release workflow now launches the
program from inside the signed app so a broken one fails the build.

Releases are built automatically by
[`.github/workflows/release.yml`](.github/workflows/release.yml) whenever
a maintainer pushes a `v*` tag (`git tag v0.2.0 && git push origin v0.2.0`),
so a new tag is all it takes to publish a new installer for every
supported OS — no one has to build it by hand.

**Building it yourself** (for development, or an OS/architecture without a
published release):

```bash
# 1. Install the localforge CLI first -- the app spawns it as a backend.
git clone https://github.com/sraveendar1/localforge.git
cd localforge
./install.sh                    # or: uv tool install .

# 2. Build and run the desktop app.
cd desktop
npm install
npm run tauri dev               # opens the app, hot-reloads on UI changes
```

For a real double-clickable app instead of dev mode:

```bash
npm run tauri build             # produces an installer under desktop/src-tauri/target/release/bundle/
```

That gives you a `.dmg`/`.app` on macOS, an `.AppImage`/`.deb` on Linux, or
an `.msi`/`.exe` on Windows, depending on what you built on — install it
like any other app for your OS from there.

**Prerequisites for building** (not needed just to use a already-built
app): Node.js + npm, Rust + Cargo, and Tauri's own OS-level dependencies —
WebKitGTK and related packages on Linux, Xcode command line tools on macOS,
the WebView2 runtime on Windows (usually already present). See [Tauri's
prerequisites guide](https://v2.tauri.app/start/prerequisites/) if
`npm run tauri dev` fails partway through with a missing system library.

`LOCALFORGE_BIN=/path/to/localforge npm run tauri dev` points the app at a
specific checkout/binary instead of whatever `localforge` resolves to on
your `PATH` — useful when developing the Python and desktop sides together.

**Packaging a standalone app:** `scripts/build_sidecar.sh` packages the
entire `localforge` CLI (Python interpreter and all) into one standalone
binary via [PyInstaller](https://pyinstaller.org), which Tauri then embeds
directly into the built app as a "sidecar" — so a downloaded release needs
no separate CLI install at all. This is exactly what
`.github/workflows/release.yml` runs before `tauri build` for every
published release; run it yourself the same way before your own
`npm run tauri build`, once per OS you're building for (PyInstaller doesn't
cross-compile):

```bash
./scripts/build_sidecar.sh      # from the repo root; needs uv
cd desktop && npm run tauri build
```

### First run: setting up inside the app

A fresh install needs no terminal. After you open a folder and trust it, if no model can run yet, the app opens a **Set up localforge** screen in place of the empty chat (the header chip says "Set up models…" until it's done):

1. **Sign in to the model that plans and reviews your work.** One card each for Anthropic (Claude), OpenAI (GPT) and Google (Gemini):
   - **Use my `claude` / `codex` / `gemini` login** appears when that app is installed on the computer. It checks that the login really answers (a signed-out one is refused with what to run) and then uses your subscription, no API key.
   - **Paste an API key.** The field hides what you type. The key is checked with the provider first (a rejected key isn't saved and says so; if the provider can't be reached it's kept with a note), stored only in localforge's private config file (`~/.config/localforge/config.env`, mode 600) and never sent back to the screen. "Get one" opens the provider's key page.
   - Either way a sensible model is chosen for you (`claude-opus-5`, Gemini's first model from your key, …) and saved as your default for every project; change it any time in Models.
2. **Or run everything on this computer.** It shows whether Ollama is installed and running and lists the models you've downloaded (one that's too big for the machine is greyed out with the reason). Ollama itself is installed from ollama.com; the app links to it.
3. **Skip for now** leaves a banner ("No model is set up yet, so tasks will fail") instead of nagging. If a task is sent with nothing set up, it fails with a plain message ("claude-opus-5 needs an Anthropic (Claude) API key or login, and none is set up yet. Use “Set up”…") instead of a library error, and the setup opens as an overlay over your conversation.

**Already set up?** The screen only appears when nothing can run yet, so a machine where you ran `localforge setup` earlier goes straight to the chat. To add or replace a key later (say a Gemini key, to send images there) or sign in with another login, open **Models → Accounts & keys…** in the left panel: the same cards, with your current orchestrator left as it is.

The coding, docs and general models need no setup: they're picked for your machine and downloaded (after asking) the first time they're needed. Nothing here replaces `localforge setup` in a terminal, which does the same for the CLI and shares the same saved settings.

### Using it

Open a project folder (the same one-time trust prompt as the CLI applies),
pick or switch the orchestrator model at any time — including mid-task,
matching `/model`'s behavior in a session — and type a task, optionally
attaching an image (📎 next to the input; needs a vision-capable
orchestrator — see "Attaching an image" below). You get the same live
picture the terminal session shows: which local model is delegated to for
each step, its output streaming in token by token, and a diff/command
prompt for every write or command with the same **Approve / Always allow /
Decline** choices as the CLI's (y)es/(n)o/(a)lways. A queue strip shows
tasks typed while one is already running, and a status bar mirrors the
CLI's always-visible activity line.

**Sidebar** (right side, and the whole panel collapses from the chevron tab on its edge, like the left one; each panel remembers whether you left it open, and the right one starts closed on a narrow window): five independently collapsible sections — click
any title to expand/collapse it — in this order: **Overall goal** (this
project's AGENTS.md summary, if it has one), **Plan** (the running task's
todos), **Active LLMs** (which model is handling each kind of work right
now, with a pulsing dot on the one generating — status only; you choose
models in the left **Models** section), **Usage and cost** (frontier tokens/cost, plus this
project's usage history), and **System** (CPU/RAM/GPU, collapsed by
default).

**Left nav** (collapsible via the chevron tab on the left edge): a list of
recently-opened project folders — click one to switch straight to it,
skipping the file dialog — and the **Models** section. (There is no command
list here: type `/` in the message box to see every command with its
description, and Tab or Enter to complete one.)

**Models** is where every model choice lives, so the whole setup is visible
in one place. The **Frontier** row is the model that plans and reviews
(clicking the model name in the header opens this section). It lists every
model already in Ollama plus each provider you have an API key or CLI login
for — the same list `/model` shows in the terminal — and a box for any other
model id. Below it are the three kinds of work (coding, docs, general), with
the toggle:

- **Automatic** (default): localforge picks the best local model for each
  kind of work. The rows are read-only but always name the real model, e.g.
  `qwen2.5-coder:7b (auto, local, installed)`.
- **Advanced**: each kind of work gets a **Change** button — Auto, a local
  model (free; ones not yet installed say how big the download is), or a paid
  cloud model through a key/login you already have (costs money per use; see
  `localforge advanced-model` below). Advanced simply means "at least one is
  pinned", so a pin made in the terminal or by typing `/advanced-model` shows
  up here as Advanced, and switching back to Automatic clears the pins.

**Models are saved per project.** The first time you change a model in a folder, localforge writes `.localforge/models.json` there: the frontier, and coding / docs / general / image / video, as plain JSON. Open that folder again — from the app, from the terminal, on the same machine — and it uses those models; change one and the file is updated. Other projects are unaffected, and a folder with no file uses your defaults (whatever `localforge setup` chose). The Models section says which it is: *"Using your defaults"* or *"Saved for this project"*. `localforge setup` itself still sets your defaults, not a project's. The file sits in `.localforge/`, which git ignores on its own, so it stays yours; delete that folder's `.gitignore` to share it with a team. It is only read for a folder you've trusted, so a cloned repo can't choose a paid model to be billed to your keys. An explicit `--model` still wins over a project's saved frontier.

**A model too big for your machine is refused.** Choosing a local model that needs more memory than this machine can give it — or whose download won't fit in the free disk space — is rejected with the reason and nothing is changed, e.g. *"qwen2.5-coder:32b can't run on this machine for coding: it needs about 24.5 GB of memory to run (even with the smallest context window), but this machine has 16 GB of memory…"*. In the Models section such models are greyed out with the reason underneath. The check runs again before every task, so a project whose saved models came from a bigger machine (or a disk that has since filled up) is rejected up front — before anything is spent or changed — naming which model is the problem and how to change it (`/model`, `/advanced-model coding auto`). Cloud models are never judged this way, and an installed model isn't faulted for disk space.

**Image and video rows.** Below coding/docs/general the Models section has an **image** row and a **video** row:

- **Image generation** is off until you pick an image model in Advanced mode (or `localforge advanced-model image <value>`). Two ways:
  - **An API key** (`api:<provider>:<model>`): OpenAI (`gpt-image-1`, `gpt-image-1-mini`) or Google AI Studio, each needing that provider's key. For Gemini the picker lists the image models *your key can actually use* (e.g. `gemini-3.1-flash-image-preview`, `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, Imagen) with the price per image where it's known (about $0.02–0.13), falling back to a built-in list if Google can't be reached. Each image is billed to the key and counted in `/usage`.
  - **Your ChatGPT login through Codex** (`cli:openai:codex-image`), *experimental*: no API key, it uses your plan's image allowance instead of per-image billing. It runs the official `codex exec` in a scratch folder, asks Codex to use its built-in image tool, and takes the image file it produces. It is **not verified against a real login yet** (Codex documents the tool but not its use in scripted `codex exec`), so if no image appears the error says what Codex printed. localforge does not call ChatGPT's private endpoint with your login token, as some third-party tools do. It appears in the picker only when the `codex` CLI is installed. **Claude's login can't do this: Claude has no image model.** Gemini's CLI only gets image generation from an extension, so it isn't offered.

  Once one is chosen the orchestrator is offered a `generate_image` tool: it describes the picture and a path, the image model draws it, and you approve saving it (the prompt says what it is and how big, since a picture has no diff). There is no local image backend yet.
- **Paid images cost money, and a subscription doesn't cover that.** An image through an API key is billed per image to that key. Google AI Pro ($19.99/month) does **not** make Gemini API calls free, and Gemini image generation has no free tier; what the plan does include is **$10 a month in Google Cloud credits** that can pay for Gemini API use once you activate them (in the Google Developer Program, applied to a Cloud project; they reset monthly and don't roll over — see [Google's page](https://developers.google.com/program)). The plan's own image and video allowance (in the Gemini app and Flow) is not available to other programs, and no supported way to use a Google login for images from code was found. So localforge helps you stay inside a budget instead:
  - **A confirmation before every paid image** saying what it costs ("about $0.039, charged to your gemini API key… this month so far: $2.10 of your $10.00 limit"). It follows Auto-approve / "Always allow" like other prompts.
  - **A monthly limit per provider**: `localforge budget gemini 10` (or `/budget gemini 10`, or "set a monthly limit" under the image row in Models). Once another image would go over it, localforge refuses **before anything is billed** and tells the orchestrator. `localforge budget` shows the month so far; `off` removes a limit. Set it to $10 to stay inside the Pro credits.
  - Spending is kept per provider and month in `~/.config/localforge/spend.json` (not per project: the credit and the bill belong to the account). The limit is enforced on known prices (Gemini's image models have one; OpenAI's are priced per token, so for them only spending already recorded can stop a call). Your provider's own dashboard remains the source of truth.
- **Video** has a row so the choice has somewhere to live, but video generation isn't built, so it can't be chosen yet.

The chat input doubles as a command line: typing `/` brings up every
session command with a description (arrow keys / Tab to pick), and each one
reuses the exact server-side handler the CLI's own slash commands use —
`/memory`, `/scratch`, `/queue`, `/stop`, `/clear`, `/usage`, `/models`,
`/installed`, `/catalog`, `/doctor`, `/scan`, `/model`, `/auto`, `/run`,
`/help`, `/compact`, `/summary`, `/tell`, `/why`, and `/stream` all work
from the box exactly as they do in a terminal session, with the reply shown
as its own message in the conversation.

**Handing off from a terminal session:** run `localforge desktop` (or type
`/desktop` inside a session) to save session memory and launch the desktop
app open to the same folder, picking up right where the terminal left off.

**Attaching an image:** click 📎 next to the chat input to attach a
screenshot or photo to your next message. This needs a vision-capable
orchestrator — an API-key model (Claude/GPT/Gemini) or Claude via CLI
login work today; other CLI logins and local Ollama models need a
vision-capable model (e.g. `llava`) picked as the orchestrator. Attaching
an image with an orchestrator that can't use it fails with a clear message
rather than silently ignoring it.

**Not in the GUI yet:** first-time onboarding (`localforge setup`) and the
`delete`/`uninstall` confirmation flows still need a terminal — run those
once from the CLI, then the desktop app picks up whatever they configured.
See [`docs/desktop_parity.md`](docs/desktop_parity.md) for the exact,
up-to-date command-by-command coverage table.

## How it works

1. **`localforge scan`** — detects OS, CPU, RAM, GPU/VRAM, and free disk space.
2. **`localforge models`** — matches your hardware against a curated catalog
   (`src/localforge/catalog_data.yaml`) to recommend the best local model per
   task type — preferring a model you already have installed over a
   fresh download, and marking which ones are installed.
3. **`localforge run "<task>"`** — the orchestrator plans the task, calls
   tools like `delegate_coding_task` / `delegate_docs_task`, and those calls
   are dispatched to the matched local model running under Ollama. Results
   flow back into the orchestrator's context until it produces a final
   answer. Every delegation is printed as it happens, e.g.:
   ```
   → delegating coding to qwen2.5-coder:14b (open-weighted model, via ollama)
   → delegating docs to mistral-nemo:12b (open-weighted model, via ollama)
   ```
   so you always know which local model is doing the actual work for a
   given task, not just that "something local" is running.

### Catching bad local output

Local models occasionally return something unusable — empty output, an
outright refusal ("I'm sorry, but..."), something absurdly short for what
was asked, or (less obviously) a well-formed result that just doesn't do
what the task asked. This is caught at three levels:

1. **A cheap heuristic tripwire in the dispatcher.** Every delegated result
   is checked for the obvious failure modes above before it's ever shown to
   the orchestrator. If flagged, `localforge` automatically retries once
   — with a different model if this hardware fits more than one for that
   modality, or the same model with reinforced instructions ("your previous
   attempt was rejected: ...") otherwise. You'll see both delegation
   attempts printed live. If the retry is still bad, the result is passed
   through wrapped in an explicit `[WARNING: ...]` tag rather than silently
   accepted.
2. **A dedicated judge model checks the result against what was actually
   asked for.** This is a real semantic check, not a string heuristic — it
   catches a result that's well-formed but doesn't do what the task asked,
   which the heuristic above can't. localforge prefers a small model
   purpose-built for evaluation (`atla/selene-mini`) if you have it
   installed; if not, it falls back to another already-installed model
   (never the one that wrote the output being judged — a model grading its
   own homework defeats the point), and skips the check entirely rather
   than have a model grade itself if nothing else is installed. **Never
   downloaded automatically, and never offered by `setup` or
   `localforge models`** — it's deliberately left out of the ordinary
   recommend/install flow (there's exactly one candidate anyway, so there's
   nothing to "recommend"); `ollama pull atla/selene-mini` yourself if you
   want the dedicated one, otherwise localforge reuses what you already
   have, or goes without. For a file write, the judge call runs *in
   parallel* with the syntax check below, so it doesn't add its own wait
   on top. A failing verdict gets the same `[WARNING: ...]` treatment,
   with the judge's specific reason attached, and never blocks the write —
   you still see the diff and decide.
3. **The orchestrator is explicitly instructed not to trust delegated
   results at face value** — to check them against what it asked for,
   treat `[WARNING: ...]`-tagged results with extra scrutiny, and delegate
   a subtask again with clearer instructions rather than passing a bad
   result through to its final answer.

The heuristic in (1) is a floor, not a correctness checker — it can't tell
if generated code actually *works*, only that it isn't obviously empty,
refused, or truncated; the judge in (2) is what catches "ran fine, but
doesn't do the thing." Every retry attempt still costs local compute, so
the Usage panel's local-token count reflects retries (and judge calls)
too, not just whichever attempt ultimately got used.

### Memory management

Each `localforge run` is stateless — it starts a fresh conversation with no
memory of any previous invocation. *Within* one run, though, a long task
can accumulate a lot of tool-result content (e.g. several rounds of
generated code); left unchecked, that would grow the context sent to the
orchestrator — and the cost of every round — without bound. Once more
than `KEEP_RECENT_TOOL_RESULTS` (currently 4) tool results have
accumulated, older ones are collapsed in place to a short placeholder
(`[superseded: earlier result, N chars -- no longer kept in full in
context]`); the assistant's own record of having made that call is never
touched, only the old result content. This bounds both context size and
per-round cost on long tasks without needing an extra summarization LLM
call.

### The summary after every task

When a task ends -- finished, stopped, or failed -- localforge prints what it actually did, in the terminal and in the desktop app:

```
╭─ Summary ─────────────────────────────────────────────╮
│ Done in 2m 14s · 7 steps                              │
│ Files changed:                                        │
│   updated src/app.py  (written by qwen2.5-coder:7b)   │
│ Delegated to: qwen2.5-coder:7b ×1                     │
│ Commands run:                                         │
│   ✗ pytest -q                                         │
│   ✓ pytest -q                                         │
│ You declined:                                         │
│   old.py                                              │
│ Problems:                                             │
│   ✗ run_command pytest -q: exit code 1 (recovered)    │
╰───────────────────────────────────────────────────────╯
```

It lists the files changed and which model wrote each, the commands run and whether they passed, anything you declined, and **every problem**, including ones the task got past on its own (marked `(recovered)`) -- a failed step, a retried orchestrator call, a usage limit. If the task ended in an error, the error text is in the summary too (`Error: ...`), along with what got done before it broke. A plain question answered without using any tools prints no summary.

### Usage metrics

Token usage is shown only when you ask for it, so it doesn't clutter every
answer. Inside a session, type `/usage`: you get the last task and, after two
or more tasks, running totals for the whole session. For a one-off run, add
`--usage` (`localforge run "..." --usage`). Failed runs still count toward
`/usage`, because they spent real tokens too. The panel is a Claude-Code-style
horizontal bar showing the open-weighted/paid split at a glance, with the
exact numbers below it:

```
╭─────────────────────────────── Usage ────────────────────────────────╮
│ ████████████████████████████████████░░░  92% open-weighted / 8% paid│
│                                                                       │
│ ■ Written by open-weighted models: 4200 tokens — never sent to or   │
│   billed by a paid API                                               │
│ ■ Written by the orchestrator (claude-opus-5): 70 tokens             │
│   Orchestrator read: 280 tokens (its instructions, the conversation) │
│   350 orchestrator tokens in all ($0.0200)                           │
╰────────────────────────────────────────────────────────────────────────╯
```

(green segment/marker = open-weighted, yellow = paid orchestrator)

The orchestrator's numbers are real: token counts come from the API
response's own `usage` field, and the dollar cost is computed by LiteLLM's
own pricing table (`litellm.completion_cost`) for whichever model you
actually used — not a guess. The "open-weighted models" figure comes from
Ollama's own `eval_count` for each generation, summed across every subtask
delegated during the run; it's the clearest proxy for "how much work
happened without touching a paid API at all" (and therefore without any
orchestrator tokens/cost for it), even though we don't attach a speculative
dollar figure to what it "would have cost" from the orchestrator, since that
depends on a model/pricing assumption we can't verify.

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

### Choosing an orchestrator model

`setup` always asks explicitly which orchestrator provider
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
`setup` automatically opens your browser straight to that
provider's API key page (Anthropic's Console, OpenAI's Platform dashboard,
or Google AI Studio) so you don't have to go find it, then you paste the
key in as usual.

**The orchestrator doesn't have to be a proprietary API at
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

Before downloading anything, `setup` checks which models
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

During `setup`, model selection isn't purely rule-based: the
catalog is first filtered down to only the models that actually fit this
machine's RAM, VRAM, *and* free disk space (`catalog.candidates()`), and
then the orchestrator itself is asked to pick the best one per modality
from that filtered list (`advisor.recommend_models()`), weighing quality
against how much headroom each choice leaves. The orchestrator can only
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
5. Asks which orchestrator provider to use and for its API key if it needs
   one (saving both to `~/.config/localforge/config.env` so you only enter
   it once), then installs and starts [Ollama](https://ollama.com) if it
   isn't already, and has the orchestrator itself pick which local models to
   pull — see "Hardware-aware model selection" below.

After that, `localforge` just works in any terminal — no repeated setup, no
manual model downloads, no re-exporting API keys. Setup, `doctor` and the
and the first-run setup all end by telling you to type `localforge` to start a session. The
installer then prints a short "what to do next" panel without opening the
session itself, so the script actually finishes. You get the same panel
whenever `localforge` runs without a real terminal attached: one version
if nothing is configured yet, and one pointing at the session once it is.

To upgrade after a new release: just re-run the one-liner or `./install.sh`
— it pulls the latest source and reinstalls.

You can also run pieces individually:
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
localforge setup                                 # one-time interactive setup (plain prompts)
localforge doctor                                # is everything set up correctly?
localforge scan                                  # what hardware do I have?
localforge models                                # what will run well on it?
localforge advanced-model                        # show what's handling coding/docs/general (auto by default)
localforge advanced-model coding qwen2.5-coder:14b            # pin a specific local model for one task type
localforge advanced-model coding api:anthropic:claude-haiku-4-5  # or send it to a paid cloud model instead
localforge advanced-model image api:openai:gpt-image-1        # turn on image generation (a paid image model)
localforge run "Build a todo REST API with docs" # do the thing
localforge run "..." --model gpt-5               # use a different orchestrator model
localforge run "..." --usage                     # also print token usage (in a session: /usage)
localforge installed                             # what local models are actually on disk, and how big
localforge delete                                # pick installed model(s) to delete, review, then confirm
localforge delete qwen2.5-coder:14b --yes        # delete a specific model without prompting
localforge uninstall                             # remove models, config, and the CLI tool (asks first)
localforge uninstall --purge-ollama              # also uninstall Ollama itself and wipe ~/.ollama
localforge theme                                 # show current theme + available options
localforge theme dark                            # switch theme (matrix / dark / light)
localforge desktop                               # hand this session off to the desktop app (see below)
```

**`localforge desktop`** (also `/desktop` in a session): launches the desktop app open to the current folder and ends this terminal session, so you can keep going in the GUI right where you left off — session memory is saved through the same path `/exit` already uses, and the GUI reads it back on start. Needs the desktop app installed already (`cd desktop && npm run tauri build`); `LOCALFORGE_DESKTOP_BIN` points it at a specific binary otherwise. Blocked while a task is running, same as `/clear`/`/uninstall`.

**`localforge advanced-model`** (also `/advanced-model` in a session, and the Advanced mode of the desktop app's Models section): coding/docs/general each independently default to *automatic* — the best-fitting installed local model, same as always, and free. This lets you override one: pin a specific local model from the catalog, or send that task type to a paid cloud model instead — either metered (an API key) or via a CLI subscription login you're already using for the orchestrator. A cloud choice needs a usable key/login already configured (`localforge setup`) and shows up as its own line in `/usage`, kept apart from local (free) and orchestrator spend since it's neither. `localforge advanced-model <type>` alone shows the current choice; `auto` clears a pin.

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
  tagged by modality (`coding`, `docs`, `general`; `judge`, used internally
  to grade other models' output, see "Catching bad local output" below; and
  reserved `image` / `video` entries), hardware requirements, and
  approximate download size; `candidates()` filters to models that fit
  RAM/VRAM/disk, `best_match()` picks the highest-quality one
  deterministically.
- `advisor.py` — lets the orchestrator pick among `candidates()` for each
  modality (via a tool call constrained with a JSON-schema `enum`, so it
  can't hallucinate a model outside the catalog), falling back to
  `best_match()` on any failure.
- `backends/` — one module per serving runtime. `ollama.py` is implemented,
  including a real progress callback fed by Ollama's streaming pull
  response (used to drive an actual download progress bar in `setup`). `comfyui.py` is a stub
  reserved for image/video generation, since those are job-based (submit →
  poll → fetch file) rather than a single request/response like text.
- `tools.py` — turns catalog + backends into tool schemas the orchestrator
  can call, and dispatches each call to the right local model. Also resolves
  the judge model (`Dispatcher.judge()`) and runs it against delegated
  output, in parallel with the syntax check for file writes.
- `orchestrator.py` — the plan → delegate → collect loop, built directly on
  LiteLLM rather than a multi-agent framework, so the delegation logic stays
  simple, provider-agnostic, and easy to step through.
- `config.py` — persists setup choices (API keys, chosen orchestrator model,
  theme) to `~/.config/localforge/config.env`, loaded automatically on
  every CLI invocation without overriding variables already set in the
  shell. `FRONTIER_PROVIDERS["local"]` maps to `None` (no API key) for the
  open-weight-orchestrator option.
- `theme.py` — the three `rich.theme.Theme` definitions (`matrix`/`dark`/`light`)
  behind `localforge theme`, keyed by semantic style names
  (`success`/`error`/`warning`/`accent`) that `cli.py` uses everywhere
  instead of literal color words.
- `cli.py` — the `localforge` command-line entry point (Typer), including
  the `setup` onboarding flow and `doctor` diagnostic.
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
