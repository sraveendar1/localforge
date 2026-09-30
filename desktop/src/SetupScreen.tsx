import { useEffect, useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import type { SetupProvider, SetupResult, SetupStatus } from "./state";

async function openLink(url: string) {
  try {
    await openUrl(url);
  } catch {
    window.open(url, "_blank", "noreferrer");
  }
}

function Link({ url, children }: { url: string; children: React.ReactNode }) {
  return (
    <button type="button" onClick={() => openLink(url)} className="text-mx-green underline hover:text-mx-bright">
      {children}
    </button>
  );
}

function ProviderCard({
  provider,
  busy,
  disabled,
  onKey,
  onLogin,
}: {
  provider: SetupProvider;
  busy: string | null;
  disabled: boolean;
  onKey: (key: string) => void;
  onLogin: () => void;
}) {
  const [key, setKey] = useState("");
  const savingKey = busy === `key:${provider.id}`;
  const checkingLogin = busy === `login:${provider.id}`;
  return (
    <div className="rounded-sm border border-mx-dim bg-mx-panel2 p-3" data-testid={`setup-${provider.id}`}>
      <div className="flex items-center justify-between">
        <span className="font-semibold text-mx-bright">{provider.label}</span>
        {provider.keySet && <span className="text-xs text-mx-green">API key saved ✓</span>}
      </div>

      {provider.cli && (
        <div className="mt-2 text-xs">
          {provider.cli.installed ? (
            <button
              type="button"
              disabled={disabled}
              onClick={onLogin}
              className="rounded-sm border border-mx-mid px-3 py-1 text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:opacity-50"
            >
              {checkingLogin ? "Checking your login…" : `Use my ${provider.cli.command} login`}
            </button>
          ) : (
            <span className="text-mx-dim">
              Have a subscription? The <span className="font-mono">{provider.cli.command}</span> app isn't installed on this computer.{" "}
              <Link url={provider.cli.installHint}>How to get it</Link>
            </span>
          )}
          {provider.cli.installed && (
            <span className="ml-2 text-mx-dim">uses your subscription, no API key needed</span>
          )}
        </div>
      )}

      <form
        className="mt-2 flex items-center gap-2"
        onSubmit={e => {
          e.preventDefault();
          if (key.trim() && !disabled) {
            onKey(key);
            setKey("");
          }
        }}
      >
        <input
          type="password"
          value={key}
          onChange={e => setKey(e.target.value)}
          placeholder={provider.keySet ? "Replace the saved API key…" : "Paste an API key…"}
          aria-label={`${provider.label} API key`}
          autoComplete="off"
          spellCheck={false}
          disabled={disabled}
          className="min-w-0 flex-1 rounded-sm border border-mx-dim bg-mx-panel px-2 py-1 text-xs text-mx-mid disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || !key.trim()}
          className="shrink-0 rounded-sm border border-mx-mid px-3 py-1 text-xs text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim"
        >
          {savingKey ? "Checking…" : "Save key"}
        </button>
      </form>
      {provider.id === "gemini" && (
        <div className="mt-1 text-xs text-mx-dim" data-testid="gemini-credit-note">
          A Google AI Pro subscription doesn't make API calls free, but it includes $10 a month in Google Cloud credits that can pay for
          them once activated (<Link url="https://developers.google.com/program">how</Link>). Gemini image generation has no free tier; you can set a
          monthly limit in Models so it never goes past that.
        </div>
      )}
      {provider.keyUrl && (
        <div className="mt-1 text-xs text-mx-dim">
          No key yet? <Link url={provider.keyUrl}>Get one</Link>. It's checked with {provider.label.split(" ")[0]} and stored only on this computer.
        </div>
      )}
    </div>
  );
}

// First-run setup, so a new user never needs a terminal for `localforge setup`:
// one model plans and reviews the work (the orchestrator), either through a
// provider account (an API key, or the login of a CLI they already use) or a
// local model. Shown while no orchestrator can run; the coding/docs/general
// models are picked automatically and downloaded (asking first) when needed.
export function SetupScreen({
  status,
  result,
  connected,
  send,
  onSkip,
  skipLabel = "Skip for now",
}: {
  status: SetupStatus;
  result: SetupResult | null;
  connected: boolean;
  send: (obj: object) => void;
  onSkip: () => void;
  skipLabel?: string;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  useEffect(() => { setBusy(null); }, [result?.n]);  // an answer arrived: whatever was in flight is done
  const anyBusy = busy !== null || !connected;
  const usable = status.ollama.models.filter(m => !m.problem);

  return (
    <div className="mx-auto flex h-full max-w-xl flex-col gap-4 overflow-y-auto px-6 py-8" data-testid="setup-screen">
      <div>
        <h1 className="text-lg font-semibold uppercase tracking-widest text-mx-bright glow">Set up localforge</h1>
        <p className="mt-2 text-sm text-mx-mid">
          localforge uses one model to plan and review your work. Choose how to sign in to it. The models that write the
          code run on this computer, are picked for you, and are downloaded (after asking you) the first time they're needed.
        </p>
        {status.orchestrator.reason && (
          <p className="mt-2 text-xs text-mx-amber" data-testid="setup-reason">Right now: {status.orchestrator.reason}.</p>
        )}
      </div>

      <div className="space-y-3">
        {status.providers.map(p => (
          <ProviderCard
            key={p.id}
            provider={p}
            busy={busy}
            disabled={anyBusy}
            onKey={key => { setBusy(`key:${p.id}`); send({ type: "save_api_key", provider: p.id, key }); }}
            onLogin={() => { setBusy(`login:${p.id}`); send({ type: "setup_use_login", provider: p.id }); }}
          />
        ))}
      </div>

      <div className="rounded-sm border border-mx-dim bg-mx-panel2 p-3" data-testid="setup-local">
        <div className="font-semibold text-mx-bright">Or run everything on this computer (free)</div>
        {!status.ollama.installed ? (
          <p className="mt-1 text-xs text-mx-dim">
            This needs <Link url="https://ollama.com/download">Ollama</Link>, which isn't installed. Install it, open it once, then come back.
          </p>
        ) : !status.ollama.running ? (
          <p className="mt-1 text-xs text-mx-dim">Ollama is installed but not running. Open the Ollama app (or run <span className="font-mono">ollama serve</span>), then come back.</p>
        ) : usable.length === 0 && status.ollama.models.length === 0 ? (
          <p className="mt-1 text-xs text-mx-dim">
            Ollama is running but has no models yet. In a terminal, <span className="font-mono">ollama pull llama3.1:8b</span> gets a good small one, then come back.
          </p>
        ) : (
          <ul className="mt-2 space-y-1 text-xs">
            {status.ollama.models.map(m => (
              <li key={m.name}>
                <button
                  type="button"
                  disabled={anyBusy || !!m.problem}
                  title={m.problem ?? undefined}
                  onClick={() => { setBusy(`local:${m.name}`); send({ type: "setup_choose_orchestrator", model: `ollama/${m.name}` }); }}
                  className={"rounded-sm border px-3 py-1 " + (m.problem ? "cursor-not-allowed border-mx-dim text-mx-dim line-through" : "border-mx-mid text-mx-green hover:border-mx-bright hover:text-mx-bright") + " disabled:opacity-60"}
                >
                  Use {m.name}
                </button>
                {m.problem && <span className="ml-2 text-mx-red">Can't run here: {m.problem}.</span>}
              </li>
            ))}
            <li className="text-mx-dim">Small local models are often unreliable at planning; a paid model above is the safer choice.</li>
          </ul>
        )}
      </div>

      {result && (
        <p
          role={result.ok ? "status" : "alert"}
          data-testid="setup-result"
          className={"whitespace-pre-wrap rounded-sm border px-3 py-2 text-sm " + (result.ok ? "border-mx-green text-mx-green" : "border-mx-red text-mx-red")}
        >
          {result.message}
        </p>
      )}

      <div className="flex items-center justify-between text-xs text-mx-dim">
        <span>You can change all of this later in Models (left panel).</span>
        <button type="button" onClick={onSkip} className="rounded-sm border border-mx-dim px-3 py-1 text-mx-mid hover:border-mx-mid hover:text-mx-bright">
          {skipLabel}
        </button>
      </div>
    </div>
  );
}
