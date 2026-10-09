import { useEffect } from "react";
import { ChecksList } from "./SetupWidgets";
import type { SetupChecks } from "./state";

// What the empty chat shows (Paperclip's "first task" idea): what is ready, what isn't and the one
// click that fixes it, then a few things to ask for. The model wizard only appears for a project
// with no saved models; this stays useful afterwards, every time a fresh conversation opens.
const EXAMPLES: [string, string][] = [
  ["Explain this project", "Explain what this project does and how it is organised."],
  ["Find bugs", "Review the code in this folder and list the bugs or risky spots you find, most serious first. Don't change anything yet."],
  ["Add tests", "Add unit tests for the most important logic in this project and tell me how to run them."],
  ["Write a README", "Write a clear README for this project: what it is, how to install it, how to run it."],
];

export function GettingStarted({
  folderName,
  checks,
  connected,
  onRequestChecks,
  onFix,
  onExample,
  onOpenWizard,
}: {
  folderName: string;
  checks: SetupChecks | null;
  connected: boolean;
  onRequestChecks: () => void;
  onFix: (fix: "planner" | "writers") => void;
  onExample: (text: string) => void;
  onOpenWizard: () => void;
}) {
  useEffect(() => { if (connected) onRequestChecks(); }, [connected]);  // eslint-disable-line react-hooks/exhaustive-deps
  const ready = !!checks && checks.ok;
  return (
    <div className="mx-auto flex h-full max-w-xl flex-col justify-center gap-5 px-6 py-6" data-testid="getting-started">
      <div className="text-center">
        <h1 className="text-sm font-semibold uppercase tracking-widest text-mx-bright glow">{ready ? "Ready when you are" : "Let's get you building"}</h1>
        <p className="mt-1 text-xs text-mx-dim">
          In <span className="font-mono text-mx-mid">{folderName}</span>: a paid model plans the work, free local models write it, and you approve every plan before anything is built.
        </p>
      </div>

      <section aria-label="Setup status" className="rounded-sm border border-mx-dim bg-mx-panel2 p-3">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-[11px] uppercase tracking-wide text-mx-mid">Setup</h2>
          <div className="flex gap-1.5">
            <button type="button" onClick={onRequestChecks} className="text-[11px] text-mx-dim underline hover:text-mx-bright">Re-check</button>
            <button type="button" onClick={onOpenWizard} className="text-[11px] text-mx-green underline hover:text-mx-bright" data-testid="gs-models">Change models…</button>
          </div>
        </div>
        <ChecksList checks={checks} onFix={onFix} />
      </section>

      <section aria-label="Things to try">
        <h2 className="mb-1.5 text-center text-[11px] uppercase tracking-wide text-mx-mid">Try asking</h2>
        <div className="flex flex-wrap justify-center gap-2" data-testid="examples">
          {EXAMPLES.map(([label, text]) => (
            <button
              key={label}
              type="button"
              title={text}
              onClick={() => onExample(text)}
              className="rounded-full border border-mx-dim bg-mx-panel2 px-3 py-1 text-xs text-mx-green hover:border-mx-mid hover:text-mx-bright"
            >
              {label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-center text-[11px] text-mx-dim">Or describe what you want in your own words below. Type <span className="font-mono text-mx-mid">/</span> for commands.</p>
      </section>
    </div>
  );
}
