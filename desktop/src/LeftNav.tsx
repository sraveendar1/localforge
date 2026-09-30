import { Curtain } from "./Curtain";
import { ModelsPanel } from "./ModelsPanel";
import type { DelegateOptions, LocalModelTarget, OrchestratorOption } from "./state";

// A collapsible left rail (chevron-toggled): recently-opened project folders
// (so switching projects doesn't mean the native file dialog every time) and
// the Models section. It used to end with a static command reference, removed
// to give the models the room: typing "/" in the message box already lists
// every command, with descriptions, and completes them.
export function LeftNav({
  open,
  onToggle,
  recentFolders,
  currentFolder,
  onSelectFolder,
  onOpenDialog,
  modelsSignal,
  frontier,
  frontierOptions,
  targets,
  delegateOptions,
  connected,
  source,
  onSetup,
  send,
}: {
  open: boolean;
  onToggle: () => void;
  recentFolders: string[];
  currentFolder: string | null;
  onSelectFolder: (path: string) => void;
  onOpenDialog: () => void;
  modelsSignal: number;  // bumped by the header's model chip to pull the Models section open
  frontier: string;
  frontierOptions: OrchestratorOption[];
  targets: { [modality: string]: LocalModelTarget };
  delegateOptions: { [modality: string]: DelegateOptions };
  connected: boolean;
  source: "project" | "defaults" | null;
  onSetup: () => void;
  send: (obj: object) => void;
}) {
  return (
    <div className="flex h-full shrink-0">
      {open && (
        <nav className="flex w-64 flex-col gap-3 overflow-y-auto border-r border-mx-dim bg-mx-panel p-3 text-xs">
          <Curtain title="Projects">
            {recentFolders.length === 0 ? (
              <p className="text-mx-dim italic">No recent projects yet.</p>
            ) : (
              <ul className="space-y-0.5">
                {recentFolders.map(f => (
                  <li key={f}>
                    <button
                      type="button"
                      title={f}
                      onClick={() => onSelectFolder(f)}
                      className={
                        "block w-full truncate rounded-sm px-1 py-0.5 text-left hover:bg-mx-dim/40 " +
                        (f === currentFolder ? "text-mx-bright" : "text-mx-mid")
                      }
                    >
                      {f.split("/").pop() || f}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <button
              type="button"
              className="mt-2 w-full rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1 text-mx-green hover:border-mx-mid hover:text-mx-bright"
              onClick={onOpenDialog}
            >
              Open folder…
            </button>
          </Curtain>
          <Curtain title="Models" openSignal={modelsSignal}>
            <ModelsPanel
              frontier={frontier}
              frontierOptions={frontierOptions}
              targets={targets}
              delegateOptions={delegateOptions}
              connected={connected}
              source={source}
              onSetup={onSetup}
              send={send}
            />
          </Curtain>
        </nav>
      )}
      <button
        type="button"
        title={open ? "Hide sidebar" : "Show projects & models"}
        onClick={onToggle}
        className="group flex w-6 shrink-0 items-center justify-center border-r border-mx-dim bg-mx-panel2 hover:bg-mx-dim"
      >
        <span className="flex h-14 w-5 items-center justify-center rounded-full border border-mx-mid bg-mx-panel text-base font-bold text-mx-green glow group-hover:border-mx-bright group-hover:text-mx-bright">
          {open ? "‹" : "›"}
        </span>
      </button>
    </div>
  );
}
