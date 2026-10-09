import { Curtain } from "./Curtain";
import { ModelsPanel } from "./ModelsPanel";
import { ExpandButton, ResizeHandle } from "./PanelResize";
import type { PanelLayout } from "./PanelResize";
import type { LocalModelTarget } from "./state";

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
  frontier,
  targets,
  connected,
  source,
  onChange,
  onManageAccounts,
  layout,
}: {
  open: boolean;
  onToggle: () => void;
  recentFolders: string[];
  currentFolder: string | null;
  onSelectFolder: (path: string) => void;
  onOpenDialog: () => void;
  frontier: string;
  targets: { [modality: string]: LocalModelTarget };
  connected: boolean;
  source: "project" | "defaults" | null;
  onChange: (which: "frontier" | string | null) => void;
  onManageAccounts: () => void;
  layout: PanelLayout;
}) {
  const { expanded } = layout;
  return (
    <div className={"flex h-full " + (expanded ? "min-w-0 flex-1" : "shrink-0")}>
      {open && (
        <nav
          className={"flex flex-col gap-3 overflow-y-auto border-r border-mx-dim bg-mx-panel p-3 text-xs " + (expanded ? "min-w-0 flex-1" : "")}
          style={expanded ? undefined : { width: layout.width }}
          data-testid="left-panel"
          data-expanded={expanded}
        >
          <div className="flex justify-end"><ExpandButton expanded={expanded} onToggle={layout.onToggleExpand} label="projects and models" /></div>
          <div className={"flex flex-col gap-3 " + (expanded ? "mx-auto w-full max-w-3xl" : "")}>
          <Curtain title="Projects">
            {recentFolders.length === 0 ? (
              <p className="text-mx-dim italic">No recent projects yet.</p>
            ) : (
              <ul className="space-y-0.5">
                {recentFolders.map(f => {
                  const current = f === currentFolder;
                  return (
                    <li key={f}>
                      <button
                        type="button"
                        title={f}
                        aria-current={current ? "true" : undefined}
                        onClick={() => onSelectFolder(f)}
                        className={
                          "block w-full truncate rounded-sm border-l-2 py-1 pl-2 pr-1 text-left hover:bg-mx-dim/40 " +
                          (current ? "border-mx-bright bg-mx-dim/30 text-mx-bright" : "border-transparent text-mx-mid")
                        }
                      >
                        {f.split("/").pop() || f}
                      </button>
                    </li>
                  );
                })}
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
          <Curtain title="Models">
            <ModelsPanel
              frontier={frontier}
              targets={targets}
              connected={connected}
              source={source}
              onChange={onChange}
              onManageAccounts={onManageAccounts}
            />
          </Curtain>
          </div>
        </nav>
      )}
      {!expanded && open && <ResizeHandle side="left" width={layout.width} onWidth={layout.onWidth} />}
      {!expanded && <button
        type="button"
        title={open ? "Hide sidebar" : "Show projects & models"}
        onClick={onToggle}
        className="group flex w-6 shrink-0 items-center justify-center border-r border-mx-dim bg-mx-panel2 hover:bg-mx-dim"
      >
        <span className="flex h-14 w-5 items-center justify-center rounded-full border border-mx-mid bg-mx-panel text-base font-bold text-mx-green glow group-hover:border-mx-bright group-hover:text-mx-bright">
          {open ? "‹" : "›"}
        </span>
      </button>}
    </div>
  );
}
