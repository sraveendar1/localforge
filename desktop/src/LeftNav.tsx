import { SLASH_COMMANDS } from "./SlashMenu";

// A collapsible left rail (chevron-toggled): a static command reference so
// new users don't have to type "/" to discover what's available, and a
// list of recently-opened project folders so switching projects doesn't
// mean going through the native file dialog every time.
export function LeftNav({
  open,
  onToggle,
  recentFolders,
  currentFolder,
  onSelectFolder,
  onOpenDialog,
}: {
  open: boolean;
  onToggle: () => void;
  recentFolders: string[];
  currentFolder: string | null;
  onSelectFolder: (path: string) => void;
  onOpenDialog: () => void;
}) {
  return (
    <div className="flex h-full shrink-0">
      {open && (
        <nav className="flex w-64 flex-col overflow-y-auto border-r border-mx-dim bg-mx-panel p-3 text-xs">
          <section className="mb-4">
            <h2 className="mb-2 border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-bright glow">Projects</h2>
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
          </section>
          <section>
            <h2 className="mb-2 border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-bright glow">Commands</h2>
            <ul className="space-y-1.5">
              {SLASH_COMMANDS.map(c => (
                <li key={c.name}>
                  <div className="font-mono font-semibold text-mx-green">{c.name}</div>
                  <div className="text-mx-dim">{c.description}</div>
                </li>
              ))}
            </ul>
          </section>
        </nav>
      )}
      <button
        type="button"
        title={open ? "Hide sidebar" : "Show projects & commands"}
        onClick={onToggle}
        className="flex w-4 shrink-0 items-center justify-center border-r border-mx-dim bg-mx-panel text-mx-dim hover:bg-mx-panel2 hover:text-mx-bright"
      >
        {open ? "‹" : "›"}
      </button>
    </div>
  );
}
