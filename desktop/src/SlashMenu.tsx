// Mirrors StdioServer.handle_help_command()'s command list (src/localforge/serve.py)
// so the chat input can suggest commands before the user hits Enter and finds
// out from the server. Keep in sync if a command is added there.
export const SLASH_COMMANDS: { name: string; description: string }[] = [
  { name: "/memory", description: "Show or clear this folder's memory" },
  { name: "/scratch", description: "List or clear this session's scratchpad" },
  { name: "/queue", description: "Show or clear the task queue" },
  { name: "/stop", description: "Stop the running task" },
  { name: "/clear", description: "Start a new session" },
  { name: "/usage", description: "Show token usage for the last task and this session" },
  { name: "/models", description: "Show best-fit local model per modality" },
  { name: "/installed", description: "Show installed models" },
  { name: "/catalog", description: "Show full model catalog" },
  { name: "/doctor", description: "Check everything's configured correctly" },
  { name: "/scan", description: "Show detected hardware" },
  { name: "/model", description: "Show or switch the orchestrator model" },
  { name: "/auto", description: "Approve file changes and commands without asking (on|off)" },
  { name: "/run", description: "Work on a task in this folder" },
  { name: "/help", description: "Show this list of commands" },
  { name: "/compact", description: "Compact the conversation history" },
  { name: "/summary", description: "Show the session summary" },
  { name: "/tell", description: "Append a note for the running task" },
  { name: "/why", description: "Explain the pending approval request" },
  { name: "/stream", description: "Toggle stream output on or off (on|off)" },
];

export function matchSlashCommands(input: string): { name: string; description: string }[] {
  if (!input.startsWith("/") || input.includes(" ")) return [];
  const needle = input.toLowerCase();
  return SLASH_COMMANDS.filter(c => c.name.startsWith(needle));
}

export function SlashMenu({
  matches,
  activeIndex,
  onPick,
}: {
  matches: { name: string; description: string }[];
  activeIndex: number;
  onPick: (name: string) => void;
}) {
  if (matches.length === 0) return null;
  return (
    <div className="mx-3 mb-1 max-h-48 overflow-y-auto rounded-sm border border-mx-dim bg-mx-panel2 text-xs">
      {matches.map((c, i) => (
        <div
          key={c.name}
          className={"flex cursor-pointer items-center gap-2 px-2 py-1 " + (i === activeIndex ? "bg-mx-dim text-mx-bright" : "text-mx-mid hover:bg-mx-dim/50")}
          onMouseDown={e => { e.preventDefault(); onPick(c.name); }}
        >
          <span className="font-mono font-semibold text-mx-green">{c.name}</span>
          <span className="truncate text-mx-dim">{c.description}</span>
        </div>
      ))}
    </div>
  );
}
