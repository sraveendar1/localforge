import type { MemoryState, ScratchFile } from "./state";

export function MemoryPanel({ memory, scratchFiles, onForget, onClearMemory, onClearScratch }: { memory: MemoryState; scratchFiles: ScratchFile[]; onForget: (name: string) => void; onClearMemory: () => void; onClearScratch: () => void }) {
  return (
    <section className="w-full space-y-2 text-xs">
      <h2 className="uppercase tracking-wide text-mx-bright border-b border-mx-dim pb-1">Memory</h2>
      {memory.narrative && <p className="text-mx-mid">{memory.narrative}</p>}
      <div className="text-mx-mid">
        {memory.facts.length > 0 ? (
          memory.facts.map((fact, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-mx-accent">{fact.name}</span>
              <span className="text-mx-mid">{fact.description ? fact.description : fact.content ? fact.content.substring(0, 50) + (fact.content.length > 50 ? "…" : "") : "No description"}</span>
              <button className="text-mx-red" onClick={() => onForget(fact.name)}>Forget</button>
            </div>
          ))
        ) : (
          <span className="text-mx-mid">No memories</span>
        )}
      </div>
      <h2 className="uppercase tracking-wide text-mx-bright border-b border-mx-dim pb-1">Scratch</h2>
      <div className="text-mx-mid">
        {scratchFiles.length > 0 ? (
          scratchFiles.map((file, i) => (
            <div key={i} className="flex items-center gap-2">
              <span>{file.path}</span>
              <span className="text-mx-mid">{(file.size / 1024).toFixed(1)} KB</span>
            </div>
          ))
        ) : (
          <span className="text-mx-mid">Empty</span>
        )}
      </div>
      <button className="text-mx-red" onClick={onClearMemory}>Clear Memory</button>
      <button className="text-mx-red" onClick={onClearScratch}>Clear Scratch</button>
    </section>
  );
}
