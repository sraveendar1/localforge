import type { Item } from "./state";
import { ItemView } from "./components";

export function ActivityStream({ items, open, round }: { items: Item[]; open: boolean; round?: number }) {
  if (items.length === 0) return null;

  const toolCount = items.filter(item => item.kind === "tool").length;
  const delegateCount = items.filter(item => item.kind === "delegate").length;

  const labels: string[] = [];
  if (items.length > 1) labels.push(`${items.length} steps`);
  if (toolCount > 0) labels.push(`${toolCount} tool${toolCount > 1 ? 's' : ''}`);
  if (delegateCount > 0) labels.push(`${delegateCount} delegation${delegateCount > 1 ? 's' : ''}`);
  if (round !== undefined) labels.push(`round ${round}`);

  return (
    <details open={open}>
      <summary className="cursor-pointer select-none list-none text-xs text-mx-mid hover:text-mx-bright">
        {labels.join(" · ")}
      </summary>
      <div className="border-l border-mx-dim pl-2 mt-1">
        {items.map((item, i) => <ItemView key={i} item={item} />)}
      </div>
    </details>
  );
}
