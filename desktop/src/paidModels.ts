import type { LocalModelTarget, OrchestratorOption } from "./state";

// The paid (cloud) models actually in play for this project, and what each is
// used for -- e.g. "claude-opus-5: orchestrator, coding" and "gpt-image-1:
// image". The right-hand panels list these instead of one "frontier model",
// because the orchestrator is no longer the only paid model: any task type can
// be sent to a paid one, and image generation always is.
export type PaidModel = { name: string; roles: string[]; how: string };

const TASK_ORDER = ["coding", "docs", "general", "image", "video"];

function howLabel(kind: string, provider: string): string {
  const p = provider ? provider[0].toUpperCase() + provider.slice(1) : "";
  return `${p} ${kind === "cli" ? "login" : "API key"}`.trim();
}

// "api:openai:gpt-image-1" -> { kind: "api", provider: "openai", model: "gpt-image-1" }
export function parseCloudTarget(target: string): { kind: string; provider: string; model: string } | null {
  const [kind, provider, ...rest] = target.split(":");
  if ((kind !== "api" && kind !== "cli") || !provider || rest.length === 0) return null;
  return { kind, provider, model: rest.join(":") };
}

export function configuredPaidModels(
  orchestrator: string,
  options: OrchestratorOption[],
  targets: { [modality: string]: LocalModelTarget },
): PaidModel[] {
  const byName = new Map<string, PaidModel>();
  const add = (name: string, role: string, how: string) => {
    const existing = byName.get(name);
    if (existing) {
      if (!existing.roles.includes(role)) existing.roles.push(role);
    } else {
      byName.set(name, { name, roles: [role], how });
    }
  };
  // The orchestrator is paid unless it's an open-weight model on this machine.
  if (orchestrator && !orchestrator.startsWith("ollama/") && !orchestrator.startsWith("ollama_chat/")) {
    const group = options.find(o => o.id === orchestrator)?.group ?? "";
    add(orchestrator, "orchestrator", group.replace(/\s*\(/, ", ").replace(/\)$/, ""));
  }
  for (const modality of TASK_ORDER) {
    const t = targets[modality];
    const cloud = t ? parseCloudTarget(t.target) : null;
    if (cloud) add(cloud.model, modality, howLabel(cloud.kind, cloud.provider));
  }
  return [...byName.values()];
}
