// "qwen2.5-coder:7b (auto, local, installed)" -> the model, and the note after it.
export function splitDescription(description: string): { name: string; note: string } {
  const at = description.indexOf(" (");
  if (at < 0 || !description.endsWith(")")) return { name: description, note: "" };
  return { name: description.slice(0, at), note: description.slice(at + 2, -1) };
}
