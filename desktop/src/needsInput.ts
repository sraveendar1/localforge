// Does the orchestrator's last reply ask the user for something? Then it is shown in orange
// as "Needs your input", so a question is never mistaken for a finished answer.
//
// Deliberately simple: a question mark ending one of the last two non-empty lines, or a closing
// request such as "let me know" / "please confirm". A false positive only colours a reply orange;
// a miss leaves it as plain text, same as before.
const REQUEST = /\b(let me know|please (confirm|specify|clarify|tell me|choose|pick|provide|share)|which (one|option|do you)|would you like|do you want|should i)\b/i;

export function asksUser(text: string): boolean {
  const lines = (text ?? "").split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) return false;
  const tail = lines.slice(-2);
  if (tail.some(l => /\?["')\]*_\s]*$/.test(l))) return true;
  return REQUEST.test(tail.join(" "));
}
