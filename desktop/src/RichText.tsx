import { useState } from "react";
import type { ReactNode } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";

// Chat text with the three things people need to act on it: links you can click (they open in
// the system browser -- inside the app window a plain <a> would navigate the app itself away),
// code blocks you can copy in one click, and `inline code` you can click to copy too.
// Only what's needed is parsed: ``` fences, [text](url), bare http(s) URLs, `code` and **bold**.

export async function openLink(url: string): Promise<void> {
  try {
    await openUrl(url);
  } catch {
    window.open(url, "_blank", "noreferrer");
  }
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // The clipboard API can be unavailable or refused; fall back to a hidden textarea.
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

export type Block = { kind: "text"; text: string } | { kind: "code"; text: string; lang: string };

// Split into prose and fenced code. A fence that is never closed (a reply still streaming in)
// is treated as code to the end, so a command being written is already copyable.
export function splitBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.split("\n");
  let prose: string[] = [];
  let code: string[] | null = null;
  let lang = "";
  const flushProse = () => { if (prose.length) blocks.push({ kind: "text", text: prose.join("\n") }); prose = []; };
  for (const line of lines) {
    const fence = /^\s*```\s*([\w+-]*)\s*$/.exec(line);
    if (fence && code === null) { flushProse(); code = []; lang = fence[1]; continue; }
    if (fence && code !== null) { blocks.push({ kind: "code", text: code.join("\n"), lang }); code = null; continue; }
    (code ?? prose).push(line);
  }
  if (code !== null) blocks.push({ kind: "code", text: code.join("\n"), lang });
  flushProse();
  return blocks;
}

const TOKEN = /(\[[^\]\n]+\]\((?:https?:\/\/|mailto:)[^)\s]+\)|https?:\/\/[^\s<>"')\]]+|`[^`\n]+`|\*\*[^*\n]+\*\*)/g;
const TRAILING = /[.,;:!?]+$/;

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [state, setState] = useState<"idle" | "done" | "failed">("idle");
  return (
    <button
      type="button"
      data-testid="copy-button"
      onClick={async () => {
        setState((await copyText(text)) ? "done" : "failed");
        setTimeout(() => setState("idle"), 1500);
      }}
      className="rounded-sm border border-mx-dim bg-mx-panel px-2 py-0.5 text-[11px] text-mx-mid hover:border-mx-mid hover:text-mx-bright"
    >
      {state === "done" ? "Copied ✓" : state === "failed" ? "Couldn't copy" : label}
    </button>
  );
}

function InlineCode({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <code
      title="Click to copy"
      data-testid="inline-code"
      onClick={async () => { if (await copyText(text)) { setCopied(true); setTimeout(() => setCopied(false), 1200); } }}
      className="cursor-copy rounded-sm border border-mx-dim bg-mx-panel2 px-1 font-mono text-[0.9em] text-mx-bright hover:border-mx-mid"
    >
      {copied ? "copied ✓" : text}
    </code>
  );
}

export function renderInline(line: string, keyBase = 0): ReactNode[] {
  const out: ReactNode[] = [];
  line.split(TOKEN).forEach((part, i) => {
    const key = `${keyBase}-${i}`;
    if (!part) return;
    const md = /^\[([^\]]+)\]\(((?:https?:\/\/|mailto:)[^)\s]+)\)$/.exec(part);
    if (md) {
      out.push(<ExternalLink key={key} url={md[2]}>{md[1]}</ExternalLink>);
    } else if (/^https?:\/\//.test(part)) {
      const trail = TRAILING.exec(part)?.[0] ?? "";
      const url = trail ? part.slice(0, -trail.length) : part;
      out.push(<ExternalLink key={key} url={url}>{url}</ExternalLink>, trail);
    } else if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      out.push(<InlineCode key={key} text={part.slice(1, -1)} />);
    } else if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      out.push(<strong key={key} className="text-mx-bright">{part.slice(2, -2)}</strong>);
    } else {
      out.push(part);
    }
  });
  return out;
}

export function ExternalLink({ url, children }: { url: string; children: ReactNode }) {
  return (
    <a
      href={url}
      title={url}
      rel="noreferrer noopener"
      target="_blank"
      className="break-all text-mx-green underline hover:text-mx-bright"
      onClick={e => { e.preventDefault(); void openLink(url); }}
    >
      {children}
    </a>
  );
}

export function CodeBlock({ text, lang }: { text: string; lang: string }) {
  return (
    <div className="my-2 overflow-hidden rounded-sm border border-mx-dim bg-black" data-testid="code-block">
      <div className="flex items-center justify-between border-b border-mx-dim bg-mx-panel2 px-2 py-0.5">
        <span className="text-[11px] uppercase tracking-wide text-mx-dim">{lang || "code"}</span>
        <CopyButton text={text} />
      </div>
      <pre className="overflow-x-auto p-2 font-mono text-xs leading-snug text-mx-bright"><code>{text}</code></pre>
    </div>
  );
}

export function RichText({ text, className = "" }: { text: string; className?: string }) {
  return (
    <div className={className}>
      {splitBlocks(text).map((b, i) =>
        b.kind === "code" ? (
          <CodeBlock key={i} text={b.text} lang={b.lang} />
        ) : (
          <div key={i} className="whitespace-pre-wrap">{b.text.split("\n").map((line, j, all) => (
            <span key={j}>{renderInline(line, j)}{j < all.length - 1 ? "\n" : ""}</span>
          ))}</div>
        ),
      )}
    </div>
  );
}
