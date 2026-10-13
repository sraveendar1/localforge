import { useEffect, useRef, useState } from "react";

// Voice input for the message box: press the mic, speak, and the words land in the box (still
// editable, never sent by themselves). It uses the web view's built-in speech recognition, so it
// works where the platform provides one (macOS/Safari-based and Chromium-based web views) and says
// so plainly where it doesn't. The macOS permission strings live in src-tauri/Info.plist.

type Recognition = {
  lang: string; continuous: boolean; interimResults: boolean;
  start(): void; stop(): void; abort(): void;
  onresult: ((e: any) => void) | null; onerror: ((e: any) => void) | null; onend: (() => void) | null;
};

export function speechSupported(): boolean {
  const w = window as any;
  return !!(w.SpeechRecognition || w.webkitSpeechRecognition);
}

// What went wrong, in words, and exactly what to do about it. The web view reports a short code
// ("not-allowed", "service-not-allowed", ...); on macOS two different switches sit behind them (the
// microphone, and the speech recognizer that turns sound into words), so the microphone is checked
// on its own first to tell them apart.
type Problem = { title: string; steps: string; code?: string };

const MAC = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform || navigator.userAgent);
const DICTATION_TIP = MAC ? " Meanwhile you can dictate with macOS itself: press the Fn key twice." : "";

const MIC_BLOCKED: Problem = {
  title: "LocalForge isn't allowed to use the microphone.",
  steps: MAC
    ? "Open System Settings → Privacy & Security → Microphone, switch LocalForge on, then quit and reopen LocalForge (macOS applies it after a restart)."
    : "Allow the microphone for LocalForge in your system's privacy settings, then reopen the app.",
};
const MESSAGES: { [code: string]: Problem } = {
  "not-allowed": {
    title: "Speech recognition isn't allowed for LocalForge.",
    steps: MAC
      ? "Open System Settings → Privacy & Security → Speech Recognition and switch LocalForge on. Also check System Settings → Keyboard → Dictation is On. Then quit and reopen LocalForge."
      : "Allow speech recognition for LocalForge in your system's privacy settings, then reopen the app.",
  },
  "service-not-allowed": {
    title: "The system's speech service is switched off.",
    steps: MAC
      ? "Turn on System Settings → Keyboard → Dictation (and Siri & Spotlight if it asks), then quit and reopen LocalForge."
      : "Turn on your system's speech or dictation service, then try again.",
  },
  "no-speech": { title: "I didn't hear anything.", steps: "Click the microphone again and start speaking right away, a little closer to the microphone." },
  "audio-capture": { title: "No microphone was found.", steps: MAC ? "Check System Settings → Sound → Input shows a microphone, then try again." : "Plug in or enable a microphone, then try again." },
  network: { title: "The speech service couldn't be reached.", steps: "Voice input sends your voice to your system's speech service, which needs an internet connection. Check your connection and try again." },
  "language-not-supported": { title: "Your language isn't supported for voice input here.", steps: "Switch the system language or dictation language to one that supports it, or type instead." },
};
const unknown = (code: string): Problem => ({
  title: "Voice input stopped unexpectedly.",
  steps: "Try again. If it keeps happening, restart LocalForge." + DICTATION_TIP,
  code,
});

export function VoiceButton({ value, onChange, disabled }: { value: string; onChange: (text: string) => void; disabled?: boolean }) {
  const [listening, setListening] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const rec = useRef<Recognition | null>(null);
  const base = useRef("");  // what was already in the box when listening began
  const supported = speechSupported();

  useEffect(() => () => rec.current?.abort(), []);

  function stop() { rec.current?.stop(); }

  function start() {
    setProblem(null);
    const w = window as any;
    const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Ctor) return setProblem({ title: "Voice input isn't available in this window.", steps: "This system's web view has no speech recognition. Type your message" + (MAC ? ", or dictate with macOS itself: press the Fn key twice." : ".") });
    // Check the microphone first, on its own: a refusal here is a microphone problem, not a speech one.
    const md = navigator.mediaDevices;
    if (md?.getUserMedia) {
      md.getUserMedia({ audio: true }).then(stream => { stream.getTracks().forEach(t => t.stop()); listen(Ctor); }, err => {
        const name = String(err?.name ?? "");
        setProblem(name === "NotFoundError" ? MESSAGES["audio-capture"] : name === "NotAllowedError" || name === "SecurityError" ? { ...MIC_BLOCKED, code: name } : unknown(name || "microphone"));
      });
    } else {
      listen(Ctor);
    }
  }

  function listen(Ctor: new () => Recognition) {
    const r: Recognition = new Ctor();
    r.lang = navigator.language || "en-US";
    r.continuous = true;
    r.interimResults = true;
    base.current = value && !/\s$/.test(value) ? value + " " : value;
    r.onresult = e => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      onChange(base.current + text.trimStart());
    };
    r.onerror = e => {
      setListening(false);
      if (e.error === "aborted") return;  // the user (or a restart) stopped it: not a problem
      setProblem(MESSAGES[e.error] ? { ...MESSAGES[e.error], code: e.error } : unknown(String(e.error)));
    };
    r.onend = () => setListening(false);
    rec.current = r;
    try { r.start(); setListening(true); } catch { setProblem({ title: "Voice input couldn't start.", steps: "Wait a second and click the microphone again." }); }
  }

  return (
    <div className="relative self-end">
      <button
        type="button"
        data-testid="voice-button"
        aria-pressed={listening}
        aria-label={listening ? "Stop voice input" : "Start voice input"}
        title={!supported ? "Voice input isn't available in this window" : listening ? "Listening… click to stop" : "Speak your message"}
        disabled={disabled}
        onClick={() => (listening ? stop() : start())}
        className={"rounded-sm border px-3 py-2 text-sm disabled:border-mx-dim disabled:text-mx-muted " + (listening ? "animate-pulse border-mx-red text-mx-red" : "border-mx-dim bg-mx-panel2 text-mx-mid hover:border-mx-mid hover:text-mx-bright")}
      >
        {listening ? "■" : "🎤"}
      </button>
      {problem && (
        <div role="alert" data-testid="voice-problem" className="absolute bottom-full left-0 z-20 mb-1 w-80 rounded-sm border border-mx-amber bg-mx-bg p-2 text-[11px]">
          <div className="font-semibold text-mx-amber" data-testid="voice-problem-title">{problem.title}</div>
          <div className="mt-1 text-mx-mid" data-testid="voice-problem-steps">{problem.steps}</div>
          {problem.code && <div className="mt-1 text-[10px] text-mx-muted">Reason code: {problem.code}</div>}
          <div className="mt-1.5 flex gap-3">
            <button type="button" className="text-mx-green underline hover:text-mx-bright" data-testid="voice-retry" onClick={() => { setProblem(null); start(); }}>Try again</button>
            <button type="button" className="text-mx-muted underline hover:text-mx-bright" onClick={() => setProblem(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
