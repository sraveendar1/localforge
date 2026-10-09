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

const MESSAGES: { [code: string]: string } = {
  "not-allowed": "Microphone access was blocked. Allow it for LocalForge in your system's privacy settings, then try again.",
  "service-not-allowed": "Speech recognition isn't allowed for LocalForge. Turn it on in your system's privacy settings.",
  "no-speech": "Didn't hear anything. Try again a little closer to the microphone.",
  "audio-capture": "No microphone was found.",
  network: "Speech recognition needs a connection on this system and couldn't reach its service.",
};

export function VoiceButton({ value, onChange, disabled }: { value: string; onChange: (text: string) => void; disabled?: boolean }) {
  const [listening, setListening] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const rec = useRef<Recognition | null>(null);
  const base = useRef("");  // what was already in the box when listening began
  const supported = speechSupported();

  useEffect(() => () => rec.current?.abort(), []);

  function stop() { rec.current?.stop(); }

  function start() {
    setProblem(null);
    const w = window as any;
    const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Ctor) return setProblem("Voice input isn't available in this window on your system. You can still dictate with your system's own dictation (macOS: press Fn twice).");
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
    r.onerror = e => { setProblem(MESSAGES[e.error] ?? `Voice input stopped (${e.error}).`); setListening(false); };
    r.onend = () => setListening(false);
    rec.current = r;
    try { r.start(); setListening(true); } catch { setProblem("Voice input couldn't start. Try again."); }
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
        className={"rounded-sm border px-3 py-2 text-sm disabled:border-mx-dim disabled:text-mx-dim " + (listening ? "animate-pulse border-mx-red text-mx-red" : "border-mx-dim bg-mx-panel2 text-mx-mid hover:border-mx-mid hover:text-mx-bright")}
      >
        {listening ? "■" : "🎤"}
      </button>
      {problem && (
        <p role="alert" data-testid="voice-problem" className="absolute bottom-full left-0 mb-1 w-64 rounded-sm border border-mx-amber bg-mx-bg p-2 text-[11px] text-mx-amber">
          {problem}
          <button type="button" className="ml-2 underline" onClick={() => setProblem(null)}>OK</button>
        </p>
      )}
    </div>
  );
}
