import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

type State = "idle" | "capturing" | "transcribing" | "injecting" | "error";

interface StatusEvent {
  state: State;
  detail?: string;
}

const LABELS: Record<State, string> = {
  idle: "Ready",
  capturing: "Listening",
  transcribing: "Transcribing",
  injecting: "Inserting",
  error: "Error",
};

const pill = document.getElementById("pill")!;
const label = document.getElementById("label")!;
const bars = Array.from(document.querySelectorAll<HTMLSpanElement>("#bars span"));
const overlay = getCurrentWindow();

/** Per-bar gain so one level still reads as a spectrum, not five clones. */
const BAR_WEIGHTS = [0.55, 0.85, 1, 0.8, 0.6];
const BAR_MIN = 4;
const BAR_MAX = 18;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let hideTimer: number | undefined;

/** Hand the bars back to the stylesheet (idle, transcribing pulse, etc.). */
function releaseBars() {
  delete pill.dataset.live;
  for (const bar of bars) bar.style.height = "";
}

await listen<StatusEvent>("status", async ({ payload }) => {
  pill.dataset.state = payload.state;
  label.textContent = payload.detail ?? LABELS[payload.state];
  releaseBars();

  clearTimeout(hideTimer);
  if (payload.state === "idle" || payload.state === "error") {
    // Linger briefly so the user sees the result, then get out of the way.
    hideTimer = setTimeout(() => overlay.hide(), payload.state === "error" ? 3000 : 1200);
  } else {
    await overlay.show();
  }
});

// Microphone level while capturing. The CSS equaliser keeps running until the
// first level arrives, then the bars follow the voice: level times a per-bar
// weight, plus a little jitter so a held vowel still moves.
await listen<number>("level", ({ payload }) => {
  if (pill.dataset.state !== "capturing" || reduceMotion) return;
  pill.dataset.live = "";
  bars.forEach((bar, i) => {
    const jitter = 0.85 + Math.random() * 0.3;
    const height = BAR_MIN + payload * (BAR_MAX - BAR_MIN) * BAR_WEIGHTS[i] * jitter;
    bar.style.height = `${Math.min(BAR_MAX, height).toFixed(1)}px`;
  });
});
