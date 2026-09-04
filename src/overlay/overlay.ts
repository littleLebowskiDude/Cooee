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
const overlay = getCurrentWindow();

let hideTimer: number | undefined;

await listen<StatusEvent>("status", async ({ payload }) => {
  pill.dataset.state = payload.state;
  label.textContent = payload.detail ?? LABELS[payload.state];

  clearTimeout(hideTimer);
  if (payload.state === "idle" || payload.state === "error") {
    // Linger briefly so the user sees the result, then get out of the way.
    hideTimer = setTimeout(() => overlay.hide(), payload.state === "error" ? 3000 : 1200);
  } else {
    await overlay.show();
  }
});
