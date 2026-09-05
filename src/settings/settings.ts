import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Strategy = "auto" | "paste" | "type";

/** Mirrors `Config` in config.rs. Fields not listed here still round-trip:
 *  the object comes back from `get_config` whole and is sent back whole. */
interface Config {
  hotkey: number[];
  model_path: string | null;
  injection: Strategy;
  dictionary: Record<string, string>;
  audio_feedback: boolean;
  asr_threads: number | null;
}

/** Mirrors `EngineInfo` in asr/mod.rs. */
interface EngineInfo {
  state: "loading" | "ready" | "failed";
  engine: string | null;
  model: string | null;
  error?: string;
}

/** Mirrors `history::Entry`. */
interface HistoryEntry {
  id: number;
  text: string;
  inference_ms: number;
  elapsed_ms: number;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- Views: history by default, settings behind the cog ------------------------

const historyView = $<HTMLDivElement>("history-view");
const settingsView = $<HTMLDivElement>("settings-view");

function showSettings(on: boolean) {
  historyView.hidden = on;
  settingsView.hidden = !on;
  window.scrollTo(0, 0);
}

$("cog").onclick = () => showSettings(settingsView.hidden);
$("done").onclick = () => showSettings(false);

// ---- History -------------------------------------------------------------------

const historyList = $<HTMLUListElement>("history");
let history: HistoryEntry[] = await invoke<HistoryEntry[]>("get_history");

function when(id: number): string {
  const d = new Date(id);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return sameDay ? time : `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

function renderHistory() {
  historyList.innerHTML = "";
  if (history.length === 0) {
    historyList.innerHTML = `<li class="empty">Nothing dictated yet. Hold the hotkey and speak.</li>`;
    return;
  }
  for (const entry of history) {
    const li = document.createElement("li");
    const text = document.createElement("p");
    text.className = "text";
    text.textContent = entry.text;
    const meta = document.createElement("div");
    meta.className = "meta";
    const stamp = document.createElement("span");
    stamp.textContent = `${when(entry.id)} · ${entry.text.length} chars · ${entry.elapsed_ms} ms`;
    const copy = document.createElement("button");
    copy.textContent = "Copy";
    copy.onclick = async () => {
      await invoke("copy_text", { text: entry.text });
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1500);
    };
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.onclick = async () => {
      await invoke("delete_history", { id: entry.id });
      history = history.filter((e) => e.id !== entry.id);
      renderHistory();
    };
    meta.append(stamp, copy, del);
    li.append(text, meta);
    historyList.append(li);
  }
}

$("clear-history").onclick = async () => {
  if (history.length === 0) return;
  await invoke("clear_history");
  history = [];
  renderHistory();
};

await listen<HistoryEntry>("history", ({ payload }) => {
  history = [payload, ...history.filter((e) => e.id !== payload.id)];
  renderHistory();
});

renderHistory();

const presetSelect = $<HTMLSelectElement>("hotkey-preset");
const customField = $<HTMLDivElement>("hotkey-custom");
const captureInput = $<HTMLInputElement>("hotkey");
const feedbackBox = $<HTMLInputElement>("feedback");
const engineEl = $<HTMLParagraphElement>("engine");
const modelInput = $<HTMLInputElement>("model");
const threadsInput = $<HTMLInputElement>("threads");
const injectionSelect = $<HTMLSelectElement>("injection");
const dictList = $<HTMLUListElement>("dict");
const statusEl = $<HTMLSpanElement>("status");

let config: Config = await invoke<Config>("get_config");

// ---- Engine status -----------------------------------------------------------

const basename = (path: string) => path.split(/[\\/]/).pop() ?? path;

function renderEngine(info: EngineInfo) {
  engineEl.dataset.state = info.state;
  const file = info.model ? basename(info.model) : null;
  switch (info.state) {
    case "loading":
      engineEl.textContent = file ? `Loading ${file}…` : "Starting…";
      break;
    case "ready":
      engineEl.textContent = file
        ? `${info.engine} · ${file}`
        : `${info.engine} engine — no model chosen, so nothing is transcribed`;
      break;
    case "failed":
      engineEl.textContent = `Could not load ${file ?? "model"}: ${info.error ?? "unknown error"}`;
      break;
  }
}

await listen<EngineInfo>("engine", ({ payload }) => renderEngine(payload));
renderEngine(await invoke<EngineInfo>("engine_info"));

$("pick").onclick = async () => {
  const picked = await invoke<string | null>("pick_model");
  if (picked) {
    config.model_path = picked;
    render();
  }
};

$("pick-dir").onclick = async () => {
  const picked = await invoke<string | null>("pick_model_dir");
  if (picked) {
    config.model_path = picked;
    render();
  }
};

$("clear").onclick = () => {
  config.model_path = null;
  render();
};

threadsInput.onchange = () => {
  const n = parseInt(threadsInput.value, 10);
  config.asr_threads = Number.isFinite(n) && n > 0 ? n : null;
  render();
};

// ---- Virtual-key codes -------------------------------------------------------

const VK = {
  SHIFT: 0x10, CONTROL: 0x11, MENU: 0x12, CAPITAL: 0x14,
  LWIN: 0x5b, RWIN: 0x5c,
  LSHIFT: 0xa0, RSHIFT: 0xa1, LCONTROL: 0xa2, RCONTROL: 0xa3, LMENU: 0xa4, RMENU: 0xa5,
} as const;

/** Mirrors `key_name` in hotkey.rs. */
const VK_NAMES: Record<number, string> = {
  0x08: "Backspace", 0x09: "Tab", 0x0d: "Enter",
  0x10: "Shift", 0x11: "Ctrl", 0x12: "Alt", 0x13: "Pause", 0x14: "Caps Lock",
  0x1b: "Esc", 0x20: "Space",
  0x5b: "Win", 0x5c: "Right Win", 0x5d: "Menu",
  0x90: "Num Lock", 0x91: "Scroll Lock",
  0xa0: "Left Shift", 0xa1: "Right Shift",
  0xa2: "Left Ctrl", 0xa3: "Right Ctrl",
  0xa4: "Left Alt", 0xa5: "Right Alt",
};

function vkName(vk: number): string {
  if (VK_NAMES[vk]) return VK_NAMES[vk];
  if ((vk >= 0x30 && vk <= 0x39) || (vk >= 0x41 && vk <= 0x5a)) return String.fromCharCode(vk);
  if (vk >= 0x70 && vk <= 0x87) return `F${vk - 0x6f}`;
  return `0x${vk.toString(16).toUpperCase()}`;
}

const chordLabel = (keys: number[]) => (keys.length ? keys.map(vkName).join("+") : "nothing (disabled)");
const sameChord = (a: number[], b: number[]) => a.length === b.length && a.every((k, i) => k === b[i]);

/** Chords offered directly. Anything else shows as "Custom". */
const PRESETS: Record<string, number[]> = {
  "ctrl-win": [VK.CONTROL, VK.LWIN],
  "caps-lock": [VK.CAPITAL],
  "right-ctrl": [VK.RCONTROL],
  "right-shift": [VK.RSHIFT],
};

function presetFor(keys: number[]): string {
  return Object.keys(PRESETS).find((id) => sameChord(PRESETS[id], keys)) ?? "custom";
}

// ---- Hotkey capture ----------------------------------------------------------

/** Browser keydown -> VK code, keeping which side a modifier was pressed on. */
function vkFromEvent(e: KeyboardEvent): number {
  const right = e.location === KeyboardEvent.DOM_KEY_LOCATION_RIGHT;
  switch (e.keyCode) {
    case VK.SHIFT: return right ? VK.RSHIFT : VK.LSHIFT;
    case VK.CONTROL: return right ? VK.RCONTROL : VK.LCONTROL;
    case VK.MENU: return right ? VK.RMENU : VK.LMENU;
    case VK.LWIN: return right ? VK.RWIN : VK.LWIN;
    default: return e.keyCode;
  }
}

/** In a chord a modifier should work from either side; alone it keeps its side
 *  so "Right Ctrl" does not silently become "any Ctrl" and fire on every Ctrl+C. */
function normaliseChord(keys: number[]): number[] {
  if (keys.length < 2) return keys;
  const generic: Record<number, number> = {
    [VK.LSHIFT]: VK.SHIFT, [VK.RSHIFT]: VK.SHIFT,
    [VK.LCONTROL]: VK.CONTROL, [VK.RCONTROL]: VK.CONTROL,
    [VK.LMENU]: VK.MENU, [VK.RMENU]: VK.MENU,
  };
  return [...new Set(keys.map((k) => generic[k] ?? k))];
}

let capturing: number[] | null = null;

function stopCapture() {
  capturing = null;
  window.removeEventListener("keydown", onCaptureDown, true);
  window.removeEventListener("keyup", onCaptureUp, true);
  captureInput.blur();
  render();
}

function onCaptureDown(e: KeyboardEvent) {
  e.preventDefault();
  if (!capturing) return;
  if (e.key === "Escape") {
    stopCapture();
    return;
  }
  const vk = vkFromEvent(e);
  if (!capturing.includes(vk)) capturing.push(vk);
  captureInput.value = chordLabel(capturing);
}

/** The first key-up ends the capture: whatever was held together is the chord. */
function onCaptureUp(e: KeyboardEvent) {
  e.preventDefault();
  if (!capturing || capturing.length === 0) return;
  config.hotkey = normaliseChord(capturing);
  stopCapture();
}

captureInput.onfocus = () => {
  capturing = [];
  captureInput.value = "Hold the keys, then release…";
  window.addEventListener("keydown", onCaptureDown, true);
  window.addEventListener("keyup", onCaptureUp, true);
};
captureInput.onblur = () => {
  if (capturing) stopCapture();
};

presetSelect.onchange = () => {
  const id = presetSelect.value;
  if (id === "custom") {
    customField.hidden = false;
    captureInput.focus();
    return;
  }
  config.hotkey = [...PRESETS[id]];
  render();
};

// ---- Everything else ---------------------------------------------------------

function render() {
  const preset = presetFor(config.hotkey);
  presetSelect.value = preset;
  customField.hidden = preset !== "custom";
  captureInput.value = chordLabel(config.hotkey);
  feedbackBox.checked = config.audio_feedback;
  modelInput.value = config.model_path ?? "";
  threadsInput.value = config.asr_threads?.toString() ?? "";
  injectionSelect.value = config.injection;

  dictList.innerHTML = "";
  const entries = Object.entries(config.dictionary ?? {});
  if (entries.length === 0) {
    dictList.innerHTML = `<li class="empty">No corrections yet.</li>`;
    return;
  }
  for (const [heard, want] of entries) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${heard}</span><span class="arrow">→</span><strong>${want}</strong>`;
    const del = document.createElement("button");
    del.textContent = "Remove";
    del.style.marginLeft = "auto";
    del.onclick = () => {
      delete config.dictionary[heard];
      render();
    };
    li.append(del);
    dictList.append(li);
  }
}

feedbackBox.onchange = () => {
  config.audio_feedback = feedbackBox.checked;
};

injectionSelect.onchange = () => {
  config.injection = injectionSelect.value as Strategy;
};

$("add").onclick = () => {
  const heard = $<HTMLInputElement>("heard").value.trim().toLowerCase();
  const want = $<HTMLInputElement>("want").value.trim();
  if (!heard || !want) return;
  config.dictionary = { ...config.dictionary, [heard]: want };
  $<HTMLInputElement>("heard").value = "";
  $<HTMLInputElement>("want").value = "";
  render();
};

$("test").onclick = async () => {
  statusEl.textContent = "Inserting in 3s — focus another app…";
  setTimeout(async () => {
    await invoke("test_injection", { text: "Cooee is wired up correctly." });
    statusEl.textContent = "Sent.";
  }, 3000);
};

$("save").onclick = async () => {
  try {
    await invoke("set_config", { new: config });
    statusEl.textContent = `Saved. Hold ${chordLabel(config.hotkey)} to dictate.`;
  } catch (e) {
    statusEl.textContent = `Failed: ${e}`;
  }
};

render();
