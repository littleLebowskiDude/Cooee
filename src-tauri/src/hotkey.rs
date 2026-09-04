//! Global push-to-talk via a low-level keyboard hook.
//!
//! # The rule that governs this file
//!
//! `WH_KEYBOARD_LL` callbacks run on the thread that installed the hook, and
//! Windows *silently removes* any hook whose callback exceeds
//! `LowLevelHooksTimeout` (~300 ms). When that happens the app appears to work
//! and then stops responding to the hotkey with no error anywhere.
//!
//! So `hook_proc` does the minimum: a few atomic loads, at most one
//! `GetAsyncKeyState` per chord key, a `try_send`, and return. It never
//! allocates, locks, logs, or touches the pipeline. All work happens on the
//! consumer side.
//!
//! # Chords
//!
//! A [`Hotkey`] is a set of virtual-key codes that must all be held together.
//! The default is Ctrl+Win, which is what Wispr Flow uses on Windows and which
//! exists on every laptop — Right Ctrl, the original default, does not.
//!
//! The chord *engages* on the key-down that completes it (every other key in
//! the chord already down, checked against the system key state) and
//! *disengages* on the first key-up of any chord key. Generic modifier codes
//! (`VK_CONTROL`, `VK_SHIFT`, `VK_MENU`) match either side; side-specific codes
//! match only that side, so a single Right Ctrl still works as it did.
//!
//! # Swallowing
//!
//! Some keys do something on their own when pressed and released with nothing
//! in between: Win opens Start, Alt focuses the menu bar, Caps Lock toggles.
//! When one of those keys is the one that completes the chord, the hook eats
//! its down *and* its later up (returning 1 instead of passing the event on),
//! so Windows never sees the key at all. When the chord is completed by some
//! other key, nothing is swallowed: Windows already saw the side-effect key go
//! down, and because a second key was pressed during its hold, its solo action
//! is suppressed anyway. Both press orders of Ctrl+Win therefore leave the
//! Start menu closed.

use crossbeam_channel::{Sender, TrySendError};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    Pressed,
    Released,
}

/// Windows virtual-key codes used by the hook and the settings UI.
pub mod vk {
    pub const SHIFT: u32 = 0x10;
    pub const CONTROL: u32 = 0x11;
    pub const MENU: u32 = 0x12;
    pub const CAPITAL: u32 = 0x14;
    pub const LWIN: u32 = 0x5B;
    pub const RWIN: u32 = 0x5C;
    pub const LSHIFT: u32 = 0xA0;
    pub const RSHIFT: u32 = 0xA1;
    pub const LCONTROL: u32 = 0xA2;
    pub const RCONTROL: u32 = 0xA3;
    pub const LMENU: u32 = 0xA4;
    pub const RMENU: u32 = 0xA5;
}

/// The keys held together to dictate. Serialises as a plain array of VK codes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Hotkey(Vec<u32>);

impl Hotkey {
    /// The hook packs the chord into one `u64`, one byte per key.
    pub const MAX_KEYS: usize = 8;

    /// Drops invalid codes and duplicates; keeps at most [`Self::MAX_KEYS`].
    pub fn new(keys: impl IntoIterator<Item = u32>) -> Self {
        let mut out: Vec<u32> = Vec::new();
        for k in keys {
            if (1..=0xFF).contains(&k) && !out.contains(&k) {
                out.push(k);
            }
        }
        out.truncate(Self::MAX_KEYS);
        Self(out)
    }

    pub fn keys(&self) -> &[u32] {
        &self.0
    }

    /// Human-readable, e.g. `Ctrl+Win`. Used by the tray tooltip.
    pub fn label(&self) -> String {
        if self.0.is_empty() {
            return "nothing (disabled)".into();
        }
        self.0
            .iter()
            .map(|&k| key_name(k))
            .collect::<Vec<_>>()
            .join("+")
    }

    fn pack(&self) -> u64 {
        self.0
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &k)| acc | ((k as u64 & 0xFF) << (8 * i)))
    }
}

impl Default for Hotkey {
    fn default() -> Self {
        Self(vec![vk::CONTROL, vk::LWIN])
    }
}

fn unpack(packed: u64) -> impl Iterator<Item = u32> {
    (0..Hotkey::MAX_KEYS)
        .map(move |i| ((packed >> (8 * i)) & 0xFF) as u32)
        .filter(|&k| k != 0)
}

/// Name for a VK code. Mirrors `VK_NAMES` in the settings UI.
pub fn key_name(vk: u32) -> String {
    let fixed = match vk {
        0x08 => "Backspace",
        0x09 => "Tab",
        0x0D => "Enter",
        0x10 => "Shift",
        0x11 => "Ctrl",
        0x12 => "Alt",
        0x13 => "Pause",
        0x14 => "Caps Lock",
        0x1B => "Esc",
        0x20 => "Space",
        0x5B => "Win",
        0x5C => "Right Win",
        0x5D => "Menu",
        0x90 => "Num Lock",
        0x91 => "Scroll Lock",
        0xA0 => "Left Shift",
        0xA1 => "Right Shift",
        0xA2 => "Left Ctrl",
        0xA3 => "Right Ctrl",
        0xA4 => "Left Alt",
        0xA5 => "Right Alt",
        0x30..=0x39 | 0x41..=0x5A => return char::from(vk as u8).to_string(),
        0x70..=0x87 => return format!("F{}", vk - 0x6F),
        _ => return format!("0x{vk:02X}"),
    };
    fixed.to_string()
}

/// Does a chord entry match a physical key? The hook always reports
/// side-specific modifier codes; a generic entry accepts either side.
fn matches(entry: u32, physical: u32) -> bool {
    entry == physical
        || match entry {
            vk::SHIFT => matches!(physical, vk::LSHIFT | vk::RSHIFT),
            vk::CONTROL => matches!(physical, vk::LCONTROL | vk::RCONTROL),
            vk::MENU => matches!(physical, vk::LMENU | vk::RMENU),
            _ => false,
        }
}

fn chord_contains(chord: impl Iterator<Item = u32>, physical: u32) -> bool {
    let mut chord = chord;
    chord.any(|e| matches(e, physical))
}

/// True when `physical` going down completes the chord: it matches one entry
/// and every other entry is already held according to `is_down`.
fn chord_completed_by(
    chord: impl Iterator<Item = u32>,
    physical: u32,
    is_down: impl Fn(u32) -> bool,
) -> bool {
    let mut hit = false;
    for entry in chord {
        if matches(entry, physical) {
            hit = true;
        } else if !is_down(entry) {
            return false;
        }
    }
    hit
}

/// Keys with a solo press-release action that must not fire when they were
/// only ever part of the chord.
const SOLO_ACTION_KEYS: &[u32] = &[vk::LWIN, vk::RWIN, vk::LMENU, vk::RMENU, vk::CAPITAL];

/// Set once before the hook is installed; read from the callback.
static TX: OnceCell<Sender<HotkeyEvent>> = OnceCell::new();
/// The active chord, packed. Replaceable at runtime without restarting.
static CHORD: AtomicU64 = AtomicU64::new(0);
/// Chord currently engaged. Also collapses key auto-repeat to one Pressed.
static IS_HELD: AtomicBool = AtomicBool::new(false);
/// Bitset over the 256 VK codes: keys whose down we ate, so we eat the up too.
static SWALLOWED: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn swallow_slot(vk: u32) -> (&'static AtomicU64, u64) {
    (&SWALLOWED[(vk as usize >> 6) & 3], 1u64 << (vk & 63))
}

fn set_swallowed(vk: u32) {
    let (slot, bit) = swallow_slot(vk);
    slot.fetch_or(bit, Ordering::Relaxed);
}

fn is_swallowed(vk: u32) -> bool {
    let (slot, bit) = swallow_slot(vk);
    slot.load(Ordering::Relaxed) & bit != 0
}

fn take_swallowed(vk: u32) -> bool {
    let (slot, bit) = swallow_slot(vk);
    slot.fetch_and(!bit, Ordering::Relaxed) & bit != 0
}

/// Non-blocking by construction. A full channel means the consumer is wedged;
/// dropping the event is strictly better than losing the hook.
#[inline]
fn emit(ev: HotkeyEvent) {
    if let Some(tx) = TX.get() {
        match tx.try_send(ev) {
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
}

/// Replace the active chord. Takes effect on the next key event.
///
/// If the old chord was engaged at the time, its release would no longer be
/// recognised, so it is released here to keep the pipeline in step.
pub fn set(hotkey: &Hotkey) {
    CHORD.store(hotkey.pack(), Ordering::Relaxed);
    if IS_HELD.swap(false, Ordering::Relaxed) {
        emit(HotkeyEvent::Released);
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, GetMessageW, SetWindowsHookExW, TranslateMessage,
        HC_ACTION, KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP,
        WM_SYSKEYDOWN, WM_SYSKEYUP,
    };

    /// System key state. Valid for keys that went down *before* the event
    /// being hooked: the low-level hook runs ahead of the state update for
    /// the current key, which is exactly why the completing key is matched
    /// against the event and only the others are looked up here.
    fn key_down(vk: u32) -> bool {
        // SAFETY: plain FFI call with no pointers.
        (unsafe { GetAsyncKeyState(vk as i32) } as u16) & 0x8000 != 0
    }

    unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code == HC_ACTION as i32 {
            let kb = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
            // Our own SendInput (Ctrl+V for paste) comes back through this
            // hook; a physical hotkey must never be confused with it.
            let injected = kb.flags.0 & LLKHF_INJECTED.0 != 0;
            if !injected {
                let vk = kb.vkCode;
                let msg = wparam.0 as u32;
                let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
                let up = msg == WM_KEYUP || msg == WM_SYSKEYUP;
                let chord = CHORD.load(Ordering::Relaxed);

                if down {
                    if IS_HELD.load(Ordering::Relaxed) {
                        // Auto-repeat of a key we ate on the way down.
                        if is_swallowed(vk) {
                            return LRESULT(1);
                        }
                    } else if chord_completed_by(unpack(chord), vk, key_down) {
                        IS_HELD.store(true, Ordering::Relaxed);
                        emit(HotkeyEvent::Pressed);
                        if SOLO_ACTION_KEYS.contains(&vk) {
                            set_swallowed(vk);
                            return LRESULT(1);
                        }
                    }
                } else if up {
                    if IS_HELD.load(Ordering::Relaxed) && chord_contains(unpack(chord), vk) {
                        IS_HELD.store(false, Ordering::Relaxed);
                        emit(HotkeyEvent::Released);
                    }
                    if take_swallowed(vk) {
                        return LRESULT(1);
                    }
                }
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    /// Installs the hook and runs a message pump forever. Call on a dedicated
    /// thread: `GetMessageW` blocks, and the hook is bound to this thread.
    pub fn run(hotkey: Hotkey, tx: Sender<HotkeyEvent>) {
        let _ = TX.set(tx);
        set(&hotkey);

        unsafe {
            let hook = match SetWindowsHookExW(WH_KEYBOARD_LL, Some(hook_proc), None, 0) {
                Ok(h) => h,
                Err(e) => {
                    tracing::error!("SetWindowsHookExW failed: {e}");
                    return;
                }
            };
            tracing::info!(hotkey = %hotkey.label(), "push-to-talk hook installed");

            // A low-level hook requires a message pump on its owning thread.
            let mut msg = MSG::default();
            while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            let _ = hook;
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::*;
    pub fn run(_hotkey: Hotkey, _tx: Sender<HotkeyEvent>) {
        tracing::warn!("global hotkey is Windows-only in this build");
    }
}

/// Spawn the hook on its own thread. Returns immediately.
pub fn spawn(hotkey: Hotkey, tx: Sender<HotkeyEvent>) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("cooee-hotkey".into())
        .spawn(move || imp::run(hotkey, tx))
        .expect("spawn hotkey thread")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn down(keys: &'static [u32]) -> impl Fn(u32) -> bool {
        move |k| keys.contains(&k)
    }

    #[test]
    fn default_is_ctrl_win() {
        assert_eq!(Hotkey::default().label(), "Ctrl+Win");
    }

    #[test]
    fn pack_round_trips_and_ignores_invalid_keys() {
        let h = Hotkey::new([vk::CONTROL, 0, vk::LWIN, vk::CONTROL, 0x1FF]);
        assert_eq!(h.keys(), &[vk::CONTROL, vk::LWIN]);
        assert_eq!(unpack(h.pack()).collect::<Vec<_>>(), h.keys());
    }

    #[test]
    fn ctrl_win_completes_in_either_order() {
        let chord = Hotkey::default();
        // Ctrl held (either side), Win arrives.
        assert!(chord_completed_by(
            chord.keys().iter().copied(),
            vk::LWIN,
            down(&[vk::CONTROL])
        ));
        // Win held, Right Ctrl arrives.
        assert!(chord_completed_by(
            chord.keys().iter().copied(),
            vk::RCONTROL,
            down(&[vk::LWIN])
        ));
        // Win alone does nothing.
        assert!(!chord_completed_by(
            chord.keys().iter().copied(),
            vk::LWIN,
            down(&[])
        ));
        // An unrelated key never completes it, even with everything held.
        assert!(!chord_completed_by(
            chord.keys().iter().copied(),
            0x41,
            down(&[vk::CONTROL, vk::LWIN])
        ));
    }

    #[test]
    fn side_specific_single_key_stays_side_specific() {
        let chord = Hotkey::new([vk::RCONTROL]);
        assert!(chord_completed_by(
            chord.keys().iter().copied(),
            vk::RCONTROL,
            down(&[])
        ));
        assert!(!chord_completed_by(
            chord.keys().iter().copied(),
            vk::LCONTROL,
            down(&[])
        ));
        assert!(chord_contains(chord.keys().iter().copied(), vk::RCONTROL));
        assert!(!chord_contains(chord.keys().iter().copied(), vk::LCONTROL));
    }

    #[test]
    fn empty_chord_never_fires() {
        let chord = Hotkey::new([]);
        assert!(!chord_completed_by(
            chord.keys().iter().copied(),
            vk::LWIN,
            |_| true
        ));
        assert_eq!(chord.label(), "nothing (disabled)");
    }

    #[test]
    fn swallow_bitset_is_per_key() {
        set_swallowed(vk::LWIN);
        assert!(is_swallowed(vk::LWIN));
        assert!(!is_swallowed(vk::RWIN));
        assert!(take_swallowed(vk::LWIN));
        assert!(!take_swallowed(vk::LWIN));
    }

    #[test]
    fn serialises_as_a_plain_array() {
        let json = serde_json::to_string(&Hotkey::default()).unwrap();
        assert_eq!(json, "[17,91]");
        let back: Hotkey = serde_json::from_str("[163]").unwrap();
        assert_eq!(back.label(), "Right Ctrl");
    }

    #[test]
    fn names_common_keys() {
        assert_eq!(key_name(0x14), "Caps Lock");
        assert_eq!(key_name(0x41), "A");
        assert_eq!(key_name(0x78), "F9");
        assert_eq!(key_name(0xE5), "0xE5");
    }
}
