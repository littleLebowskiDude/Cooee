//! Getting text into whatever window currently has focus.
//!
//! There is no single strategy that works everywhere, so we ship two and choose
//! per-utterance:
//!
//! | Strategy      | Good at                              | Fails on                        |
//! |---------------|--------------------------------------|---------------------------------|
//! | `SendInput`   | Any focused control, keeps clipboard  | Slow past a few hundred chars;  |
//! |               |                                       | some Electron apps drop chars   |
//! | Clipboard+^V  | Long text, instant, format-preserving | Secure fields that block paste  |
//!
//! Either way we save and restore the user's clipboard — silently eating it is
//! the fastest way to make a dictation app feel hostile.

use anyhow::{Context, Result};

/// Above this length, typing char-by-char becomes perceptibly slow.
const PASTE_THRESHOLD: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    /// Synthesise Unicode keystrokes.
    Type,
    /// Stage on the clipboard and send Ctrl+V.
    Paste,
    /// Paste when long, type when short.
    #[default]
    Auto,
}

pub fn inject(text: &str, strategy: Strategy) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let chosen = match strategy {
        Strategy::Auto if text.chars().count() > PASTE_THRESHOLD => Strategy::Paste,
        Strategy::Auto => Strategy::Type,
        explicit => explicit,
    };

    match chosen {
        Strategy::Paste => imp::paste(text),
        _ => imp::type_text(text),
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE,
        VIRTUAL_KEY, VK_CONTROL, VK_V,
    };

    fn key_event(vk: VIRTUAL_KEY, scan: u16, flags: u32) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: scan,
                    dwFlags: windows::Win32::UI::Input::KeyboardAndMouse::KEYBD_EVENT_FLAGS(flags),
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    /// Sends each character as a Unicode keystroke. Non-BMP characters (emoji)
    /// arrive as surrogate pairs, which `encode_utf16` gives us for free.
    pub fn type_text(text: &str) -> Result<()> {
        let mut inputs = Vec::with_capacity(text.len() * 2);
        for unit in text.encode_utf16() {
            inputs.push(key_event(VIRTUAL_KEY(0), unit, KEYEVENTF_UNICODE.0));
            inputs.push(key_event(
                VIRTUAL_KEY(0),
                unit,
                KEYEVENTF_UNICODE.0 | KEYEVENTF_KEYUP.0,
            ));
        }

        let size = std::mem::size_of::<INPUT>() as i32;
        // SAFETY: `inputs` is a valid, correctly-sized INPUT array.
        let sent = unsafe { SendInput(&inputs, size) };
        if sent as usize != inputs.len() {
            anyhow::bail!("SendInput sent {sent} of {} events", inputs.len());
        }
        Ok(())
    }

    pub fn paste(text: &str) -> Result<()> {
        let mut clipboard = arboard::Clipboard::new().context("failed to open clipboard")?;
        // Best-effort save; an image or empty clipboard just means nothing to restore.
        let previous = clipboard.get_text().ok();

        clipboard
            .set_text(text)
            .context("failed to stage text on clipboard")?;

        let size = std::mem::size_of::<INPUT>() as i32;
        let keys = [
            key_event(VK_CONTROL, 0, 0),
            key_event(VK_V, 0, 0),
            key_event(VK_V, 0, KEYEVENTF_KEYUP.0),
            key_event(VK_CONTROL, 0, KEYEVENTF_KEYUP.0),
        ];
        // SAFETY: fixed-size, correctly-initialised INPUT array.
        let sent = unsafe { SendInput(&keys, size) };
        if sent as usize != keys.len() {
            anyhow::bail!("SendInput sent {sent} of {} events for Ctrl+V", keys.len());
        }

        // The target app reads the clipboard asynchronously after the keystroke,
        // so restoring immediately would race it.
        if let Some(prev) = previous {
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(250));
                if let Ok(mut cb) = arboard::Clipboard::new() {
                    let _ = cb.set_text(prev);
                }
            });
        }
        Ok(())
    }
}

#[cfg(not(windows))]
mod imp {
    use super::*;
    pub fn type_text(_: &str) -> Result<()> {
        anyhow::bail!("text injection is Windows-only in this build")
    }
    pub fn paste(_: &str) -> Result<()> {
        anyhow::bail!("text injection is Windows-only in this build")
    }
}
