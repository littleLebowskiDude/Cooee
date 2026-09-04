//! Persisted settings. Lives at %APPDATA%/cooee/config.json.

use crate::hotkey::Hotkey;
use crate::inject::Strategy;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Keys held together to dictate. Defaults to Ctrl+Win.
    pub hotkey: Hotkey,
    /// Path to a whisper.cpp GGML model. `None` means the mock engine.
    pub model_path: Option<PathBuf>,
    pub injection: Strategy,
    pub dictionary: Dictionary,
    /// Play a short tone on capture start/stop.
    pub audio_feedback: bool,
    /// Threads for whisper inference. `None` picks cores-1.
    ///
    /// More is not always faster: on a busy machine, oversubscribing makes
    /// ggml's threads fight the scheduler and latency gets *worse*.
    pub asr_threads: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: Hotkey::default(),
            model_path: None,
            injection: Strategy::default(),
            dictionary: Dictionary::default(),
            audio_feedback: true,
            asr_threads: None,
        }
    }
}

/// Case-insensitive whole-word replacements applied after filler removal.
///
/// This is what makes the app feel personal: whisper will never spell your
/// colleagues' names or your company's jargon right, and correcting the same
/// word twice is the moment a dictation app loses someone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Dictionary {
    entries: BTreeMap<String, String>,
}

impl Dictionary {
    pub fn insert(&mut self, heard: &str, want: &str) {
        self.entries.insert(heard.to_lowercase(), want.to_string());
    }

    pub fn remove(&mut self, heard: &str) {
        self.entries.remove(&heard.to_lowercase());
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.entries.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Replaces whole words only, so a short entry can't corrupt a longer word.
    pub fn apply(&self, text: &str) -> String {
        if self.entries.is_empty() {
            return text.to_string();
        }
        text.split_inclusive(char::is_whitespace)
            .map(|token| {
                let trailing: String = token
                    .chars()
                    .rev()
                    .take_while(|c| c.is_whitespace())
                    .collect();
                let word = token.trim_end();
                // Preserve punctuation attached to the word.
                let core = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
                let Some(replacement) = self.entries.get(&core.to_lowercase()) else {
                    return token.to_string();
                };
                format!("{}{}", word.replacen(core, replacement, 1), trailing)
            })
            .collect()
    }
}

/// Upgrades older on-disk shapes in place before deserialising.
///
/// 0.1.0 stored a single `hotkey_vk`; chords replaced it with a `hotkey`
/// array. A one-key array behaves exactly as the old field did.
fn migrate(value: &mut serde_json::Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    if let Some(vk) = obj.remove("hotkey_vk") {
        if !obj.contains_key("hotkey") {
            obj.insert("hotkey".into(), serde_json::json!([vk]));
        }
    }
}

pub fn config_path() -> Result<PathBuf> {
    let dir = dirs::config_dir()
        .context("could not resolve the user config directory")?
        .join("cooee");
    std::fs::create_dir_all(&dir).context("could not create the config directory")?;
    Ok(dir.join("config.json"))
}

impl Config {
    /// Missing or corrupt config falls back to defaults rather than failing to
    /// start — a dictation app that won't launch is worse than one with reset
    /// settings.
    pub fn load() -> Self {
        let Ok(raw) = config_path().and_then(|p| Ok(std::fs::read_to_string(p)?)) else {
            return Self::default();
        };
        serde_json::from_str::<serde_json::Value>(&raw)
            .map(|mut value| {
                migrate(&mut value);
                value
            })
            .and_then(serde_json::from_value)
            .unwrap_or_else(|e| {
                tracing::warn!("config is unreadable ({e}); using defaults");
                Self::default()
            })
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_replaces_whole_words_only() {
        let mut d = Dictionary::default();
        d.insert("cat", "dog");
        assert_eq!(d.apply("the cat sat"), "the dog sat");
        // "catalogue" must not become "dogalogue".
        assert_eq!(d.apply("the catalogue"), "the catalogue");
    }

    #[test]
    fn dictionary_preserves_attached_punctuation() {
        let mut d = Dictionary::default();
        d.insert("claude", "Claude");
        assert_eq!(d.apply("hi claude, hello"), "hi Claude, hello");
    }

    #[test]
    fn defaults_round_trip_through_json() {
        let json = serde_json::to_string(&Config::default()).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hotkey, Hotkey::default());
    }

    #[test]
    fn migrates_single_key_hotkey_vk_to_a_chord() {
        let mut value = serde_json::json!({ "hotkey_vk": 163, "injection": "auto" });
        migrate(&mut value);
        let cfg: Config = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.hotkey, Hotkey::new([0xA3]));
        assert_eq!(cfg.hotkey.label(), "Right Ctrl");
    }

    #[test]
    fn migration_never_overrides_an_explicit_chord() {
        let mut value = serde_json::json!({ "hotkey_vk": 163, "hotkey": [17, 91] });
        migrate(&mut value);
        let cfg: Config = serde_json::from_value(value).unwrap();
        assert_eq!(cfg.hotkey, Hotkey::default());
    }
}
