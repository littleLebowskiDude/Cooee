//! Persisted settings. Lives at %APPDATA%/cooee/config.json.

use crate::hotkey::Hotkey;
use crate::inject::Strategy;
use crate::pipeline::CaptureMode;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Keys held together to dictate. Defaults to Ctrl+Win.
    pub hotkey: Hotkey,
    /// Whether the chord is held for the utterance or tapped to start and stop.
    pub capture_mode: CaptureMode,
    /// Path to a whisper.cpp GGML model. `None` means the mock engine.
    pub model_path: Option<PathBuf>,
    pub injection: Strategy,
    /// Per-application overrides for [`Self::injection`], keyed by executable
    /// name (`outlook.exe`). Empty by default, so an upgraded install behaves
    /// exactly as it did before there were profiles.
    pub profiles: BTreeMap<String, Strategy>,
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
            capture_mode: CaptureMode::default(),
            model_path: None,
            injection: Strategy::default(),
            profiles: BTreeMap::new(),
            dictionary: Dictionary::default(),
            audio_feedback: true,
            asr_threads: None,
        }
    }
}

impl Config {
    /// The insertion strategy for the application about to receive the text.
    ///
    /// Falls back to the global setting whenever there is no profile for the
    /// app, or no app could be identified at all, so the feature can only ever
    /// change behaviour for an app the user has explicitly named.
    ///
    /// Matching ignores case: keys are written lowercased by the settings UI,
    /// but the file is plain JSON and people edit it by hand.
    pub fn strategy_for(&self, exe: Option<&str>) -> Strategy {
        let Some(exe) = exe else {
            return self.injection;
        };
        self.profiles
            .iter()
            .find(|(app, _)| app.eq_ignore_ascii_case(exe))
            .map(|(_, strategy)| *strategy)
            .unwrap_or(self.injection)
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

    /// Longest prompt handed to the engine, in characters. Whisper's prompt
    /// budget is a couple of hundred tokens; well under that, and short
    /// enough that a near-silent clip is unlikely to be answered with the
    /// prompt itself.
    const PROMPT_MAX_CHARS: usize = 200;

    /// The target spellings, as text for the engine to treat as preceding
    /// context: `Claude, Cooee.` Corrections still apply afterwards, so a
    /// word the prompt fails to tip is caught as before. `None` when empty.
    pub fn prompt(&self) -> Option<String> {
        let mut seen: Vec<String> = Vec::new();
        let mut terms: Vec<&str> = Vec::new();
        for want in self.entries.values() {
            let want = want.trim();
            let key = want.to_lowercase();
            if want.is_empty() || seen.contains(&key) {
                continue;
            }
            seen.push(key);
            terms.push(want);
        }
        if terms.is_empty() {
            return None;
        }

        let mut out = String::new();
        for term in terms {
            let sep = if out.is_empty() { "" } else { ", " };
            if out.len() + sep.len() + term.len() + 1 > Self::PROMPT_MAX_CHARS {
                break;
            }
            out.push_str(sep);
            out.push_str(term);
        }
        if out.is_empty() {
            return None;
        }
        out.push('.');
        Some(out)
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

/// `%APPDATA%\cooee`, created if missing. Config and history live here.
pub fn config_dir() -> Result<PathBuf> {
    let dir = dirs::config_dir()
        .context("could not resolve the user config directory")?
        .join("cooee");
    std::fs::create_dir_all(&dir).context("could not create the config directory")?;
    Ok(dir)
}

pub fn config_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.json"))
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
    fn prompt_lists_targets_once_and_ends_with_a_full_stop() {
        let mut d = Dictionary::default();
        d.insert("kui", "Cooee");
        d.insert("cooey", "Cooee");
        d.insert("claw'd", "Claude");
        assert_eq!(d.prompt().as_deref(), Some("Claude, Cooee."));
    }

    #[test]
    fn prompt_is_none_when_there_is_nothing_to_say() {
        assert_eq!(Dictionary::default().prompt(), None);
        let mut d = Dictionary::default();
        d.insert("x", "   ");
        assert_eq!(d.prompt(), None);
    }

    #[test]
    fn prompt_stays_within_budget() {
        let mut d = Dictionary::default();
        for i in 0..100 {
            d.insert(&format!("heard{i:03}"), &format!("Term{i:03}"));
        }
        let p = d.prompt().unwrap();
        assert!(p.len() <= Dictionary::PROMPT_MAX_CHARS, "{} chars", p.len());
        assert!(p.ends_with('.'));
        assert!(p.starts_with("Term000, Term001"));
    }

    #[test]
    fn defaults_round_trip_through_json() {
        let json = serde_json::to_string(&Config::default()).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hotkey, Hotkey::default());
    }

    #[test]
    fn no_profiles_means_the_global_strategy_everywhere() {
        let cfg = Config {
            injection: Strategy::Auto,
            ..Default::default()
        };
        // The upgrade path: an existing config has no profiles map at all.
        assert_eq!(cfg.strategy_for(Some("outlook.exe")), Strategy::Auto);
        assert_eq!(cfg.strategy_for(None), Strategy::Auto);
    }

    #[test]
    fn a_profile_overrides_only_its_own_app() {
        let mut cfg = Config {
            injection: Strategy::Auto,
            ..Default::default()
        };
        cfg.profiles
            .insert("windowsterminal.exe".into(), Strategy::Type);
        assert_eq!(
            cfg.strategy_for(Some("windowsterminal.exe")),
            Strategy::Type
        );
        assert_eq!(cfg.strategy_for(Some("outlook.exe")), Strategy::Auto);
        // An unidentifiable app must never pick up someone else's profile.
        assert_eq!(cfg.strategy_for(None), Strategy::Auto);
    }

    #[test]
    fn profile_lookup_ignores_case() {
        let mut cfg = Config {
            injection: Strategy::Auto,
            ..Default::default()
        };
        // As a hand-edited config might spell it.
        cfg.profiles.insert("OUTLOOK.EXE".into(), Strategy::Paste);
        assert_eq!(cfg.strategy_for(Some("outlook.exe")), Strategy::Paste);
    }

    #[test]
    fn profiles_survive_a_json_round_trip() {
        let mut cfg = Config::default();
        cfg.profiles.insert("code.exe".into(), Strategy::Paste);
        let back: Config = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(back.strategy_for(Some("code.exe")), Strategy::Paste);
    }

    #[test]
    fn a_1_0_config_still_loads_and_gains_empty_profiles() {
        // Exactly the shape 1.0.1 wrote, with no capture_mode and no profiles.
        let raw = r#"{
            "hotkey": [17, 91],
            "model_path": null,
            "injection": "auto",
            "dictionary": {},
            "audio_feedback": true,
            "asr_threads": null
        }"#;
        let cfg: Config = serde_json::from_str(raw).unwrap();
        assert!(cfg.profiles.is_empty());
        assert_eq!(
            cfg.capture_mode,
            CaptureMode::Hold,
            "hold must stay default"
        );
        assert_eq!(cfg.strategy_for(Some("anything.exe")), Strategy::Auto);
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
