//! What was dictated, so it can be copied again.
//!
//! Text lands wherever the cursor was on release, which is sometimes the
//! wrong window or nowhere at all. The history keeps every transcript that
//! made it past polishing, newest first, in `%APPDATA%\cooee\history.json`,
//! and the main window shows it with a copy button per entry. Capped so the
//! file cannot grow without bound; nothing here is worth more than a few
//! hundred utterances of scrollback.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Entries kept. At a few hundred bytes each this is well under a megabyte.
const CAP: usize = 500;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    /// Milliseconds since the Unix epoch when the dictation finished; also
    /// the id the UI deletes by. Bumped by one if two land in the same ms.
    pub id: u64,
    pub text: String,
    /// Wall-clock time the engine spent.
    pub inference_ms: u64,
    /// Release to text ready, before insertion.
    pub elapsed_ms: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
struct File {
    /// Oldest first on disk.
    entries: Vec<Entry>,
}

pub struct History {
    file: File,
    path: Option<PathBuf>,
}

impl History {
    /// Loads the file, or starts empty: history that cannot be read is not
    /// worth refusing to dictate over.
    pub fn load() -> Self {
        let path = crate::config::config_dir().ok().map(|d| d.join("history.json"));
        let file = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();
        Self { file, path }
    }

    /// In-memory only, for tests.
    pub fn in_memory() -> Self {
        Self { file: File::default(), path: None }
    }

    pub fn push(&mut self, text: String, inference_ms: u64, elapsed_ms: u64) -> Entry {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = self.file.entries.last().map(|e| e.id).unwrap_or(0);
        let entry = Entry { id: now.max(last + 1), text, inference_ms, elapsed_ms };
        self.file.entries.push(entry.clone());
        if self.file.entries.len() > CAP {
            let excess = self.file.entries.len() - CAP;
            self.file.entries.drain(..excess);
        }
        self.save();
        entry
    }

    /// Newest first, as the window lists them.
    pub fn entries(&self) -> Vec<Entry> {
        self.file.entries.iter().rev().cloned().collect()
    }

    pub fn remove(&mut self, id: u64) {
        self.file.entries.retain(|e| e.id != id);
        self.save();
    }

    pub fn clear(&mut self) {
        self.file.entries.clear();
        self.save();
    }

    fn save(&self) {
        if let Some(path) = &self.path {
            if let Err(e) = self.write(path) {
                tracing::warn!("could not save history: {e:#}");
            }
        }
    }

    fn write(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string(&self.file)?;
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_first_and_ids_unique() {
        let mut h = History::in_memory();
        let a = h.push("one".into(), 1, 2);
        let b = h.push("two".into(), 1, 2);
        assert!(b.id > a.id);
        let list = h.entries();
        assert_eq!(list[0].text, "two");
        assert_eq!(list[1].text, "one");
    }

    #[test]
    fn capped_at_the_oldest_end() {
        let mut h = History::in_memory();
        for i in 0..(CAP + 10) {
            h.push(format!("{i}"), 0, 0);
        }
        let list = h.entries();
        assert_eq!(list.len(), CAP);
        assert_eq!(list[0].text, format!("{}", CAP + 9));
        assert_eq!(list.last().unwrap().text, "10");
    }

    #[test]
    fn remove_and_clear() {
        let mut h = History::in_memory();
        let a = h.push("keep".into(), 0, 0);
        let b = h.push("drop".into(), 0, 0);
        h.remove(b.id);
        assert_eq!(h.entries(), vec![a.clone()]);
        h.clear();
        assert!(h.entries().is_empty());
    }

    #[test]
    fn round_trips_through_json() {
        let mut h = History::in_memory();
        h.push("hello".into(), 3, 4);
        let json = serde_json::to_string(&h.file).unwrap();
        let back: File = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entries, h.file.entries);
    }
}
