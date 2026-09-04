//! Whisper's tokenizer, decode side only: GPT-2 byte-level BPE.
//!
//! Token strings in `vocab.json` are byte sequences mapped through GPT-2's
//! printable-unicode table (space is `Ġ`, and so on). Decoding is that table
//! run backwards, then UTF-8. Encoding (for prompts) is not implemented.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    tokens: Vec<Option<String>>,
    unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("vocab.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&raw).context("parse vocab.json")?;
        let size = vocab.values().max().map(|&m| m as usize + 1).unwrap_or(0);
        let mut tokens = vec![None; size];
        for (tok, id) in vocab {
            tokens[id as usize] = Some(tok);
        }
        Ok(Self {
            tokens,
            unicode_to_byte: bytes_to_unicode().into_iter().map(|(b, c)| (c, b)).collect(),
        })
    }

    /// Text for a token sequence, dropping every id at or above `first_special`
    /// (whisper keeps all its control tokens at the top of the vocabulary,
    /// starting at end-of-text).
    pub fn decode(&self, ids: &[u32], first_special: u32) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id >= first_special {
                continue;
            }
            if let Some(Some(tok)) = self.tokens.get(id as usize) {
                bytes.extend(tok.chars().filter_map(|c| self.unicode_to_byte.get(&c).copied()));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's `bytes_to_unicode`: the 256 byte values as printable characters.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let printable: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut out = Vec::with_capacity(256);
    let mut next = 256u32;
    for b in 0..=255u8 {
        let c = if printable.contains(&(b as u32)) {
            b as u32
        } else {
            next += 1;
            next - 1
        };
        out.push((b, char::from_u32(c).expect("valid code point")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_matches_gpt2() {
        let table: HashMap<u8, char> = bytes_to_unicode().into_iter().collect();
        assert_eq!(table[&b' '], 'Ġ');
        assert_eq!(table[&b'a'], 'a');
        assert_eq!(table[&b'\n'], 'Ċ');
        assert_eq!(table[&0xC3], 'Ã');
        assert_eq!(table[&0xA9], '©');
        assert_eq!(table.len(), 256);
    }

    #[test]
    fn decodes_bytes_and_skips_specials() {
        let tokenizer = Tokenizer {
            tokens: vec![Some("Hello".into()), Some("Ġcaf".into()), Some("Ã©".into()), Some("<|endoftext|>".into())],
            unicode_to_byte: bytes_to_unicode().into_iter().map(|(b, c)| (c, b)).collect(),
        };
        assert_eq!(tokenizer.decode(&[0, 1, 2, 3], 3), "Hello café");
    }

    /// Against the real vocabulary, when the export is present.
    #[test]
    fn decodes_the_sample_transcript() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/whisper-base.en-onnx");
        if !dir.join("vocab.json").exists() {
            eprintln!("skipped: no {}", dir.display());
            return;
        }
        let tokenizer = Tokenizer::load(&dir).unwrap();
        let ids = [50257, 50362, 383, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256];
        assert_eq!(tokenizer.decode(&ids, 50256).trim(), "The quick brown fox jumps over the lazy dog.");
    }
}
