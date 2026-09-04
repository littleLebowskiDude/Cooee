//! Whisper's tokenizer: GPT-2 byte-level BPE, both directions.
//!
//! Token strings in `vocab.json` are byte sequences mapped through GPT-2's
//! printable-unicode table (space is `Ġ`, and so on). Decoding is that table
//! run backwards, then UTF-8. Encoding needs `merges.txt` from the same
//! model repository; without it the tokenizer still decodes, and the
//! engine runs without the dictionary prompt.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    tokens: Vec<Option<String>>,
    vocab: HashMap<String, u32>,
    /// Merge pairs by rank, from `merges.txt`. `None` when the file is absent.
    merges: Option<HashMap<(String, String), usize>>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("vocab.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&raw).context("parse vocab.json")?;
        let merges = match std::fs::read_to_string(model_dir.join("merges.txt")) {
            Ok(text) => Some(parse_merges(&text)),
            Err(_) => None,
        };
        Ok(Self::from_parts(vocab, merges))
    }

    fn from_parts(vocab: HashMap<String, u32>, merges: Option<HashMap<(String, String), usize>>) -> Self {
        let size = vocab.values().max().map(|&m| m as usize + 1).unwrap_or(0);
        let mut tokens = vec![None; size];
        for (tok, &id) in &vocab {
            tokens[id as usize] = Some(tok.clone());
        }
        let byte_to_unicode = bytes_to_unicode();
        let unicode_to_byte = byte_to_unicode.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Self {
            tokens,
            vocab,
            merges,
            byte_to_unicode,
            unicode_to_byte,
        }
    }

    /// Whether `encode` is available (`merges.txt` was present).
    pub fn can_encode(&self) -> bool {
        self.merges.is_some()
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

    /// Token ids for `text`, as GPT-2's encoder produces them. `None` without
    /// `merges.txt`. Whisper expects prompt text to start with a space.
    pub fn encode(&self, text: &str) -> Option<Vec<u32>> {
        let merges = self.merges.as_ref()?;
        let mut out = Vec::new();
        for piece in pre_tokenize(text) {
            let symbols: Vec<String> = piece
                .bytes()
                .map(|b| self.byte_to_unicode[b as usize].to_string())
                .collect();
            for sym in bpe(symbols, merges) {
                match self.vocab.get(&sym) {
                    Some(&id) => out.push(id),
                    // Every byte is in the vocabulary, so a miss means a
                    // merge produced a token the vocab lacks; fall back to bytes.
                    None => out.extend(sym.chars().filter_map(|c| self.vocab.get(&c.to_string()).copied())),
                }
            }
        }
        Some(out)
    }
}

fn parse_merges(text: &str) -> HashMap<(String, String), usize> {
    text.lines()
        .filter(|l| !l.starts_with("#version") && !l.trim().is_empty())
        .enumerate()
        .filter_map(|(rank, line)| {
            let mut it = line.split(' ');
            Some(((it.next()?.to_string(), it.next()?.to_string()), rank))
        })
        .collect()
}

/// Byte-pair merge a word's symbols, lowest-ranked pair first.
fn bpe(mut word: Vec<String>, ranks: &HashMap<(String, String), usize>) -> Vec<String> {
    while word.len() > 1 {
        let best = word
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| ranks.get(&(w[0].clone(), w[1].clone())).map(|&r| (r, i)))
            .min();
        let Some((_, i)) = best else { break };
        let merged = format!("{}{}", word[i], word[i + 1]);
        word.splice(i..i + 2, [merged]);
    }
    word
}

/// GPT-2's pre-tokenizer pattern, by hand (the `regex` crate has no
/// lookahead for the whitespace rule):
///
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
fn pre_tokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let end_of = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < n {
        let start = chars[i].0;
        let c = chars[i].1;
        // Contractions
        if c == '\'' {
            let rest: String = chars[i + 1..(i + 3).min(n)].iter().map(|x| x.1).collect();
            let len = ["re", "ve", "ll"].iter().find(|s| rest.starts_with(*s)).map(|_| 2).or_else(|| {
                ["s", "t", "m", "d"].iter().find(|s| rest.starts_with(*s)).map(|_| 1)
            });
            if let Some(len) = len {
                pieces.push(&text[start..end_of(i + 1 + len)]);
                i += 1 + len;
                continue;
            }
        }
        // Optional single space, then a run of one class
        let (word_start, first) = if c == ' ' && i + 1 < n { (i + 1, chars[i + 1].1) } else { (i, c) };
        let class = char_class(first);
        if class != Class::Space {
            let mut j = word_start + 1;
            while j < n && char_class(chars[j].1) == class {
                j += 1;
            }
            pieces.push(&text[start..end_of(j)]);
            i = j;
            continue;
        }
        // Whitespace run. If something follows it, the last whitespace char
        // is left to prefix the next piece (the `(?!\S)` rule).
        let mut j = i;
        while j < n && chars[j].1.is_whitespace() {
            j += 1;
        }
        let stop = if j < n && j - i > 1 { j - 1 } else { j };
        pieces.push(&text[start..end_of(stop)]);
        i = stop;
    }
    pieces
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

fn char_class(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Number
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// GPT-2's `bytes_to_unicode`: the 256 byte values as printable characters.
fn bytes_to_unicode() -> [char; 256] {
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..256u32 {
        let c = if printable(b) {
            b
        } else {
            next += 1;
            next - 1
        };
        table[b as usize] = char::from_u32(c).expect("valid code point");
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_matches_gpt2() {
        let table = bytes_to_unicode();
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'a' as usize], 'a');
        assert_eq!(table[b'\n' as usize], 'Ċ');
        assert_eq!(table[0xC3], 'Ã');
        assert_eq!(table[0xA9], '©');
    }

    #[test]
    fn pre_tokenizer_follows_the_gpt2_pattern() {
        assert_eq!(pre_tokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pre_tokenize("Hello  world"), vec!["Hello", " ", " world"]);
        assert_eq!(pre_tokenize("it's 2026, isn't it?"), vec!["it", "'s", " 2026", ",", " isn", "'t", " it", "?"]);
        assert_eq!(pre_tokenize(" Claude, Cooee."), vec![" Claude", ",", " Cooee", "."]);
        assert_eq!(pre_tokenize("a\n\nb"), vec!["a", "\n", "\n", "b"]);
        assert_eq!(pre_tokenize("x  "), vec!["x", "  "]);
    }

    #[test]
    fn decodes_bytes_and_skips_specials() {
        let vocab: HashMap<String, u32> =
            [("Hello", 0), ("Ġcaf", 1), ("Ã©", 2), ("<|endoftext|>", 3)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let tokenizer = Tokenizer::from_parts(vocab, None);
        assert_eq!(tokenizer.decode(&[0, 1, 2, 3], 3), "Hello café");
        assert!(!tokenizer.can_encode());
        assert_eq!(tokenizer.encode("x"), None);
    }

    fn real_tokenizer() -> Option<Tokenizer> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/whisper-base.en-onnx");
        if !dir.join("merges.txt").exists() {
            eprintln!("skipped: no {}", dir.display());
            return None;
        }
        Some(Tokenizer::load(&dir).unwrap())
    }

    /// Against the real vocabulary, when the export is present.
    #[test]
    fn decodes_the_sample_transcript() {
        let Some(tokenizer) = real_tokenizer() else { return };
        let ids = [50257, 50362, 383, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256];
        assert_eq!(tokenizer.decode(&ids, 50256).trim(), "The quick brown fox jumps over the lazy dog.");
    }

    /// Expected ids: the model's own output for the first sentence, and
    /// openai/gpt-2's `encoder.py` for the rest (bench scratch, 2026-09-04).
    #[test]
    fn encodes_like_gpt2() {
        let Some(tokenizer) = real_tokenizer() else { return };
        assert!(tokenizer.can_encode());
        let cases: [(&str, &[u32]); 4] = [
            (" The quick brown fox jumps over the lazy dog.", &[383, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]),
            (" Claude, Cooee.", &[40559, 11, 1766, 78, 1453, 13]),
            ("Hello  world", &[15496, 220, 995]),
            ("it's 2026, isn't it?", &[270, 338, 1160, 2075, 11, 2125, 470, 340, 30]),
        ];
        for (text, ids) in cases {
            assert_eq!(tokenizer.encode(text).unwrap(), ids, "{text:?}");
            assert_eq!(tokenizer.decode(ids, 50256), text);
        }
    }

    #[test]
    fn non_ascii_round_trips() {
        let Some(tokenizer) = real_tokenizer() else { return };
        for text in [" naïve café", "日本語 テキスト", " emoji 🎉 here"] {
            let ids = tokenizer.encode(text).unwrap();
            assert_eq!(tokenizer.decode(&ids, 50256), text);
        }
    }
}
