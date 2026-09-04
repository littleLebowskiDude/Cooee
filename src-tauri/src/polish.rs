//! Raw transcript -> text a person would have typed.
//!
//! This is the layer that separates a dictation *product* from a speech-to-text
//! *demo*, and it is deliberately rule-based for now: an LLM pass would need
//! either a network call (ruled out — this build is local-only) or a second
//! resident model. Rules get us most of the perceived quality at ~5 ms, and the
//! seam below is where a local small model slots in later.

use crate::config::Dictionary;

/// Disfluencies removed when they stand alone as whole words.
const FILLERS: &[&str] = &["um", "uh", "erm", "uhh", "umm", "er", "ah"];

pub fn polish(raw: &str, dict: &Dictionary) -> String {
    let mut text = raw.trim().to_string();
    if text.is_empty() {
        return text;
    }

    text = strip_fillers(&text);
    text = apply_spoken_commands(&text);
    text = dict.apply(&text);
    text = capitalise_sentences(&text);
    text = collapse_whitespace(&text);
    text
}

/// Only removes fillers as standalone words — "umbrella" and "another" survive.
///
/// Speech puts fillers between commas ("probably, uh, move"), so dropping the
/// word alone leaves an orphaned comma ("probably, move"). When a removed filler
/// was comma-delimited on both sides, its trailing comma goes with it.
fn strip_fillers(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();

    for word in text.split_whitespace() {
        let bare: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();

        if !FILLERS.contains(&bare.as_str()) {
            out.push(word.to_string());
            continue;
        }

        // The filler is dropped. If the previous word ended in a comma and this
        // filler also carried one, that pair wrapped the filler — drop one.
        let filler_had_comma = word.ends_with(',');
        if filler_had_comma {
            if let Some(prev) = out.last_mut() {
                if prev.ends_with(',') {
                    prev.pop();
                }
            }
        }
    }

    out.join(" ")
}

/// Spoken punctuation. Whisper already emits most punctuation, but people who
/// come from other dictation tools reach for these out of habit.
fn apply_spoken_commands(text: &str) -> String {
    const RULES: &[(&str, &str)] = &[
        (" new line ", "\n"),
        (" new paragraph ", "\n\n"),
        (" comma ", ", "),
        (" full stop ", ". "),
        (" period ", ". "),
        (" question mark ", "? "),
        (" exclamation mark ", "! "),
    ];
    // Pad so leading/trailing occurrences match the same rule shape.
    let padded = format!(" {text} ");
    let mut out = padded;
    for (spoken, literal) in RULES {
        // Case-insensitive replace without pulling in a regex dependency.
        // ASCII-only lowering: byte-length preserving, so `idx` stays valid for `out`.
        while let Some(idx) = out.to_ascii_lowercase().find(spoken) {
            out.replace_range(idx..idx + spoken.len(), literal);
        }
    }
    out.trim().to_string()
}

/// Capitalises the first letter of the text and the start of each sentence.
///
/// A sentence break requires terminal punctuation *followed by whitespace*.
/// Without that second condition every decimal point starts a new sentence and
/// "version 2.0 is out" becomes "version 2.0 Is out".
fn capitalise_sentences(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut start_of_sentence = true;
    // Saw `.`/`?`/`!`; only a following space confirms it ended a sentence.
    let mut pending_break = false;

    for ch in text.chars() {
        if start_of_sentence && ch.is_alphabetic() {
            out.extend(ch.to_uppercase());
            start_of_sentence = false;
            pending_break = false;
            continue;
        }

        out.push(ch);
        if ch == '\n' {
            start_of_sentence = true;
            pending_break = false;
        } else if matches!(ch, '.' | '?' | '!') {
            pending_break = true;
        } else if ch.is_whitespace() {
            if pending_break {
                start_of_sentence = true;
                pending_break = false;
            }
        } else {
            // A non-space right after `.` means it was a decimal or abbreviation.
            pending_break = false;
            // A digit is sentence content, so the sentence has now begun and a
            // later letter must not be capitalised ("3.0s", not "3.0S").
            // Opening brackets and quotes deliberately fall through, keeping the
            // flag so `"hello` and `[hello` still capitalise their first letter.
            if ch.is_alphanumeric() {
                start_of_sentence = false;
            }
        }
    }
    out
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_space = false;
    for ch in text.chars() {
        // Newlines are meaningful (new paragraph); spaces are not.
        if ch == '\n' {
            out.push(ch);
            last_was_space = true;
        } else if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict() -> Dictionary {
        Dictionary::default()
    }

    #[test]
    fn removes_standalone_fillers_only() {
        assert_eq!(polish("um so uh the umbrella", &dict()), "So the umbrella");
    }

    #[test]
    fn comma_wrapped_filler_leaves_no_orphan_comma() {
        // "probably, uh, move" must not become "probably, move".
        assert_eq!(
            polish("we should probably, uh, move the meeting", &dict()),
            "We should probably move the meeting"
        );
        // A comma that was doing real work survives.
        assert_eq!(
            polish("first, um, we ship, then we test", &dict()),
            "First we ship, then we test"
        );
    }

    #[test]
    fn capitalises_each_sentence() {
        assert_eq!(
            polish("hello there. how are you?", &dict()),
            "Hello there. How are you?"
        );
    }

    #[test]
    fn applies_spoken_punctuation() {
        assert_eq!(polish("hello comma world", &dict()), "Hello, world");
    }

    #[test]
    fn dictionary_corrections_win() {
        let mut d = Dictionary::default();
        d.insert("claw'd", "Claude");
        assert_eq!(polish("i use claw'd daily", &d), "I use Claude daily");
    }

    #[test]
    fn decimals_do_not_start_a_new_sentence() {
        assert_eq!(polish("version 2.0 is out", &dict()), "Version 2.0 is out");
        assert_eq!(
            polish("3.0s of audio captured", &dict()),
            "3.0s of audio captured"
        );
    }

    #[test]
    fn capitalises_first_letter_after_opening_punctuation() {
        assert_eq!(polish("\"hello there", &dict()), "\"Hello there");
        assert_eq!(polish("(hello there", &dict()), "(Hello there");
    }

    #[test]
    fn empty_input_is_safe() {
        assert_eq!(polish("   ", &dict()), "");
    }
}
