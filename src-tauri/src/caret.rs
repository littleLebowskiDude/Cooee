//! What sits beside the caret in the focused field, so dictated text does not
//! run into it. Dictating mid-sentence should read "…the report today" and
//! not "…the reporttoday".
//!
//! Read through UI Automation's text pattern: the focused element's caret
//! range, widened by one character on each side. Nothing is selected, moved
//! or copied. Apps that expose no text pattern (terminals, some Electron
//! apps) simply get the text unchanged.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Neighbours {
    /// The character immediately before the caret (or the selection).
    pub before: Option<char>,
    /// The character immediately after.
    pub after: Option<char>,
}

/// The neighbours of the caret in the focused field, if it can be read.
pub fn neighbours() -> Option<Neighbours> {
    imp::neighbours()
}

/// `text` with a space on either side where it would otherwise touch a word
/// or closing punctuation. Openers on the left ("(", a quote, a dash) and
/// closers on the right (".", ",", ")") are left snug.
pub fn pad(text: &str, n: &Neighbours) -> String {
    let first = text.chars().next();
    let last = text.chars().last();
    let (Some(first), Some(last)) = (first, last) else {
        return text.to_string();
    };
    const OPENERS: &str = "([{\"'\u{201C}\u{2018}/-\u{2013}\u{2014}<";
    const CLOSERS: &str = ".,;:!?)]}\"'\u{201D}\u{2019}>";
    let space_before = match n.before {
        Some(c) => !c.is_whitespace() && !OPENERS.contains(c) && (first.is_alphanumeric() || OPENERS.contains(first)),
        None => false,
    };
    let space_after = match n.after {
        Some(c) => !c.is_whitespace() && !CLOSERS.contains(c) && !last.is_whitespace(),
        None => false,
    };
    let mut out = String::with_capacity(text.len() + 2);
    if space_before {
        out.push(' ');
    }
    out.push_str(text);
    if space_after {
        out.push(' ');
    }
    out
}

#[cfg(windows)]
mod imp {
    use super::Neighbours;
    use windows::core::Interface;
    use windows::Win32::Foundation::BOOL;
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED};
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationTextPattern, IUIAutomationTextPattern2, IUIAutomationTextRange,
        TextPatternRangeEndpoint_End, TextPatternRangeEndpoint_Start, TextUnit_Character, UIA_TextPattern2Id,
        UIA_TextPatternId,
    };

    pub fn neighbours() -> Option<Neighbours> {
        // SAFETY: COM calls on this thread, every interface checked for
        // failure and released on drop.
        unsafe {
            // The pipeline thread may already hold a COM apartment (the audio
            // stack); a changed-mode result is fine for UIA.
            let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
            let uia: IUIAutomation = CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok()?;
            let element = uia.GetFocusedElement().ok()?;
            tracing::debug!(
                name = %element.CurrentName().map(|s| s.to_string()).unwrap_or_default(),
                class = %element.CurrentClassName().map(|s| s.to_string()).unwrap_or_default(),
                control = ?element.CurrentControlType().map(|t| t.0).unwrap_or(0),
                text_pattern = element.GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId).is_ok(),
                text_pattern2 = element.GetCurrentPatternAs::<IUIAutomationTextPattern2>(UIA_TextPattern2Id).is_ok(),
                "focused element"
            );
            let range = caret_range(&element)?;
            let n = Neighbours {
                before: edge_char(&range, true),
                after: edge_char(&range, false),
            };
            tracing::debug!(?n, "caret neighbours");
            Some(n)
        }
    }

    /// The caret (a degenerate range) or the selection, which an insert replaces.
    unsafe fn caret_range(element: &windows::Win32::UI::Accessibility::IUIAutomationElement) -> Option<IUIAutomationTextRange> {
        if let Ok(p2) = element.GetCurrentPatternAs::<IUIAutomationTextPattern2>(UIA_TextPattern2Id) {
            let mut active = BOOL(0);
            if let Ok(r) = p2.GetCaretRange(&mut active) {
                return Some(r);
            }
            if let Some(r) = first_selection(&p2.cast::<IUIAutomationTextPattern>().ok()?) {
                return Some(r);
            }
        }
        let p = element.GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId).ok()?;
        first_selection(&p)
    }

    unsafe fn first_selection(p: &IUIAutomationTextPattern) -> Option<IUIAutomationTextRange> {
        let sel = p.GetSelection().ok()?;
        if sel.Length().ok()? < 1 {
            return None;
        }
        sel.GetElement(0).ok()
    }

    /// One character outside the range at its start (`before`) or end.
    unsafe fn edge_char(range: &IUIAutomationTextRange, before: bool) -> Option<char> {
        let probe = range.Clone().ok()?;
        let moved = if before {
            // Collapse to the start, then widen it one character back.
            probe.MoveEndpointByRange(TextPatternRangeEndpoint_End, &probe.Clone().ok()?, TextPatternRangeEndpoint_Start).ok()?;
            probe.MoveEndpointByUnit(TextPatternRangeEndpoint_Start, TextUnit_Character, -1).ok()?
        } else {
            probe.MoveEndpointByRange(TextPatternRangeEndpoint_Start, &probe.Clone().ok()?, TextPatternRangeEndpoint_End).ok()?;
            probe.MoveEndpointByUnit(TextPatternRangeEndpoint_End, TextUnit_Character, 1).ok()?
        };
        if moved == 0 {
            return None;
        }
        let text = probe.GetText(4).ok()?.to_string();
        if before { text.chars().last() } else { text.chars().next() }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn neighbours() -> Option<super::Neighbours> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(before: Option<char>, after: Option<char>) -> Neighbours {
        Neighbours { before, after }
    }

    #[test]
    fn touching_a_word_on_the_left_gets_a_space() {
        assert_eq!(pad("today", &n(Some('t'), None)), " today");
        assert_eq!(pad("today", &n(Some('.'), None)), " today");
    }

    #[test]
    fn whitespace_or_openers_on_the_left_stay_snug() {
        assert_eq!(pad("today", &n(Some(' '), None)), "today");
        assert_eq!(pad("today", &n(Some('\n'), None)), "today");
        assert_eq!(pad("today", &n(Some('('), None)), "today");
        assert_eq!(pad("today", &n(Some('"'), None)), "today");
        assert_eq!(pad("today", &n(None, None)), "today");
    }

    #[test]
    fn punctuation_led_text_never_gets_a_leading_space() {
        assert_eq!(pad(", and then", &n(Some('t'), None)), ", and then");
        assert_eq!(pad(".", &n(Some('t'), None)), ".");
    }

    #[test]
    fn touching_a_word_on_the_right_gets_a_space() {
        assert_eq!(pad("today", &n(None, Some('a'))), "today ");
        assert_eq!(pad("today", &n(None, Some('.'))), "today");
        assert_eq!(pad("today", &n(None, Some(' '))), "today");
        assert_eq!(pad("today ", &n(None, Some('a'))), "today ");
    }

    #[test]
    fn both_sides() {
        assert_eq!(pad("brown", &n(Some('e'), Some('f'))), " brown ");
        assert_eq!(pad("", &n(Some('e'), Some('f'))), "");
    }
}
