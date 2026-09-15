//! Which application has focus, so insertion can be tuned to it.
//!
//! One global insertion strategy cannot be right everywhere. The table at the
//! top of `inject.rs` lists two strategies with opposite failure modes, and
//! which one is correct depends entirely on the window about to receive the
//! text: some Electron apps drop synthesised keystrokes, and fields that
//! refuse a paste need exactly those keystrokes.
//!
//! The executable is read from the foreground window rather than through UI
//! Automation, because UIA is precisely what fails on the apps most likely to
//! need an override — a terminal exposes no text pattern at all, which is
//! already why `caret.rs` gives up on it.

/// Lowercased file name of the foreground window's executable, for example
/// `windowsterminal.exe`.
///
/// `None` when there is no foreground window, or when the process cannot be
/// opened — an elevated process cannot be queried by an unelevated one, and
/// that is not an error worth surfacing. Callers fall back to the global
/// strategy.
pub fn foreground_exe() -> Option<String> {
    imp::foreground_path().as_deref().map(exe_name)
}

/// Last path segment, lowercased, so a profile key matches however Windows
/// happens to report the path.
fn exe_name(path: &str) -> String {
    path.rsplit(['\\', '/'])
        .next()
        .unwrap_or(path)
        .to_lowercase()
}

#[cfg(windows)]
mod imp {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    /// Full path of the foreground window's executable.
    pub fn foreground_path() -> Option<String> {
        // SAFETY: every handle is checked and closed; the buffer is sized and
        // its length is taken from what the call reports written.
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.0.is_null() {
                return None;
            }

            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid == 0 {
                return None;
            }

            // LIMITED_INFORMATION is the weakest right that can still read an
            // image name, so this works without any elevation of our own.
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;

            let mut buf = [0u16; 1024];
            let mut len = buf.len() as u32;
            let queried = QueryFullProcessImageNameW(
                process,
                PROCESS_NAME_WIN32,
                PWSTR(buf.as_mut_ptr()),
                &mut len,
            );
            let _ = CloseHandle(process);
            queried.ok()?;

            Some(String::from_utf16_lossy(&buf[..len as usize]))
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn foreground_path() -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::exe_name;

    #[test]
    fn takes_the_file_name_and_lowercases_it() {
        assert_eq!(
            exe_name(r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE"),
            "outlook.exe"
        );
        assert_eq!(
            exe_name(r"C:\Windows\System32\WindowsTerminal.exe"),
            "windowsterminal.exe"
        );
    }

    #[test]
    fn copes_with_a_bare_name_or_forward_slashes() {
        assert_eq!(exe_name("Code.exe"), "code.exe");
        assert_eq!(exe_name("/usr/bin/Thing"), "thing");
        assert_eq!(exe_name(""), "");
    }
}
