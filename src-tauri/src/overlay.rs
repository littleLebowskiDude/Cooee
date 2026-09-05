//! Puts the HUD where the user is looking.
//!
//! The overlay is a transient window with no fixed home. Left to Windows it
//! opens wherever the last window opened, often on the wrong monitor. Instead
//! it is moved, just before every show, to the bottom-centre of the work area
//! (the screen minus the taskbar) of whichever monitor holds the focused
//! window — which is the app the text is about to land in.

use tauri::{AppHandle, Manager, Monitor, PhysicalPosition, Runtime};

/// Gap between the pill and the taskbar, in logical pixels.
const BOTTOM_MARGIN: f64 = 48.0;

/// Best-effort: a positioning failure must never stop the overlay showing.
pub fn place<R: Runtime>(app: &AppHandle<R>) {
    let Some(overlay) = app.get_webview_window("overlay") else {
        return;
    };
    let Some(monitor) = target_monitor(app) else {
        return;
    };
    let (Ok(current_scale), Ok(size)) = (overlay.scale_factor(), overlay.outer_size()) else {
        return;
    };

    // The overlay's physical size reflects the monitor it is on *now*. Go via
    // logical pixels so a move to a monitor with a different scale centres
    // correctly rather than by the old size.
    let scale = monitor.scale_factor();
    let width = size.width as f64 / current_scale * scale;
    let height = size.height as f64 / current_scale * scale;

    let work = monitor.work_area();
    let x = work.position.x as f64 + (work.size.width as f64 - width) / 2.0;
    let y = work.position.y as f64 + work.size.height as f64 - height - BOTTOM_MARGIN * scale;

    if let Err(e) = overlay.set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32))
    {
        tracing::debug!("could not position overlay: {e}");
    }
}

/// Centres the settings window on the monitor under the cursor, just before
/// it is shown. Windows otherwise opens it wherever it last was, which on a
/// laptop that has been undocked can be a monitor that is no longer there.
pub fn place_settings<R: Runtime>(app: &AppHandle<R>) {
    let Some(window) = app.get_webview_window("settings") else {
        return;
    };
    let cursor = app.cursor_position().ok();
    let under_cursor = cursor.and_then(|p| app.monitor_from_point(p.x, p.y).ok().flatten());
    let monitor = under_cursor.or_else(|| app.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else {
        return;
    };
    tracing::debug!(
        cursor = ?cursor.map(|p| (p.x, p.y)),
        monitor = ?monitor.name(),
        at = ?(monitor.position().x, monitor.position().y),
        scale = monitor.scale_factor(),
        "placing settings"
    );
    let (Ok(current_scale), Ok(size)) = (window.scale_factor(), window.outer_size()) else {
        return;
    };
    let scale = monitor.scale_factor();
    let width = size.width as f64 / current_scale * scale;
    let height = size.height as f64 / current_scale * scale;
    let work = monitor.work_area();
    let x = work.position.x as f64 + (work.size.width as f64 - width) / 2.0;
    let y = work.position.y as f64 + (work.size.height as f64 - height) / 2.0;
    if let Err(e) = window.set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32)) {
        tracing::debug!("could not position settings: {e}");
    }
}

/// The monitor under the focused window, else the primary.
fn target_monitor<R: Runtime>(app: &AppHandle<R>) -> Option<Monitor> {
    if let Some((x, y)) = imp::foreground_centre() {
        if let Ok(Some(monitor)) = app.monitor_from_point(x, y) {
            return Some(monitor);
        }
    }
    app.primary_monitor().ok().flatten()
}

#[cfg(windows)]
mod imp {
    use windows::Win32::Foundation::RECT;
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowRect};

    /// Centre of the foreground window in physical screen pixels. A window
    /// straddling two monitors is assigned by its centre, which is the one
    /// the user is most likely looking at.
    pub fn foreground_centre() -> Option<(f64, f64)> {
        // SAFETY: plain Win32 calls; `rect` is a valid out-pointer.
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.0.is_null() {
                return None;
            }
            let mut rect = RECT::default();
            GetWindowRect(hwnd, &mut rect).ok()?;
            Some((
                (rect.left + rect.right) as f64 / 2.0,
                (rect.top + rect.bottom) as f64 / 2.0,
            ))
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn foreground_centre() -> Option<(f64, f64)> {
        None
    }
}
