//! System tray icon.
//!
//! Cooee has no primary window — the overlay is transient and the settings
//! window starts hidden. Without a tray icon the app is completely invisible
//! once installed, which reads as "it didn't launch".

use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, Runtime,
};

/// Brings the settings window to the foreground, creating focus if hidden.
fn show_settings<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("settings") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

const TRAY_ID: &str = "cooee";

fn tooltip(hotkey_label: &str) -> String {
    format!("Cooee — hold {hotkey_label} to dictate")
}

/// Keeps the tooltip truthful after the hotkey is changed in settings.
pub fn set_hotkey_label<R: Runtime>(app: &AppHandle<R>, hotkey_label: &str) {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let _ = tray.set_tooltip(Some(tooltip(hotkey_label)));
    }
}

pub fn build<R: Runtime>(app: &AppHandle<R>, hotkey_label: &str) -> tauri::Result<()> {
    let settings = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Cooee", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&settings, &separator, &quit])?;

    let icon = app
        .default_window_icon()
        .cloned()
        .ok_or_else(|| tauri::Error::AssetNotFound("default window icon".into()))?;

    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .tooltip(tooltip(hotkey_label))
        .menu(&menu)
        // Left click opens settings directly; the menu is on right click.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "settings" => show_settings(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_settings(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}
