//! Cooee — push-to-talk local dictation for Windows.

pub mod asr;
pub mod audio;
pub mod caret;
pub mod config;
pub mod focus;
pub mod history;
pub mod hotkey;
pub mod inject;
pub mod overlay;
pub mod pipeline;
pub mod polish;
pub mod tone;
pub mod tray;
pub mod vad;

use crate::asr::{EngineInfo, EngineSlot, EngineState};
use crate::config::Config;
use crate::pipeline::{Observer, Pipeline, State, StatusEvent};
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};

/// Bridges pipeline transitions to the webview as a `status` event.
struct WindowObserver(AppHandle);

impl Observer for WindowObserver {
    fn on_status(&self, event: StatusEvent) {
        // Move the pill before the webview shows it, so it never flashes at
        // its old position first.
        if event.state == State::Capturing {
            overlay::place(&self.0);
        }
        if let Err(e) = self.0.emit("status", &event) {
            tracing::debug!("could not emit status: {e}");
        }
    }

    fn on_level(&self, level: f32) {
        // Twenty a second while capturing; a dropped one is invisible.
        let _ = self.0.emit("level", level);
    }

    fn on_dictated(&self, dictated: pipeline::Dictated) {
        let entry = self.0.state::<AppState>().history.write().push(
            dictated.text,
            dictated.inference_ms,
            dictated.elapsed_ms,
        );
        if let Err(e) = self.0.emit("history", &entry) {
            tracing::debug!("could not emit history: {e}");
        }
    }
}

/// Shared application state exposed to Tauri commands.
pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub engine: EngineSlot,
    pub engine_info: Arc<RwLock<EngineInfo>>,
    pub history: Arc<RwLock<history::History>>,
}

fn publish_engine_info(app: &AppHandle, info: EngineInfo) {
    *app.state::<AppState>().engine_info.write() = info.clone();
    if let Err(e) = app.emit("engine", &info) {
        tracing::debug!("could not emit engine info: {e}");
    }
}

/// Loads `model` on a background thread and swaps it into the slot.
///
/// Dictation keeps working on the previous engine until the new one is ready.
/// On failure the previous engine stays; if there was none (first load), the
/// mock engine is installed so the rest of the app is still exercisable, and
/// the error is shown in settings.
fn load_engine(app: AppHandle, model: Option<PathBuf>, threads: Option<usize>) {
    let spawned = std::thread::Builder::new()
        .name("cooee-model".into())
        .spawn(move || {
            publish_engine_info(
                &app,
                EngineInfo {
                    state: EngineState::Loading,
                    engine: None,
                    model: model.clone(),
                    error: None,
                },
            );

            let slot = app.state::<AppState>().engine.clone();
            let info = match asr::build_engine(model.as_deref(), threads) {
                Ok(engine) => {
                    let name = engine.name().to_string();
                    *slot.write() = Some(Arc::from(engine));
                    EngineInfo {
                        state: EngineState::Ready,
                        engine: Some(name),
                        model,
                        error: None,
                    }
                }
                Err(error) => {
                    tracing::error!("model load failed: {error}");
                    let mut current = slot.write();
                    let engine = current.get_or_insert_with(|| Arc::new(asr::mock::MockEngine));
                    EngineInfo {
                        state: EngineState::Failed,
                        engine: Some(engine.name().to_string()),
                        model,
                        error: Some(error),
                    }
                }
            };
            publish_engine_info(&app, info);
        });
    if let Err(e) = spawned {
        tracing::error!("could not spawn model loader: {e}");
    }
}

#[tauri::command]
fn get_config(state: tauri::State<AppState>) -> Config {
    state.config.read().clone()
}

#[tauri::command]
fn set_config(
    new: Config,
    state: tauri::State<AppState>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    // The hook reads its chord atomically, so a new hotkey applies at once.
    hotkey::set(&new.hotkey);
    tray::set_hotkey_label(&app, &new.hotkey.label());

    let model_changed = {
        let old = state.config.read();
        old.model_path != new.model_path || old.asr_threads != new.asr_threads
    };
    if model_changed {
        load_engine(app.clone(), new.model_path.clone(), new.asr_threads);
    }

    *state.config.write() = new;
    state.config.read().save().map_err(|e| e.to_string())
}

#[tauri::command]
fn engine_info(state: tauri::State<AppState>) -> EngineInfo {
    state.engine_info.read().clone()
}

/// Native file picker for a whisper.cpp GGML model. Async so the blocking
/// dialog runs on a worker thread rather than the event loop.
#[tauri::command]
async fn pick_model(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, AppState>,
) -> Result<Option<PathBuf>, String> {
    use tauri_plugin_dialog::DialogExt;

    let mut dialog = window
        .dialog()
        .file()
        .set_title("Choose a whisper.cpp model")
        .add_filter("whisper.cpp model (ggml-*.bin)", &["bin"])
        .set_parent(&window);
    let start_in = state
        .config
        .read()
        .model_path
        .as_ref()
        .and_then(|p| p.parent().map(PathBuf::from));
    if let Some(dir) = start_in {
        dialog = dialog.set_directory(dir);
    }

    dialog
        .blocking_pick_file()
        .map(|f| f.into_path())
        .transpose()
        .map_err(|e| e.to_string())
}

/// Native folder picker for an ONNX export (encoder on the NPU); see
/// docs/NPU.md for the layout. Async for the same reason as `pick_model`.
#[tauri::command]
async fn pick_model_dir(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, AppState>,
) -> Result<Option<PathBuf>, String> {
    use tauri_plugin_dialog::DialogExt;

    let mut dialog = window
        .dialog()
        .file()
        .set_title("Choose a whisper ONNX model folder")
        .set_parent(&window);
    let start_in = state
        .config
        .read()
        .model_path
        .as_ref()
        .and_then(|p| p.parent().map(PathBuf::from));
    if let Some(dir) = start_in {
        dialog = dialog.set_directory(dir);
    }

    dialog
        .blocking_pick_folder()
        .map(|f| f.into_path())
        .transpose()
        .map_err(|e| e.to_string())
}

/// Lets the settings UI verify injection without speaking.
#[tauri::command]
fn test_injection(text: String, state: tauri::State<AppState>) -> Result<(), String> {
    let strategy = state.config.read().injection;
    inject::inject(&text, strategy).map_err(|e| e.to_string())
}

/// The app with focus right now, for the settings UI's detect affordance.
/// Nobody should have to know that Windows Terminal's executable is called
/// `WindowsTerminal.exe` in order to give it a profile.
#[tauri::command]
fn foreground_app() -> Option<String> {
    focus::foreground_exe()
}

/// Newest first.
#[tauri::command]
fn get_history(state: tauri::State<AppState>) -> Vec<history::Entry> {
    state.history.read().entries()
}

/// Everything the app holds on this machine, and where.
///
/// The point of the panel this feeds is that "nothing leaves the machine" is
/// unfalsifiable from the outside. Naming every file the app writes, and
/// offering to hand them over or delete them, is the part a user can check.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DataReport {
    config_dir: Option<String>,
    config_path: Option<String>,
    history_path: Option<String>,
    history_entries: usize,
    history_bytes: Option<u64>,
    /// The model actually being used, if any. Local file or folder.
    model_path: Option<String>,
    /// Engine currently loaded, so the panel does not have to guess.
    engine: Option<String>,
}

#[tauri::command]
fn data_report(state: tauri::State<AppState>) -> DataReport {
    let history = state.history.read();
    let show = |p: &std::path::Path| p.display().to_string();
    DataReport {
        config_dir: config::config_dir().ok().map(|p| show(&p)),
        config_path: config::config_path().ok().map(|p| show(&p)),
        history_path: history.path().map(show),
        history_entries: history.len(),
        history_bytes: history.bytes(),
        model_path: state.config.read().model_path.as_deref().map(show),
        engine: state.engine_info.read().engine.clone(),
    }
}

/// Writes every transcript to a file the user chooses. Async so the blocking
/// dialog stays off the event loop, as with the model pickers.
#[tauri::command]
async fn export_history(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, AppState>,
) -> Result<Option<PathBuf>, String> {
    use tauri_plugin_dialog::DialogExt;

    // Serialise before the dialog: holding the lock across an await would
    // block every dictation for as long as the picker is open.
    let json = state
        .history
        .read()
        .export_json()
        .map_err(|e| e.to_string())?;

    let stamp = chrono_stamp();
    let Some(path) = window
        .dialog()
        .file()
        .set_title("Export dictation history")
        .set_file_name(format!("cooee-history-{stamp}.json"))
        .add_filter("JSON", &["json"])
        .set_parent(&window)
        .blocking_save_file()
        .map(|f| f.into_path())
        .transpose()
        .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };

    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(Some(path))
}

/// `YYYY-MM-DD` for today, without pulling in a date library for one filename.
fn chrono_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days since the Unix epoch to a calendar date, by Howard Hinnant's
/// `civil_from_days`. Split out from [`chrono_stamp`] so it can be tested
/// against known dates: a filename is low stakes, but a date routine written
/// from memory is exactly the kind of thing that is quietly wrong for years.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Opens the folder holding config and history in Explorer, so "it is all in
/// these two files" can be verified rather than taken on trust.
#[tauri::command]
fn open_data_folder() -> Result<(), String> {
    let dir = config::config_dir().map_err(|e| e.to_string())?;
    // explorer.exe reports a non-zero exit code even when it succeeds, so the
    // status is deliberately not checked; only a failure to spawn is an error.
    std::process::Command::new("explorer.exe")
        .arg(&dir)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn delete_history(id: u64, state: tauri::State<AppState>) {
    state.history.write().remove(id);
}

#[tauri::command]
fn clear_history(state: tauri::State<AppState>) {
    state.history.write().clear();
}

/// Puts a past dictation on the clipboard, for pasting wherever it was
/// meant to go.
#[tauri::command]
fn copy_text(text: String) -> Result<(), String> {
    arboard::Clipboard::new()
        .and_then(|mut c| c.set_text(text))
        .map_err(|e| e.to_string())
}

pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cooee=info".into()),
        )
        .init();

    let config = Arc::new(RwLock::new(Config::load()));
    tracing::info!("starting cooee");

    let engine: EngineSlot = Arc::new(RwLock::new(None));
    let state = AppState {
        config: config.clone(),
        engine: engine.clone(),
        engine_info: Arc::new(RwLock::new(EngineInfo {
            state: EngineState::Loading,
            engine: None,
            model: config.read().model_path.clone(),
            error: None,
        })),
        history: Arc::new(RwLock::new(history::History::load())),
    };

    tauri::Builder::default()
        // A second instance would install its own keyboard hook and every
        // dictation would be typed twice. Focus the existing one instead.
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            tray::show_settings(app);
        }))
        .manage(state)
        // Closing the settings window must hide it, not destroy it: a
        // destroyed window cannot be shown again, so the tray's Settings
        // entry went dead after the first close.
        .on_window_event(|window, event| {
            if window.label() == "settings" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            set_config,
            engine_info,
            pick_model,
            pick_model_dir,
            test_injection,
            get_history,
            delete_history,
            clear_history,
            copy_text,
            data_report,
            export_history,
            open_data_folder,
            foreground_app
        ])
        .setup(move |app| {
            let handle = app.handle().clone();

            // Without this the app has no visible UI at all once installed.
            tray::build(&handle, &config.read().hotkey.label())?;

            // Off the startup path: a large model takes seconds to load and
            // the tray icon should not wait for it.
            let cfg = config.read();
            load_engine(handle.clone(), cfg.model_path.clone(), cfg.asr_threads);
            drop(cfg);

            // Bounded: if the pipeline ever wedges, the hook drops events rather
            // than blocking and getting itself uninstalled by Windows.
            let (tx, rx) = crossbeam_channel::bounded(32);

            hotkey::spawn(config.read().hotkey.clone(), tx);

            let pipeline = Pipeline {
                config: config.clone(),
                engine: engine.clone(),
                observer: Arc::new(WindowObserver(handle)),
            };
            std::thread::Builder::new()
                .name("cooee-pipeline".into())
                .spawn(move || pipeline.run(rx))?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running cooee");
}

#[cfg(test)]
mod tests {
    use super::civil_from_days;

    #[test]
    fn converts_days_since_the_epoch_to_a_date() {
        // Anchors chosen for the cases the algorithm gets wrong when it is
        // misremembered: the epoch itself, a leap day, a century boundary
        // that *is* a leap year, and one that is not.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(789), (1972, 2, 29));
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(20_710), (2026, 9, 14));
        // 2100 is not a leap year, so day 47541 is 1 March and not 29 Feb.
        assert_eq!(civil_from_days(47_541), (2100, 3, 1));
    }

    #[test]
    fn every_day_of_a_year_round_trips_in_order() {
        // Walks 1999 into 2001 across the 2000 leap day, checking the parts
        // stay in range and the sequence never repeats or skips.
        let mut previous = civil_from_days(10_500);
        for day in 10_501..11_500 {
            let (y, m, d) = civil_from_days(day);
            assert!((1..=12).contains(&m), "month {m} out of range on day {day}");
            assert!((1..=31).contains(&d), "day {d} out of range on day {day}");
            assert!(
                (y, m, d) > previous,
                "day {day} gave {:?} after {:?}",
                (y, m, d),
                previous
            );
            previous = (y, m, d);
        }
    }
}
