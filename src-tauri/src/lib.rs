//! Cooee — push-to-talk local dictation for Windows.

pub mod asr;
pub mod audio;
pub mod config;
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
}

/// Shared application state exposed to Tauri commands.
pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub engine: EngineSlot,
    pub engine_info: Arc<RwLock<EngineInfo>>,
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
    };

    tauri::Builder::default()
        // A second instance would install its own keyboard hook and every
        // dictation would be typed twice. Focus the existing one instead.
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("settings") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            get_config,
            set_config,
            engine_info,
            pick_model,
            pick_model_dir,
            test_injection
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
