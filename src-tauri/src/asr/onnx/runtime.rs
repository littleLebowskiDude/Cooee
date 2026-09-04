//! Process-wide ONNX Runtime setup: load `onnxruntime.dll`, register the QNN
//! plugin EP, find the NPU. Done once; every engine and the bench share it.
//!
//! The DLLs are looked for in `runtime\` beside the executable first (the
//! installed layout), then beside it, then at `ORT_DYLIB_PATH` /
//! `QNN_EP_PATH`, then in the pip packages' site-packages (the dev-machine
//! layout, see docs/NPU.md).

use anyhow::{anyhow, Context, Result};
use once_cell::sync::OnceCell;
use ort::device::Device;
use ort::environment::Environment;
use ort::logging::LogLevel;
use ort::memory::DeviceType;
use std::path::PathBuf;

/// The plugin EP takes its name from the registration; the options prefix
/// and `Device::ep()` both report it. This is the name the pip package uses.
pub const QNN_EP: &str = "QNNExecutionProvider";

pub struct Runtime {
    pub ort_dll: PathBuf,
    pub qnn_ep: Option<PathBuf>,
    /// A QNN NPU device was enumerated after registering the plugin.
    pub npu: bool,
}

static RUNTIME: OnceCell<std::result::Result<Runtime, String>> = OnceCell::new();

/// Loads the runtime on first call; later calls return the same outcome.
pub fn init() -> Result<&'static Runtime> {
    RUNTIME
        .get_or_init(|| init_once().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

fn init_once() -> Result<Runtime> {
    let ort_dll = locate("ORT_DYLIB_PATH", "onnxruntime.dll", &["onnxruntime", "capi", "onnxruntime.dll"])
        .context("onnxruntime.dll not found beside the app, at ORT_DYLIB_PATH, or in a pip install")?;
    let committed = ort::init_from(&ort_dll)
        .map_err(|e| anyhow!("load {}: {e}", ort_dll.display()))?
        .with_name("cooee")
        .commit();
    if !committed {
        tracing::debug!("ort environment was already committed; reusing it");
    }
    let env = Environment::current()?;
    // ORT's own logger writes to stderr. At Warning it reports every QNN
    // fusion it could not do and calls the cached context "suboptimal" (it
    // runs at the same speed); neither is actionable, so errors only.
    env.set_log_level(LogLevel::Error);

    let qnn_ep = locate(
        "QNN_EP_PATH",
        "onnxruntime_providers_qnn.dll",
        &["onnxruntime_qnn", "onnxruntime_providers_qnn.dll"],
    );
    let qnn_ep = match qnn_ep {
        Some(path) => match env.register_ep_library(QNN_EP, &path) {
            // The handle only matters for unregistering; dropping it keeps the EP.
            Ok(_) => Some(path),
            Err(e) => {
                tracing::warn!("QNN EP at {} did not register ({e}); encoder stays on the CPU", path.display());
                None
            }
        },
        None => {
            tracing::warn!("no QNN EP library found; encoder stays on the CPU");
            None
        }
    };
    let npu = qnn_ep.is_some() && npu_devices(&env).next().is_some();
    for d in env.devices() {
        let hw = d.hardware_device();
        tracing::info!(ep = d.ep().unwrap_or("?"), ty = ?hw.ty(), vendor = hw.vendor().unwrap_or("?"), "onnxruntime device");
    }
    tracing::info!(ort = %ort_dll.display(), npu, "onnxruntime loaded");
    Ok(Runtime { ort_dll, qnn_ep, npu })
}

/// QNN devices backed by the NPU. `Device` borrows the environment and is
/// not `Send`, so callers enumerate on the thread that builds the session.
pub fn npu_devices(env: &Environment) -> impl Iterator<Item = Device<'_>> + '_ {
    env.devices()
        .filter(|d| d.ep().ok() == Some(QNN_EP) && d.hardware_device().ty() == DeviceType::NPU)
}

fn locate(var: &str, beside_exe: &str, site_rel: &[&str]) -> Option<PathBuf> {
    // The installers put bundle resources under `runtime\` in the install
    // directory (tauri.onnx.conf.json); a bare copy beside the exe also works.
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from)) {
        for p in [dir.join("runtime").join(beside_exe), dir.join(beside_exe)] {
            if p.exists() {
                return Some(p);
            }
        }
    }
    if let Some(v) = std::env::var_os(var) {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    let mut p = site_packages()?;
    p.extend(site_rel);
    p.exists().then_some(p)
}

/// The pip packages are the dev-time source of the DLLs. Bundling is later.
fn site_packages() -> Option<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")?;
    let programs = PathBuf::from(local).join("Programs").join("Python");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(programs)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("Lib").join("site-packages"))
        .filter(|p| p.join("onnxruntime").is_dir())
        .collect();
    dirs.sort();
    dirs.pop()
}
