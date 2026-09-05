//! Prints what `caret::neighbours` sees in whatever field has focus, after a
//! short delay to switch to it.
//!
//!     cargo run --release --example caret_probe
fn main() {
    tracing_subscriber::fmt().with_env_filter("cooee=debug").with_writer(std::io::stdout).init();
    std::thread::sleep(std::time::Duration::from_secs(3));
    match cooee_lib::caret::neighbours() {
        Some(n) => println!("before={:?} after={:?} -> {:?}", n.before, n.after, cooee_lib::caret::pad("dictated", &n)),
        None => println!("no text pattern on the focused element"),
    }
}
