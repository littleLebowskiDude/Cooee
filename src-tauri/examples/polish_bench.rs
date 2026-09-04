//! Baseline for the polish comparison: times the rule-based pass over the
//! shared corpus so Phi Silica is measured against identical input.
//!
//!   cargo run --release --example polish_bench -- bench/transcripts.json

use cooee_lib::config::Dictionary;
use cooee_lib::polish::polish;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/transcripts.json".into());
    let raw = std::fs::read_to_string(&path)?;
    let cases: serde_json::Value = serde_json::from_str(&raw)?;
    let cases = cases.as_array().expect("corpus must be an array");

    let dict = Dictionary::default();
    let mut total_us = 0u128;

    println!("{:-<78}", "");
    for case in cases {
        let id = case["id"].as_str().unwrap_or("?");
        let input = case["raw"].as_str().unwrap_or("");

        // Warm, then measure: these are microsecond-scale operations.
        for _ in 0..100 {
            std::hint::black_box(polish(std::hint::black_box(input), &dict));
        }
        let t = std::time::Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            std::hint::black_box(polish(std::hint::black_box(input), &dict));
        }
        let us = t.elapsed().as_micros() / iterations;
        total_us += us;

        println!("[{id}]  {us} us");
        println!("  in : {input}");
        println!("  out: {}", polish(input, &dict));
        println!();
    }
    println!("{:-<78}", "");
    println!(
        "rules: {} us mean per transcript",
        total_us / cases.len() as u128
    );
    Ok(())
}
