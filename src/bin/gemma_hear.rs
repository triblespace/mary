//! Gemma 4 audio understanding, end-to-end inside mary:
//! audio file → symphonia decode → rubato resample 16 kHz → log-mel
//! (AudioFeatureExtractor) → Gemma 4 audio tower → embedder → text decoder →
//! generation, all via the shared [`mary::models::gemma::gemma4::hear::Hearing`]
//! seam (`gemma_listen` runs the same seam in a live loop). Weights come ONLY
//! from a native model-collection pile (create one with `mary import <model>
//! --pile <path> --key <key>`); `config.json` + `tokenizer.json` stay small
//! files resolved from the local HF snapshot of `--model` (default E4B).
//! Explicit `--config-json` and `--tokenizer-json` bypass that resolution for
//! pinned offline controls. This uses the same hearing-specific backend as
//! the resident faculty: CUDA on Linux, WGPU elsewhere. The
//! audio path is parity-gated per stage against
//! HF goldens (`gemma_audio_parity`, cos = 1.0 features/tower/embedder/cascade,
//! shards AND pile, 2026-07-10).
//!
//! Besides transcribing, this bin doubles as the hearing feasibility
//! meter: it prints the audio soft-token rate, the encode realtime factor
//! (cold = shader compile, then warm), per-run end-to-end latency, and RSS —
//! the numbers that decide whether an always-on chunked hearing loop fits
//! next to the realtime voice + the 31B on one machine. `--repeat N` re-runs
//! `understand` on the warm stack for contention-robust timing (best-of-N).
//!
//! Usage:
//!   cargo run --release --features gemma --bin gemma_hear -- \
//!     --pile /path/to/gemma_e4b.pile --audio /tmp/sample.flac \
//!     --prompt "Transcribe exactly what is being said." --tokens 60 --repeat 3
//!
//! `--pile` falls back to the `GEMMA_PILE` env var.

use burn::prelude::*;
use mary::models::gemma::gemma4::audio_load::load_audio_16k_mono;
use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::hear::Hearing;
use mary::nn::backend::hear::{Device, B};
use mary::persist::load_gemma4_hearing_from_pile;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Resolve a SMALL side-file (config.json / tokenizer.json) from the local HF
/// snapshot. Weights never come from here — they load from the pile.
fn find_hf_file(model_id: &str, filename: &str) -> String {
    let o = Command::new("python3")
        .args([
            "-c",
            &format!(
                "from huggingface_hub import hf_hub_download; print(hf_hub_download('{}', '{}'))",
                model_id, filename
            ),
        ])
        .output()
        .unwrap();
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|s| s == k)
        .map(|i| args[i + 1].clone())
}

/// Current process resident-set size in GiB (macOS `ps`). GPU allocations on
/// Apple silicon are unified memory, so this is the real system footprint.
fn rss_gib() -> f64 {
    let pid = std::process::id().to_string();
    let o = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .unwrap();
    let kb: f64 = String::from_utf8(o.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap_or(0.0);
    kb / 1048576.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let audio_path = arg(&args, "--audio").expect("need --audio <path>");
    let question =
        arg(&args, "--prompt").unwrap_or_else(|| "What is being said in this audio?".into());
    let max_new = arg(&args, "--tokens")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(150);
    let repeat = arg(&args, "--repeat")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let pile = arg(&args, "--pile")
        .or_else(|| std::env::var("GEMMA_PILE").ok())
        .unwrap_or_else(|| {
            eprintln!(
                "gemma_hear: pass --pile <gemma.pile> or set GEMMA_PILE (create one with mary import)"
            );
            std::process::exit(2);
        });
    let model_id = arg(&args, "--model").unwrap_or_else(|| "google/gemma-4-E4B-it".into());

    let device = Device::default();

    // --- Load audio from disk (symphonia + rubato) ---
    println!("Loading {audio_path}...");
    let wave =
        load_audio_16k_mono(Path::new(&audio_path)).unwrap_or_else(|e| panic!("audio load: {e}"));
    let audio_secs = wave.len() as f64 / 16_000.0;
    println!("  {} samples @ 16 kHz ({audio_secs:.2}s)", wave.len());

    // --- Load model + audio tower + embedder (weights: pile-only) ---
    let config_path =
        arg(&args, "--config-json").unwrap_or_else(|| find_hf_file(&model_id, "config.json"));
    let mut config = Gemma4Config::load(Path::new(&config_path));
    // This is the hearing path: skip the vision tower so the footprint numbers
    // reflect hearing alone (decoder + audio tower + embedder).
    config.vision_config = None;
    let tokenizer_path =
        arg(&args, "--tokenizer-json").unwrap_or_else(|| find_hf_file(&model_id, "tokenizer.json"));
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();

    println!("Loading model from pile {pile}...");
    let t_load = Instant::now();
    let (model, _vision, tower, embedder) = load_gemma4_hearing_from_pile::<B>(
        Path::new(&pile),
        mary::selection::ModelSelector::Source {
            source: &model_id,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
        config,
        &device,
    )
    .unwrap_or_else(|e| panic!("pile load: {e}"));
    println!(
        "Loaded in {:.1}s. RSS {:.2} GiB",
        t_load.elapsed().as_secs_f64(),
        rss_gib()
    );

    // --- Transcriber-stage meter: soft-token rate + encode realtime factor ---
    // Cold pass includes GPU kernel compilation; warm is the steady state an
    // always-on hearing loop would see. The readback forces GPU sync, so the
    // timing covers the full features→tower→embedder→CPU path.
    let fe = mary::models::gemma::gemma4::audio_preprocess::AudioFeatureExtractor::new();
    let (feat, _mask, n_frames) = fe.extract(&wave);
    let mut n_soft = 0usize;
    let mut warm_secs = f64::INFINITY;
    for pass in ["cold", "warm"] {
        let t = Instant::now();
        let input =
            Tensor::<B, 1>::from_floats(&feat[..], &device).reshape([1, n_frames, fe.feature_size]);
        let out = tower.forward(input);
        let [_, n_tok, mh] = out.dims();
        let emb = embedder.forward(out.reshape([n_tok, mh]));
        let v: Vec<f32> = emb.to_data().to_vec().unwrap();
        let secs = t.elapsed().as_secs_f64();
        let finite = v.iter().filter(|x| x.is_finite()).count();
        println!(
            "[stt] encode ({pass}): {n_frames} mel frames → {n_tok} soft tokens in {secs:.3}s \
             ({finite}/{} finite)",
            v.len()
        );
        n_soft = n_tok;
        warm_secs = secs;
    }
    println!(
        "[stt] soft-token rate: {:.2} tok/s of audio ({audio_secs:.2}s → {n_soft} tokens)",
        n_soft as f64 / audio_secs
    );
    println!(
        "[stt] encode RTF (warm): {:.0}x realtime ({warm_secs:.3}s for {audio_secs:.2}s)",
        audio_secs / warm_secs
    );

    let hearing = Hearing::new(model, tower, embedder, tokenizer, device);
    println!("\n--- Generating ---\n");

    let mut best = f64::INFINITY;
    for run in 1..=repeat {
        let t = Instant::now();
        let text = hearing.understand(&wave, &question, max_new, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        });
        let secs = t.elapsed().as_secs_f64();
        if secs < best {
            best = secs;
        }
        println!(
            "\n\n[run {run}/{repeat}] {} chars in {secs:.2}s ({:.2}x realtime; \
             hear path: features → tower → embed → prefill → greedy decode)",
            text.len(),
            audio_secs / secs
        );
    }
    if repeat > 1 {
        println!(
            "[stt] best-of-{repeat} understand: {best:.2}s ({:.2}x realtime)",
            audio_secs / best
        );
    }
    println!("[stt] final RSS: {:.2} GiB", rss_gib());
}
