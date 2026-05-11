//! Stage-level benchmark for the conversational assistant.
//!
//! Each pipeline stage (ASR → LLM → TTS) is exercised in isolation so you
//! can attribute end-to-end latency to a specific module:
//!
//! - **ASR**: transcribe a known WAV. Report wall time, audio duration,
//!   real-time factor (RTF), and decoded text.
//! - **LLM**: generate up to N tokens from a fixed prompt. Report TTFT,
//!   total time, token count, tokens/sec.
//! - **TTS**: synthesize a fixed phrase. Report wall time, output audio
//!   duration, RTF.
//!
//! Run:
//! ```text
//! cargo run --release --features "wgpu,cli,hub,llm" --bin voxtral-bench-assistant -- \
//!   --all \
//!   --asr-audio test_data/mary_had_lamb.wav \
//!   --llm-model "D:/Files/.../Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use tracing::info;

use voxtral_mini_realtime::assistant::config::{
    AssistantConfig, AudioConfig, LatencyConfig, VadConfig,
};

#[derive(Parser)]
#[command(name = "voxtral-bench-assistant")]
#[command(about = "Per-stage benchmark for the real-time assistant")]
struct Args {
    /// Run all three stages (ASR + LLM + TTS).
    #[arg(long, conflicts_with_all = &["asr", "llm", "tts"])]
    all: bool,
    /// Run only the ASR stage.
    #[arg(long)]
    asr: bool,
    /// Run only the LLM stage.
    #[arg(long)]
    llm: bool,
    /// Run only the TTS stage.
    #[arg(long)]
    tts: bool,

    /// ASR GGUF model.
    #[arg(long, default_value = "models/voxtral-q4.gguf")]
    asr_gguf: PathBuf,
    /// ASR audio file (16 kHz mono WAV).
    #[arg(long, default_value = "test_data/mary_had_lamb.wav")]
    asr_audio: PathBuf,
    /// Voxtral tokenizer JSON.
    #[arg(long, default_value = "models/tekken.json")]
    asr_tokenizer: PathBuf,

    /// Local LLM GGUF (Qwen2 family).
    #[cfg(feature = "llm")]
    #[arg(long)]
    llm_model: Option<PathBuf>,
    /// Tokenizer for the LLM. Auto-downloaded if missing.
    #[cfg(feature = "llm")]
    #[arg(long, default_value = "models/qwen2.5-0.5b-tokenizer.json")]
    llm_tokenizer: PathBuf,
    /// HuggingFace repo for the LLM tokenizer fallback.
    #[cfg(feature = "llm")]
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
    llm_hf_repo: String,
    /// Prompt for the LLM benchmark.
    #[cfg(feature = "llm")]
    #[arg(long, default_value = "What is two plus two? Reply with only the number.")]
    llm_prompt: String,
    /// Maximum tokens for the LLM benchmark.
    #[cfg(feature = "llm")]
    #[arg(long, default_value_t = 64)]
    llm_max_tokens: usize,

    /// TTS GGUF model.
    #[arg(long, default_value = "models/voxtral-tts-q4-gguf/voxtral-tts-q4.gguf")]
    tts_gguf: PathBuf,
    /// Voice preset name.
    #[arg(long, default_value = "casual_female")]
    voice: String,
    /// Voice preset directory.
    #[arg(long, default_value = "models/voxtral-tts-q4-gguf/voice_embedding")]
    voices_dir: PathBuf,
    /// Phrase to synthesize.
    #[arg(long, default_value = "Hello, the assistant is online.")]
    tts_text: String,

    /// Hybrid RTX→iGPU split.
    #[arg(long)]
    hybrid: bool,
    /// Enable Shannon-Prime VHT2 KV cache compression.
    #[arg(long)]
    shannon_prime: bool,

    /// Number of warm-up iterations before timed runs.
    #[arg(long, default_value_t = 1)]
    warmup: u32,
    /// Number of timed iterations.
    #[arg(long, default_value_t = 3)]
    iters: u32,
}

#[derive(Serialize, Default, Clone)]
struct StageReport {
    runs: Vec<RunResult>,
    median_ms: f64,
    median_rtf: Option<f64>,
    median_ttft_ms: Option<f64>,
    median_toks_per_sec: Option<f64>,
}

#[derive(Serialize, Default, Clone)]
struct RunResult {
    elapsed_ms: f64,
    audio_duration_s: Option<f64>,
    text: Option<String>,
    n_tokens: Option<u32>,
    ttft_ms: Option<u64>,
}

#[derive(Serialize, Default)]
struct FinalReport {
    asr: Option<StageReport>,
    #[cfg(feature = "llm")]
    llm: Option<StageReport>,
    tts: Option<StageReport>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    let run_asr = args.all || args.asr;
    let run_tts = args.all || args.tts;
    #[cfg(feature = "llm")]
    let run_llm = args.all || args.llm;
    #[cfg(not(feature = "llm"))]
    let run_llm = false;

    if !run_asr && !run_llm && !run_tts {
        bail!("specify at least one of --all, --asr, --llm, --tts");
    }

    let mut report = FinalReport::default();
    if run_asr {
        match bench_asr(&args) {
            Ok(r) => report.asr = Some(r),
            Err(e) => eprintln!("[asr] error: {e}"),
        }
    }
    #[cfg(feature = "llm")]
    if run_llm {
        match bench_llm(&args) {
            Ok(r) => report.llm = Some(r),
            Err(e) => eprintln!("[llm] error: {e}"),
        }
    }
    if run_tts {
        match bench_tts(&args) {
            Ok(r) => report.tts = Some(r),
            Err(e) => eprintln!("[tts] error: {e}"),
        }
    }

    println!("\n=== JSON ===\n{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// ASR
// ---------------------------------------------------------------------------

fn bench_asr(args: &Args) -> Result<StageReport> {
    use voxtral_mini_realtime::audio::io::load_wav;
    use voxtral_mini_realtime::audio::resample::resample_to_16k;

    println!("\n=== ASR ===");
    if !args.asr_gguf.exists() {
        bail!("ASR GGUF missing: {}", args.asr_gguf.display());
    }
    let audio = load_wav(args.asr_audio.to_str().unwrap()).context("load asr audio")?;
    let audio = if audio.sample_rate != 16_000 {
        resample_to_16k(&audio).context("resample to 16k")?
    } else {
        audio
    };
    let audio_duration_s = audio.samples.len() as f64 / 16_000.0;
    println!(
        "audio: {}s ({} samples), {} Hz",
        format_f(audio_duration_s),
        audio.samples.len(),
        16_000
    );

    let cfg = base_cfg(args);
    let cfg_arc = std::sync::Arc::new(cfg);

    // Warm up.
    for _ in 0..args.warmup {
        let _ = voxtral_mini_realtime::assistant::asr::transcribe(&cfg_arc, &audio.samples)?;
    }
    let mut runs = Vec::new();
    for i in 0..args.iters {
        let t0 = Instant::now();
        let text = voxtral_mini_realtime::assistant::asr::transcribe(&cfg_arc, &audio.samples)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rtf = (elapsed_ms / 1000.0) / audio_duration_s;
        info!(
            iter = i,
            elapsed_ms = format_f(elapsed_ms),
            rtf = format_f(rtf),
            "asr"
        );
        println!(
            "  iter {i}: {} ms  RTF={}  text=\"{}\"",
            format_f(elapsed_ms),
            format_f(rtf),
            text.chars().take(80).collect::<String>()
        );
        runs.push(RunResult {
            elapsed_ms,
            audio_duration_s: Some(audio_duration_s),
            text: Some(text),
            ..Default::default()
        });
    }
    Ok(summarize(runs, Some(audio_duration_s), None, None))
}

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

#[cfg(feature = "llm")]
fn bench_llm(args: &Args) -> Result<StageReport> {
    use voxtral_mini_realtime::assistant::llm::{self, LlmConfig, LlmEvent};

    println!("\n=== LLM ===");
    let model_path = args
        .llm_model
        .clone()
        .context("--llm-model is required for the LLM benchmark")?;
    if !model_path.exists() {
        bail!("LLM GGUF missing: {}", model_path.display());
    }

    let lcfg = LlmConfig {
        gguf_path: model_path,
        tokenizer_path: args.llm_tokenizer.clone(),
        hf_repo: args.llm_hf_repo.clone(),
        max_new_tokens: args.llm_max_tokens,
        temperature: 0.7,
        top_p: 0.9,
        seed: 42,
        system_prompt: "Reply with one short sentence.".to_string(),
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<LlmEvent>();
    let handle = llm::spawn(lcfg, tx)?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let mut runs = Vec::new();
    let total_iters = args.warmup + args.iters;
    for i in 0..total_iters {
        let is_warm = i < args.warmup;
        handle
            .prompt_tx
            .send(args.llm_prompt.clone())
            .context("LLM prompt_tx closed")?;
        let t0 = Instant::now();
        let mut reply = String::new();
        let mut n_tokens = 0u32;
        let mut ttft_ms: Option<u64> = None;
        let mut total_ms_from_llm: Option<u64> = None;
        rt.block_on(async {
            while let Some(evt) = rx.recv().await {
                match evt {
                    LlmEvent::Token(piece) => {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(t0.elapsed().as_millis() as u64);
                        }
                        reply.push_str(&piece);
                    }
                    LlmEvent::Done {
                        n_tokens: n,
                        total_ms,
                        ..
                    } => {
                        n_tokens = n;
                        total_ms_from_llm = Some(total_ms);
                        break;
                    }
                }
            }
        });
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let toks_per_sec = if total_ms_from_llm.unwrap_or(0) > 0 {
            (n_tokens as f64) * 1000.0 / (total_ms_from_llm.unwrap() as f64)
        } else {
            0.0
        };
        let tag = if is_warm { "warmup" } else { "timed " };
        println!(
            "  {tag} {i}: {} ms  TTFT={} ms  n={n_tokens}  {} tok/s  reply=\"{}\"",
            format_f(elapsed_ms),
            ttft_ms.map(|m| m.to_string()).unwrap_or_else(|| "-".into()),
            format_f(toks_per_sec),
            reply.replace('\n', " ").chars().take(80).collect::<String>()
        );
        if !is_warm {
            runs.push(RunResult {
                elapsed_ms,
                audio_duration_s: None,
                text: Some(reply),
                n_tokens: Some(n_tokens),
                ttft_ms,
            });
        }
    }
    Ok(summarize_llm(runs))
}

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------

fn bench_tts(args: &Args) -> Result<StageReport> {
    println!("\n=== TTS ===");
    if !args.tts_gguf.exists() {
        bail!("TTS GGUF missing: {}", args.tts_gguf.display());
    }
    let cfg = base_cfg(args);
    let cfg_arc = std::sync::Arc::new(cfg);

    for _ in 0..args.warmup {
        let _ = voxtral_mini_realtime::assistant::tts::synthesize(&cfg_arc, &args.tts_text)?;
    }
    let mut runs = Vec::new();
    for i in 0..args.iters {
        let t0 = Instant::now();
        let samples =
            voxtral_mini_realtime::assistant::tts::synthesize(&cfg_arc, &args.tts_text)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let audio_duration_s = samples.len() as f64 / 24_000.0;
        let rtf = if audio_duration_s > 0.0 {
            (elapsed_ms / 1000.0) / audio_duration_s
        } else {
            f64::NAN
        };
        println!(
            "  iter {i}: {} ms  audio={}s  RTF={}",
            format_f(elapsed_ms),
            format_f(audio_duration_s),
            format_f(rtf)
        );
        runs.push(RunResult {
            elapsed_ms,
            audio_duration_s: Some(audio_duration_s),
            ..Default::default()
        });
        let _ = i;
    }
    let last_dur = runs.last().and_then(|r| r.audio_duration_s);
    Ok(summarize(runs, last_dur, None, None))
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

fn base_cfg(args: &Args) -> AssistantConfig {
    AssistantConfig {
        asr_gguf: args.asr_gguf.clone(),
        tokenizer_path: args.asr_tokenizer.clone(),
        tts_gguf: Some(args.tts_gguf.clone()),
        voices_dir: args.voices_dir.clone(),
        voice: args.voice.clone(),
        audio: AudioConfig::default(),
        vad: VadConfig::default(),
        latency: LatencyConfig::default(),
        hybrid: args.hybrid,
        shannon_prime: args.shannon_prime,
        max_kv_tokens: 4096,
        tui: false,
        #[cfg(feature = "llm")]
        llm: None,
    }
}

fn summarize(
    runs: Vec<RunResult>,
    audio_duration_s: Option<f64>,
    _ttft: Option<f64>,
    _tps: Option<f64>,
) -> StageReport {
    let median_ms = median(runs.iter().map(|r| r.elapsed_ms).collect());
    let median_rtf =
        audio_duration_s.map(|d| (median_ms / 1000.0) / d.max(1e-9));
    StageReport {
        runs,
        median_ms,
        median_rtf,
        median_ttft_ms: None,
        median_toks_per_sec: None,
    }
}

#[cfg(feature = "llm")]
fn summarize_llm(runs: Vec<RunResult>) -> StageReport {
    let median_ms = median(runs.iter().map(|r| r.elapsed_ms).collect());
    let median_ttft_ms = median(
        runs.iter()
            .filter_map(|r| r.ttft_ms.map(|m| m as f64))
            .collect(),
    );
    let total_tokens: f64 = runs
        .iter()
        .filter_map(|r| r.n_tokens.map(|n| n as f64))
        .sum::<f64>();
    let total_ms: f64 = runs.iter().map(|r| r.elapsed_ms).sum();
    let toks_per_sec = if total_ms > 0.0 {
        Some(total_tokens * 1000.0 / total_ms)
    } else {
        None
    };
    StageReport {
        runs,
        median_ms,
        median_rtf: None,
        median_ttft_ms: Some(median_ttft_ms),
        median_toks_per_sec: toks_per_sec,
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n.is_multiple_of(2) {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    } else {
        xs[n / 2]
    }
}

fn format_f(x: f64) -> String {
    if x.abs() < 0.01 {
        format!("{x:.4}")
    } else if x.abs() < 10.0 {
        format!("{x:.3}")
    } else if x.abs() < 1000.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.0}")
    }
}
