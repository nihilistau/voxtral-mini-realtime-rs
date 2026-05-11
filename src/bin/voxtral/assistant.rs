//! `voxtral assistant` subcommand — real-time conversational assistant.
//!
//! Phase 1: validates the audio loop wiring (mic → VAD → stub ASR → stub TTS
//! → speaker). Real ASR/TTS land in Phase 2; LLM in Phase 2 too.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use voxtral_mini_realtime::assistant::{
    config::{AudioConfig, LatencyConfig, VadConfig},
    AssistantConfig, AssistantOrchestrator,
};

#[derive(Parser)]
pub struct Args {
    /// Q4 GGUF ASR model.
    #[arg(long, default_value = "models/voxtral-q4.gguf")]
    pub asr_gguf: PathBuf,

    /// Q4 GGUF TTS model. Defaults to the bundled location;
    /// omit explicitly with --no-tts to disable.
    #[arg(long, default_value = "models/voxtral-tts-q4-gguf/voxtral-tts-q4.gguf")]
    pub tts_gguf: Option<PathBuf>,

    /// Tokenizer JSON (Tekken). Tries `models/tekken.json` then falls back
    /// to the TTS GGUF's parent directory.
    #[arg(long, default_value = "models/tekken.json")]
    pub tokenizer: PathBuf,

    /// Voice preset directory.
    #[arg(long, default_value = "models/voxtral-tts-q4-gguf/voice_embedding")]
    pub voices_dir: PathBuf,

    /// Voice preset name.
    #[arg(long, default_value = "casual_female")]
    pub voice: String,

    /// Mic input device name (default: system default).
    #[arg(long)]
    pub mic: Option<String>,

    /// Speaker output device name (default: system default).
    #[arg(long)]
    pub speaker: Option<String>,

    /// Mic input sample rate (after resample). Voxtral wants 16 kHz.
    #[arg(long, default_value_t = 16_000)]
    pub input_rate: u32,

    /// Speaker output sample rate. Voxtral TTS is 24 kHz.
    #[arg(long, default_value_t = 24_000)]
    pub output_rate: u32,

    /// Hybrid RTX↔iGPU split.
    #[arg(long)]
    pub hybrid: bool,

    /// Enable Shannon-Prime VHT2 KV-cache compression.
    #[arg(long)]
    pub shannon_prime: bool,

    /// Hard cap on LLM KV-cache size (tokens).
    #[arg(long, default_value_t = 4096)]
    pub max_kv_tokens: usize,

    /// Render the Sesame-style TUI (Phase 5; falls back to logs for Phase 1).
    #[arg(long)]
    pub tui: bool,

    /// VAD energy threshold (RMS, 0..1). Lower = more sensitive.
    #[arg(long, default_value_t = 0.015)]
    pub vad_threshold: f32,

    /// Disable pre-warmup pass.
    #[arg(long)]
    pub no_prewarm: bool,

    /// Disable the under-ambient room tone.
    #[arg(long)]
    pub no_ambient: bool,

    /// Disable the initial connection sound.
    #[arg(long)]
    pub no_connection_sound: bool,

    /// Local LLM GGUF (Qwen2 family). Omit for echo mode.
    #[cfg(feature = "llm")]
    #[arg(long)]
    pub llm_model: Option<PathBuf>,

    /// Tokenizer JSON for the LLM (auto-downloaded from HF if missing).
    #[cfg(feature = "llm")]
    #[arg(long, default_value = "models/qwen2.5-0.5b-tokenizer.json")]
    pub llm_tokenizer: PathBuf,

    /// HuggingFace repo for the LLM tokenizer fallback.
    #[cfg(feature = "llm")]
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
    pub llm_hf_repo: String,

    /// System prompt prepended to every LLM turn.
    #[cfg(feature = "llm")]
    #[arg(
        long,
        default_value = "You are a concise spoken assistant. Reply in one short sentence."
    )]
    pub system_prompt: String,

    /// Max tokens per LLM reply.
    #[cfg(feature = "llm")]
    #[arg(long, default_value_t = 120)]
    pub llm_max_tokens: usize,

    /// LLM sampling temperature. 0 = greedy.
    #[cfg(feature = "llm")]
    #[arg(long, default_value_t = 0.7)]
    pub llm_temperature: f64,
}

pub fn run(args: Args) -> Result<()> {
    #[cfg(feature = "llm")]
    let llm_cfg = args.llm_model.as_ref().map(|path| {
        voxtral_mini_realtime::assistant::llm::LlmConfig {
            gguf_path: path.clone(),
            tokenizer_path: args.llm_tokenizer.clone(),
            hf_repo: args.llm_hf_repo.clone(),
            max_new_tokens: args.llm_max_tokens,
            temperature: args.llm_temperature,
            top_p: 0.9,
            seed: 42,
            system_prompt: args.system_prompt.clone(),
        }
    });

    let cfg = AssistantConfig {
        asr_gguf: args.asr_gguf,
        tokenizer_path: args.tokenizer,
        tts_gguf: args.tts_gguf,
        voices_dir: args.voices_dir,
        voice: args.voice,
        audio: AudioConfig {
            input_rate_hz: args.input_rate,
            output_rate_hz: args.output_rate,
            input_chunk_ms: 20,
            output_jitter_ms: 80,
            input_device: args.mic,
            output_device: args.speaker,
        },
        vad: VadConfig {
            energy_threshold: args.vad_threshold,
            ..VadConfig::default()
        },
        latency: LatencyConfig {
            prewarm: !args.no_prewarm,
            ambient_tail: !args.no_ambient,
            connection_sound: !args.no_connection_sound,
            ..LatencyConfig::default()
        },
        hybrid: args.hybrid,
        shannon_prime: args.shannon_prime,
        max_kv_tokens: args.max_kv_tokens,
        tui: args.tui,
        #[cfg(feature = "llm")]
        llm: llm_cfg,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;

    runtime.block_on(AssistantOrchestrator::new(cfg).run())
}
