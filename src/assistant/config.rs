//! Static configuration knobs for the assistant.
//!
//! Everything here is read once at startup and held in `Arc<AssistantConfig>`.

use std::path::PathBuf;
use std::time::Duration;

/// Top-level configuration for an assistant session.
#[derive(Debug, Clone)]
pub struct AssistantConfig {
    /// GGUF model path for Voxtral ASR (Q4).
    pub asr_gguf: PathBuf,
    /// Tokenizer JSON path (Tekken).
    pub tokenizer_path: PathBuf,
    /// TTS GGUF model path (Q4). If `None`, ASR-only mode is used.
    pub tts_gguf: Option<PathBuf>,
    /// Voice preset directory for TTS.
    pub voices_dir: PathBuf,
    /// Selected voice preset name.
    pub voice: String,

    /// Audio I/O configuration.
    pub audio: AudioConfig,
    /// VAD / barge-in thresholds.
    pub vad: VadConfig,
    /// Latency-masking knobs.
    pub latency: LatencyConfig,

    /// Use hybrid RTX↔iGPU split (encoder on discrete, decoder on integrated).
    pub hybrid: bool,
    /// Enable Shannon-Prime VHT2 KV-cache compression.
    pub shannon_prime: bool,
    /// Hard cap on LLM KV-cache size (tokens) to keep USM allocations stable.
    pub max_kv_tokens: usize,

    /// Render the Sesame-style TUI instead of plain log output.
    pub tui: bool,
}

/// Mic and speaker configuration.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate fed into ASR. Voxtral expects 16 kHz mono.
    pub input_rate_hz: u32,
    /// Sample rate produced by TTS / consumed by speaker. Voxtral TTS is 24 kHz mono.
    pub output_rate_hz: u32,
    /// Mic capture chunk size in milliseconds (granularity into the pipeline).
    pub input_chunk_ms: u32,
    /// Speaker jitter buffer depth in milliseconds.
    pub output_jitter_ms: u32,
    /// Optional explicit input device name (None = default).
    pub input_device: Option<String>,
    /// Optional explicit output device name (None = default).
    pub output_device: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_rate_hz: 16_000,
            output_rate_hz: 24_000,
            input_chunk_ms: 20,
            output_jitter_ms: 80,
            input_device: None,
            output_device: None,
        }
    }
}

/// Voice activity detection / barge-in tuning.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Frames of detected speech required to enter `SpeechStart` (hysteresis on).
    pub speech_start_frames: u8,
    /// Frames of detected silence required to enter `SpeechEnd` (hysteresis off).
    pub speech_end_frames: u8,
    /// Energy threshold for the Phase 1 placeholder VAD (RMS in normalized [0,1]).
    pub energy_threshold: f32,
    /// Spectral entropy ceiling above which a frame is considered noise/silence.
    pub entropy_ceiling: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_start_frames: 3,
            speech_end_frames: 8,
            energy_threshold: 0.015,
            entropy_ceiling: 5.0,
        }
    }
}

/// Latency-masking parameters.
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    /// If LLM has not emitted a token after this delay, inject a filler.
    pub filler_after: Duration,
    /// Cross-fade duration when interrupting TTS playback.
    pub interrupt_fade: Duration,
    /// Run a dummy ASR+LLM+TTS pass on startup to JIT kernels.
    pub prewarm: bool,
    /// Loop ambient room tone under the conversation to keep the audio driver warm.
    pub ambient_tail: bool,
    /// Play the connection sound on session start.
    pub connection_sound: bool,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            filler_after: Duration::from_millis(100),
            interrupt_fade: Duration::from_millis(30),
            prewarm: true,
            ambient_tail: true,
            connection_sound: true,
        }
    }
}

impl AssistantConfig {
    /// Number of samples in one input chunk.
    pub fn input_chunk_samples(&self) -> usize {
        (self.audio.input_rate_hz as usize * self.audio.input_chunk_ms as usize) / 1000
    }

    /// Number of samples in the speaker jitter buffer.
    pub fn output_jitter_samples(&self) -> usize {
        (self.audio.output_rate_hz as usize * self.audio.output_jitter_ms as usize) / 1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_chunk_samples_at_16k_20ms() {
        let cfg = AssistantConfig {
            asr_gguf: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            tts_gguf: None,
            voices_dir: PathBuf::new(),
            voice: String::new(),
            audio: AudioConfig::default(),
            vad: VadConfig::default(),
            latency: LatencyConfig::default(),
            hybrid: false,
            shannon_prime: false,
            max_kv_tokens: 4096,
            tui: false,
        };
        assert_eq!(cfg.input_chunk_samples(), 320);
        assert_eq!(cfg.output_jitter_samples(), 1920);
    }
}
