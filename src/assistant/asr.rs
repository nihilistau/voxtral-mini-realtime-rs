//! ASR adapter — Voxtral Q4 GGUF transcription.
//!
//! On first call: lazily loads the Q4 Voxtral model + Tekken tokenizer +
//! mel extractor, then caches them in a process-global `OnceLock<Mutex<...>>`.
//! Subsequent calls reuse the warmed-up state.
//!
//! Falls back to a stub transcript when:
//!   - `cfg.asr_gguf` doesn't exist on disk, or
//!   - the tokenizer can't be loaded.
//!
//! This keeps the assistant runnable end-to-end even before models are
//! downloaded — useful for VAD/TUI/audio-loop development.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use tracing::{info, warn};

use crate::assistant::config::AssistantConfig;
use crate::audio::mel::{MelConfig, MelSpectrogram};
use crate::audio::pad::{pad_audio, PadConfig};
use crate::audio::AudioBuffer;
use crate::gguf::loader::Q4ModelLoader;
use crate::models::time_embedding::TimeEmbedding;
use crate::tokenizer::VoxtralTokenizer;

type Backend = Wgpu;

struct AsrState {
    device: burn::backend::wgpu::WgpuDevice,
    model: crate::gguf::model::Q4VoxtralModel,
    tokenizer: VoxtralTokenizer,
    mel: MelSpectrogram,
    pad_config: PadConfig,
    t_embed: Tensor<Backend, 3>,
}

static ASR: OnceLock<Mutex<Option<Result<AsrState, String>>>> = OnceLock::new();

/// Transcribe one utterance of 16 kHz mono samples to text.
pub fn transcribe(cfg: &AssistantConfig, samples: &[f32]) -> Result<String> {
    if samples.is_empty() {
        return Ok(String::new());
    }
    let lazy = ASR.get_or_init(|| Mutex::new(None));
    let mut guard = lazy.lock().map_err(|_| anyhow!("ASR state mutex poisoned"))?;
    if guard.is_none() {
        match load_state(cfg) {
            Ok(state) => {
                info!("Voxtral ASR state ready");
                *guard = Some(Ok(state));
            }
            Err(e) => {
                warn!(error = %e, "Voxtral ASR unavailable; falling back to stub");
                *guard = Some(Err(e.to_string()));
            }
        }
    }
    match guard.as_ref().unwrap() {
        Ok(_) => {
            // Drop the immutable borrow so we can grab a mutable one.
            let state = guard.as_mut().unwrap().as_mut().unwrap();
            state.transcribe(samples)
        }
        Err(_) => Ok(stub_transcript(samples)),
    }
}

impl AsrState {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let mut audio = AudioBuffer::new(samples.to_vec(), 16_000);
        audio.peak_normalize(0.95);

        let padded = pad_audio(&audio, &self.pad_config);
        let mel_frames = self.mel.compute_log(&padded.samples);
        if mel_frames.is_empty() {
            return Ok(String::new());
        }
        let n_frames = mel_frames.len();
        let n_mels = mel_frames[0].len();

        // [n_frames, n_mels] → [1, n_mels, n_frames]
        let mut transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (fi, frame) in mel_frames.iter().enumerate() {
            for (mi, &v) in frame.iter().enumerate() {
                transposed[mi][fi] = v;
            }
        }
        let flat: Vec<f32> = transposed.into_iter().flatten().collect();
        let mel_tensor: Tensor<Backend, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(flat, [1, n_mels, n_frames]),
            &self.device,
        );

        let generated = if self.model.is_hybrid() {
            self.model
                .transcribe_streaming_hybrid(mel_tensor, self.t_embed.clone())
        } else {
            self.model
                .transcribe_streaming(mel_tensor, self.t_embed.clone())
        };

        // Voxtral tokens < 1000 are control / streaming-pad tokens; > 1000 are
        // real text tokens.
        let text_ids: Vec<u32> = generated
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();
        let text = self
            .tokenizer
            .decode(&text_ids)
            .map_err(|e| anyhow!("tokenizer decode: {e:?}"))?;
        Ok(text.trim().to_string())
    }
}

fn load_state(cfg: &AssistantConfig) -> Result<AsrState> {
    if !cfg.asr_gguf.exists() {
        return Err(anyhow!("ASR GGUF not found at {}", cfg.asr_gguf.display()));
    }
    // Tokenizer fallback: next to the GGUF if the configured path doesn't exist.
    let tokenizer_path: PathBuf = if cfg.tokenizer_path.exists() {
        cfg.tokenizer_path.clone()
    } else {
        cfg.asr_gguf
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("tekken.json")
    };
    if !tokenizer_path.exists() {
        return Err(anyhow!("tokenizer not found at {}", tokenizer_path.display()));
    }

    // Hybrid = encoder on DiscreteGpu(0), decoder on IntegratedGpu(0) — matches
    // the existing `transcribe --hybrid` flag behavior.
    let device = if cfg.hybrid {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(0)
    } else {
        burn::backend::wgpu::WgpuDevice::default()
    };
    info!(?device, gguf = %cfg.asr_gguf.display(), "Loading Voxtral ASR Q4");

    let mut loader = Q4ModelLoader::from_file(&cfg.asr_gguf).context("open ASR GGUF")?;
    let mut model = if cfg.hybrid {
        let decoder_device = burn::backend::wgpu::WgpuDevice::IntegratedGpu(0);
        loader
            .load_hybrid(&device, &decoder_device)
            .context("load Q4 hybrid")?
    } else {
        loader.load(&device).context("load Q4 model")?
    };
    if cfg.shannon_prime || cfg.hybrid {
        let head_dim = model.decoder().head_dim();
        info!(head_dim, "Enabling Shannon-Prime VHT2 KV cache compression");
        model.enable_shannon_prime(head_dim);
    }

    info!(path = %tokenizer_path.display(), "Loading Voxtral tokenizer");
    let tokenizer =
        VoxtralTokenizer::from_file(&tokenizer_path).context("load Voxtral tokenizer")?;

    let mel = MelSpectrogram::new(MelConfig::voxtral());
    let pad_config = PadConfig::voxtral();
    let time_embed = TimeEmbedding::new(3072);
    // delay=6 tokens (~480 ms lookahead) is the recommended sweet spot from CLAUDE.md.
    let t_embed = time_embed.embed::<Backend>(6.0, &device);

    Ok(AsrState {
        device,
        model,
        tokenizer,
        mel,
        pad_config,
        t_embed,
    })
}

fn stub_transcript(samples: &[f32]) -> String {
    let duration_s = samples.len() as f32 / 16_000.0;
    format!("(stub transcript, {duration_s:.2}s of audio)")
}
