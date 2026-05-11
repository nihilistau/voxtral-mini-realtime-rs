//! TTS adapter — Voxtral Q4 GGUF pipeline.
//!
//! On first call: lazily loads the Q4 backbone + flow-matching transformer +
//! codec decoder + tokenizer + the configured voice preset, then caches them
//! in a process-global `OnceLock<Mutex<...>>`. Subsequent calls reuse the
//! warmed-up state.
//!
//! Falls back to a synthesized 880 Hz blip when:
//!   - `cfg.tts_gguf` is None (assistant launched without a TTS model), or
//!   - the model file at `cfg.tts_gguf` doesn't exist, or
//!   - the voice / tokenizer can't be loaded (warning logged once).
//!
//! Called from the orchestrator's pipeline via `tokio::task::spawn_blocking`,
//! so blocking the thread inside is fine. We use `pollster::block_on` to
//! drive the async GPU calls inside `generate_async`.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use tracing::{info, warn};

use crate::assistant::config::AssistantConfig;
use crate::gguf::Q4TtsModelLoader;
use crate::tokenizer::TekkenEncoder;
use crate::tts::config::{AudioCodebookLayout, TtsSpecialTokens};
use crate::tts::embeddings::AudioCodebookEmbeddings;
use crate::tts::voice::load_voice_from_bytes;

type Backend = Wgpu;

/// Resources held across calls. Wrapped in a `Mutex` because Burn tensors
/// and the WgpuDevice are `Send` but not `Sync`; serializing synth calls
/// is fine since the orchestrator pipeline is sequential.
struct TtsState {
    device: burn::backend::wgpu::WgpuDevice,
    backbone: crate::gguf::tts_model::Q4TtsBackbone,
    fm: crate::gguf::tts_model::Q4FmTransformer,
    codec: crate::tts::codec::CodecDecoder<Backend>,
    codebook: AudioCodebookEmbeddings<Backend>,
    voice_embed: Tensor<Backend, 2>,
    tokenizer: TekkenEncoder,
    special: TtsSpecialTokens,
    max_frames: usize,
}

static TTS: OnceLock<Mutex<Option<Result<TtsState, String>>>> = OnceLock::new();

/// Synthesize text to 24 kHz mono PCM. Lazy-loads the model on first call.
pub fn synthesize(cfg: &AssistantConfig, text: &str) -> Result<Vec<f32>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let lazy = TTS.get_or_init(|| Mutex::new(None));
    let mut guard = lazy.lock().map_err(|_| anyhow!("TTS state mutex poisoned"))?;
    if guard.is_none() {
        match load_state(cfg) {
            Ok(state) => {
                info!(text_len = text.len(), "Voxtral TTS state ready");
                *guard = Some(Ok(state));
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "Voxtral TTS unavailable; falling back to synthesized blip"
                );
                *guard = Some(Err(e.to_string()));
            }
        }
    }
    match guard.as_mut().unwrap() {
        Ok(state) => state.synthesize(text),
        Err(_) => Ok(synth_blip(cfg, text)),
    }
}

impl TtsState {
    fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self.tokenizer.encode(text);
        let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let s = &self.special;
        let bos = self
            .backbone
            .embed_tokens_from_ids(&[s.bos_token_id as i32], 1, 1);
        let begin_audio = self
            .backbone
            .embed_tokens_from_ids(&[s.begin_audio_token_id as i32], 1, 1);
        let next_audio_text =
            self.backbone
                .embed_tokens_from_ids(&[s.next_audio_text_token_id as i32], 1, 1);
        let repeat_audio_text =
            self.backbone
                .embed_tokens_from_ids(&[s.repeat_audio_text_token_id as i32], 1, 1);
        let text_embeds = self
            .backbone
            .embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());

        let input_sequence = Tensor::cat(
            vec![
                bos,
                begin_audio.clone(),
                self.voice_embed.clone().unsqueeze_dim::<3>(0),
                next_audio_text,
                text_embeds,
                repeat_audio_text,
                begin_audio,
            ],
            1,
        );

        let frames = pollster::block_on(self.backbone.generate_async(
            input_sequence,
            &self.fm,
            &self.codebook,
            self.max_frames,
        ))
        .map_err(|e| anyhow!("TTS generation failed: {e}"))?;

        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let n_frames = frames.len();
        let semantic_indices: Vec<usize> = frames.iter().map(|f| f.semantic_idx).collect();
        let mut acoustic_data = Vec::with_capacity(n_frames * 36);
        for frame in &frames {
            for &level in &frame.acoustic_levels {
                acoustic_data.push(level as f32);
            }
        }
        let acoustic_tensor: Tensor<Backend, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(acoustic_data, [n_frames, 36]),
            &self.device,
        );
        let waveform = self.codec.decode(&semantic_indices, acoustic_tensor);
        let [_batch, total_samples] = waveform.dims();
        let data = waveform.to_data();
        let slice = data
            .as_slice::<f32>()
            .map_err(|e| anyhow!("waveform readback: {e:?}"))?;
        let mut samples = slice[..total_samples].to_vec();

        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1e-6 {
            let gain = 0.95 / peak;
            for s in &mut samples {
                *s *= gain;
            }
        }
        Ok(samples)
    }
}

fn load_state(cfg: &AssistantConfig) -> Result<TtsState> {
    let gguf_path = cfg
        .tts_gguf
        .clone()
        .context("no TTS GGUF path configured")?;
    if !gguf_path.exists() {
        return Err(anyhow!("TTS GGUF not found at {}", gguf_path.display()));
    }

    // Voice + tokenizer paths fall back to the TTS GGUF's directory if the
    // explicit cfg paths point at the ASR-side defaults.
    let voice_path: PathBuf = {
        let preferred = cfg.voices_dir.join(format!("{}.safetensors", cfg.voice));
        if preferred.exists() {
            preferred
        } else {
            let sibling = gguf_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join("voice_embedding")
                .join(format!("{}.safetensors", cfg.voice));
            if sibling.exists() {
                sibling
            } else {
                return Err(anyhow!(
                    "voice preset '{}' not found at {} or {}",
                    cfg.voice,
                    preferred.display(),
                    cfg.voices_dir.display()
                ));
            }
        }
    };

    let tokenizer_path: PathBuf = if cfg.tokenizer_path.exists() {
        cfg.tokenizer_path.clone()
    } else {
        gguf_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("tekken.json")
    };
    if !tokenizer_path.exists() {
        return Err(anyhow!("tokenizer not found at {}", tokenizer_path.display()));
    }

    let device = match (cfg.hybrid, &cfg.audio) {
        (true, _) => burn::backend::wgpu::WgpuDevice::DiscreteGpu(0),
        _ => burn::backend::wgpu::WgpuDevice::default(),
    };
    info!(?device, gguf = %gguf_path.display(), "Loading Voxtral TTS Q4");

    let mut loader = Q4TtsModelLoader::from_file(&gguf_path).context("open TTS GGUF")?;
    let (backbone, mut fm, codec) = loader.load(&device).context("load TTS Q4 model")?;
    // Euler 3 = real-time setting (RTF ~14x on Q4 per CLAUDE.md).
    fm.set_euler_steps(3);

    info!(path = %tokenizer_path.display(), "Loading Tekken tokenizer");
    let tokenizer = TekkenEncoder::from_file(&tokenizer_path).context("load Tekken tokenizer")?;

    info!(path = %voice_path.display(), "Loading voice preset");
    let voice_bytes = std::fs::read(&voice_path).context("read voice safetensors")?;
    let voice_embed: Tensor<Backend, 2> =
        load_voice_from_bytes(&voice_bytes, 3072, &device).context("parse voice safetensors")?;

    let codebook = AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        AudioCodebookLayout::default(),
    );
    let special = TtsSpecialTokens::default();

    Ok(TtsState {
        device,
        backbone,
        fm,
        codec,
        codebook,
        voice_embed,
        tokenizer,
        special,
        max_frames: 2000,
    })
}

/// Fallback synthesizer: 150–650 ms 880 Hz blip with envelope, used when the
/// Voxtral TTS isn't available.
fn synth_blip(cfg: &AssistantConfig, text: &str) -> Vec<f32> {
    let sr = cfg.audio.output_rate_hz as f32;
    let dur_s = 0.15f32 + (text.len().min(80) as f32) * 0.01;
    let n = (sr * dur_s) as usize;
    let amp = 0.4 * ((text.len().min(50) as f32) / 50.0).min(1.0) + 0.1;
    let freq = 880.0f32;
    let two_pi = std::f32::consts::TAU;
    let mut out = Vec::with_capacity(n);
    let env_n = (sr * 0.02) as usize;
    for i in 0..n {
        let env = if i < env_n {
            i as f32 / env_n as f32
        } else if i > n.saturating_sub(env_n) {
            (n - i) as f32 / env_n as f32
        } else {
            1.0
        };
        out.push((two_pi * freq * (i as f32) / sr).sin() * amp * env);
    }
    out
}
