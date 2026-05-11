//! TTS adapter.
//!
//! Phase 1: stub — produces a short 880 Hz "blip" tone so the speaker path can
//! be validated without model setup.
//!
//! Phase 2 will replace this with a model-owner thread that loads the Q4
//! Voxtral TTS pipeline once and streams audio chunks as they synthesize.

use anyhow::Result;

use crate::assistant::config::AssistantConfig;

/// Synthesize text to 24 kHz mono PCM. Phase-1 stub emits a 150 ms 880 Hz
/// sine envelope scaled by `text.len()` (longer text → louder, up to -6 dB).
pub fn synthesize(cfg: &AssistantConfig, text: &str) -> Result<Vec<f32>> {
    let sr = cfg.audio.output_rate_hz as f32;
    let dur_s = 0.15f32 + (text.len().min(80) as f32) * 0.01;
    let n = (sr * dur_s) as usize;
    let amp = 0.4 * ((text.len().min(50) as f32) / 50.0).min(1.0) + 0.1;
    let freq = 880.0f32;
    let two_pi = std::f32::consts::TAU;
    let mut out = Vec::with_capacity(n);
    // 20 ms attack / release envelope to avoid clicks.
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
    Ok(out)
}
