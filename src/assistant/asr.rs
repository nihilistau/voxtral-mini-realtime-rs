//! ASR adapter.
//!
//! Phase 1: stub — returns a placeholder transcript with the sample count so
//! the audio loop can be validated end-to-end without model setup.
//!
//! Phase 2 will replace this with a model-owner thread that loads the Q4
//! Voxtral model once and serves multiple utterances over a sync mpsc.

use anyhow::Result;

use crate::assistant::config::AssistantConfig;

/// Transcribe a single utterance.
///
/// Called from `tokio::task::spawn_blocking` because the real ASR path
/// (added in Phase 2) drives Burn/cubecl which is synchronous.
pub fn transcribe(_cfg: &AssistantConfig, samples: &[f32]) -> Result<String> {
    let duration_s = samples.len() as f32 / 16_000.0;
    // Placeholder echo string so the rest of the pipeline has something to chew on.
    Ok(format!("(stub transcript, {duration_s:.2}s of audio)"))
}
