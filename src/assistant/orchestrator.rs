//! Top-level assistant supervisor.
//!
//! Owns the tokio runtime entry point, all task handles, and the session
//! state machine. Phase 1 only wires the audio loop with a placeholder
//! energy-based VAD and an echo "LLM" (transcript → TTS). ASR and TTS are
//! invoked via `spawn_blocking` because Burn/cubecl operations require a
//! blocking context.
//!
//! Later phases extend `run()` with the real LLM task, VHT2-based VAD,
//! filler manager, mixer, and Sesame-style TUI.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

use crate::assistant::audio_in::{self, PcmChunk};
use crate::assistant::audio_out::{self, AudioChunk, AudioOutCmd};
use crate::assistant::config::AssistantConfig;
use crate::assistant::state::SessionState;

/// Boundary event in the input stream — speech started, speech ended.
#[derive(Debug, Clone)]
enum VadEvent {
    SpeechStart,
    SpeechEnd,
}

/// The orchestrator. Construct, then `run().await`.
pub struct AssistantOrchestrator {
    cfg: Arc<AssistantConfig>,
}

impl AssistantOrchestrator {
    pub fn new(cfg: AssistantConfig) -> Self {
        Self { cfg: Arc::new(cfg) }
    }

    /// Run the assistant until Ctrl-C or fatal error.
    pub async fn run(self) -> Result<()> {
        let cfg = self.cfg.clone();
        info!(
            tui = cfg.tui,
            hybrid = cfg.hybrid,
            shannon_prime = cfg.shannon_prime,
            "Starting assistant session"
        );

        // ---- State channel (every task can read SessionState via watch) -----
        let (state_tx, state_rx) = watch::channel(SessionState::Idle);

        // ---- Audio I/O ------------------------------------------------------
        let (audio_chunk_tx, audio_chunk_rx) = mpsc::channel::<AudioChunk>(64);
        let out_handle = audio_out::spawn(cfg.clone(), audio_chunk_rx)
            .context("spawning speaker output")?;

        let (in_handle, mic_rx) = audio_in::spawn(cfg.clone()).context("spawning mic capture")?;

        // Forward mic samples into the input ring buffer for waveform viz.
        // (Phase 1 keeps this minimal — the TUI hook lands in Phase 5.)

        // ---- Placeholder VAD + utterance assembly --------------------------
        // For Phase 1 we use a simple RMS energy threshold over each 20 ms
        // chunk with `speech_start_frames / speech_end_frames` hysteresis.
        // Phase 3 swaps this for VHT2 spectral VAD.
        let (vad_tx, vad_rx) = mpsc::channel::<VadEvent>(16);
        let (utterance_tx, utterance_rx) = mpsc::channel::<Vec<f32>>(4);
        spawn_vad(cfg.clone(), mic_rx, vad_tx.clone(), utterance_tx.clone());

        // ---- ASR → echo → TTS task -----------------------------------------
        spawn_pipeline(
            cfg.clone(),
            utterance_rx,
            audio_chunk_tx.clone(),
            state_tx.clone(),
        );

        // ---- Barge-in supervisor -------------------------------------------
        // When VAD fires SpeechStart while the assistant is Speaking, flush
        // the audio output and reset to Listening. Phase 3 will also cancel
        // an in-flight LLM/TTS task; for Phase 1 we only stop the speaker.
        {
            let mut vad_rx = vad_rx;
            let mut state_rx_clone = state_rx.clone();
            let cmd_tx = out_handle.cmd_tx.clone();
            let state_tx = state_tx.clone();
            tokio::spawn(async move {
                while let Some(evt) = vad_rx.recv().await {
                    match evt {
                        VadEvent::SpeechStart => {
                            let cur = *state_rx_clone.borrow_and_update();
                            if matches!(cur, SessionState::Speaking) {
                                info!("Barge-in detected; flushing speaker");
                                let _ = cmd_tx.send(AudioOutCmd::Flush);
                                let _ = state_tx.send(SessionState::Interrupted);
                                let _ = state_tx.send(SessionState::Listening);
                            } else if matches!(cur, SessionState::Idle) {
                                let _ = state_tx.send(SessionState::Listening);
                            }
                        }
                        VadEvent::SpeechEnd => {
                            debug!("VAD speech end");
                        }
                    }
                }
            });
        }

        // ---- Initial state -------------------------------------------------
        let _ = state_tx.send(SessionState::Listening);

        // Wait for Ctrl-C.
        tokio::signal::ctrl_c()
            .await
            .context("listening for Ctrl-C")?;
        info!("Shutdown requested");

        // Drop handles to stop streams.
        drop(in_handle);
        drop(out_handle);
        drop(audio_chunk_tx);
        drop(state_tx);
        let _ = state_rx; // silence unused

        Ok(())
    }
}

/// Energy-based VAD with hysteresis. Accumulates speech samples into one
/// utterance and emits the utterance buffer when speech-end is detected.
fn spawn_vad(
    cfg: Arc<AssistantConfig>,
    mut mic_rx: mpsc::Receiver<PcmChunk>,
    vad_tx: mpsc::Sender<VadEvent>,
    utterance_tx: mpsc::Sender<Vec<f32>>,
) {
    tokio::spawn(async move {
        let mut in_speech = false;
        let mut above = 0u8;
        let mut below = 0u8;
        let mut buf: Vec<f32> = Vec::with_capacity(cfg.audio.input_rate_hz as usize * 8);
        let thr = cfg.vad.energy_threshold;
        let start_n = cfg.vad.speech_start_frames;
        let end_n = cfg.vad.speech_end_frames;
        // Always keep a 200 ms pre-roll so we don't clip the first phoneme.
        let preroll_samples = (cfg.audio.input_rate_hz as usize * 200) / 1000;
        let mut preroll: Vec<f32> = Vec::with_capacity(preroll_samples);

        while let Some(chunk) = mic_rx.recv().await {
            let rms = rms(&chunk.samples);
            let active = rms > thr;
            if active {
                above = above.saturating_add(1);
                below = 0;
            } else {
                below = below.saturating_add(1);
                above = 0;
            }

            if !in_speech {
                // Maintain rolling pre-roll.
                if preroll.len() + chunk.samples.len() > preroll_samples {
                    let drop = preroll.len() + chunk.samples.len() - preroll_samples;
                    preroll.drain(..drop.min(preroll.len()));
                }
                preroll.extend_from_slice(&chunk.samples);

                if above >= start_n {
                    in_speech = true;
                    buf.clear();
                    buf.extend_from_slice(&preroll);
                    preroll.clear();
                    let _ = vad_tx.send(VadEvent::SpeechStart).await;
                }
            } else {
                buf.extend_from_slice(&chunk.samples);
                if below >= end_n {
                    in_speech = false;
                    let utt = std::mem::take(&mut buf);
                    let _ = vad_tx.send(VadEvent::SpeechEnd).await;
                    if !utt.is_empty() {
                        let _ = utterance_tx.send(utt).await;
                    }
                }
            }
        }
    });
}

#[inline]
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Phase-1 pipeline: utterance → ASR → echo → TTS → speaker.
fn spawn_pipeline(
    cfg: Arc<AssistantConfig>,
    mut utterance_rx: mpsc::Receiver<Vec<f32>>,
    audio_chunk_tx: mpsc::Sender<AudioChunk>,
    state_tx: watch::Sender<SessionState>,
) {
    tokio::spawn(async move {
        while let Some(utt) = utterance_rx.recv().await {
            let _ = state_tx.send(SessionState::Thinking);
            let t0 = Instant::now();
            // ASR via spawn_blocking — Burn/cubecl needs a blocking context.
            let cfg_a = cfg.clone();
            let asr_result = tokio::task::spawn_blocking(move || crate::assistant::asr::transcribe(&cfg_a, &utt)).await;
            let transcript = match asr_result {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    warn!(?e, "ASR failed");
                    let _ = state_tx.send(SessionState::Listening);
                    continue;
                }
                Err(e) => {
                    warn!(?e, "ASR task panicked");
                    let _ = state_tx.send(SessionState::Listening);
                    continue;
                }
            };
            let asr_ms = t0.elapsed().as_millis();
            info!(asr_ms, %transcript, "ASR done");

            if transcript.trim().is_empty() {
                let _ = state_tx.send(SessionState::Listening);
                continue;
            }

            // Phase-1 "LLM" = echo. Phase 2 swaps this for candle.
            let reply = transcript.clone();

            // TTS: Phase 1 generates the whole reply, then streams chunks.
            let cfg_t = cfg.clone();
            let tts_result = tokio::task::spawn_blocking(move || {
                crate::assistant::tts::synthesize(&cfg_t, &reply)
            })
            .await;
            let audio = match tts_result {
                Ok(Ok(a)) => a,
                Ok(Err(e)) => {
                    warn!(?e, "TTS failed");
                    let _ = state_tx.send(SessionState::Listening);
                    continue;
                }
                Err(e) => {
                    warn!(?e, "TTS task panicked");
                    let _ = state_tx.send(SessionState::Listening);
                    continue;
                }
            };

            let _ = state_tx.send(SessionState::Speaking);
            // Stream in 20 ms chunks.
            let chunk = (cfg.audio.output_rate_hz as usize * 20) / 1000;
            for window in audio.chunks(chunk) {
                if audio_chunk_tx
                    .send(AudioChunk {
                        samples: window.to_vec(),
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
            let _ = state_tx.send(SessionState::Listening);
        }
    });
}
