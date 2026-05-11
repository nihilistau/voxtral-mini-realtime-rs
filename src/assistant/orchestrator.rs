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
use tokio::sync::{broadcast, mpsc, watch};
use tracing::{debug, info, warn};

use crate::assistant::audio_in::{self, PcmChunk};
use crate::assistant::audio_out::{self, AudioChunk, AudioOutCmd};
use crate::assistant::config::AssistantConfig;
use crate::assistant::filler;
use crate::assistant::mixer::{self, MixerCmd};
use crate::assistant::state::SessionState;
use crate::assistant::vad::{Vad, VadFrame};
use crate::tui::assistant_view::{
    AssistantViewState, SharedAssistantViewState, TranscriptRole,
};

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

/// Shared event bus made available to optional subscribers (e.g. the TUI).
/// Returned by [`AssistantOrchestrator::event_bus`] before `run()` is called.
#[derive(Clone)]
pub struct EventBus {
    /// Every VHT2-analyzed mic frame: RMS, entropy, flatness, power spectrum.
    pub vad: broadcast::Sender<VadFrame>,
    /// Session state transitions.
    pub state: watch::Sender<SessionState>,
    /// Finalized transcripts (after ASR).
    pub transcripts: broadcast::Sender<String>,
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

        // ---- Event bus broadcasts (subscribed by TUI later) -----------------
        let (vad_frame_tx, _) = broadcast::channel::<VadFrame>(256);
        let (transcript_tx, _) = broadcast::channel::<String>(32);

        // ---- Audio I/O + mixer ----------------------------------------------
        // Speaker pulls from the *mixed* stream produced by the mixer task.
        // Synthesis sources (voice / filler / connection / ambient) push into
        // the mixer; the mixer sums and forwards 20 ms chunks to audio_out.
        let (mixed_tx, mixed_rx) = mpsc::channel::<AudioChunk>(64);
        let out_handle = audio_out::spawn(cfg.clone(), mixed_rx)
            .context("spawning speaker output")?;
        let mixer_handle = mixer::spawn(cfg.clone(), mixed_tx);

        let (in_handle, mic_rx) = audio_in::spawn(cfg.clone()).context("spawning mic capture")?;

        // ---- Connection sound + ambient room tone --------------------------
        filler::play_connection(cfg.as_ref(), &mixer_handle.connection_tx).await;
        filler::spawn_ambient(cfg.clone(), mixer_handle.ambient_tx.clone());
        // Filler manager — watches state for Thinking + filler_after timer.
        filler::spawn(
            cfg.clone(),
            state_rx.clone(),
            mixer_handle.filler_tx.clone(),
        );

        // ---- Optional TUI ---------------------------------------------------
        let tui_state: Option<SharedAssistantViewState> = if cfg.tui {
            let st = Arc::new(std::sync::Mutex::new(AssistantViewState::new(
                3.0,
                cfg.audio.input_rate_hz,
            )));
            let st_for_thread = st.clone();
            std::thread::spawn(move || {
                if let Err(e) = crate::tui::assistant_view::run(st_for_thread) {
                    warn!(?e, "TUI exited with error");
                }
            });
            Some(st)
        } else {
            None
        };

        // ---- VHT2 VAD + utterance assembly ---------------------------------
        let (vad_tx, vad_rx) = mpsc::channel::<VadEvent>(16);
        let (utterance_tx, utterance_rx) = mpsc::channel::<Vec<f32>>(4);
        // Bridge: tee mic chunks to both the VAD task and the TUI mic ring.
        let (mic_to_vad_tx, mic_to_vad_rx) = mpsc::channel::<PcmChunk>(64);
        {
            let tui = tui_state.clone();
            tokio::spawn(async move {
                let mut mic_rx = mic_rx;
                while let Some(chunk) = mic_rx.recv().await {
                    if let Some(ref state) = tui {
                        if let Ok(mut s) = state.lock() {
                            s.mic_buf.push_slice(&chunk.samples);
                        }
                    }
                    if mic_to_vad_tx.send(chunk).await.is_err() {
                        break;
                    }
                }
            });
        }
        spawn_vad(
            cfg.clone(),
            mic_to_vad_rx,
            vad_tx.clone(),
            utterance_tx.clone(),
            vad_frame_tx.clone(),
        );

        // ---- TUI forwarders -------------------------------------------------
        if let Some(ref tui) = tui_state {
            // VAD frames → TUI spectrum/metrics.
            {
                let tui = tui.clone();
                let mut rx = vad_frame_tx.subscribe();
                tokio::spawn(async move {
                    while let Ok(frame) = rx.recv().await {
                        if let Ok(mut s) = tui.lock() {
                            s.rms = frame.rms;
                            s.entropy = frame.entropy;
                            s.flatness = frame.flatness;
                            s.vht2_power = frame.power;
                        }
                    }
                });
            }
            // Transcripts → TUI history.
            {
                let tui = tui.clone();
                let mut rx = transcript_tx.subscribe();
                tokio::spawn(async move {
                    while let Ok(text) = rx.recv().await {
                        if let Ok(mut s) = tui.lock() {
                            s.push_transcript(TranscriptRole::User, text.clone());
                            s.push_transcript(TranscriptRole::Assistant, text);
                        }
                    }
                });
            }
            // State transitions → TUI state pill.
            {
                let tui = tui.clone();
                let mut rx = state_rx.clone();
                tokio::spawn(async move {
                    loop {
                        let label = (*rx.borrow_and_update()).label();
                        if let Ok(mut s) = tui.lock() {
                            s.state_label = label;
                        }
                        if rx.changed().await.is_err() {
                            break;
                        }
                    }
                });
            }
        }

        // ---- ASR → echo → TTS task -----------------------------------------
        // Voice goes into the mixer's voice channel (not directly to speaker).
        spawn_pipeline(
            cfg.clone(),
            utterance_rx,
            mixer_handle.voice_tx.clone(),
            state_tx.clone(),
            transcript_tx.clone(),
        );

        // ---- Barge-in supervisor -------------------------------------------
        // When VAD fires SpeechStart while the assistant is Speaking, flush
        // both the mixer's voice queue (drops queued TTS) and the speaker's
        // jitter buffer (silences the in-flight chunk). Phase 3b will also
        // cancel the LLM and roll back its KV cache here.
        {
            let mut vad_rx = vad_rx;
            let mut state_rx_clone = state_rx.clone();
            let out_cmd_tx = out_handle.cmd_tx.clone();
            let mixer_cmd_tx = mixer_handle.cmd_tx.clone();
            let state_tx = state_tx.clone();
            tokio::spawn(async move {
                while let Some(evt) = vad_rx.recv().await {
                    match evt {
                        VadEvent::SpeechStart => {
                            let cur = *state_rx_clone.borrow_and_update();
                            if matches!(cur, SessionState::Speaking) {
                                info!("Barge-in detected; flushing voice + speaker");
                                let _ = mixer_cmd_tx.send(MixerCmd::FlushVoice);
                                let _ = out_cmd_tx.send(AudioOutCmd::Flush);
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
        drop(mixer_handle);
        drop(state_tx);
        let _ = state_rx; // silence unused

        Ok(())
    }
}

/// VHT2-driven VAD task. Accumulates mic samples into 32 ms windows, runs
/// spectral analysis on each, broadcasts the [`VadFrame`] (for the TUI),
/// emits Start/End events, and assembles complete utterances.
fn spawn_vad(
    cfg: Arc<AssistantConfig>,
    mut mic_rx: mpsc::Receiver<PcmChunk>,
    vad_tx: mpsc::Sender<VadEvent>,
    utterance_tx: mpsc::Sender<Vec<f32>>,
    frame_tx: broadcast::Sender<VadFrame>,
) {
    tokio::spawn(async move {
        let mut vad = Vad::new(cfg.vad.clone());
        let mut in_speech = false;
        let mut utt: Vec<f32> = Vec::with_capacity(cfg.audio.input_rate_hz as usize * 8);
        // Keep a 200 ms pre-roll so we don't clip the first phoneme.
        let preroll_samples = (cfg.audio.input_rate_hz as usize * 200) / 1000;
        let mut preroll: Vec<f32> = Vec::with_capacity(preroll_samples);

        while let Some(chunk) = mic_rx.recv().await {
            // Maintain rolling pre-roll regardless of state — when speech
            // starts we prepend whatever is in it.
            if !in_speech {
                if preroll.len() + chunk.samples.len() > preroll_samples {
                    let drop = preroll.len() + chunk.samples.len() - preroll_samples;
                    preroll.drain(..drop.min(preroll.len()));
                }
                preroll.extend_from_slice(&chunk.samples);
            } else {
                utt.extend_from_slice(&chunk.samples);
            }

            for frame in vad.push(&chunk.samples) {
                // Best-effort broadcast — no subscribers is fine.
                let _ = frame_tx.send(frame.clone());

                match (in_speech, frame.is_speech) {
                    (false, true) => {
                        in_speech = true;
                        utt.clear();
                        utt.extend_from_slice(&preroll);
                        preroll.clear();
                        let _ = vad_tx.send(VadEvent::SpeechStart).await;
                    }
                    (true, false) => {
                        in_speech = false;
                        let captured = std::mem::take(&mut utt);
                        let _ = vad_tx.send(VadEvent::SpeechEnd).await;
                        if !captured.is_empty() {
                            let _ = utterance_tx.send(captured).await;
                        }
                    }
                    _ => {}
                }
            }
        }
    });
}

/// Phase-1 pipeline: utterance → ASR → echo → TTS → speaker.
fn spawn_pipeline(
    cfg: Arc<AssistantConfig>,
    mut utterance_rx: mpsc::Receiver<Vec<f32>>,
    audio_chunk_tx: mpsc::Sender<AudioChunk>,
    state_tx: watch::Sender<SessionState>,
    transcript_tx: broadcast::Sender<String>,
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
            let _ = transcript_tx.send(transcript.clone());

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
