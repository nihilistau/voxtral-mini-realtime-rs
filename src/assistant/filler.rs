//! Filler audio manager: keeps the speaker warm during LLM "thinking" pauses.
//!
//! Listens for `Thinking` session-state transitions on a `watch` channel
//! and starts a `filler_after` timer (default 100 ms). If `Thinking` is
//! still active when the timer fires, push a random filler clip into the
//! mixer's filler channel. The mixer plays it under the (eventual) voice.
//!
//! If `Thinking` ends quickly (TTFT < threshold), the timer is cancelled
//! and no filler plays. If multiple `Thinking` cycles occur in succession,
//! each gets its own timer + a non-repeating selection.

use std::sync::Arc;

use tokio::sync::{mpsc, watch};
use tokio::time::{self, Duration};
use tracing::debug;

use crate::assistant::assets;
use crate::assistant::audio_out::AudioChunk;
use crate::assistant::config::AssistantConfig;
use crate::assistant::state::SessionState;

/// Spawn the filler manager.
///
/// - `state_rx`: subscribes to session state.
/// - `filler_tx`: where to push filler audio (mixer's filler channel).
pub fn spawn(
    cfg: Arc<AssistantConfig>,
    mut state_rx: watch::Receiver<SessionState>,
    filler_tx: mpsc::Sender<AudioChunk>,
) {
    tokio::spawn(async move {
        let bank = assets::synth_fillers(cfg.audio.output_rate_hz);
        if bank.is_empty() {
            return;
        }
        let delay = cfg.latency.filler_after;
        loop {
            // Wait for a state change.
            if state_rx.changed().await.is_err() {
                break;
            }
            let cur = *state_rx.borrow_and_update();
            if cur != SessionState::Thinking {
                continue;
            }
            // Race the timer against another state change.
            let sleep = time::sleep(delay);
            tokio::pin!(sleep);
            tokio::select! {
                _ = &mut sleep => {
                    // TTFT exceeded; inject one filler.
                    let samples = assets::pick_filler(&bank);
                    if !samples.is_empty() {
                        debug!(len = samples.len(), "Injecting filler");
                        let _ = filler_tx.send(AudioChunk { samples }).await;
                    }
                }
                changed = state_rx.changed() => {
                    if changed.is_err() { break; }
                    // State left Thinking before the timer; nothing to do.
                }
            }
        }
    });
}

/// Push the connection sound into the mixer's connection channel.
/// One-shot — call once on session start.
pub async fn play_connection(cfg: &AssistantConfig, tx: &mpsc::Sender<AudioChunk>) {
    if !cfg.latency.connection_sound {
        return;
    }
    let samples = assets::synth_connection(cfg.audio.output_rate_hz);
    let _ = tx.send(AudioChunk { samples }).await;
}

/// Continuously push the ambient room-tone loop into the mixer.
/// Runs until the channel is closed.
pub fn spawn_ambient(cfg: Arc<AssistantConfig>, tx: mpsc::Sender<AudioChunk>) {
    if !cfg.latency.ambient_tail {
        return;
    }
    tokio::spawn(async move {
        let loop_samples = assets::synth_ambient_loop(cfg.audio.output_rate_hz);
        if loop_samples.is_empty() {
            return;
        }
        // Push the loop ~10 times per second worth (60 ms each); the mixer
        // drains samples at output rate so this won't run away.
        let chunk = (cfg.audio.output_rate_hz as usize * 60) / 1000;
        let mut pos = 0usize;
        let mut ticker = time::interval(Duration::from_millis(60));
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let mut buf = Vec::with_capacity(chunk);
            for _ in 0..chunk {
                buf.push(loop_samples[pos]);
                pos = (pos + 1) % loop_samples.len();
            }
            if tx.send(AudioChunk { samples: buf }).await.is_err() {
                break;
            }
        }
    });
}
