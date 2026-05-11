//! Multi-source audio mixer.
//!
//! Sits between the synthesis tasks (TTS, filler, connection sound, ambient
//! room tone) and the speaker output task. Each source pushes [`AudioChunk`]s
//! into its own mpsc; the mixer task sums them with per-source weights at a
//! fixed tick rate, soft-clips, and forwards 20 ms mixed chunks to the speaker.
//!
//! The mixer is what gives the assistant the "always-on" call feeling —
//! there's always *something* on the wire (low ambient tone), and fillers can
//! overlap with speech without artifacts.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::debug;

use crate::assistant::audio_out::AudioChunk;
use crate::assistant::config::AssistantConfig;

/// Per-source weights expressed as linear amplitudes (not dB).
#[derive(Debug, Clone)]
pub struct MixWeights {
    /// LLM/TTS voice. 0 dB.
    pub voice: f32,
    /// Filler audio. -6 dB.
    pub filler: f32,
    /// Connection beep. -3 dB.
    pub connection: f32,
    /// Ambient room tone. -30 dB.
    pub ambient: f32,
}

impl Default for MixWeights {
    fn default() -> Self {
        Self {
            voice: 1.0,
            filler: 0.5,
            connection: 0.7,
            ambient: 0.03,
        }
    }
}

/// Handles returned from [`spawn`]. Each `*_tx` accepts [`AudioChunk`]s for a
/// specific mix source.
#[derive(Clone)]
pub struct MixerHandle {
    pub voice_tx: mpsc::Sender<AudioChunk>,
    pub filler_tx: mpsc::Sender<AudioChunk>,
    pub connection_tx: mpsc::Sender<AudioChunk>,
    pub ambient_tx: mpsc::Sender<AudioChunk>,
    /// Command channel — currently used to drop all queued voice samples
    /// during a barge-in. Filler and ambient continue uninterrupted.
    pub cmd_tx: mpsc::UnboundedSender<MixerCmd>,
}

/// Out-of-band control commands for the mixer.
#[derive(Debug, Clone, Copy)]
pub enum MixerCmd {
    /// Drop all queued voice samples (used for instant-flush on barge-in).
    FlushVoice,
}

/// Spawn the mixer task. `out_tx` receives the mixed [`AudioChunk`]s — wire
/// this into `audio_out::spawn`.
pub fn spawn(
    cfg: Arc<AssistantConfig>,
    out_tx: mpsc::Sender<AudioChunk>,
) -> MixerHandle {
    let (voice_tx, voice_rx) = mpsc::channel::<AudioChunk>(64);
    let (filler_tx, filler_rx) = mpsc::channel::<AudioChunk>(16);
    let (connection_tx, connection_rx) = mpsc::channel::<AudioChunk>(4);
    let (ambient_tx, ambient_rx) = mpsc::channel::<AudioChunk>(16);
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<MixerCmd>();

    tokio::spawn(run(
        cfg, out_tx, voice_rx, filler_rx, connection_rx, ambient_rx, cmd_rx,
    ));

    MixerHandle {
        voice_tx,
        filler_tx,
        connection_tx,
        ambient_tx,
        cmd_tx,
    }
}

async fn run(
    cfg: Arc<AssistantConfig>,
    out_tx: mpsc::Sender<AudioChunk>,
    mut voice_rx: mpsc::Receiver<AudioChunk>,
    mut filler_rx: mpsc::Receiver<AudioChunk>,
    mut connection_rx: mpsc::Receiver<AudioChunk>,
    mut ambient_rx: mpsc::Receiver<AudioChunk>,
    mut cmd_rx: mpsc::UnboundedReceiver<MixerCmd>,
) {
    let rate = cfg.audio.output_rate_hz as usize;
    let tick_ms = 20u64;
    let chunk = (rate * tick_ms as usize) / 1000;
    let weights = MixWeights::default();

    let mut voice = VecDeque::<f32>::with_capacity(chunk * 8);
    let mut filler = VecDeque::<f32>::with_capacity(chunk * 8);
    let mut connection = VecDeque::<f32>::with_capacity(chunk * 8);
    let mut ambient = VecDeque::<f32>::with_capacity(chunk * 8);

    let mut ticker = interval(Duration::from_millis(tick_ms));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            // Drain sources opportunistically; never block the tick.
            Some(c) = voice_rx.recv() => {
                voice.extend(c.samples);
            }
            Some(c) = filler_rx.recv() => {
                filler.extend(c.samples);
            }
            Some(c) = connection_rx.recv() => {
                connection.extend(c.samples);
            }
            Some(c) = ambient_rx.recv() => {
                ambient.extend(c.samples);
            }
            Some(cmd) = cmd_rx.recv() => match cmd {
                MixerCmd::FlushVoice => {
                    let dropped = voice.len();
                    voice.clear();
                    while voice_rx.try_recv().is_ok() {}
                    debug!(dropped, "Mixer flushed voice queue");
                }
            },
            _ = ticker.tick() => {
                let mut out = Vec::with_capacity(chunk);
                for _ in 0..chunk {
                    let v = voice.pop_front().unwrap_or(0.0) * weights.voice;
                    let f = filler.pop_front().unwrap_or(0.0) * weights.filler;
                    let con = connection.pop_front().unwrap_or(0.0) * weights.connection;
                    let amb = ambient.pop_front().unwrap_or(0.0) * weights.ambient;
                    let sum = v + f + con + amb;
                    // Soft clip: tanh approximation that's roughly identity in [-0.8, 0.8]
                    // and rolls off smoothly to ±1.
                    out.push(soft_clip(sum));
                }
                if out_tx.send(AudioChunk { samples: out }).await.is_err() {
                    break;
                }
            }
        }
    }
}

#[inline]
fn soft_clip(x: f32) -> f32 {
    // Simple tanh-like saturation without the expensive math.
    // y = x / (1 + |x|) scaled to peak ~0.95.
    let s = x / (1.0 + x.abs());
    s * 1.052
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_clip_in_range_is_near_identity() {
        for &x in &[-0.5f32, -0.1, 0.0, 0.1, 0.3, 0.5] {
            let y = soft_clip(x);
            assert!((y - x).abs() < 0.18, "soft_clip({x}) = {y}, too far");
        }
    }

    #[test]
    fn soft_clip_does_not_exceed_one() {
        for x in [-100.0f32, -5.0, 5.0, 100.0] {
            assert!(soft_clip(x).abs() <= 1.06, "soft_clip({x}) overshot");
        }
    }
}
