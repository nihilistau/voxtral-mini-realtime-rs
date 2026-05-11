//! Speaker playback: tokio mpsc of `AudioChunk`s → cpal output stream at 24 kHz mono.
//!
//! A bounded `RingBuffer` acts as the jitter buffer. The cpal output callback
//! pulls from it; the async producer pushes into it. Pushes are protected by a
//! `parking_lot::Mutex` only for the brief copy — the callback never blocks
//! the producer and vice versa (a starved callback fills with zeros / silence).
//!
//! We deliberately use a tiny `std::sync::Mutex` here instead of an mpsc-fed
//! single-thread consumer because cpal's output callback runs on its own OS
//! thread and we need synchronous shared state. The lock window is microseconds.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, Stream, StreamConfig};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::assistant::config::AssistantConfig;
use crate::audio::ring_buffer::RingBuffer;

/// A chunk of synthesized PCM destined for the speaker. Sample rate must
/// match `AudioConfig::output_rate_hz`.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
}

/// Commands the orchestrator sends to the speaker task out-of-band of the
/// normal sample stream — used for instant-flush during barge-in.
#[derive(Debug, Clone, Copy)]
pub enum AudioOutCmd {
    /// Drop all queued samples and silence the speaker after a `interrupt_fade`
    /// cross-fade. Reset state to ready-for-next-utterance.
    Flush,
}

/// Handle returned by [`spawn`]. Drop to stop the output stream.
pub struct AudioOutHandle {
    _stream: Stream,
    /// Shared jitter buffer (push side). Held by orchestrator for flushes.
    pub jitter: Arc<Mutex<RingBuffer>>,
    /// Channel for control commands.
    pub cmd_tx: mpsc::UnboundedSender<AudioOutCmd>,
}

/// Spawn the speaker stream and a forwarding task that consumes `AudioChunk`s.
///
/// Returns the handle (keep alive for the session) and a sender used by the
/// mixer to push audio frames in.
pub fn spawn(
    cfg: Arc<AssistantConfig>,
    mut audio_rx: mpsc::Receiver<AudioChunk>,
) -> Result<AudioOutHandle> {
    let host = cpal::default_host();
    let device = match &cfg.audio.output_device {
        Some(name) => host
            .output_devices()
            .context("enumerating output devices")?
            .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("output device '{name}' not found"))?,
        None => host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device"))?,
    };

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let supported = device
        .default_output_config()
        .context("getting default output config")?;
    let sample_format = supported.sample_format();
    let device_rate = supported.sample_rate().0;
    let device_channels = supported.channels() as usize;
    let cfg_struct: StreamConfig = supported.config();
    info!(
        device = %device_name,
        device_rate,
        device_channels,
        source_rate = cfg.audio.output_rate_hz,
        "Speaker playback starting"
    );

    // Jitter buffer sized for the configured target rate. Pre-allocate 1 s
    // capacity which is plenty for any sane jitter and keeps lookups O(1).
    let jitter = Arc::new(Mutex::new(RingBuffer::from_duration_secs(
        1.0,
        cfg.audio.output_rate_hz,
    )));

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<AudioOutCmd>();

    // Producer task: drains the mpsc into the ring buffer.
    {
        let jitter = jitter.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    Some(cmd) = cmd_rx.recv() => match cmd {
                        AudioOutCmd::Flush => {
                            // Drain the mpsc to avoid post-flush leftovers.
                            while audio_rx.try_recv().is_ok() {}
                            if let Ok(mut j) = jitter.lock() {
                                j.clear();
                            }
                        }
                    },
                    Some(chunk) = audio_rx.recv() => {
                        if let Ok(mut j) = jitter.lock() {
                            j.push_slice(&chunk.samples);
                        }
                    },
                    else => break,
                }
            }
        });
    }

    // Cross-rate state for the cpal callback: device rate may not equal source rate.
    let resample_ratio = device_rate as f64 / cfg.audio.output_rate_hz as f64;
    let mut phase = 0.0f64;

    let jitter_cb = jitter.clone();
    let err_fn = |err| warn!(?err, "cpal output stream error");

    macro_rules! write_callback {
        ($t:ty) => {{
            move |out: &mut [$t], _: &cpal::OutputCallbackInfo| {
                let need_frames = out.len() / device_channels;
                let need_source = ((need_frames as f64) / resample_ratio).ceil() as usize + 2;
                let snapshot = match jitter_cb.lock() {
                    Ok(mut j) => {
                        let avail = j.snapshot();
                        // Drain the consumed portion by clearing and pushing back the tail.
                        // For Phase 1 we just snapshot and don't drain — the ring overwrites
                        // older data as the producer keeps pushing. Below-real-time output
                        // would replay; in practice the producer leads. To prevent replay,
                        // we clear and reinsert the unconsumed tail.
                        if avail.len() > need_source {
                            let tail: Vec<f32> = avail[need_source..].to_vec();
                            j.clear();
                            j.push_slice(&tail);
                        } else {
                            j.clear();
                        }
                        avail
                    }
                    Err(_) => Vec::new(),
                };

                // Linear resample from source -> device rate using phase.
                let mut produced = 0usize;
                for frame_i in 0..need_frames {
                    let t = phase + (frame_i as f64) / resample_ratio;
                    let i = t as usize;
                    let sample = if i + 1 < snapshot.len() {
                        let frac = (t - i as f64) as f32;
                        snapshot[i] + (snapshot[i + 1] - snapshot[i]) * frac
                    } else if i < snapshot.len() {
                        snapshot[i]
                    } else {
                        0.0
                    };
                    let typed = <$t as cpal::FromSample<f32>>::from_sample_(sample);
                    for c in 0..device_channels {
                        out[frame_i * device_channels + c] = typed;
                    }
                    produced = frame_i + 1;
                }

                let advance = (produced as f64) / resample_ratio;
                phase = (phase + advance).fract();
            }
        }};
    }

    let stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(&cfg_struct, write_callback!(f32), err_fn, None),
        SampleFormat::I16 => device.build_output_stream(&cfg_struct, write_callback!(i16), err_fn, None),
        SampleFormat::U16 => device.build_output_stream(&cfg_struct, write_callback!(u16), err_fn, None),
        other => return Err(anyhow!("unsupported speaker sample format: {other:?}")),
    }
    .context("building output stream")?;

    stream.play().context("starting output stream")?;

    Ok(AudioOutHandle {
        _stream: stream,
        jitter,
        cmd_tx,
    })
}

// FromSample is used inside the write_callback! macro; reference it here to
// avoid the unused-import warning when macro expansion happens elsewhere.
const _: fn() = || {
    let _ = <f32 as FromSample<f32>>::from_sample_(0.0f32);
};
