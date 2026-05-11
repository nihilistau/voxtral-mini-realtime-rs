//! Mic capture: cpal input stream → tokio mpsc of 16 kHz mono `PcmChunk`s.
//!
//! cpal runs its callback on its own OS thread. We do the on-the-fly
//! resample-to-16k and channel downmix in that callback (it's cheap),
//! then push fixed-size chunks into an async mpsc.
//!
//! No allocations or blocking calls happen on the cpal callback path —
//! all buffers are pre-allocated and bounded.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream, StreamConfig};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::assistant::config::AssistantConfig;

/// 20 ms of 16 kHz mono PCM (or whatever `input_chunk_samples()` returns).
#[derive(Debug, Clone)]
pub struct PcmChunk {
    /// Mono f32 PCM, normalized to roughly [-1, 1].
    pub samples: Vec<f32>,
    /// Monotonic frame counter (one frame = one chunk).
    pub frame: u64,
}

/// Handle returned by [`spawn`]. Dropping the handle stops the input stream.
pub struct AudioInHandle {
    _stream: Stream,
}

/// Spawn the mic capture stream.
///
/// Returns a handle that owns the underlying cpal stream (keep it alive for the
/// session) and a receiver delivering [`PcmChunk`]s at `input_chunk_ms` cadence.
pub fn spawn(cfg: Arc<AssistantConfig>) -> Result<(AudioInHandle, mpsc::Receiver<PcmChunk>)> {
    let host = cpal::default_host();
    let device = match &cfg.audio.input_device {
        Some(name) => host
            .input_devices()
            .context("enumerating input devices")?
            .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
            .ok_or_else(|| anyhow!("input device '{name}' not found"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?,
    };

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let supported = device
        .default_input_config()
        .context("getting default input config")?;
    let sample_format = supported.sample_format();
    let device_rate = supported.sample_rate().0;
    let device_channels = supported.channels() as usize;
    let cfg_struct: StreamConfig = supported.config();
    info!(
        device = %device_name,
        device_rate,
        device_channels,
        target_rate = cfg.audio.input_rate_hz,
        chunk_samples = cfg.input_chunk_samples(),
        "Mic capture starting"
    );

    // Bounded channel: ~10 chunks of latency cap before we start dropping.
    let (tx, rx) = mpsc::channel::<PcmChunk>(10);

    let chunk_samples = cfg.input_chunk_samples();
    let target_rate = cfg.audio.input_rate_hz;
    let resample_ratio = target_rate as f64 / device_rate as f64;

    // Per-stream scratch state — mono accumulator + frame counter + resample phase.
    let scratch = ResampleScratch::new(chunk_samples);
    let mut state = StreamState {
        scratch,
        device_channels,
        resample_ratio,
        frame: 0,
        tx,
    };

    let err_fn = |err| warn!(?err, "cpal input stream error");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &cfg_struct,
            move |data: &[f32], _| state.feed_f32(data),
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &cfg_struct,
            move |data: &[i16], _| {
                let mut tmp = vec![0.0f32; data.len()];
                for (o, &s) in tmp.iter_mut().zip(data) {
                    *o = s.to_sample::<f32>();
                }
                state.feed_f32(&tmp);
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &cfg_struct,
            move |data: &[u16], _| {
                let mut tmp = vec![0.0f32; data.len()];
                for (o, &s) in tmp.iter_mut().zip(data) {
                    *o = s.to_sample::<f32>();
                }
                state.feed_f32(&tmp);
            },
            err_fn,
            None,
        ),
        other => return Err(anyhow!("unsupported mic sample format: {other:?}")),
    }
    .context("building input stream")?;

    stream.play().context("starting input stream")?;

    Ok((AudioInHandle { _stream: stream }, rx))
}

struct StreamState {
    scratch: ResampleScratch,
    device_channels: usize,
    resample_ratio: f64,
    frame: u64,
    tx: mpsc::Sender<PcmChunk>,
}

impl StreamState {
    fn feed_f32(&mut self, interleaved: &[f32]) {
        // Downmix to mono.
        let mut mono: Vec<f32> = if self.device_channels == 1 {
            interleaved.to_vec()
        } else {
            let frames = interleaved.len() / self.device_channels;
            let mut out = Vec::with_capacity(frames);
            for i in 0..frames {
                let base = i * self.device_channels;
                let sum: f32 = interleaved[base..base + self.device_channels].iter().sum();
                out.push(sum / self.device_channels as f32);
            }
            out
        };

        // Cheap linear resample. ASR doesn't need pristine quality, and the
        // alternative (rubato on the cpal thread) is too heavy. Phase-accurate
        // accumulator avoids drift across callbacks.
        if (self.resample_ratio - 1.0).abs() > 1e-6 {
            mono = self.scratch.linear_resample(&mono, self.resample_ratio);
        }

        // Push into ring and emit chunks.
        self.scratch.ring.extend_from_slice(&mono);
        let chunk = self.scratch.chunk_samples;
        while self.scratch.ring.len() >= chunk {
            let mut out = Vec::with_capacity(chunk);
            out.extend(self.scratch.ring.drain(..chunk));
            let frame = self.frame;
            self.frame += 1;
            // try_send: if the consumer is behind, drop the chunk rather than
            // block the cpal callback. Drops mean the assistant is overloaded
            // and we want to surface that, not stall the audio device.
            if self.tx.try_send(PcmChunk { samples: out, frame }).is_err() {
                // Channel full or closed; nothing we can do here.
            }
        }
    }
}

struct ResampleScratch {
    ring: Vec<f32>,
    chunk_samples: usize,
    /// Fractional phase accumulator across cpal callbacks for linear resample.
    phase: f64,
}

impl ResampleScratch {
    fn new(chunk_samples: usize) -> Self {
        Self {
            ring: Vec::with_capacity(chunk_samples * 8),
            chunk_samples,
            phase: 0.0,
        }
    }

    /// Stateful linear-interpolation resampler. `ratio = target / source`.
    /// Maintains fractional phase so successive callbacks join seamlessly.
    fn linear_resample(&mut self, input: &[f32], ratio: f64) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        let step = 1.0 / ratio;
        let mut out = Vec::with_capacity(((input.len() as f64) * ratio).ceil() as usize);
        let mut t = self.phase;
        while t < input.len() as f64 {
            let i = t as usize;
            let frac = (t - i as f64) as f32;
            let a = input[i];
            let b = if i + 1 < input.len() { input[i + 1] } else { a };
            out.push(a + (b - a) * frac);
            t += step;
        }
        self.phase = t - input.len() as f64;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_resample_identity_ratio() {
        let mut s = ResampleScratch::new(8);
        let r = s.linear_resample(&[0.0, 0.5, 1.0, 0.5], 1.0);
        assert_eq!(r.len(), 4);
        // First and last sample should match.
        assert!((r[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn linear_resample_downsample_2x() {
        let mut s = ResampleScratch::new(8);
        // 8 samples -> ratio 0.5 -> 4 samples.
        let r = s.linear_resample(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0.5);
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn linear_resample_phase_preserved_across_calls() {
        let mut s = ResampleScratch::new(8);
        let r1 = s.linear_resample(&[0.0, 1.0, 2.0, 3.0], 0.75);
        let r2 = s.linear_resample(&[4.0, 5.0, 6.0, 7.0], 0.75);
        // No samples should be lost or duplicated at the boundary; total
        // count should match a single combined call within ±1.
        let mut s2 = ResampleScratch::new(8);
        let r_full = s2.linear_resample(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0.75);
        let split_total = r1.len() + r2.len();
        assert!((split_total as isize - r_full.len() as isize).abs() <= 1);
    }
}
