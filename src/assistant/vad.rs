//! VHT2-based Voice Activity Detection.
//!
//! Speech vs. background noise is hard to separate with energy alone: a quiet
//! conversation in a noisy room can have lower RMS than the room itself. The
//! VHT2 power spectrum carries shape information that energy doesn't:
//!
//! - **Voiced speech** concentrates energy in a few low/mid bands → low
//!   spectral entropy `H(X)`, low spectral flatness.
//! - **Unvoiced speech** (fricatives) has higher entropy but a recognizable
//!   non-flat shape.
//! - **Stationary noise** (fans, hum, room tone) is approximately flat across
//!   bands → high flatness, high entropy.
//!
//! We combine three features into a single speech score:
//! - RMS above a floor (gates out absolute silence)
//! - `H(X)` below a ceiling (filters out flat noise)
//! - flatness below a ceiling (catches non-flat but loud noise like clicks)
//!
//! All three must be true for a frame to count as speech, with hysteresis on
//! the frame count (3 above / 8 below by default — see [`VadConfig`]).

use crate::assistant::config::VadConfig;
use crate::models::layers::shannon_prime::{spectral_entropy, spectral_flatness, vht2_f32_inplace};

/// One window of mic samples after windowing + VHT2 + feature extraction.
#[derive(Debug, Clone)]
pub struct VadFrame {
    /// RMS amplitude of the windowed samples in [0, 1] (approximately).
    pub rms: f32,
    /// Shannon entropy of the VHT2 power distribution, in bits. For a
    /// `WINDOW` of 512, H ≤ log₂(512) = 9.
    pub entropy: f32,
    /// Spectral flatness in [0, 1]. 1 = flat (noise), 0 = pure tone.
    pub flatness: f32,
    /// True if VAD decided this frame is speech.
    pub is_speech: bool,
    /// Raw VHT2 power coefficients (for TUI spectrum display). Length =
    /// `WINDOW / 2` (we discard the upper Hartley reflection).
    pub power: Vec<f32>,
}

/// Window size in samples at the mic input rate. Power-of-two for the fast
/// VHT2 SIMD path. 512 samples @ 16 kHz = 32 ms — within real-time budget.
pub const WINDOW: usize = 512;

/// Streaming VAD. Owns a small scratch buffer and accumulates 32 ms windows.
/// One instance per session — keep alive for the lifetime of the orchestrator.
pub struct Vad {
    cfg: VadConfig,
    scratch: Vec<f32>,
    hann: Vec<f32>,
    above_streak: u8,
    below_streak: u8,
    /// Adaptive noise floor (RMS, exponentially smoothed during silence).
    noise_floor: f32,
}

impl Vad {
    pub fn new(cfg: VadConfig) -> Self {
        let hann: Vec<f32> = (0..WINDOW)
            .map(|i| {
                0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (WINDOW as f32 - 1.0)).cos()
            })
            .collect();
        Self {
            cfg,
            scratch: Vec::with_capacity(WINDOW * 2),
            hann,
            above_streak: 0,
            below_streak: 0,
            noise_floor: 0.001,
        }
    }

    /// Push raw mic samples; drains complete windows and returns one
    /// [`VadFrame`] per window analyzed.
    pub fn push(&mut self, samples: &[f32]) -> Vec<VadFrame> {
        self.scratch.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.scratch.len() >= WINDOW {
            let mut window: Vec<f32> = self.scratch.drain(..WINDOW).collect();
            out.push(self.analyze(&mut window));
        }
        out
    }

    fn analyze(&mut self, window: &mut [f32]) -> VadFrame {
        // Pre-windowing RMS — what the energy detector would see.
        let rms = rms(window);

        // Apply Hann window to reduce spectral leakage.
        for (s, &h) in window.iter_mut().zip(self.hann.iter()) {
            *s *= h;
        }

        // VHT2 in place. Self-inverse; for VAD we only care about the
        // coefficient *magnitudes*, not reconstructing the signal.
        vht2_f32_inplace(window);

        // Power = coeff² in the lower half (Hartley is its own mirror).
        let half = WINDOW / 2;
        let mut power = Vec::with_capacity(half);
        for &c in &window[..half] {
            power.push(c * c);
        }

        let entropy = spectral_entropy(&power.iter().map(|p| p.sqrt()).collect::<Vec<_>>());
        let flatness = spectral_flatness(&power);

        // Adaptive noise floor: slowly track RMS during silence.
        let active = self.is_speech_features(rms, entropy, flatness);
        if !active && rms > 0.0 {
            self.noise_floor = self.noise_floor * 0.97 + rms * 0.03;
        }

        // Hysteresis on the final decision.
        if active {
            self.above_streak = self.above_streak.saturating_add(1);
            self.below_streak = 0;
        } else {
            self.below_streak = self.below_streak.saturating_add(1);
            self.above_streak = 0;
        }

        let is_speech = if self.above_streak >= self.cfg.speech_start_frames {
            true
        } else if self.below_streak >= self.cfg.speech_end_frames {
            false
        } else {
            // Sticky: maintain previous decision while in transition.
            self.above_streak > 0
        };

        VadFrame {
            rms,
            entropy,
            flatness,
            is_speech,
            power,
        }
    }

    fn is_speech_features(&self, rms: f32, entropy: f32, flatness: f32) -> bool {
        // Energy must clear the larger of the configured threshold or the
        // adaptive noise floor + headroom (3x noise floor).
        let energy_ok = rms > self.cfg.energy_threshold && rms > self.noise_floor * 3.0;
        let entropy_ok = entropy < self.cfg.entropy_ceiling;
        let flatness_ok = flatness < 0.7;
        energy_ok && entropy_ok && flatness_ok
    }

    /// Current adaptive noise floor estimate (RMS). Useful for the TUI footer.
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }
}

#[inline]
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn make_tone(n: usize, freq: f32, sr: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / sr).sin() * amp)
            .collect()
    }

    fn make_white_noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        // xorshift, deterministic
        let mut x = seed.wrapping_mul(2685821657736338717);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = (x as f32) / (u64::MAX as f32) * 2.0 - 1.0;
                v * amp
            })
            .collect()
    }

    #[test]
    fn silence_is_not_speech() {
        let mut vad = Vad::new(VadConfig::default());
        let frames = vad.push(&make_silence(WINDOW * 4));
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(!f.is_speech, "silence flagged as speech: {f:?}");
        }
    }

    #[test]
    fn tone_is_speech_after_hysteresis() {
        let mut vad = Vad::new(VadConfig::default());
        // Loud 300 Hz tone is tonal (low flatness, low entropy) and loud.
        let frames = vad.push(&make_tone(WINDOW * 8, 300.0, 16_000.0, 0.3));
        // After hysteresis (3 frames), later frames should be speech.
        assert!(frames.last().unwrap().is_speech);
    }

    #[test]
    fn white_noise_is_not_speech() {
        let mut vad = Vad::new(VadConfig::default());
        // Loud white noise: flat spectrum, high entropy, should be rejected
        // even though RMS is high.
        let frames = vad.push(&make_white_noise(WINDOW * 8, 0.3, 42));
        assert!(
            !frames.iter().all(|f| f.is_speech),
            "white noise should not be flagged as speech in every frame: {frames:?}"
        );
    }
}
