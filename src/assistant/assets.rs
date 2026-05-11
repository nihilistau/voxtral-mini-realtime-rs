//! Procedurally-synthesized "always-on" audio assets.
//!
//! Real Sesame-style polish needs hand-crafted recordings: rising connection
//! tone, gentle room ambience, natural "uhh/um" fillers in the assistant's
//! own voice. Until those exist on disk in `assets/`, we synthesize
//! reasonable approximations so the latency-masking infrastructure works
//! end-to-end without needing external files.
//!
//! A future commit can override [`load_or_synth_*`] to read WAVs from
//! `assets/` when present.

use rand::seq::SliceRandom;

/// Short rising chirp (~250 ms) emitted on session start. Two-tone sweep
/// from 480 → 720 Hz with a gentle attack/release envelope.
pub fn synth_connection(sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let dur = 0.25;
    let n = (sr * dur) as usize;
    let f0 = 480.0f32;
    let f1 = 720.0f32;
    let env_n = (sr * 0.04) as usize;
    let two_pi = std::f32::consts::TAU;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f32 / n as f32;
        // Linear sweep in frequency space.
        let f = f0 + (f1 - f0) * frac;
        phase += two_pi * f / sr;
        let env = if i < env_n {
            i as f32 / env_n as f32
        } else if i > n.saturating_sub(env_n) {
            (n - i) as f32 / env_n as f32
        } else {
            1.0
        };
        // Mix fundamental + slight second harmonic for warmth.
        let s = (phase.sin() * 0.8 + (2.0 * phase).sin() * 0.2) * 0.6 * env;
        out.push(s);
    }
    out
}

/// One ~600 ms loop of soft, mostly-inaudible room tone. Output ambient
/// volume is set by the mixer (-30 dB by default); the loop itself is
/// near-flat noise with a low-pass shape so it's not hissy.
pub fn synth_ambient_loop(sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let dur = 0.6;
    let n = (sr * dur) as usize;
    // xorshift for deterministic, gentle noise.
    let mut x: u64 = 0x00C0_FFEE_BEEF_u64;
    let mut prev = 0.0f32;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let v = ((x as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
        // 1-pole low-pass to remove high-frequency hiss.
        prev = prev * 0.92 + v * 0.08;
        out.push(prev * 0.5);
    }
    // Trim ends to zero-cross for seamless looping.
    let fade = (sr * 0.02) as usize;
    for i in 0..fade.min(out.len()) {
        let f = i as f32 / fade as f32;
        let last = out.len() - 1 - i;
        out[i] *= f;
        out[last] *= f;
    }
    out
}

/// A small bank of synthesized "uhh"/"um"/"hmm" fillers. Each one is a few
/// shaped formant tones lasting 200–350 ms. Returns a Vec of variants the
/// filler manager picks from at random.
pub fn synth_fillers(sample_rate: u32) -> Vec<Vec<f32>> {
    let sr = sample_rate as f32;
    // (duration_s, formant1_hz, formant2_hz, has_hum)
    let recipes: &[(f32, f32, f32, bool)] = &[
        (0.30, 220.0, 700.0, true),   // "uhh"
        (0.22, 260.0, 850.0, true),   // "um"
        (0.32, 200.0, 600.0, false),  // "mmm"
        (0.25, 240.0, 900.0, true),   // "uh"
    ];
    recipes
        .iter()
        .map(|&(dur, f1, f2, hum)| synth_filler_word(sr, dur, f1, f2, hum))
        .collect()
}

fn synth_filler_word(sr: f32, dur: f32, f1: f32, f2: f32, hum: bool) -> Vec<f32> {
    let n = (sr * dur) as usize;
    let two_pi = std::f32::consts::TAU;
    let env_attack = (sr * 0.05) as usize;
    let env_release = (sr * 0.08) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let env = if i < env_attack {
            i as f32 / env_attack as f32
        } else if i > n.saturating_sub(env_release) {
            (n - i) as f32 / env_release as f32
        } else {
            1.0
        };
        // Two formants + optional very-low fundamental hum.
        let f0 = if hum { (two_pi * 110.0 * t).sin() * 0.3 } else { 0.0 };
        let a = (two_pi * f1 * t).sin() * 0.5;
        let b = (two_pi * f2 * t).sin() * 0.2;
        out.push((a + b + f0) * env * 0.5);
    }
    out
}

/// Pick one filler at random from the bank.
pub fn pick_filler(bank: &[Vec<f32>]) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    bank.choose(&mut rng).cloned().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_has_audible_energy() {
        let s = synth_connection(24_000);
        assert!(!s.is_empty());
        let peak = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(peak > 0.1, "connection peak too quiet: {peak}");
        assert!(peak < 1.0, "connection peak unclipped: {peak}");
    }

    #[test]
    fn ambient_is_low_volume() {
        let s = synth_ambient_loop(24_000);
        let peak = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(peak < 0.6, "ambient peak too loud (mixer expects -30 dB scaling): {peak}");
    }

    #[test]
    fn fillers_bank_nonempty() {
        let bank = synth_fillers(24_000);
        assert!(bank.len() >= 3);
        for f in &bank {
            assert!(!f.is_empty());
        }
    }
}
