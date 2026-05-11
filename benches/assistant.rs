//! Microbenchmarks for the real-time assistant's per-tick hot paths.
//!
//! Measures the CPU-side pieces that run on the audio thread or inside the
//! tokio runtime — VAD, spectral entropy, soft-clip mixing, VHT2 transform.
//! These all need to fit comfortably inside their respective tick budgets:
//!
//! | Component         | Budget         | Reason                                   |
//! |-------------------|----------------|------------------------------------------|
//! | VAD (32 ms window)| < 1 ms         | runs per 32 ms mic frame at 16 kHz       |
//! | Mixer (20 ms tick)| < 200 µs       | runs every 20 ms at 24 kHz output rate   |
//! | VHT2 (head_dim=128)| < 30 µs        | runs per KV write — many per LLM token   |
//!
//! Run: `cargo bench --features "wgpu,cli,hub,assistant" assistant`.

use std::f32::consts::TAU;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use voxtral_mini_realtime::assistant::config::VadConfig;
use voxtral_mini_realtime::assistant::vad::{Vad, WINDOW};
use voxtral_mini_realtime::models::layers::shannon_prime::{
    spectral_entropy, spectral_flatness, vht2_f32_inplace,
};

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

fn sine(n: usize, freq: f32, sr: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (TAU * freq * (i as f32) / sr).sin() * amp)
        .collect()
}

fn xor_noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_685_821_657_736_338_717);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x as f32) / (u64::MAX as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

// ---------------------------------------------------------------------------
// VHT2 — the spectral transform underneath VAD + KV compression
// ---------------------------------------------------------------------------

fn bench_vht2(c: &mut Criterion) {
    let mut group = c.benchmark_group("vht2_inplace");
    for &n in &[64usize, 128, 256, 512, 1024] {
        let data: Vec<f32> = sine(n, 100.0, 16_000.0, 0.5);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut d| vht2_f32_inplace(black_box(&mut d)),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_spectral_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_entropy");
    for &n in &[64usize, 128, 256, 512] {
        let data = sine(n, 100.0, 16_000.0, 0.5);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, data| {
            b.iter(|| spectral_entropy(black_box(data)));
        });
    }
    group.finish();
}

fn bench_spectral_flatness(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_flatness");
    for &n in &[64usize, 128, 256, 512] {
        // Power values, not raw coefficients.
        let data: Vec<f32> = sine(n, 100.0, 16_000.0, 0.5)
            .iter()
            .map(|x| x * x)
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, data| {
            b.iter(|| spectral_flatness(black_box(data)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// VAD — full analyze pipeline: Hann window + VHT2 + entropy + flatness
// ---------------------------------------------------------------------------

fn bench_vad_one_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad");
    // 32 ms of 16 kHz mono = WINDOW samples
    let speech = sine(WINDOW, 280.0, 16_000.0, 0.3);
    let silence = vec![0.0f32; WINDOW];
    let noise = xor_noise(WINDOW, 0.3, 1234);

    group.throughput(Throughput::Elements(WINDOW as u64));
    group.bench_function("voiced_tone_window", |b| {
        let mut vad = Vad::new(VadConfig::default());
        // Pre-warm so the steady-state cost is what we measure.
        let _ = vad.push(&speech);
        b.iter(|| {
            let _ = vad.push(black_box(&speech));
        });
    });
    group.bench_function("silence_window", |b| {
        let mut vad = Vad::new(VadConfig::default());
        let _ = vad.push(&silence);
        b.iter(|| {
            let _ = vad.push(black_box(&silence));
        });
    });
    group.bench_function("noise_window", |b| {
        let mut vad = Vad::new(VadConfig::default());
        let _ = vad.push(&noise);
        b.iter(|| {
            let _ = vad.push(black_box(&noise));
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Mixer soft-clip sum — runs on every mixed output sample
// ---------------------------------------------------------------------------

fn mix_and_clip(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = a[i] + b[i] * 0.5 + c[i] * 0.7 + d[i] * 0.03;
        // Same soft clip as mixer::soft_clip.
        out.push((s / (1.0 + s.abs())) * 1.052);
    }
    out
}

fn bench_mixer_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixer_tick");
    // 24 kHz × 20 ms = 480 samples per tick
    for &n in &[480usize, 960] {
        let voice = sine(n, 200.0, 24_000.0, 0.6);
        let filler = sine(n, 350.0, 24_000.0, 0.4);
        let conn = sine(n, 600.0, 24_000.0, 0.5);
        let amb = xor_noise(n, 0.1, 42);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n}_samples")),
            &(voice, filler, conn, amb),
            |b, (v, f, c, a)| {
                b.iter(|| mix_and_clip(black_box(v), black_box(f), black_box(c), black_box(a)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_vht2,
    bench_spectral_entropy,
    bench_spectral_flatness,
    bench_vad_one_window,
    bench_mixer_tick,
);
criterion_main!(benches);
