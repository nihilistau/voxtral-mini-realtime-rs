# Assistant Microbenchmark Results

**Captured:** 2026-05-11  
**Platform:** Windows 11, Rust 1.92.0, x86_64-pc-windows-msvc  
**SIMD:** AVX2+AVX-512 runtime-dispatched paths (per `shannon_prime.rs`)  
**Command:**
```bash
cargo bench --features "wgpu,cli,hub,assistant" --bench assistant -- --warm-up-time 1 --measurement-time 3
```

## VHT2 in-place transform — `vht2_f32_inplace`

| N    | Median   | Throughput     |
| ---- | -------- | -------------- |
| 128  | 369 ns   | 347 Melem/s    |
| 256  | 664 ns   | 386 Melem/s    |
| 512  | 1.41 µs  | 362 Melem/s    |
| 1024 | 2.56 µs  | 400 Melem/s    |

## Spectral entropy — `spectral_entropy`

| N   | Median   | Throughput     |
| --- | -------- | -------------- |
| 64  | 692 ns   | 92 Melem/s     |
| 128 | 1.37 µs  | 93 Melem/s     |
| 256 | 2.77 µs  | 92 Melem/s     |
| 512 | 5.55 µs  | 92 Melem/s     |

## Spectral flatness — `spectral_flatness`

| N   | Median   |
| --- | -------- |
| 64  | 380 ns   |
| 128 | 656 ns   |
| 256 | 1.38 µs  |
| 512 | 2.79 µs  |

## VAD full analyze (32 ms window @ 16 kHz = 512 samples)

End-to-end: Hann window × VHT2 × power × spectral_entropy × spectral_flatness + hysteresis.

| Input         | Median   | Per-32-ms budget |
| ------------- | -------- | ---------------- |
| Silence       | 3.86 µs  | 0.012 %          |
| Voiced tone   | 6.17 µs  | 0.019 %          |
| White noise   | 6.76 µs  | 0.021 %          |

## Mixer tick (4-source sum + soft-clip)

| Samples per tick   | Median   | Per-20-ms budget |
| ------------------ | -------- | ---------------- |
| 480  (24 kHz)      | 759 ns   | 0.0038 %         |
| 960  (48 kHz)      | 1.45 µs  | 0.0073 %         |

## Headroom summary

Every hot-path operation runs in single-digit microseconds against
millisecond-scale budgets. The conversational latency floor is set by:

1. ASR encoder+decoder forward pass (GPU; hundreds of ms)
2. LLM forward pass + sampling (CPU candle; depends on tok/sec)
3. TTS backbone + codec (GPU; hundreds of ms)
4. VAD hysteresis frames (≥ 8 × 32 ms = 256 ms end-of-speech)

Not CPU bookkeeping. The VHT2 + entropy + mixer + soft-clip
infrastructure is essentially free relative to the model inference.
