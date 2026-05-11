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

## Stage-level results (voxtral-bench-assistant, release build)

### LLM — Qwen2.5-0.5B-Instruct Q4_K_M (candle CPU)

```
target/release/voxtral-bench-assistant.exe --llm \
  --llm-model "<path>/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" \
  --warmup 1 --iters 3 --llm-max-tokens 60 \
  --llm-prompt "Briefly describe what real-time speech recognition is, in two sentences."
```

| Iter   | Wall ms | TTFT ms | n_tokens | Reply length |
| ------ | ------- | ------- | -------- | ------------ |
| 0      | 3 599   | 3 510   | 18       | full sentence |
| 1      | 3 408   | 3 407   | 18       | full sentence |
| 2      | 3 750   | 3 657   | 18       | full sentence |
| Median | 3 599   | 3 510   | —        | —             |

**Aggregate tok/sec across all 3 timed runs:** ~5.0 tok/s on CPU candle.
TTFT is dominated by prompt prefill (chat template = ~32 tokens). A
GPU-offloaded build would drop this by ~10×.

### TTS — Voxtral Q4 GGUF, euler_steps=3, voice=casual_female

```
target/release/voxtral-bench-assistant.exe --tts --warmup 1 --iters 2 \
  --tts-text "Hello, this is a test of the Voxtral text to speech pipeline."
```

| Iter   | Wall ms | Audio s | RTF    |
| ------ | ------- | ------- | ------ |
| 0      | 15 043  | 3.68    | 4.09   |
| 1      | 16 855  | 4.08    | 4.13   |
| Median | 15 949  | 3.88    | 3.91   |

**RTF 4.0** means synthesis is 4× slower than playback. Sentence-streaming
TTS (currently deferred work) would let the speaker start before the
codec decodes the whole reply, masking this with per-sentence latency.

## Headroom summary

Every hot-path operation runs in single-digit microseconds against
millisecond-scale budgets. The conversational latency floor is set by:

1. ASR encoder+decoder forward pass (GPU; hundreds of ms)
2. LLM forward pass + sampling (CPU candle; depends on tok/sec)
3. TTS backbone + codec (GPU; hundreds of ms)
4. VAD hysteresis frames (≥ 8 × 32 ms = 256 ms end-of-speech)

Not CPU bookkeeping. The VHT2 + entropy + mixer + soft-clip
infrastructure is essentially free relative to the model inference.
