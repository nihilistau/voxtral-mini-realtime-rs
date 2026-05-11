# Performance & Storage Tiering

Targets, measured numbers, and how to reproduce them — plus the question of
when (if ever) Optane storage actually helps a real-time assistant of this
size.

## Hot-path budgets

The assistant has three independent ticks running concurrently. Each one has
a budget set by the audio device and the inference cadence:

| Hot path                       | Budget         | Reason                                                                |
| ------------------------------ | -------------- | --------------------------------------------------------------------- |
| Mic chunk (20 ms @ 16 kHz)     | < 20 ms wall   | cpal callback must return before the next chunk arrives               |
| Mixer tick (20 ms @ 24 kHz)    | < 20 ms wall   | runs every 20 ms; missing a tick = audio dropout                      |
| VAD analyze (32 ms @ 16 kHz)   | < 32 ms wall   | one per Hann-windowed VHT2 frame                                      |
| Per-LLM-token CPU work         | budget-free    | bounded by GGUF model size — measured via tok/sec instead             |

The mic→VAD→ASR flow lives in the tokio runtime, while the speaker side
runs on its own cpal thread. Microbenchmarks measure the *steady-state CPU
cost* of the per-tick math, ignoring channel send/recv overhead which is
dwarfed by it.

## Microbenchmark results

Captured on Windows 11, Rust 1.92.0, x86_64-pc-windows-msvc, AVX2+AVX-512
runtime-dispatched SIMD path engaged (per the runtime detection in
`shannon_prime.rs`). Reproduce with:

```bash
cargo bench --features "wgpu,cli,hub,assistant" --bench assistant
```

### VHT2 in-place transform

The Vilenkin–Hartley transform is the backbone of VAD frame analysis and the
Shannon-Prime KV-cache compression. Power-of-2 sizes hit the SIMD path.

| N    | median time | throughput     |
| ---- | ----------- | -------------- |
| 128  | 369 ns      | 347 Melem/s    |
| 256  | 664 ns      | 386 Melem/s    |
| 512  | 1.41 µs     | 362 Melem/s    |
| 1024 | 2.56 µs     | 400 Melem/s    |

VAD uses N=512 (32 ms @ 16 kHz). KV-cache compression uses N=128 (head_dim).

### Spectral entropy & flatness

These run on every VAD frame to gate speech vs. noise.

| N   | spectral_entropy | spectral_flatness |
| --- | ---------------- | ----------------- |
| 64  | 692 ns           | 380 ns            |
| 128 | 1.37 µs          | 656 ns            |
| 256 | 2.77 µs          | 1.38 µs           |
| 512 | 5.55 µs          | 2.79 µs           |

### Full VAD analyze (Hann + VHT2 + entropy + flatness)

Steady-state cost per 32 ms mic window. Decision-stage cost varies by signal
type because flatness short-circuits earlier on silence.

| Input          | median time | per-32 ms budget % |
| -------------- | ----------- | ------------------ |
| Silence        | 3.86 µs     | 0.012 %            |
| Voiced tone    | 6.17 µs     | 0.019 %            |
| White noise    | 6.76 µs     | 0.021 %            |

VAD costs ~0.02 % of its tick budget — 5000× headroom.

### Mixer per-tick

20 ms tick of 4-source mixing (voice + filler + connection + ambient) with
soft-clip:

| Samples per tick | median time | per-20 ms budget % |
| ---------------- | ----------- | ------------------ |
| 480 (24 kHz)     | 759 ns      | 0.0038 %           |
| 960 (48 kHz)     | 1.45 µs     | 0.0073 %           |

Mixer costs 1/25000th of its budget. Soft-clip is `x / (1 + |x|) * 1.052`
which the compiler vectorizes cleanly.

## Stage-level benchmarks (RTF, TTFT, tok/sec)

The `voxtral-bench-assistant` binary times each pipeline stage in isolation
so end-to-end latency can be attributed to a specific module.

```bash
# All three stages with the default model paths (works because models/
# is a junction to D:/F/.../models/):
cargo run --release --features "wgpu,cli,hub,llm" --bin voxtral-bench-assistant -- \
  --all --llm-model "D:/Files/Models/lmstudio-community/Qwen 2.5 coder 0.5b-1b-3b-14b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

# Just one stage:
cargo run --release --features "wgpu,cli,hub,assistant" --bin voxtral-bench-assistant -- --asr
cargo run --release --features "wgpu,cli,hub,llm"        --bin voxtral-bench-assistant -- --llm \
  --llm-model "D:/Files/.../Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
cargo run --release --features "wgpu,cli,hub,assistant" --bin voxtral-bench-assistant -- --tts
```

Reported per stage:

| Stage | Metrics                                                            |
| ----- | ------------------------------------------------------------------ |
| ASR   | wall time, audio duration, RTF, decoded text                       |
| LLM   | wall time, TTFT, n_tokens, tokens/sec                              |
| TTS   | wall time, output audio duration, RTF                              |

The binary outputs human-readable lines plus a `=== JSON ===` block at the
end suitable for CI assertions.

## Conversational SLA

The end-user-perceptible latency is the time from end-of-utterance to first
audio sample out of the speaker. With `voxtral assistant --tui` the TUI's
footer already shows the TTFT for the last turn in milliseconds.

Target on the Beast Canyon NUC (Intel UHD iGPU + RTX 2060) with the L0
hybrid pipeline + Shannon-Prime KV compression engaged, per `state.md`:

| Hop                                                       | Target ms |
| --------------------------------------------------------- | --------- |
| Mic → VAD speech-end fires (after 8-frame hysteresis)     | ~256 ms   |
| ASR encode (RTX) on a 2 s utterance                       | ~80 ms    |
| ASR decode + tokenize (L0 iGPU, Shannon-Prime KV)         | ~300 ms   |
| LLM TTFT (Qwen2.5-0.5B Q4, CPU candle)                    | ~120 ms   |
| TTS first audio frame (Voxtral Q4, euler-steps 3)         | ~150 ms   |
| **Total end-of-speech → first speaker sample**            | **~900 ms** |

Below 900 ms is the "comfortable conversation" threshold in voice-UX
research; below 600 ms is "feels like a person." Filler injection at 100 ms
masks the gap when LLM TTFT > 100 ms, so perceived TTFT drops to that bound.

## Storage tiering — does Optane matter here?

Short answer: **not for the current size class**.

### The arithmetic

| Component             | Size       | mmap residency         |
| --------------------- | ---------- | ---------------------- |
| Voxtral Q4 ASR        | 2.5 GB     | full                   |
| Voxtral Q4 TTS        | 2.7 GB     | full                   |
| Qwen2.5-0.5B Q4       | 379 MB     | full                   |
| Tokenizers + voices   | < 100 MB   | full                   |
| LLM KV cache (4 k)    | ~50 MB     | hot per-turn           |
| **Working set total** | **~5.7 GB** | comfortably in DRAM   |

On any machine with ≥ 16 GB RAM, the OS page cache holds all four GGUFs in
the file cache after first-touch, so subsequent mmap reads are DRAM-speed
(~100 ns / cache line). Optane (~2 µs page-in) is only faster than DRAM in
the *first-touch* path and only beats NVMe (~50 µs page-in) for cold loads.

### When Optane would actually pay off

The Optane M10 acceleration story makes sense in three scenarios — none of
which apply to today's pipeline:

1. **MoE expert paging**. A 27 B-parameter mixture-of-experts model with
   active experts swapped in per token can't keep all weights in DRAM. The
   `state.md` "MoE expert paging" + ping-pong buffer design is real future
   work that needs Optane.

2. **Multi-model hot-swap**. Switching between a 270 M router and a 7 B
   deep model on demand. Loading 7 B weights from NVMe is ~5 s; from Optane
   is ~250 ms. Today we only ship the 0.5 B model.

3. **KV cache spill for long contexts**. 32 k-token context on a small
   model can exceed the per-process RAM budget; spilling old KV pages to
   Optane vs NVMe shaves orders of magnitude off the swap cost. With our
   4 k cap and ~50 MB worst-case KV, this never spills.

### Current recommendation

- Put the GGUFs on whatever drive is fastest *for cold start* (Optane M10
  > NVMe > SATA SSD). After first launch they live in the page cache
  anyway and the storage tier becomes irrelevant.
- The `models/` directory in this worktree is a Windows junction to
  `D:/F/shannon-prime-repos/voxtral-mini-realtime-rs/models/` — if your
  Optane drive is mounted as e.g. `O:`, you can swap the junction target
  for `O:/voxtral/models/` and rebuild the page cache there:

  ```powershell
  Remove-Item models -Recurse -Force
  New-Item -ItemType Junction -Path models -Target "O:\voxtral\models"
  ```

- If/when we ship the MoE expert paging or multi-model router, the
  `Q4ModelLoader` already mmap's via `BufReader<File>`, so swapping storage
  is transparent — we'd add an LRU page-out policy in the SP-SVM Engine
  layer, not the model loader. See `docs/SHANNON_PRIME_SVM_ENGINE.md` for
  the architectural sketch.

## Reproducibility

```bash
# Microbenchmarks — Criterion HTML reports in target/criterion/
cargo bench --features "wgpu,cli,hub,assistant" --bench assistant

# Stage benchmarks — JSON to stdout
cargo run --release --features "wgpu,cli,hub,llm" --bin voxtral-bench-assistant -- --all \
  --llm-model <path-to-qwen-gguf>

# Existing ASR pipeline bench (encoder + decoder timing)
cargo run --release --features "wgpu,cli,hub" --bin e2e-bench -- \
  --audio test_data/mary_had_lamb.wav --gguf models/voxtral-q4.gguf --tokenizer models/tekken.json
```

All benchmark commands honor the standard Criterion flags (`--warm-up-time`,
`--measurement-time`, `--sample-size`).
