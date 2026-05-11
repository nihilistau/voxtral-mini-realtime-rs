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

## Conversational SLA — measured vs target

The end-user-perceptible latency is the time from end-of-utterance to first
audio sample out of the speaker. With `voxtral assistant --tui` the TUI's
footer already shows the TTFT for the last turn in milliseconds.

### Measured on this workstation (Windows 11 / RTX + Intel UHD / candle CPU LLM)

Captured 2026-05-11 via `voxtral-bench-assistant --llm --tts`. Median of 3
iterations after 1 warmup. Hardware: this developer's box, not the target
Beast Canyon NUC.

| Stage                                                       | Measured       |
| ----------------------------------------------------------- | -------------- |
| LLM TTFT (Qwen2.5-0.5B-Instruct Q4_K_M, candle CPU)         | **~3 500 ms**  |
| LLM throughput (open-ended prompt, 18 tokens)               | **~5 tok/s**   |
| LLM total wall for a 15-token short reply                   | **~3 600 ms**  |
| TTS wall for 3.68 s of audio (Voxtral Q4, euler-steps 3)    | **~15 000 ms** |
| TTS RTF (synthesis time / audio duration)                   | **~4.0**       |

Reality check on those numbers:

- **LLM TTFT is dominated by prompt prefill on CPU.** Candle's quantized
  Qwen2 forward pass costs ~100–200 ms per layer iteration; the prefill
  forward over a ~32-token chat-templated prompt takes most of the TTFT.
  GPU offload (candle `cuda` feature) would drop this by ~10×.
- **TTS at RTF 4.0 means a 5 s reply takes ~20 s to synthesize.** The
  speaker hears nothing until the codec decode completes — sentence-level
  streaming TTS (deferred work) would push first-audio latency down to
  the per-sentence cost (~3–5 s for the first clause).
- **The filler manager at 100 ms is essential**, not optional. With
  measured TTFT of 3.5 s, every turn would feel broken without an
  "uhh" mask while the LLM thinks.

### Target on Beast Canyon NUC (RTX 2060 + Intel UHD, L0 + SP-SVM)

Aspirational targets from `state.md` once the L0 hybrid path is wired into
the assistant. The current orchestrator uses the wgpu backend for ASR/TTS
and CPU for LLM; the L0 path is sketched in but not yet plumbed into the
assistant feature.

| Hop                                                       | Target ms |
| --------------------------------------------------------- | --------- |
| Mic → VAD speech-end fires (8-frame hysteresis)           | ~256 ms   |
| ASR encode (RTX) on 2 s utterance                         | ~80 ms    |
| ASR decode + tokenize (L0 iGPU, Shannon-Prime KV)         | ~300 ms   |
| LLM TTFT (Qwen2.5-0.5B Q4 + GPU offload)                  | ~120 ms   |
| TTS first audio frame (Voxtral Q4, streaming, euler 3)    | ~150 ms   |
| **Total end-of-speech → first speaker sample**            | **~900 ms** |

Closing the gap from measured 4-5 s to target ~900 ms needs three pieces
in order of impact: (1) GPU-offloaded LLM (~30× TTFT improvement on this
hardware), (2) sentence-streaming TTS (cuts perceived latency by ~5×),
(3) L0 hybrid ASR for the iGPU decoder (~2× decode speedup per the
existing benchmarks in `benchmarks/BENCHMARK_RESULTS.md`).

## Memory & storage tiering

This box has four tiers. The right answer is: **put the hot weights in VRAM,
let DRAM be the spill, use Optane for cold-start / MoE / multi-model, and
NVMe for archival only.**

### Tier latencies — actual numbers

QD1 random read latencies, the metric that matters for autoregressive
inference (every token = a chain of tiny dependent loads). These are the
real "time-to-first-bit" numbers, not the marketing sequential throughput.

| Tier                          | Random read latency | Capacity here | Notes                                  |
| ----------------------------- | ------------------- | ------------- | -------------------------------------- |
| RTX 2060 VRAM                 | **~5 ns** (GPU-local) / **~1 µs** (PCIe round-trip from CPU) | **12 GB** | Where the model *should* live          |
| DDR4/5 DRAM                   | **~80–100 ns**      | 16–32 GB      | Page cache after mmap first-touch      |
| Optane M10 (NVMe-attached)    | **~1 µs** (random 4K, QD1) | 16–64 GB | Byte-addressable-feel for small reads  |
| TLC NVMe                      | **~50–100 µs** (random 4K, QD1) | TB+ | Sequential is fine, random is awful   |
| SATA SSD                      | ~150 µs             | TB+           | Don't                                  |

The headline: Optane is ~50–100× faster than NVMe at QD1 small random
reads. That's the *only* metric that matters when an attention layer chases
KV pointers or an LLM samples from a tail of weights. Sequential throughput
benchmarks don't measure what we care about.

VRAM-from-GPU is *3 orders of magnitude* faster than DRAM-from-CPU for the
operations the model actually runs (per-thread vector loads with massive
parallelism). The PCIe ~1 µs round-trip only matters when the CPU has to
touch a buffer the GPU owns — minimize those crossings, which is exactly
what the SP-SVM USM design is for.

### The arithmetic with this hardware

| Component             | Size       | Best tier  | Reason                                |
| --------------------- | ---------- | ---------- | ------------------------------------- |
| Voxtral Q4 ASR        | 2.5 GB     | **VRAM**   | hot during every utterance            |
| Voxtral Q4 TTS        | 2.7 GB     | **VRAM**   | hot during every reply                |
| Qwen2.5-0.5B Q4       | 379 MB     | **VRAM**   | hot during every think step           |
| Voice presets         | < 100 MB   | DRAM       | one-shot per turn                     |
| Tokenizers            | < 50 MB    | DRAM       | CPU-side                              |
| LLM KV cache (4 k)    | ~50 MB     | **VRAM**   | every-token read/write                |
| **GPU-resident total**| **~5.6 GB** | of 12 GB available — 6 GB headroom |

**Everything fits in VRAM with 6 GB to spare.** That's the right loadout.

Today the LLM runs on CPU through candle (no CUDA feature wired in), so
the 0.5 B + KV cache + workspace lives in DRAM and chews through 100–200 ms
per forward pass. Fixing that is the highest-impact change to TTFT —
candle's `cuda` feature, `Device::Cuda(0)`, done.

### When does Optane actually help?

Now that we've put VRAM at the top, Optane's role is narrower but real:

1. **Cold start.** mmap'd weights page in at ~1 µs/page from Optane vs
   ~50 µs from NVMe. For a 2.5 GB ASR GGUF that's 600k pages → ~0.6 s
   from Optane vs ~30 s from NVMe on a cold boot. After the page cache
   warms up, DRAM serves subsequent reads and the source disk is
   irrelevant.

2. **Multi-model hot-swap.** Today we ship one 0.5 B router. The
   architecture is meant to scale to a 270 M router + 7 B deep model
   pair, swapping in the deep model on demand. Loading 7 B from NVMe is
   ~5 s; from Optane is ~150 ms.

3. **KV-cache spill for long contexts.** A 32 k-token context with a
   reasonable model can exceed VRAM. Spilling old KV pages to Optane vs
   NVMe is the difference between "noticeable hitch" and "session-killing
   stall." With our current 4 k cap and small models this never trips.

4. **MoE expert paging.** A future mixture-of-experts (the `state.md`
   roadmap mentions 27 B MoE). Active experts swap in per token; Optane's
   ~1 µs random read latency makes this viable, NVMe's ~50 µs doesn't.
   See `docs/SHANNON_PRIME_SVM_ENGINE.md`.

### Current recommendation

- **Wire the candle `cuda` feature and pin the LLM to `Device::Cuda(0)`.**
  Single biggest TTFT win available. Enabled via the `llm-cuda` cargo
  feature:

  ```bash
  # Windows + CUDA 13.2 + MSVC 2022: nvcc's CCCL header requires the
  # standard-conforming preprocessor in cl.exe. Set this env var so nvcc
  # forwards /Zc:preprocessor to its host compiler:
  set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor

  cargo build --release --features "wgpu,cli,hub,llm-cuda" --bin voxtral
  ```

  The `pick_device()` helper in `src/assistant/llm.rs` tries
  `Device::new_cuda(0)` first when the feature is on and falls back to
  CPU with a warning if no CUDA card is reachable.
- For ASR + TTS, Burn already picks the wgpu adapter — that already
  uses VRAM. The `--hybrid` flag splits encoder (RTX) + decoder (Intel
  UHD) for memory pressure, which is the right move when running all
  three models simultaneously.
- Put the GGUFs on the fastest random-read tier you have (Optane M10
  > NVMe > SATA). The `models/` junction makes this swappable:

  ```powershell
  Remove-Item models -Recurse -Force
  New-Item -ItemType Junction -Path models -Target "O:\voxtral\models"
  ```

- If/when MoE expert paging or multi-model routing ships, the
  `Q4ModelLoader` already mmap's via `BufReader<File>` so the storage
  tier is transparent — LRU page-out lives in the SP-SVM Engine layer,
  not the model loader.

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
