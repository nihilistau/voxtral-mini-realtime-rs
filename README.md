# Voxtral Mini 4B Realtime (Rust) — Shannon-Prime Fork

[![HuggingFace ASR](https://img.shields.io/badge/%F0%9F%A4%97-ASR_Model-yellow)](https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf)
[![HuggingFace TTS](https://img.shields.io/badge/%F0%9F%A4%97-TTS_Model-yellow)](https://huggingface.co/TrevorJS/voxtral-tts-q4-gguf)
[![ASR Demo](https://img.shields.io/badge/%F0%9F%8E%99%EF%B8%8F-ASR_Demo-blue)](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime)
[![TTS Demo](https://img.shields.io/badge/%F0%9F%94%8A-TTS_Demo-purple)](https://huggingface.co/spaces/TrevorJS/voxtral-4b-tts)

Streaming speech recognition and text-to-speech running natively and in the browser. A pure Rust implementation of Mistral's [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (ASR) and [Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) models using the [Burn](https://burn.dev) ML framework.

> **Fork of [TrevorS/voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs)** — this fork adds real-time waveform visualization (browser Canvas + CLI TUI), Shannon-Prime VHT2 KV cache compression, and a full documentation suite.

### Fork Additions

| Feature | Description |
|---------|-------------|
| Real-time waveform (browser) | Canvas-based scrolling waveform with peak-bucketed downsampling, 60fps |
| Real-time waveform (CLI TUI) | ratatui + crossterm Unicode block-character rendering via `--tui` flag |
| Shannon-Prime VHT2 | Vilenkin-Hartley Transform KV cache compression (~4.6x) |
| Level Zero iGPU backend | Zero-copy USM decode on Intel iGPU — 5.2x faster than wgpu on same hardware |
| Hybrid RTX→L0 pipeline | Encoder on RTX (wgpu), decoder on iGPU (Level Zero), zero-copy KV cache |
| Shared ring buffer | `src/audio/ring_buffer.rs` — circular buffer with peak-bucketed snapshot |
| Documentation suite | Setup guide, usage reference, WASM API docs in `docs/` |

## Benchmarks

NVIDIA DGX Spark (GB10, LPDDR5x).

### ASR (Speech Recognition)

16s test audio, 3-run average:

| Path | Encode | Decode | Total | RTF | Tok/s | Memory |
|------|--------|--------|-------|-----|-------|--------|
| **Q4 GGUF native** | 1021 ms | 5578 ms | 6629 ms | **0.416** | **19.4** | 703 MB |
| BF16 native | 887 ms | 23689 ms | 24607 ms | 1.543 | 4.6 | 9.2 GB |
| Q4 GGUF WASM | — | — | ~225 s | ~14.1 | ~0.5 | (browser) |

- **8.49% WER** on FLEURS English (647 utterances), vs. Mistral's reported 4.90% at f32

### TTS (Text-to-Speech)

"The quick brown fox jumps over the lazy dog" (9 tokens), casual_female voice:

| Path | Euler Steps | Gen Time | Audio | RTF | Model Size |
|------|-------------|----------|-------|-----|------------|
| **Q4 GGUF native** | 3 | 3.7s | 3.84s | **0.97** | 2.67 GB |
| Q4 GGUF native | 4 | 5.0s | 4.96s | 1.01 | 2.67 GB |
| BF16 native | 3 | 10.4s | 2.72s | 3.82 | ~8 GB |
| BF16 native | 8 | 20.6s | 2.96s | 6.97 | ~8 GB |
| Q4 GGUF WASM | 8 | 367s | 3.52s | 104 | 2.67 GB |

- **RTF** < 1.0 means faster-than-real-time synthesis
- Q4 at 3 Euler steps achieves **real-time** with perfect Whisper large-v3 transcription
- Optimizations: batched CFG (2× → batch=2), fused QKV+gate/up projections, pre-allocated KV cache
- Q4 model load: 3.9s native, 9.2s WASM (including shard download over localhost)
- 20 preset voices across 9 languages. Use `--euler-steps` to tune speed/quality tradeoff


# RTF Benchmark Results — Voxtral Mini Q4 GGUF

**Hardware:** Intel NUC 11 Extreme (Beast Canyon)
- **Discrete GPU:** NVIDIA GeForce RTX 2060 (12 GB VRAM)
- **Integrated GPU:** Intel UHD Graphics (shared system memory)
- **CPU:** Intel Core i9-11900KB
- **OS:** Windows, Vulkan backend

**Model:** Voxtral Mini 4B Q4_0 GGUF (~2.5 GB)

**Date:** 2026-05-06

---

## Summary

| Mode | 3.4s Audio | 34s Audio | 120s Audio | Notes |
|------|-----------|-----------|------------|-------|
| **Discrete (RTX)** | 1.91 RTF | 0.63 RTF | **0.55 RTF** | Fastest. Real-time at ≥30s audio |
| Discrete + SP | 1.42 RTF | 0.97 RTF | 1.39 RTF | SP overhead hurts when VRAM is available |  

### Level Zero iGPU Backend (SP-SVM Engine)

## Detailed Results

### Short Audio (3.4s — "Mary had a little lamb")

| Mode | Pre (ms) | Enc (ms) | Xfer (ms) | Dec (ms) | Total (ms) | RTF | Tok/s |
|------|---------|---------|----------|---------|-----------|-----|-------|
| discrete | 372 | 2,099 | 0 | 4,111 | 6,582 | 1.91 | 7.3 |
| discrete+SP | 24 | 1,435 | 0 | 3,423 | 4,882 | 1.42 | 8.8 |
| integrated+SP | 71 | 26,756 | 0 | 24,101 | 50,928 | 14.80 | 1.2 |
| hybrid | 23 | 1,523 | 339 | 23,390 | 25,275 | 7.35 | 1.3 |
| hybrid+pipe | 23 | 1,181 | 0 | 23,638 | 25,200 | 7.33 | 1.3 |

### Medium Audio (34.4s — 10x concatenation)

| Mode | Pre (ms) | Enc (ms) | Xfer (ms) | Dec (ms) | Total (ms) | RTF | Tok/s |
|------|---------|---------|----------|---------|-----------|-----|-------|
| discrete | 420 | 6,121 | 0 | 15,053 | 21,593 | 0.63 | 14.8 |
| discrete+SP | 403 | 5,954 | 0 | 27,014 | 33,371 | 0.97 | 8.3 |
| hybrid | 423 | 5,280 | 357 | 106,444 | 112,503 | 3.27 | 2.1 |
| hybrid+pipe | 391 | 5,356 | 319 | 106,377 | 112,443 | 3.27 | 2.1 |

### Long Audio (120.4s — 35x concatenation)

| Mode | Pre (ms) | Enc (ms) | Xfer (ms) | Dec (ms) | Total (ms) | RTF | Tok/s |
|------|---------|---------|----------|---------|-----------|-----|-------|
| discrete | 326 | 19,137 | 0 | 47,079 | 66,542 | 0.55 | 16.2 |
| discrete+SP | 344 | 18,728 | 0 | 147,753 | 166,825 | 1.39 | 5.2 |
| hybrid | 326 | 17,778 | 297 | 423,914 | 442,315 | 3.67 | 1.8 |
| hybrid+pipe | 325 | 17,770 | 324 | 424,309 | 442,727 | 3.68 | 1.8 |

---

## Analysis

### Scaling with Audio Length

Discrete mode improves dramatically with longer audio — from 1.91 RTF (3.4s) to 0.55 RTF (120s). This is because the fixed model-load and warmup costs amortize over more audio. At 120s, the RTX 2060 transcribes at 1.8x real-time speed.

### Shannon-Prime Overhead

On discrete GPU, Shannon-Prime VHT2 compression adds significant decode overhead (3.1x slower at 120s). The VHT2 compress/decompress cycles on every KV cache access dominate when VRAM isn't constrained. SP's value is enabling inference on memory-constrained devices (iGPU), not throughput optimization.

### Hybrid Decode Bottleneck

The iGPU decode is 7-9x slower than RTX decode. This completely dominates the total time, making the encode phase (which runs at RTX speed) irrelevant to the overall RTF.

### Pipeline Overlap

Pipelined hybrid shows virtually no improvement over non-pipelined hybrid. The reason: encode time (~18s for 120s audio) is dwarfed by decode time (~424s). Even if you perfectly overlap all encode work with decode work, you save at most 18s out of 442s total — a 4% improvement, within measurement noise.

### When to Use Each Mode

- **Discrete:** Best throughput. Use when RTX has available VRAM (~2.5 GB)
- **Hybrid:** When RTX VRAM is needed for other workloads (rendering, other models). Frees 2.5 GB RTX VRAM at cost of 6.6x slower inference
- **Integrated-only:** Only when no discrete GPU is available. Too slow for real-time use
- **Shannon-Prime:** Only beneficial on memory-constrained devices. Do not enable on discrete GPU

---

## Level Zero Zero-Copy Backend (SP-SVM Engine)

**Branch:** `svm-zero-copy`  
**Date:** 2026-05-06  
**Hardware:** Same NUC Beast Canyon (Intel UHD Graphics, 32 EUs)

The Level Zero backend bypasses wgpu/Vulkan entirely for iGPU decode, using Intel's native L0 API with USM (Unified Shared Memory) for true zero-copy operation between CPU and iGPU.

### L0 Hybrid Results (RTX Encode → L0 iGPU Decode)

Test audio: 3.4s "Mary had a little lamb"

| Metric | wgpu Hybrid | L0 Hybrid | Improvement |
|--------|-------------|-----------|-------------|
| Encode (RTX) | 1,523 ms | 1,217 ms | 1.25x (warmup pass) |
| Decode (iGPU) | 23,390 ms | 15,535 ms | 1.5x |
| Per-token steady-state | ~340 ms | 229.4 ms | 1.48x |
| Total RTF | 7.35 | 4.98 | 1.48x |
| Decode-only RTF | — | 2.87 | — |

### L0 Decode-Only Results (Pure iGPU, 20 tokens)

| Metric | wgpu iGPU | L0 iGPU | Improvement |
|--------|-----------|---------|-------------|
| Per-token | ~1200 ms (14.80 RTF) | 229 ms | **5.2x** |
| RTF | 14.80 | 2.87 | **5.2x** |

### Why L0 Is Faster

The 5.2x improvement over wgpu on the same 32-EU iGPU comes from eliminating abstraction overhead:

1. **USM zero-copy** — CPU (VHT2 compress/decompress) and GPU (Q4 matmul) operate on the same physical DRAM pointers. wgpu creates staging buffers + fences even on UMA hardware.
2. **Pre-created kernel pool** — 3 kernels reused across all dispatches. wgpu recompiles pipelines per shape variant.
3. **Reusable command list** — `zeCommandListReset` → append → close → submit → sync. Avoids create/destroy overhead per dispatch.
4. **Warmup pass** — Primes L0 kernel JIT and USM page faults before timed execution.
5. **Zero-alloc KV write** — Direct `copy_from_slice` into USM buffers, no heap allocation per token.

### L0 Build & Run

```bash
# Build
cargo build --release --features "wgpu,cli,hub,l0"

# Smoke test (validates L0 pipeline)
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-smoke

# Q4 matmul correctness
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-q4-test

# Pure L0 decode benchmark (no encoder)
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-decode -- \
  --gguf models/voxtral-q4.gguf --tokens 20

# Full hybrid: RTX encode → L0 iGPU decode
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-hybrid -- \
  --gguf models/voxtral-q4.gguf --audio test_data/mary_had_lamb.wav

# Single-layer microbenchmark
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-bench
```

### Architecture Summary

```
RTX 2060 (Vulkan/wgpu)          Intel UHD (Level Zero)
┌────────────────────┐          ┌────────────────────────────────┐
│  Mel → Encoder     │          │  26-layer autoregressive decode │
│  → Adapter         │──f32──→  │  Q4 matmul (SPIR-V kernel)     │
│  (audio embeddings)│  xfer    │  + CPU RoPE/Attention/SwiGLU    │
└────────────────────┘          │  + VHT2 KV compression (USM)    │
                                └────────────────────────────────┘
                                         │
                                    USM Shared Memory
                                    (zero-copy CPU↔GPU)
```

### Per-Token Breakdown (26 layers)

| Operation | Location | Time |
|-----------|----------|------|
| Q4 matmul (QKV, O, gate/up, down) | iGPU | ~180 ms |
| RoPE + GQA attention | CPU | ~30 ms |
| SwiGLU + RMSNorm + residuals | CPU | ~15 ms |
| VHT2 compress/decompress (KV) | CPU (on USM) | ~4 ms |
| **Total per token** | | **~229 ms** |


Try the demos: [ASR (speech-to-text)](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime) | [TTS (text-to-speech)](https://huggingface.co/spaces/TrevorJS/voxtral-4b-tts)

## Quick Start

### Native CLI

```bash
# Download ASR model weights (~9 GB BF16 or ~2.5 GB Q4)
uv run --with huggingface_hub \
  hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir models/voxtral
uv run --with huggingface_hub \
  hf download TrevorJS/voxtral-mini-realtime-gguf --local-dir models/

# Transcribe audio (BF16 or Q4)
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  transcribe --audio audio.wav --model models/voxtral
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  transcribe --audio audio.wav --gguf models/voxtral-q4.gguf

# With real-time TUI waveform display
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  transcribe --audio audio.wav --gguf models/voxtral-q4.gguf --tui
```

### Browser Demo

```bash
# Build WASM package
wasm-pack build --target web --no-default-features --features wasm

# Generate self-signed cert (WebGPU requires secure context)
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout /tmp/voxtral-key.pem -out /tmp/voxtral-cert.pem \
  -days 7 -nodes -subj "/CN=localhost"

# Start dev server
bun serve.mjs
```

Open `https://localhost:8443`, accept the certificate, and click **Load from Server** to download the model shards. Record from your microphone or upload a WAV file to transcribe.

Hosted demos: [ASR on HuggingFace Spaces](https://huggingface.co/spaces/TrevorJS/voxtral-mini-realtime) | [TTS on HuggingFace Spaces](https://huggingface.co/spaces/TrevorJS/voxtral-4b-tts)

### Text-to-Speech

```bash
# Download TTS model weights (~8 GB BF16 or ~2.67 GB Q4)
uv run --with huggingface_hub \
  hf download mistralai/Voxtral-4B-TTS-2603 --local-dir models/voxtral-tts
uv run --with huggingface_hub \
  hf download TrevorJS/voxtral-tts-q4-gguf voxtral-tts-q4.gguf --local-dir models

# Synthesize speech (BF16 or Q4)
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --voice casual_female
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --voice casual_female --gguf models/voxtral-tts-q4.gguf

# Real-time with 3 Euler steps
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- \
  speak --text "Hello world" --gguf models/voxtral-tts-q4.gguf --euler-steps 3

# List available voices
cargo run --release --features "wgpu,cli,hub" --bin voxtral -- speak --list-voices
```

20 preset voices across 9 languages. The TTS pipeline runs backbone (Ministral 3B) autoregressive decoding, flow-matching acoustic prediction, and codec synthesis to produce 24 kHz audio.

### Level Zero Backend (Intel iGPU)

```bash
# Requires Intel GPU with Level Zero driver (Windows or Linux)
# Branch: svm-zero-copy

# Pure L0 decode benchmark
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-decode -- \
  --gguf models/voxtral-q4.gguf --tokens 20

# Full hybrid pipeline: RTX encode → L0 iGPU decode
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-hybrid -- \
  --gguf models/voxtral-q4.gguf --audio test_data/mary_had_lamb.wav
```

The Level Zero backend implements the SP-SVM (Shannon-Prime Shared Virtual Memory) engine: Q4 matmul kernels dispatched via Intel Level Zero on USM shared memory, with VHT2 KV cache compression operating in-place on the same pointers — zero copies between CPU and iGPU.

## Architecture

```
Audio (16kHz mono)
  -> Mel spectrogram [B, 128, T]
    -> Causal encoder (32 layers, 1280 dim, sliding window 750)
      -> Conv 4x downsample -> Reshape [B, T/16, 5120]
        -> Adapter [B, T/16, 3072]
          -> Autoregressive decoder (26 layers, 3072 dim, GQA 32Q/8KV)
            -> Token IDs -> Text
```

### Two Inference Paths

| | BF16 (native) | Q4 GGUF (native + browser) |
|---|---|---|
| Weights | SafeTensors (~9 GB) | GGUF Q4_0 (~2.5 GB) |
| Linear ops | Burn tensor matmul | Custom WGSL shader (fused dequant + matmul) |
| Embeddings | f32 tensor (1.5 GiB) | Q4 on GPU (216 MB) + CPU bytes for lookups |
| Browser | No | Yes (WASM + WebGPU) |

### Q4 Padding Workaround

The upstream mistral-common library left-pads audio with 32 silence tokens (at 12.5 Hz). After the mel/conv/reshape pipeline, this covers only 16 of the 38 decoder prefix positions with silence — the remaining 22 contain actual audio. The f32 model handles this fine, but Q4_0 quantization makes the decoder sensitive to speech content in the prefix: audio that starts immediately with speech (mic recordings, clips with no leading silence) produces all-pad tokens instead of text.

The left padding is increased to 76 tokens, which maps to exactly 38 decoder tokens of silence and covers the full streaming prefix. See [`src/audio/pad.rs`](src/audio/pad.rs) for details.

### WASM Constraints Solved

Running a 4B model in a browser tab required solving five hard constraints:

1. **2 GB allocation limit** — `ShardedCursor` reads across multiple `Vec<u8>` buffers
2. **4 GB address space** — Two-phase loading: parse weights, drop reader, then finalize
3. **1.5 GiB embedding table** — Q4 embeddings on GPU + CPU-side row lookups
4. **No sync GPU readback** — All tensor reads use `into_data_async().await`
5. **256 workgroup invocation limit** — Patched cubecl-wgpu to cap reduce kernel workgroups

## Building

```bash
# Native (default features: wgpu + native-tokenizer)
cargo build --release

# With all features
cargo build --release --features "wgpu,cli,hub"

# WASM
wasm-pack build --target web --no-default-features --features wasm
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `wgpu` (default) | GPU backend via Burn/CubeCL (WebGPU, Vulkan, Metal) |
| `native-tokenizer` (default) | Tekken BPE encoding via tiktoken (WASM-compatible) |
| `wasm` | Browser support: wasm-bindgen, WebGPU device init, JS bindings |
| `cli` | CLI binary with clap + indicatif |
| `hub` | HuggingFace Hub model downloads |
| `l0` | Intel Level Zero backend for zero-copy iGPU decode (Windows, requires Intel GPU driver) |

## Testing

```bash
# Unit + integration tests (requires GPU for full suite)
cargo test --features "wgpu,cli,hub"

# Lint
cargo clippy --features "wgpu,cli,hub" -- -D warnings
cargo clippy --no-default-features --features wasm --target wasm32-unknown-unknown -- -D warnings

# E2E browser test (requires Playwright + model shards)
bunx playwright test tests/e2e_browser.spec.ts
```

GPU-dependent tests (model layer shapes, Q4 matmul, WGSL shader correctness) are skipped in CI since GitHub Actions runners lack a GPU adapter. These tests run locally on any machine with Vulkan, Metal, or WebGPU support.

## Model Preparation

### Q4 GGUF Sharding (for browser)

GGUF files must be split into shards of 512 MB or less to stay under the browser's `ArrayBuffer` limit:

```bash
# ASR shards
split -b 512m models/voxtral-q4.gguf models/voxtral-q4-shards/shard-

# TTS shards (quantize first, then shard)
uv run --with safetensors --with torch --with numpy --with packaging \
  scripts/quantize_tts_gguf.py models/voxtral-tts/ -o models/voxtral-tts-q4.gguf
split -b 512m models/voxtral-tts-q4.gguf models/voxtral-tts-q4-shards/shard-
```

The dev server discovers shards from `models/voxtral-q4-shards/` (ASR) and `models/voxtral-tts-q4-shards/` (TTS).

## Project Structure

```
src/
  audio/            # Mel spectrogram, chunking, resampling, padding, ring buffer
    ring_buffer.rs  # Shared circular buffer for waveform visualization
  models/           # BF16 model: encoder, decoder, adapter, attention, RoPE, KV cache
    layers/
      shannon_prime.rs  # VHT2 KV cache compression (Shannon-Prime)
  gguf/             # Q4 GGUF: reader, loader, model, tensor, WGSL shader, tests
  web/              # WASM bindings: VoxtralQ4, initWgpuDevice, async decode loop
  tts/              # TTS pipeline: backbone, flow matching, codec, voice presets
  tokenizer/        # Tekken tokenizer: decode (ASR) + encode (TTS via tiktoken)
  l0/               # Level Zero backend: zero-copy iGPU decode (SP-SVM engine)
    mod.rs          # L0 module root, feature-gated
    device.rs       # L0 device discovery and context creation
    usm.rs          # USM shared memory allocator + KV cache
    decode.rs       # L0DecodeContext: kernel pool, reusable cmd list, VHT2
    kernel.rs       # Module/kernel creation and dispatch
    ocl_compile.rs  # OpenCL C → native binary compilation
    spirv_gen.rs    # Q4 matmul OpenCL kernel source
    q4_decoder.rs   # Full 26-layer decoder (bypasses Burn/wgpu)
  tui/              # Terminal UI: waveform widget, event loop, shared state
    mod.rs          # TuiState + run_tui() event loop
    waveform_widget.rs  # Unicode block-char waveform renderer
  bin/voxtral/
    transcribe.rs   # ASR CLI binary (--tui flag for waveform display)
    speak.rs        # TTS CLI binary

space/              # Browser demo: index.html, worker.js, voxtral-client.js
  waveform.js       # Canvas-based scrolling waveform renderer
tests/              # Integration tests + Playwright E2E spec
scripts/            # Dev scripts: reference implementations, weight inspection
patches/            # cubecl-wgpu workgroup size fix for WebGPU
docs/               # Documentation suite
  SETUP.md          # Installation and build guide
  USAGE.md          # CLI and API usage reference
  WASM_API.md       # Browser JavaScript API docs
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Setup Guide](docs/SETUP.md)** — Installation, prerequisites, model downloads, and troubleshooting
- **[Usage Guide](docs/USAGE.md)** — CLI commands, Rust API examples, browser quickstart
- **[WASM API Reference](docs/WASM_API.md)** — VoxtralClient and WaveformRenderer JavaScript APIs

## License

Apache-2.0
