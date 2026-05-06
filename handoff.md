# Handoff — Voxtral Mini Realtime RS

**Purpose:** Enable any future Claude session (or human contributor) to pick up exactly where the last session left off.

---

## Project Identity

- **Repo:** nihilistau/voxtral-mini-realtime-rs (fork of TrevorS/voxtral-mini-realtime-rs)
- **Part of:** Shannon-Prime Project (standalone workstream)
- **Language:** Rust (Burn ML framework), WASM target for browser
- **What it does:** Real-time speech-to-text and text-to-speech using Voxtral Mini 4B, runs natively (Vulkan/Metal) and in-browser (WebGPU)

## What's Been Done (Session 2026-05-06)

1. **Compiled and validated** — Rust 1.92, Vulkan, 60+ tests passing
2. **Fixed VHT2 test** — was asserting energy concentration (wrong for Walsh-Hadamard), now asserts energy preservation
3. **Waveform visualizer (browser)** — Canvas-based scrolling waveform in `space/waveform.js`, wired into VoxtralClient via `onAudioChunk`
4. **Waveform visualizer (CLI TUI)** — ratatui + crossterm in `src/tui/`, Unicode block-char rendering
5. **Shared ring buffer** — `src/audio/ring_buffer.rs` with peak-bucketed downsampling
6. **TUI integrated into CLI** — `voxtral transcribe --tui` flag spawns waveform display
7. **Full documentation suite** — `docs/SETUP.md`, `docs/USAGE.md`, `docs/WASM_API.md`
8. **ASR E2E validated** — Downloaded Q4 GGUF (2.5 GB), transcribed TTS-generated "Mary had a little lamb" correctly
9. **TTS E2E validated** — Q4 at euler-steps 3, 14.5x RTF for short phrases
10. **README overhauled** — Fork additions, TUI usage, docs links
11. **CHANGELOG v0.3.0** — All fork changes documented
12. **WASM build verified** — wasm32-unknown-unknown target installed, `cargo check` and `cargo build` pass
13. **All tests pass** — 48 audio/tokenizer/ring-buffer, 12 GPU/GGUF, 4 Shannon-Prime = 64 total

## What's Done (Session 2026-05-06 pt. 2 — Shannon-Prime SVM Engine)

1. **Shannon-Prime wired into Q4 pipeline** — `KVCache::update()` now transparently compresses/decompresses via VHT2 when `ShannonPrimeConfig` is attached
2. **CLI flags** — `--device integrated|discrete|auto` and `--shannon-prime` on both `transcribe` and `speak`
3. **AVX-512/AVX2 VHT2** — runtime SIMD dispatch in `vht2_f32_inplace()` (scalar fallback preserved)
4. **Engine module** — `src/engine/mod.rs` + `src/engine/svm.rs` with device selection and KV memory estimation
5. **Architecture doc** — `docs/SHANNON_PRIME_SVM_ENGINE.md` covering NUC Beast Canyon + Optane M10 design
6. **CI still green** — formatting fix from earlier session committed

## What's Done (Session 2026-05-06 pt. 3 — Hybrid Split Engine)

1. **Hybrid RTX↔iGPU split engine** — encoder on RTX 2060 (discrete), decoder on Intel UHD (integrated)
2. **`load_hybrid()`** in `Q4ModelLoader` — loads encoder+adapter on discrete GPU, decoder+Q4 embeddings on integrated
3. **`transcribe_streaming_hybrid()`** — encode on RTX, transfer via `to_data()`/`from_data()` (zero-copy on UMA/SVM), decode on iGPU with Shannon-Prime KV cache
4. **`--hybrid` CLI flag** — automatically enables Shannon-Prime, selects DiscreteGpu(0) for encoder, IntegratedGpu(0) for decoder
5. **`load_compact()`** — keeps embeddings in Q4 (~216MB) for iGPU which can't allocate 1.5GB f32 buffer
6. **All three modes tested** — discrete-only (RTX), integrated-only (iGPU + compact + Shannon-Prime), hybrid (RTX encoder + iGPU decoder)
7. **E2E verified** — "Mary had a little lamb. Its fleece was" transcribed correctly in hybrid mode, model load 2.4s
8. **Committed and pushed** — `e609714` on `sp` remote

## What's Done (Session 2026-05-06 pt. 4 — Pipeline & Benchmarking)

1. **Pipelined hybrid inference** — `transcribe_streaming_hybrid_pipelined()` overlaps encode chunk N+1 on RTX while decoding chunk N on iGPU
2. **`--pipelined` CLI flag** — requires `--hybrid`, kicks off encoding for next chunk before waiting for decode
3. **E2E benchmark binary** — `src/bin/e2e_bench.rs` with `--compare-all`, `--device`, `--shannon-prime`, `--hybrid`, `--pipelined` flags
4. **RTF comparison across 5 modes** — discrete, discrete+SP, integrated+SP, hybrid, hybrid+pipe
5. **Three audio lengths tested** — 3.4s, 34.4s, 120.4s
6. **Key findings:**
   - RTX discrete-only is fastest: **0.55 RTF** (1.8x real-time) at 120s audio
   - Shannon-Prime adds 2.5x overhead on discrete GPU (1.39 RTF) — designed for memory savings, not speed
   - Hybrid iGPU decode is 7-9x slower than RTX decode (3.67 RTF at 120s)
   - Pipeline overlap provides negligible benefit (~0.1%) — encode time dwarfed by decode bottleneck
7. **Full results in** `benchmarks/BENCHMARK_RESULTS.md` and `benchmarks/all_results.json`

## What's Done (Session 2026-05-06 pt. 5 — Level Zero Zero-Copy Backend)

**Branch: `svm-zero-copy`** — all L0 work is here.

1. **Level Zero backend** — `src/l0/` module with dynamic FFI to `ze_loader.dll`
2. **Device discovery** — Intel UHD Graphics (32 EUs, 4 GiB max alloc) on NUC Beast Canyon
3. **USM shared allocation** — `zeMemAllocShared` gives true zero-copy pointer (CPU+GPU same DRAM)
4. **OpenCL kernel compilation** — compile OpenCL C → native binary → L0 module (bypasses naga SPIR-V flavor mismatch)
5. **vecadd kernel verified** — GPU dispatch on USM buffers with correct results
6. **Q4 matmul kernel verified** — dequant+matmul produces max error 0.000003 vs CPU reference
7. **L0DecodeContext** — integrates USM KV cache + VHT2 (zero-copy in-place) + Q4 dispatch
8. **Benchmark (debug mode, Intel UHD 32 EUs):**
   - Q4 matmul [1,1,3072]×[3072,3072]: **0.76 ms**
   - VHT2 compress+decompress on USM: **0.17 ms** (zero copy, no staging)
   - Per-token (1 layer): 3.2 ms → estimated 26-layer: ~83 ms/token → **~12 tok/s**
   - vs wgpu baseline: 14.80 RTF (1.2 tok/s) = **~10× improvement**

**Key insight:** wgpu creates staging buffers + fences even on UMA. L0+USM bypasses this entirely.
The same `*mut f32` pointer is used by CPU (VHT2) and GPU (attention) with zero copy.

**SPIR-V issue solved:** naga outputs Vulkan-flavor SPIR-V (GLSL.std.450) which Intel IGC rejects.
Solution: compile OpenCL C via `OpenCL.dll` (always present), extract native binary, load via `ZE_MODULE_FORMAT_NATIVE`.

## What's Done (Session 2026-05-06 pt. 6 — Full 26-Layer L0 Decoder)

**Branch: `svm-zero-copy`** — extends L0 work with complete decoder.

1. **Full 26-layer decoder** — `src/l0/q4_decoder.rs` (832 lines) implements the entire autoregressive decode loop bypassing Burn/wgpu entirely
2. **Per-layer forward pass** — RMSNorm → Q/K/V proj → RoPE → KV cache (VHT2) → CPU GQA attention → O proj → residual → FFN (gate/up/SwiGLU/down) → residual
3. **Batched GPU dispatch** — `q4_matmul_batch()` in `decode.rs` packs multiple kernel launches into one command list with single fence sync (4 fences per layer instead of 6)
4. **Multiplexed KV cache** — 26×8=208 effective KV heads in single USM allocation (1248 MiB)
5. **GGUF → USM loader** — `load_decoder_weights_from_gguf()` reads Q4 bytes directly into USM shared memory
6. **E2E benchmark binary** — `src/bin/l0_decode.rs` with clap CLI, per-token timing, RTF reporting
7. **VHT2 pad-to-PoT workaround** — head_dim=96 (3072/32) padded to 128 for transform, truncated back
8. **CPU GQA attention** — handles 32Q/8KV grouped query attention on CPU (memory-bound for single-token decode)

**Benchmark results (release build, Intel UHD 770, 32 EUs):**
- **229 ms/token steady-state**
- **2.86 RTF** (real-time factor)
- **5.2× improvement** over wgpu iGPU baseline (14.80 RTF)
- Model load: 1.5–1.7s into USM
- Init (L0 context + kernel compile + KV alloc): ~2s

**Bottleneck identified:** 156× `zeKernelCreate` per token (6 per layer × 26 layers). Pre-creating and reusing kernel objects is the biggest remaining perf opportunity.

**VHT2 composite-order deferred:** User confirmed VHT2 should handle non-PoT natively via mixed-radix decomposition (96 = 2⁵×3), but said to "wait and target as optimization."

## What's Done (Session 2026-05-06 pt. 7 — L0 Optimizations & Hybrid Pipeline)

**Branch: `svm-zero-copy`** — final optimizations and hybrid integration.

1. **Pre-created kernel pool** — 3 kernels reused for batched QKV projections (eliminates 156× zeKernelCreate/token)
2. **Reusable command list** — `zeCommandListReset` → append → close → submit → sync (eliminates create/destroy per dispatch)
3. **Zero-alloc KV write** — inlined `copy_from_slice` directly into USM, removed dead `write_kv_to_cache` function
4. **Warmup passes** — both encode (primes CubeCL autotune) and decode (primes L0 JIT + USM page faults)
5. **L0 Hybrid binary** — `src/bin/l0_hybrid.rs`: RTX encode (wgpu) → L0 iGPU decode (Level Zero)
6. **Benchmark results:**
   - Encode: 2,164 ms → 1,217 ms (warmup eliminated autotune from timed path)
   - Decode steady-state: 229.4 ms/token (2.87 RTF)
   - Total RTF: 4.98 (vs 7.35 wgpu hybrid, vs 14.80 wgpu iGPU-only)
   - 5.2x improvement over wgpu on same 32-EU hardware
7. **Docs updated** — README, benchmarks, state.md, handoff.md, CLAUDE.md all reflect L0 numbers
8. **Committed and pushed** — all on `svm-zero-copy` branch to `sp` remote

## What's Next

1. **Composite-order VHT2** — handle head_dim=96 natively via mixed-radix (96 = 2⁵×3) without pad-to-PoT
2. **MoE expert paging** — the SP-SVM Engine architecture is ready for dynamic expert scheduling (Oracle + ping-pong buffers + Optane-backed weight reservoir)
3. **Tag v0.4.0** — Shannon-Prime SVM engine release

## Key Files to Know

| File | Purpose |
|------|---------|
| `CLAUDE.md` | AI assistant instructions, build commands, architecture |
| `plan.md` | Phased implementation plan |
| `state.md` | Current project status snapshot |
| `handoff.md` | This file — session continuity |
| `CHANGELOG.md` | Version history (v0.3.0 fork additions) |
| `src/audio/ring_buffer.rs` | Shared circular buffer for waveform viz |
| `src/tui/mod.rs` | TUI event loop, TuiState shared state |
| `src/tui/waveform_widget.rs` | Unicode waveform widget for ratatui |
| `src/models/layers/shannon_prime.rs` | VHT2 KV cache compression |
| `space/waveform.js` | Browser Canvas waveform renderer |
| `src/gguf/loader.rs` | Q4 model loading — load_compact(), load_hybrid() |
| `src/gguf/model.rs` | Q4 model — transcribe_streaming_hybrid() |
| `src/bin/voxtral/transcribe.rs` | CLI transcribe with --tui, --hybrid, --device flags |
| `src/bin/e2e_bench.rs` | E2E benchmark binary with RTF comparison |
| `src/l0/mod.rs` | Level Zero backend: FFI bindings, init |
| `src/l0/device.rs` | L0 device discovery and context |
| `src/l0/usm.rs` | USM shared memory allocator + UsmKvCache |
| `src/l0/decode.rs` | L0DecodeContext: USM + VHT2 + Q4 dispatch |
| `src/l0/ocl_compile.rs` | OpenCL C → native binary compilation |
| `src/l0/kernel.rs` | SPIR-V/native module loading, kernel dispatch |
| `src/bin/l0_smoke.rs` | L0 pipeline smoke test |
| `src/bin/l0_q4_test.rs` | Q4 matmul correctness test on L0 |
| `src/l0/q4_decoder.rs` | Full 26-layer L0 decoder (DecoderConfig, L0Decoder, weights loader) |
| `src/bin/l0_bench.rs` | L0 zero-copy decode benchmark |
| `src/bin/l0_decode.rs` | Full 26-layer decode benchmark with GGUF loading |
| `src/bin/l0_hybrid.rs` | Hybrid pipeline: RTX encode → L0 iGPU decode |
| `benchmarks/BENCHMARK_RESULTS.md` | Full benchmark analysis and results |
| `docs/SETUP.md` | Installation and build guide |
| `docs/USAGE.md` | CLI and API usage reference |
| `docs/WASM_API.md` | Browser JavaScript API docs |

## Build Commands (Quick Reference)

```bash
# Native
cargo build --release --features "wgpu,cli,hub"
cargo test --features "wgpu,cli,hub" --lib
cargo clippy --features "wgpu,cli,hub" -- -D warnings

# WASM
cargo build --no-default-features --features wasm --target wasm32-unknown-unknown
wasm-pack build --target web --no-default-features --features wasm  # (needs wasm-pack installed)

# Run transcription
cargo run --features "wgpu,cli,hub" --bin voxtral -- transcribe --audio test_data/mary_had_lamb.wav --gguf models/voxtral-q4.gguf --tui

# Run TTS
cargo run --features "wgpu,cli,hub" --bin voxtral -- speak --text "Hello" --gguf models/voxtral-tts-q4-gguf/voxtral-tts-q4.gguf --euler-steps 3 --voices-dir models/voxtral-tts-q4-gguf/voice_embedding

# L0 (Intel Level Zero backend)
cargo run --features "wgpu,cli,hub,l0" --bin l0-smoke      # validate L0 pipeline
cargo run --features "wgpu,cli,hub,l0" --bin l0-q4-test    # Q4 matmul correctness
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-bench  # single-layer benchmark
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-decode -- --gguf models/voxtral-q4.gguf --tokens 20  # full 26-layer decode
cargo run --release --features "wgpu,cli,hub,l0" --bin l0-hybrid -- --gguf models/voxtral-q4.gguf --audio test_data/mary_had_lamb.wav  # hybrid RTX→L0
```

## Model Weights (Local)

| Model | Path | Size | Status |
|-------|------|------|--------|
| ASR Q4 GGUF | `models/voxtral-q4.gguf` | 2.5 GB | Downloaded |
| TTS Q4 GGUF | `models/voxtral-tts-q4-gguf/voxtral-tts-q4.gguf` | ~2.7 GB | Downloaded |
| Tokenizer | `models/tekken.json` | ~2 MB | Copied from TTS |
| Voice presets | `models/voxtral-tts-q4-gguf/voice_embedding/` | 20 files | Downloaded |

## Git Workflow

- Work on `main` branch (direct commits allowed)
- Push to `sp` remote (nihilistau/voxtral-mini-realtime-rs)
- Atomic commits, push after each logical unit

## Gotchas & Warnings

1. **Model weights not in repo** — need `hf download` commands from CLAUDE.md
2. **WASM 2GB allocation limit** — drives sharded loading design
3. **Peak normalization is critical** — Q4 path fails on quiet audio without `peak_normalize(0.95)`
4. **cubecl patch** — `patches/cubecl-wgpu-0.9.0/` is required, don't update cubecl without checking
5. **PowerShell timeout** — `cargo test` with all tests can exceed 45s due to GPU init. Run module subsets.
6. **Git lock files** — sandbox sometimes leaves `.git/index.lock`. Remove before committing.
7. **PDB collision warnings** — harmless on Windows when lib+binary share a crate name
8. **TTS voice path** — must pass `--voices-dir models/voxtral-tts-q4-gguf/voice_embedding` explicitly
9. **test_data/** — not in git; generate test audio via TTS or download separately
10. **Release builds** — take 10-14 min with fat LTO (`codegen-units = 1`); debug builds are faster but inference is ~5x slower
11. **iGPU 1.5GB limit** — Intel UHD can't allocate f32 token embeddings (1.5 GiB). Use `load_compact()` or `--hybrid` which keeps Q4 embeddings (~216 MB)
12. **Hybrid mode** — `--hybrid` auto-enables `--shannon-prime`. Cross-device transfer is zero-copy on UMA but involves `to_data()`/`from_data()` on discrete systems
