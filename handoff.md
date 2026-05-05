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
5. **`load_compact()`** — keeps embeddings in Q4 (~216MB) for iGP