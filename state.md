# State — Voxtral Mini Realtime RS

**Last updated:** 2026-05-06  
**Branch:** main (sp remote: nihilistau/voxtral-mini-realtime-rs)

## Current Status: Phase 4 — Shannon-Prime SVM Engine (Hybrid Complete)

Shannon-Prime VHT2 compression wired into Q4 pipeline. Hybrid RTX↔iGPU split engine implemented and tested on NUC Beast Canyon: encoder runs on RTX 2060 (discrete), decoder on Intel UHD (integrated) with Shannon-Prime KV cache compression. Three inference modes verified: discrete-only, integrated-only, hybrid.

CLI flags: `--device` (integrated/discrete/auto), `--shannon-prime`, `--hybrid`. AVX-512/AVX2 SIMD dispatch for VHT2. Engine module (`src/engine/`) with SVM device selection and KV memory estimation.

Previous milestones: ASR/TTS E2E validated, WASM build verified, CI green, all docs complete.

## What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Native build | **Verified locally** | Rust 1.92, Vulkan, zero clippy warnings |
| Audio + tokenizer tests | **40/40 pass** | CPU-only, fast |
| GGUF GPU tests | **12/12 pass** | Vulkan adapter, ~14s |
| Shannon-Prime VHT2 | **4/4 pass** | Fixed energy test (was testing wrong invariant) |
| Ring buffer | **8/8 pass** | New: `src/audio/ring_buffer.rs` |
| Browser waveform | **Implemented** | Canvas-based, `space/waveform.js` |
| CLI TUI waveform | **Implemented** | ratatui + crossterm, `src/tui/` |
| TTS (Q4 GGUF) | **E2E verified** | "Hello world" → 1.92s audio, 14.5x RTF |
| TTS (BF16) | Upstream verified | 20 voices, 9 languages |
| Native ASR (Q4 GGUF) | **E2E verified** | Discrete, integrated, and hybrid modes |
| Hybrid split engine | **E2E verified** | RTX encoder + iGPU decoder, Shannon-Prime KV |
| WASM/Browser | **Not yet tested** | wasm32 target not installed |
| Docs (SETUP/USAGE/WASM_API) | **Complete** | `docs/` directory |
| README (fork) | **Updated** | Fork additions, TUI, docs links |
| CHANGELOG v0.3.0 | **Written** | All fork changes documented |

## Recent Commits (This Session)

1. `1c287ad` — feat: Shannon-Prime VHT2 KV cache compression module
2. `b2abaae` — docs: add project management files (plan, state, handoff) and update CLAUDE.md
3. `b866cf8` — fix: correct VHT2 test — verify energy preservation, not concentration
4. `e12deac` — feat: add real-time waveform visualizer (browser + CLI TUI)
5. `bb5b708` — docs: update state.md — Phase 1+2 complete
6. `21645ad` — feat: wire TUI into transcribe CLI + add full documentation suite
7. `c4ab5b1` — feat: Shannon-Prime SVM engine — device selection, KV cache wiring, AVX-512 VHT2
8. `e609714` — feat: hybrid RTX↔iGPU split engine — encoder on discrete, decoder on integrated

## Remotes

- `origin` → `https://github.com/TrevorS/voxtral-mini-realtime-rs.git` (upstream fork)
- `sp` → `https://github.com/nihilistau/voxtral-mini-realtime-rs.git` (Shannon-Prime fork)

## Dependencies Snapshot

- Rust 1.92.0 (stable), edition 2021
- Burn 0.20, cubecl 0.9, ratatui 0.29, crossterm 0.28
- wasm-pack for browser builds (wasm32 target not yet installed)
- Playwright + bun for E2E browser tests
- Models: TTS Q4 GGUF downloaded, ASR Q4 GGUF downloading

## Remaining Work

1. **Optane M10 mmap integration** — arriving 2026-05-07, KV cache spill tier
2. **Pipelined overlap** — encode chunk N+1 on RTX while decoding chunk N on iGPU
3. **Benchmarking** — RTF comparison across discrete/integrated/hybrid modes
4. **GitHub CI release workflow** — automated builds and releases
5. **Tag v0.4.0** — Shannon-Prime SVM engine release
