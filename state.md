# State — Voxtral Mini Realtime RS

**Last updated:** 2026-05-06  
**Branch:** main (sp remote: nihilistau/voxtral-mini-realtime-rs)

## Current Status: Phase 4 — Shannon-Prime SVM Engine

Shannon-Prime VHT2 compression now wired into the live Q4 inference pipeline. CLI flags `--device` (integrated/discrete/auto) and `--shannon-prime` added. AVX-512/AVX2 SIMD dispatch for VHT2. Engine module created (`src/engine/`) with SVM device selection and KV memory estimation. Architecture document written for NUC Beast Canyon deployment.

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
| Native ASR (Q4 GGUF) | **Downloading model** | ~2.5 GB, curl in progress |
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

1. **ASR E2E test** — finish download, run `voxtral transcribe` with `--tui` flag
2. **WASM build** — install wasm32 target, verify wasm-pack build
3. **TUI in speak command** — wire TuiState into `voxtral speak`
4. **Tag v0.3.0** — after E2E validation
