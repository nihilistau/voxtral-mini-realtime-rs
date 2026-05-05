# State — Voxtral Mini Realtime RS

**Last updated:** 2026-05-06  
**Branch:** main (sp remote: nihilistau/voxtral-mini-realtime-rs)

## Current Status: Phase 2 Complete — Waveform Visualizer Implemented

Native build compiles and passes all tests. Waveform visualizer added for both browser (Canvas) and CLI (ratatui TUI). Shannon-Prime VHT2 tests fixed and passing. Ready for Phase 3 (documentation) and Phase 4 (E2E validation).

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
| Native ASR (BF16) | Upstream verified | SafeTensors, full precision |
| Native ASR (Q4 GGUF) | Upstream verified | Custom WGSL shader, ~2.5 GB |
| WASM/Browser ASR | Upstream verified | WebGPU, sharded loading |
| TTS (BF16) | Upstream verified | 20 voices, 9 languages |
| TTS (Q4) | Upstream verified | Euler-steps 3, real-time |

## Recent Commits (This Session)

1. `b2abaae` — docs: add project management files (plan, state, handoff) and update CLAUDE.md
2. `b866cf8` — fix: correct VHT2 test — verify energy preservation, not concentration
3. `e12deac` — feat: add real-time waveform visualizer (browser + CLI TUI)

## Remotes

- `origin` → `https://github.com/TrevorS/voxtral-mini-realtime-rs.git` (upstream fork)
- `sp` → `https://github.com/nihilistau/voxtral-mini-realtime-rs.git` (Shannon-Prime fork)

## Dependencies Snapshot

- Rust 1.92.0 (stable), edition 2021
- Burn 0.20, cubecl 0.9, ratatui 0.29, crossterm 0.28
- wasm-pack for browser builds (wasm32 target not yet installed)
- Playwright + bun for E2E browser tests
- Models: ~2.5 GB (Q4 GGUF), ~9 GB (BF16 SafeTensors), ~8 GB (TTS)

## Remaining Work

1. **Phase 3 — Documentation:** README overhaul, setup guide, usage guide, API reference, WASM docs
2. **Phase 4 — Integration:** Wire TUI into CLI transcribe/speak commands, E2E test with model weights
3. **WASM build:** Install wasm32 target, verify wasm-pack build still works
4. **Model weights:** Download for full E2E testing
