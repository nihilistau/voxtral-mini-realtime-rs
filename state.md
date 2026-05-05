# State — Voxtral Mini Realtime RS

**Last updated:** 2026-05-06  
**Branch:** main (sp remote: nihilistau/voxtral-mini-realtime-rs)

## Current Status: Pre-compilation / Feature Addition

The repo is a fork of TrevorS/voxtral-mini-realtime-rs with one additional commit from a previous Shannon-Prime session that added a VHT2 KV cache compression module (`src/models/layers/shannon_prime.rs`). The upstream codebase is mature — 230 tests, Criterion benchmarks, working WASM path — but has not yet been compiled by the current operator.

## What Works (Per Git History)

| Component | Status | Notes |
|-----------|--------|-------|
| Native ASR (BF16) | Upstream verified | SafeTensors, full precision |
| Native ASR (Q4 GGUF) | Upstream verified | Custom WGSL shader, ~2.5 GB |
| WASM/Browser ASR | Upstream verified | WebGPU, sharded loading |
| TTS (BF16) | Upstream verified | 20 voices, 9 languages |
| TTS (Q4) | Upstream verified | Euler-steps 3, real-time |
| Shannon-Prime VHT2 | Added, untested locally | KV cache compression 4.6x |
| Waveform Visualizer | **Not started** | New feature to add |

## Remotes

- `origin` → `https://github.com/TrevorS/voxtral-mini-realtime-rs.git` (upstream fork)
- `sp` → `https://github.com/nihilistau/voxtral-mini-realtime-rs.git` (Shannon-Prime fork)

## Dependencies Snapshot

- Rust edition 2021, Burn 0.20, cubecl 0.9
- wasm-pack for browser builds
- Playwright + bun for E2E browser tests
- Models: ~2.5 GB (Q4 GGUF), ~9 GB (BF16 SafeTensors), ~8 GB (TTS)

## Blockers / Open Questions

1. Need to confirm Rust toolchain is installed and correct version
2. Need model weights downloaded (or skip model-dependent tests initially)
3. Shannon-Prime module compiles in isolation but integration test needed
4. Waveform visualizer design not yet specified
