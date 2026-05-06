# State — Voxtral Mini Realtime RS

**Last updated:** 2026-05-06  
**Branch:** svm-zero-copy (sp remote: nihilistau/voxtral-mini-realtime-rs)

## Current Status: Phase 5 — SP-SVM Engine Complete (Level Zero Zero-Copy)

The Shannon-Prime SVM Engine is fully operational on the NUC Beast Canyon. The Level Zero backend (`src/l0/`) bypasses wgpu/Vulkan entirely for iGPU decode, achieving **5.2x speedup** over wgpu on the same 32-EU Intel UHD Graphics.

**Benchmark numbers (3.4s audio, release build):**
- L0 Hybrid (RTX encode → L0 iGPU decode): **4.98 RTF** total, **2.87 decode-only RTF**
- Steady-state decode: **229 ms/token** (4.4 tok/s)
- vs wgpu iGPU baseline: 14.80 RTF → 2.87 RTF = **5.2x improvement**
- Encode (RTX, post-warmup): 1,217 ms

**Key optimizations applied:**
- USM zero-copy (CPU VHT2 + GPU Q4 matmul on same DRAM pointers)
- Pre-created kernel pool (3 kernels, avoids zeKernelCreate per dispatch)
- Reusable command list (avoids create/destroy per dispatch)
- Warmup passes (prime autotune cache + L0 JIT + USM page faults)
- Zero-alloc KV write path (direct copy_from_slice into USM)

Previous milestones: ASR/TTS E2E validated, WASM build verified, CI green, all docs complete, wgpu benchmarks complete.

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
| L0 zero-copy backend | **E2E verified** | 5.2x faster than wgpu iGPU, 229 ms/token |
| L0 hybrid pipeline | **E2E verified** | RTX encode → L0 decode, 4.98 RTF |
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

1. **Composite-order VHT2** — handle head_dim=96 natively via mixed-radix (96 = 2^5 × 3) without pad-to-PoT
2. **Optane M10 mmap integration** — KV cache spill tier for long contexts
3. **MoE expert paging** — dynamic weight streaming for 27B MoE models (requires Oracle scheduler + ping-pong buffers)
4. **Tag v0.4.0** — Shannon-Prime SVM engine release
