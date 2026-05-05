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

## What's Next

1. **E2E test with model weights** — download Q4 GGUF, run `voxtral transcribe` on test audio, verify output
2. **TUI in speak command** — wire TuiState into `voxtral speak` for TTS waveform display
3. **WASM build verification** — install wasm32 target, run `wasm-pack build`, verify browser demo
4. **README overhaul** — fork-specific README with screenshots and feature matrix
5. **CHANGELOG entry** — document v0.3.0 waveform feature
6. **Tag release** — v0.3.0 after E2E validation

## Key Files to Know

| File | Purpose |
|------|---------|
| `CLAUDE.md` | AI assistant instructions, build commands, architecture |
| `plan.md` | Phased implementation plan |
| `state.md` | Current project status snapshot |
| `src/audio/ring_buffer.rs` | Shared circular buffer for waveform viz |
| `src/tui/mod.rs` | TUI event loop, TuiState shared state |
| `src/tui/waveform_widget.rs` | Unicode waveform widget for ratatui |
| `space/waveform.js` | Browser Canvas waveform renderer |
| `src/bin/voxtral/transcribe.rs` | CLI transcribe with --tui flag |
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
wasm-pack build --target web --no-default-features --features wasm

# Run transcription
cargo run --features "wgpu,cli,hub" --bin voxtral -- transcribe --audio test_data/mary_had_lamb.wav --gguf models/voxtral-q4.gguf --tui
```

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
