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

## What's Next

1. **Wire TUI into speak command** — TuiState into `voxtral speak` for TTS waveform display
2. **WASM full build** — install wasm-pack, run `wasm-pack build --target web`, verify browser demo
3. **Browser E2E test** — load model shards in browser, transcribe test audio
4. **Tag v0.3.0 release** — after full E2E validation
5. **CI fix investigation** — GitHub Actions may need Vulkan/GPU setup or skip adjustments

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
cargo build --no-default-features --features wasm --target wasm32-unknown-unknown
wasm-pack build --target web --no-default-features --features wasm  # (needs wasm-pack installed)

# Run transcription
cargo run --features "wgpu,cli,hub" --bin voxtral -- transcribe --audio test_data/mary_had_lamb.wav --gguf models/voxtral-q4.gguf --tui

# Run TTS
cargo run --features "wgpu,cli,hub" --bin voxtral -- speak --text "Hello" --gguf models/voxtral-tts-q4-gguf/voxtral-tts-q4.gguf --euler-steps 3 --voices-dir models/voxtral-tts-q4-gguf/voice_embedding
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
10. **Release builds** — take 60-90s due to LTO; debug builds are faster but inference is ~5x slower
