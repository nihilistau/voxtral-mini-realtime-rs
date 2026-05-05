# Handoff — Voxtral Mini Realtime RS

**Purpose:** Enable any future Claude session (or human contributor) to pick up exactly where the last session left off.

---

## Project Identity

- **Repo:** nihilistau/voxtral-mini-realtime-rs (fork of TrevorS/voxtral-mini-realtime-rs)
- **Part of:** Shannon-Prime Project (standalone workstream)
- **Language:** Rust (Burn ML framework), WASM target for browser
- **What it does:** Real-time speech-to-text and text-to-speech using Voxtral Mini 4B, runs natively (Vulkan/Metal) and in-browser (WebGPU)

## What's Been Done

1. **Upstream fork** with full ASR + TTS pipeline (BF16 and Q4 GGUF paths)
2. **Shannon-Prime VHT2 compression** added to KV cache layer (`src/models/layers/shannon_prime.rs`) — 4.6x compression, +0.04% PPL improvement claimed
3. **Planning documents** created: `plan.md`, `state.md`, this file

## What's Next (Immediate)

1. **Compile the project** — operator is new to Rust, may need toolchain setup
2. **Run tests** — 230 tests should pass without model weights (most are unit tests)
3. **Add waveform visualizer** — browser (Canvas) + CLI (ratatui TUI)
4. **Document everything** — full suite: dev docs, user docs, API reference, changelog

## Key Files to Know

| File | Purpose |
|------|---------|
| `CLAUDE.md` | AI assistant instructions, build commands, architecture |
| `plan.md` | Phased implementation plan |
| `state.md` | Current project status snapshot |
| `src/gguf/` | Q4 quantized inference (WASM-capable) |
| `src/models/` | BF16 model components |
| `src/tts/` | Text-to-speech pipeline |
| `src/web/` | WASM bindings |
| `src/audio/` | Audio processing (mel, resampling, normalization) |
| `space/` | Browser demo app (ASR) |
| `space-tts/` | Browser demo app (TTS) |
| `src/models/layers/shannon_prime.rs` | VHT2 KV cache compression |

## Build Commands (Quick Reference)

```bash
# Native
cargo build --features "wgpu,cli,hub"
cargo test --features "wgpu,cli,hub"

# WASM
wasm-pack build --target web --no-default-features --features wasm

# Lint
cargo clippy --features "wgpu,cli,hub" -- -D warnings
```

## Git Workflow

- Work on `main` branch (direct commits allowed per CLAUDE.md)
- Push to `sp` remote (nihilistau/voxtral-mini-realtime-rs)
- Atomic commits, push after each phase completes
- Tag releases at milestones

## Gotchas & Warnings

1. **Model weights not in repo** — need `hf download` commands from CLAUDE.md. Most unit tests don't need them.
2. **WASM 2GB allocation limit** — drives the sharded loading design. Don't combine shards.
3. **Peak normalization is critical** — Q4 path fails on quiet audio without it. See `AudioBuffer::peak_normalize(0.95)`.
4. **cubecl patch** — `patches/cubecl-wgpu-0.9.0/` is required. Don't update cubecl without checking the patch.
5. **Operator context** — user is new to Rust but experienced in other domains. Explain Rust-specific concepts when relevant.

## Open Questions for Next Session

- What Rust toolchain version is installed? (need stable + wasm32 target)
- Are NVIDIA drivers / Vulkan SDK available for GPU tests?
- Does the user want to download model weights now or defer to later?
- Waveform visualizer: any specific visual style preferences?
