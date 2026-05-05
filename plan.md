# Plan — Voxtral Mini Realtime RS

**Created:** 2026-05-06  
**Goal:** Get the project building, add a real-time waveform visualizer (browser + CLI), document everything, and ship.

---

## Phase 1: Compile & Validate (Current)

### 1.1 Environment Setup
- Verify Rust toolchain (stable, wasm32-unknown-unknown target)
- Install wasm-pack if not present
- Confirm GPU support (Vulkan on Windows)

### 1.2 Native Build
- `cargo build --features "wgpu,cli,hub"`
- Fix any compilation errors (dependency resolution, feature flags)
- Run `cargo clippy` — zero warnings

### 1.3 Test Suite
- `cargo test --features "wgpu,cli,hub"` (unit tests, no model needed)
- Identify which tests require model weights and skip gracefully
- Confirm Shannon-Prime module passes its own tests

### 1.4 WASM Build
- `wasm-pack build --target web --no-default-features --features wasm`
- Verify pkg/ output is produced

---

## Phase 2: Waveform Visualizer

### 2.1 Design
- Browser: Canvas-based real-time waveform in the existing `space/index.html`
- CLI: ratatui-based TUI waveform during transcription/TTS
- Shared: audio ring buffer abstraction that both renderers consume

### 2.2 Browser Waveform
- Add `<canvas id="waveform">` to space/index.html
- `WaveformRenderer` class in new `space/waveform.js`:
  - Draws PCM amplitude as a scrolling waveform
  - Color-coded: input (blue), output/TTS (green)
  - Responds to mic input in real-time + file playback
- Wire into `VoxtralClient` event system (onAudioChunk callback)

### 2.3 CLI Waveform (TUI)
- Add `ratatui` + `crossterm` as optional deps behind `cli` feature
- New module `src/tui/` with:
  - `WaveformWidget` — renders PCM as Unicode block chars
  - `TranscriptionView` — live text below the waveform
- Integrate into `voxtral transcribe` and `voxtral speak` commands
- Graceful fallback: if terminal doesn't support TUI, just print text

### 2.4 Shared Audio Buffer
- `src/audio/ring_buffer.rs` — lock-free ring buffer (or `Arc<Mutex<VecDeque>>`)
- Both browser (via WASM bindgen) and CLI consume the same abstraction
- Configurable window size (default: 2 seconds at 16kHz = 32,000 samples)

---

## Phase 3: Documentation

### 3.1 Dev Documentation
- Update CLAUDE.md with waveform visualizer architecture
- Inline rustdoc on all new public APIs
- Architecture decision records in `docs/`

### 3.2 User-Facing Documentation
- README.md overhaul: quick start, screenshots, feature matrix
- `docs/SETUP.md` — step-by-step for Windows/Linux/macOS
- `docs/USAGE.md` — CLI reference, browser usage, voice list

### 3.3 API Reference
- `cargo doc --features "wgpu,cli,hub"` builds cleanly
- Key types documented with examples
- WASM JS API documented in `docs/WASM_API.md`

### 3.4 Changelog
- Maintain CHANGELOG.md (keep existing, append new entries)
- Follow Keep a Changelog format

---

## Phase 4: Integration & Ship

### 4.1 E2E Validation
- Native transcription with test audio
- TTS output sounds correct
- Browser loads, transcribes, shows waveform
- TUI waveform renders during CLI usage

### 4.2 Git Hygiene
- Atomic commits per logical change
- Push to `sp` remote after each phase completes
- Tag releases (v0.3.0 for waveform feature)

### 4.3 CI Considerations
- Existing CI via cargo-dist should still pass
- Add clippy check for new code
- WASM build in CI

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-06 | Both browser + CLI waveform | Full coverage of both usage paths |
| 2026-05-06 | ratatui for TUI | Mature, well-maintained, crossterm backend works on Windows |
| 2026-05-06 | Ring buffer shared abstraction | Avoid duplicating audio buffering logic |
| 2026-05-06 | Canvas (not WebGL) for browser | Simpler, sufficient for waveform, no extra deps |
