//! Real-time conversational assistant orchestrator.
//!
//! A tokio-based async pipeline that wires the existing Voxtral ASR and TTS
//! together with cpal mic/speaker I/O into a Sesame-AI-style call experience:
//!
//! ```text
//! mic ──► audio_in ──► (mpsc PcmChunk) ──► VAD ─┬─► ASR ─► (Transcript) ─┐
//!                                               │                         │
//!                                               └─► (Speech/Silence)      ▼
//!                                                                       LLM
//!                                                                         │
//!                                                                         ▼
//!                                          mixer ◄─── TTS ◄── (LlmToken) ─┘
//!                                            │
//!                                            ▼
//!                                       audio_out ──► speaker
//! ```
//!
//! All inter-task communication uses `tokio::sync::mpsc` (no Arc<Mutex> in the hot path).
//! The orchestrator owns the global session state machine and supervises tasks.
//!
//! Gated behind the `assistant` feature and only built for non-wasm targets.

pub mod asr;
pub mod assets;
pub mod audio_in;
pub mod audio_out;
pub mod config;
pub mod filler;
pub mod mixer;
pub mod orchestrator;
pub mod state;
pub mod tts;
pub mod vad;

pub use config::AssistantConfig;
pub use orchestrator::AssistantOrchestrator;
pub use state::SessionState;
