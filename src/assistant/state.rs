//! Session state machine for the assistant.
//!
//! Distributed to all tasks via a `tokio::sync::watch` channel so any task
//! can react to transitions without polling or holding a mutex.

use std::fmt;

/// High-level state of the assistant session.
///
/// Transitions are driven by the orchestrator in response to events:
/// - `Idle → Listening`: session start (or end of TTS playback)
/// - `Listening → Thinking`: VAD detected end-of-speech, ASR finalized
/// - `Thinking → Speaking`: first audio sample emitted from TTS
/// - `Speaking → Listening`: TTS playback completed, or interrupted
/// - `* → Interrupted → Listening`: user barge-in detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// Initial state, pre-warmup not yet finished or session not started.
    Idle,
    /// Capturing user audio, waiting for end-of-utterance.
    Listening,
    /// User finished talking; running ASR → LLM, no audio out yet.
    Thinking,
    /// TTS is producing audio that's being mixed to the speaker.
    Speaking,
    /// Interrupt detected mid-`Speaking`; in the middle of an instant-flush.
    /// Brief transient state; orchestrator drives back to `Listening`.
    Interrupted,
}

impl SessionState {
    /// Human-readable short label for TUI display.
    pub fn label(&self) -> &'static str {
        match self {
            SessionState::Idle => "IDLE",
            SessionState::Listening => "LISTENING",
            SessionState::Thinking => "THINKING",
            SessionState::Speaking => "SPEAKING",
            SessionState::Interrupted => "INTERRUPTED",
        }
    }
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_round_trip() {
        for s in [
            SessionState::Idle,
            SessionState::Listening,
            SessionState::Thinking,
            SessionState::Speaking,
            SessionState::Interrupted,
        ] {
            assert!(!s.label().is_empty());
            assert_eq!(s.to_string(), s.label());
        }
    }
}
