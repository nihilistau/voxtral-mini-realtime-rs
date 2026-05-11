//! Token-ahead routing + agent-tier affinity (CosySim patterns).
//!
//! Two pieces, both pulled from CosySim's `engine/lmstudio/router.py`
//! and `engine/pipeline/token_router.py`:
//!
//! 1. **`AgentAffinity`** — bounded LRU mapping `voice_id → preferred
//!    tier`. The orchestrator records which voice was last used; the
//!    router biases toward keeping that voice's KV cache warm.
//! 2. **`TokenAheadDispatcher`** — fires `PreWarm` events as early
//!    tokens arrive in the LLM stream, so downstream tasks (voice
//!    swap, tool prep) can start before the full intent is confirmed.
//!
//! v2 ships the infrastructure; concrete pre-warm actions land in
//! later commits once we have actual tools and multi-voice routing.

use std::collections::VecDeque;
use std::time::Instant;

/// Compute tier — where a turn runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Big model on the discrete GPU (RTX). Default for `act` / multi-
    /// turn dialogue.
    GpuPrimary,
    /// Small CPU model for utility / classification tasks.
    CpuUtility,
    /// Tiny router model (Gemma 270M) on the iGPU. Speculative draft
    /// candidate.
    CpuRouter,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::GpuPrimary => "gpu_primary",
            Tier::CpuUtility => "cpu_utility",
            Tier::CpuRouter => "cpu_router",
        }
    }
}

/// Bounded-LRU `voice_id → Tier` affinity tracker.
///
/// Sticky: once a voice runs on a tier, prefer that tier on subsequent
/// turns so the LLM's KV cache (for that character's persona prefix)
/// stays warm. Evicting an entry means the next turn for that voice
/// pays the prefill cost again.
pub struct AgentAffinity {
    capacity: usize,
    /// (voice_id, tier, last_used). VecDeque keeps insertion order so
    /// LRU eviction is O(N) but capacity is small (256 by default).
    entries: VecDeque<(String, Tier, Instant)>,
}

impl AgentAffinity {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    /// Get the cached tier for a voice, or `None` if unknown.
    pub fn get(&self, voice_id: &str) -> Option<Tier> {
        self.entries
            .iter()
            .find(|(v, _, _)| v == voice_id)
            .map(|(_, t, _)| *t)
    }

    /// Record that `voice_id` just ran on `tier`. Updates last-used or
    /// inserts. Evicts the LRU entry if at capacity.
    pub fn record(&mut self, voice_id: &str, tier: Tier) {
        let now = Instant::now();
        // Find existing.
        if let Some(idx) = self.entries.iter().position(|(v, _, _)| v == voice_id) {
            self.entries[idx] = (voice_id.to_string(), tier, now);
            return;
        }
        // Insert new; evict LRU if needed.
        if self.entries.len() >= self.capacity {
            // Evict the entry with the oldest last_used.
            if let Some(lru) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, _, t))| *t)
                .map(|(i, _)| i)
            {
                self.entries.remove(lru);
            }
        }
        self.entries.push_back((voice_id.to_string(), tier, now));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for AgentAffinity {
    fn default() -> Self {
        Self::new(256)
    }
}

// ---------------------------------------------------------------------------
// Token-ahead dispatcher
// ---------------------------------------------------------------------------

/// Pre-warm signals emitted as early tokens arrive. Each one tells
/// downstream tasks to start preparing a resource before the full
/// intent is confirmed.
///
/// CosySim's `_DEFAULT_INTENT_MAP` covers tool-call intents like
/// `"selfie" → "generate_image_request"`. Ours stays voice-focused.
#[derive(Debug, Clone)]
pub enum PreWarm {
    /// Swap to this voice for the rest of the reply.
    Voice(String),
    /// Higher euler-step quality (for narration / story mode).
    HighQualityTts,
    /// Faster euler-step (for very short responses).
    FastTts,
}

/// Dispatcher inspects each token piece as it arrives and emits
/// `PreWarm` signals.
pub struct TokenAheadDispatcher {
    /// True once we've sent enough pre-warm decisions for this turn.
    saturated: bool,
    /// Buffer of tokens since turn start. Pattern matchers run against
    /// the lowercase prefix.
    prefix: String,
}

impl TokenAheadDispatcher {
    pub fn new() -> Self {
        Self {
            saturated: false,
            prefix: String::with_capacity(64),
        }
    }

    /// Called on every LLM token piece. Returns at most one PreWarm
    /// signal per turn. After saturation, becomes a no-op until reset.
    pub fn observe(&mut self, piece: &str) -> Option<PreWarm> {
        if self.saturated {
            return None;
        }
        self.prefix.push_str(piece);
        // Truncate so the matcher cost is bounded.
        if self.prefix.len() > 128 {
            self.saturated = true;
            return None;
        }
        let p = self.prefix.to_lowercase();
        // Very small rule set; replace with a real classifier later.
        if p.contains("once upon") || p.contains("let me tell you a story") {
            self.saturated = true;
            return Some(PreWarm::HighQualityTts);
        }
        if (p.starts_with("yes")
            || p.starts_with("no")
            || p.starts_with("sure")
            || p.starts_with("ok"))
            && self.prefix.len() > 16
        {
            // Short response detected; only saturate after a few
            // tokens to avoid false-positive on "Yes, however,…".
            self.saturated = true;
            return Some(PreWarm::FastTts);
        }
        None
    }

    pub fn reset(&mut self) {
        self.saturated = false;
        self.prefix.clear();
    }
}

impl Default for TokenAheadDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinity_round_trip() {
        let mut a = AgentAffinity::new(4);
        assert!(a.get("alice").is_none());
        a.record("alice", Tier::GpuPrimary);
        assert_eq!(a.get("alice"), Some(Tier::GpuPrimary));
    }

    #[test]
    fn affinity_evicts_lru_at_capacity() {
        let mut a = AgentAffinity::new(2);
        a.record("alice", Tier::GpuPrimary);
        std::thread::sleep(std::time::Duration::from_millis(5));
        a.record("bob", Tier::GpuPrimary);
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Touch alice so bob becomes LRU.
        a.record("alice", Tier::GpuPrimary);
        a.record("carol", Tier::GpuPrimary);
        assert_eq!(a.get("bob"), None);
        assert!(a.get("alice").is_some());
        assert!(a.get("carol").is_some());
    }

    #[test]
    fn dispatcher_detects_story_mode() {
        let mut d = TokenAheadDispatcher::new();
        let mut got = None;
        for piece in ["Once ", "upon ", "a ", "time"] {
            if let Some(p) = d.observe(piece) {
                got = Some(p);
                break;
            }
        }
        assert!(matches!(got, Some(PreWarm::HighQualityTts)));
    }

    #[test]
    fn dispatcher_emits_once() {
        let mut d = TokenAheadDispatcher::new();
        let first = d.observe("Once upon a time in a far away land");
        assert!(matches!(first, Some(PreWarm::HighQualityTts)));
        // After saturation no more emissions.
        for _ in 0..5 {
            assert!(d.observe(" more text").is_none());
        }
    }
}
