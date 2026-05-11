//! Contextual dialog repairs.
//!
//! Generic "uhh / um / mmm" fillers (`assistant::assets::synth_fillers`)
//! are fine as a latency mask but feel robotic. Real conversation
//! repairs are *contextual*:
//!
//! - **Backchannel** ("mhm", "yeah") — acknowledge the user is still
//!   talking; useful for medium pauses where you're not interrupting.
//! - **Continue** ("go on", "and?") — explicit invitation to resume
//!   after the user's pause exceeds the speech-end timeout.
//! - **Interrupt-ack** ("oh, sorry — go ahead") — when our barge-in
//!   handler fires.
//! - **Self-correct** ("wait, I think I misheard") — when we get a
//!   second utterance very fast after replying (likely correction).
//! - **Cold-start patience** ("hang on, getting set up") — initial
//!   load mask, buys 10–20 s of perceived progress vs silence.
//! - **Filler** (the original generic uhh/um) — pure latency mask
//!   between Thinking start and first LLM token.
//!
//! Selection is a small state machine, NOT a classifier. Inputs: the
//! orchestrator's current `SessionState`, time since last transition,
//! whether barge-in just fired, whether this is a cold session.
//!
//! Implementation note: the *audio* for each repair is synthesized via
//! the existing procedural `assets::synth_filler_word`-style functions
//! at session start, then cached in DRAM. A future commit can override
//! to load real WAVs from `assets/repairs/<kind>/*.wav`.

use std::time::{Duration, Instant};

use rand::seq::SliceRandom;

use crate::assistant::assets;
use crate::assistant::state::SessionState;

/// A repair category. Each maps to a small bank of audio clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairKind {
    /// Pure latency mask while LLM is still thinking; the old "uhh".
    Filler,
    /// "Mhm", "yeah" — passive acknowledgement.
    Backchannel,
    /// "Go on", "and?" — explicit continue prompt.
    Continue,
    /// "Sorry, you were saying" — fires after barge-in cancels TTS.
    InterruptAck,
    /// "Wait, did you mean…" — fires when a fresh utterance lands < 2 s
    /// after our previous reply (likely correction / disambiguation).
    SelfCorrect,
    /// "Hang on, getting set up" — masks cold-start model loading.
    ColdStartPatience,
}

impl RepairKind {
    pub fn label(self) -> &'static str {
        match self {
            RepairKind::Filler => "filler",
            RepairKind::Backchannel => "backchannel",
            RepairKind::Continue => "continue",
            RepairKind::InterruptAck => "interrupt_ack",
            RepairKind::SelfCorrect => "self_correct",
            RepairKind::ColdStartPatience => "cold_start_patience",
        }
    }
}

/// Pre-synthesized banks of repair audio. One bank per [`RepairKind`].
/// Built once at session start; cheap clones thereafter.
pub struct RepairBank {
    pub filler: Vec<Vec<f32>>,
    pub backchannel: Vec<Vec<f32>>,
    pub continue_: Vec<Vec<f32>>,
    pub interrupt_ack: Vec<Vec<f32>>,
    pub self_correct: Vec<Vec<f32>>,
    pub cold_start: Vec<Vec<f32>>,
}

impl RepairBank {
    /// Procedurally generate every repair clip. ~250 ms total at 24 kHz.
    /// Each bank holds 2–4 variants so back-to-back repairs don't repeat.
    pub fn synth(sr: u32) -> Self {
        Self {
            // Existing "uhh / um / mmm / uh" bank.
            filler: assets::synth_fillers(sr),
            // Backchannels are shorter, lower-pitched.
            backchannel: vec![
                short_word(sr, 0.18, 150.0, 500.0),
                short_word(sr, 0.16, 170.0, 540.0),
                short_word(sr, 0.20, 140.0, 480.0),
            ],
            // "Go on" / "and" — slight upward inflection.
            continue_: vec![
                two_syllable(sr, 0.30, 200.0, 700.0, 240.0, 820.0),
                two_syllable(sr, 0.26, 220.0, 750.0, 260.0, 880.0),
            ],
            // Interrupt-ack — falling pitch, breathy attack.
            interrupt_ack: vec![
                two_syllable(sr, 0.32, 260.0, 800.0, 200.0, 600.0),
                two_syllable(sr, 0.30, 280.0, 850.0, 210.0, 620.0),
            ],
            // Self-correct — slightly questioning rise.
            self_correct: vec![
                two_syllable(sr, 0.34, 200.0, 700.0, 270.0, 880.0),
                two_syllable(sr, 0.32, 220.0, 740.0, 290.0, 920.0),
            ],
            // Cold-start patience — longer, more neutral.
            cold_start: vec![
                two_syllable(sr, 0.45, 180.0, 600.0, 200.0, 640.0),
                two_syllable(sr, 0.50, 200.0, 640.0, 180.0, 620.0),
                two_syllable(sr, 0.40, 190.0, 620.0, 210.0, 660.0),
            ],
        }
    }

    /// Pick one variant for the given kind. Falls back to filler if a
    /// bank is somehow empty.
    pub fn pick(&self, kind: RepairKind) -> Vec<f32> {
        let bank: &Vec<Vec<f32>> = match kind {
            RepairKind::Filler => &self.filler,
            RepairKind::Backchannel => &self.backchannel,
            RepairKind::Continue => &self.continue_,
            RepairKind::InterruptAck => &self.interrupt_ack,
            RepairKind::SelfCorrect => &self.self_correct,
            RepairKind::ColdStartPatience => &self.cold_start,
        };
        let mut rng = rand::thread_rng();
        bank.choose(&mut rng).cloned().unwrap_or_else(|| {
            self.filler
                .choose(&mut rng)
                .cloned()
                .unwrap_or_default()
        })
    }
}

/// Decision state. Owned by the orchestrator's filler manager.
#[derive(Debug, Clone)]
pub struct RepairContext {
    /// Most recent session-state transition timestamp.
    pub state_changed_at: Instant,
    /// Current session state.
    pub state: SessionState,
    /// Last user utterance ended at this instant; used to detect
    /// fast-follow corrections.
    pub last_user_end: Option<Instant>,
    /// Last assistant reply ended (TTS playback complete) at this instant.
    pub last_reply_end: Option<Instant>,
    /// True for the first ~10 s of a session — biases toward
    /// `ColdStartPatience`.
    pub cold_session: bool,
    /// Counter of consecutive repairs of the same kind, so the
    /// orchestrator doesn't loop the same clip.
    pub last_kind: Option<RepairKind>,
    pub consecutive: u8,
}

impl RepairContext {
    pub fn new() -> Self {
        Self {
            state_changed_at: Instant::now(),
            state: SessionState::Idle,
            last_user_end: None,
            last_reply_end: None,
            cold_session: true,
            last_kind: None,
            consecutive: 0,
        }
    }

    /// Note a session-state transition; resets the timer.
    pub fn transition(&mut self, new_state: SessionState) {
        self.state = new_state;
        self.state_changed_at = Instant::now();
        // Cold session lifts after the first transition out of Idle.
        if !matches!(new_state, SessionState::Idle) && self.cold_session {
            // Stay cold for ~10 s, the typical model-load window.
            // Caller is expected to flip this off explicitly when
            // pre-warm completes.
        }
    }

    /// Mark cold-start mask done.
    pub fn end_cold_session(&mut self) {
        self.cold_session = false;
    }
}

impl Default for RepairContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Decide which repair (if any) to fire right now.
///
/// Returns `None` if no repair is appropriate. The caller is responsible
/// for actually pushing the audio into the mixer's filler channel.
///
/// Time arguments are decoupled (vs. reading the clock inside) for
/// testability.
pub fn decide(
    ctx: &RepairContext,
    now: Instant,
    barge_in_just_fired: bool,
    llm_ttft_exceeded: bool,
) -> Option<RepairKind> {
    // Highest priority: interruption acknowledgement always wins.
    if barge_in_just_fired {
        return Some(RepairKind::InterruptAck);
    }

    // Cold-start mask covers the first ~10 s of the session, fires every
    // ~3 s while in Thinking or Idle to keep the "I'm here, loading"
    // feeling alive.
    if ctx.cold_session {
        let dwell = now.duration_since(ctx.state_changed_at);
        if dwell >= Duration::from_secs(2)
            && matches!(ctx.state, SessionState::Thinking | SessionState::Idle)
        {
            return Some(RepairKind::ColdStartPatience);
        }
    }

    // Fast-follow correction: user spoke again within 2 s of our last
    // reply — likely they're correcting us.
    if let (Some(reply_end), SessionState::Thinking) =
        (ctx.last_reply_end, ctx.state)
    {
        if now.duration_since(reply_end) < Duration::from_millis(2000)
            && ctx
                .last_user_end
                .is_some_and(|u| u > reply_end)
        {
            return Some(RepairKind::SelfCorrect);
        }
    }

    // LLM TTFT exceeded the 100 ms filler budget: bridge with generic
    // filler. This is the most common case.
    if llm_ttft_exceeded && matches!(ctx.state, SessionState::Thinking) {
        return Some(RepairKind::Filler);
    }

    // Listening + dwell > 2 s with no transcript yet: "go on".
    if matches!(ctx.state, SessionState::Listening) {
        let dwell = now.duration_since(ctx.state_changed_at);
        if dwell >= Duration::from_millis(2500) && ctx.last_user_end.is_some() {
            return Some(RepairKind::Continue);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Audio primitives — simple two-formant + envelope, same DSP as `assets`
// ---------------------------------------------------------------------------

fn short_word(sr: u32, dur: f32, f1: f32, f2: f32) -> Vec<f32> {
    let sr = sr as f32;
    let n = (sr * dur) as usize;
    let two_pi = std::f32::consts::TAU;
    let attack = (sr * 0.03) as usize;
    let release = (sr * 0.05) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let env = if i < attack {
            i as f32 / attack as f32
        } else if i > n.saturating_sub(release) {
            (n - i) as f32 / release as f32
        } else {
            1.0
        };
        let t = i as f32 / sr;
        let a = (two_pi * f1 * t).sin() * 0.5;
        let b = (two_pi * f2 * t).sin() * 0.2;
        out.push((a + b) * env * 0.5);
    }
    out
}

fn two_syllable(
    sr: u32,
    dur: f32,
    f1a: f32,
    f2a: f32,
    f1b: f32,
    f2b: f32,
) -> Vec<f32> {
    let half = dur * 0.45;
    let gap = dur * 0.10;
    let mut a = short_word(sr, half, f1a, f2a);
    let pad: Vec<f32> = vec![0.0; (sr as f32 * gap) as usize];
    let b = short_word(sr, half, f1b, f2b);
    a.extend(pad);
    a.extend(b);
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_synth_populates_all_kinds() {
        let bank = RepairBank::synth(24_000);
        assert!(!bank.filler.is_empty());
        assert!(!bank.backchannel.is_empty());
        assert!(!bank.continue_.is_empty());
        assert!(!bank.interrupt_ack.is_empty());
        assert!(!bank.self_correct.is_empty());
        assert!(!bank.cold_start.is_empty());
        // Every clip has non-trivial energy.
        for (name, clips) in [
            ("filler", &bank.filler),
            ("backchannel", &bank.backchannel),
            ("continue", &bank.continue_),
            ("interrupt_ack", &bank.interrupt_ack),
            ("self_correct", &bank.self_correct),
            ("cold_start", &bank.cold_start),
        ] {
            for (i, clip) in clips.iter().enumerate() {
                assert!(
                    !clip.is_empty(),
                    "{name}/{i} is empty"
                );
                let peak = clip.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                assert!(peak > 0.05, "{name}/{i} peak too quiet: {peak}");
            }
        }
    }

    #[test]
    fn decide_returns_interrupt_ack_on_barge_in() {
        let ctx = RepairContext::new();
        let now = Instant::now();
        assert_eq!(
            decide(&ctx, now, true, false),
            Some(RepairKind::InterruptAck)
        );
    }

    #[test]
    fn decide_filler_when_ttft_exceeded_in_thinking() {
        let mut ctx = RepairContext::new();
        ctx.cold_session = false; // skip cold-start mask
        ctx.state = SessionState::Thinking;
        ctx.state_changed_at = Instant::now();
        assert_eq!(
            decide(&ctx, Instant::now(), false, true),
            Some(RepairKind::Filler)
        );
    }

    #[test]
    fn decide_continue_after_long_listening_pause() {
        let mut ctx = RepairContext::new();
        ctx.cold_session = false;
        ctx.state = SessionState::Listening;
        let three_s_ago = Instant::now() - Duration::from_millis(3000);
        ctx.state_changed_at = three_s_ago;
        ctx.last_user_end = Some(three_s_ago);
        assert_eq!(
            decide(&ctx, Instant::now(), false, false),
            Some(RepairKind::Continue)
        );
    }

    #[test]
    fn decide_no_repair_in_quiet_idle() {
        let mut ctx = RepairContext::new();
        ctx.cold_session = false;
        ctx.state = SessionState::Idle;
        assert_eq!(decide(&ctx, Instant::now(), false, false), None);
    }

    #[test]
    fn decide_cold_start_patience_during_thinking() {
        let mut ctx = RepairContext::new();
        ctx.state = SessionState::Thinking;
        ctx.state_changed_at = Instant::now() - Duration::from_secs(3);
        assert_eq!(
            decide(&ctx, Instant::now(), false, false),
            Some(RepairKind::ColdStartPatience)
        );
    }
}
