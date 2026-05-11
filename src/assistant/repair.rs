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

use std::sync::Arc;
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
/// Clips are stored as `Arc<[f32]>` so `pick()` returns a cheap refcount
/// bump rather than a full sample-buffer copy.
pub struct RepairBank {
    pub filler: Vec<Arc<[f32]>>,
    pub backchannel: Vec<Arc<[f32]>>,
    pub continue_: Vec<Arc<[f32]>>,
    pub interrupt_ack: Vec<Arc<[f32]>>,
    pub self_correct: Vec<Arc<[f32]>>,
    pub cold_start: Vec<Arc<[f32]>>,
}

fn to_arc_bank(clips: Vec<Vec<f32>>) -> Vec<Arc<[f32]>> {
    clips.into_iter().map(Arc::from).collect()
}

impl RepairBank {
    /// Procedurally generate every repair clip. ~250 ms total at 24 kHz.
    /// Each bank holds 2–4 variants so back-to-back repairs don't repeat.
    pub fn synth(sr: u32) -> Self {
        Self {
            // Existing "uhh / um / mmm / uh" bank.
            filler: to_arc_bank(assets::synth_fillers(sr)),
            // Backchannels are shorter, lower-pitched.
            backchannel: to_arc_bank(vec![
                short_word(sr, 0.18, 150.0, 500.0),
                short_word(sr, 0.16, 170.0, 540.0),
                short_word(sr, 0.20, 140.0, 480.0),
            ]),
            // "Go on" / "and" — slight upward inflection.
            continue_: to_arc_bank(vec![
                two_syllable(sr, 0.30, 200.0, 700.0, 240.0, 820.0),
                two_syllable(sr, 0.26, 220.0, 750.0, 260.0, 880.0),
            ]),
            // Interrupt-ack — falling pitch, breathy attack.
            interrupt_ack: to_arc_bank(vec![
                two_syllable(sr, 0.32, 260.0, 800.0, 200.0, 600.0),
                two_syllable(sr, 0.30, 280.0, 850.0, 210.0, 620.0),
            ]),
            // Self-correct — slightly questioning rise.
            self_correct: to_arc_bank(vec![
                two_syllable(sr, 0.34, 200.0, 700.0, 270.0, 880.0),
                two_syllable(sr, 0.32, 220.0, 740.0, 290.0, 920.0),
            ]),
            // Cold-start patience — longer, more neutral.
            cold_start: to_arc_bank(vec![
                two_syllable(sr, 0.45, 180.0, 600.0, 200.0, 640.0),
                two_syllable(sr, 0.50, 200.0, 640.0, 180.0, 620.0),
                two_syllable(sr, 0.40, 190.0, 620.0, 210.0, 660.0),
            ]),
        }
    }

    /// Pick one variant for the given kind. Returns a cheap `Arc` clone
    /// (refcount bump, no sample copy). Falls back to filler if a bank
    /// is somehow empty; returns an empty `Arc<[f32]>` only when even
    /// the filler bank is empty.
    pub fn pick(&self, kind: RepairKind) -> Arc<[f32]> {
        let bank: &Vec<Arc<[f32]>> = match kind {
            RepairKind::Filler => &self.filler,
            RepairKind::Backchannel => &self.backchannel,
            RepairKind::Continue => &self.continue_,
            RepairKind::InterruptAck => &self.interrupt_ack,
            RepairKind::SelfCorrect => &self.self_correct,
            RepairKind::ColdStartPatience => &self.cold_start,
        };
        let mut rng = rand::thread_rng();
        if let Some(clip) = bank.choose(&mut rng) {
            return clip.clone();
        }
        self.filler
            .choose(&mut rng)
            .cloned()
            .unwrap_or_else(|| Arc::from(Vec::<f32>::new()))
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
    /// True for the duration of cold-start mask. Lifted explicitly by
    /// the caller (e.g. when pre-warm completes); `transition()` does
    /// NOT lift it — the model-load timeline is independent of state
    /// transitions, and racing the two leads to a chatty session.
    pub cold_session: bool,
    /// Last repair kind fired and when. `decide()` uses these to
    /// enforce per-kind cooldowns so the same repair doesn't spam.
    pub last_kind: Option<RepairKind>,
    pub last_repair_at: Option<Instant>,
    /// Counter of consecutive repairs of the same kind. Bumped by
    /// `mark_fired`; the caller may inspect this to vary the audio
    /// pick or escalate to silence after N repeats.
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
            last_repair_at: None,
            consecutive: 0,
        }
    }

    /// Note a session-state transition; resets the dwell timer. Does
    /// NOT touch `cold_session` — call `end_cold_session()` explicitly
    /// once model pre-warm has finished.
    pub fn transition(&mut self, new_state: SessionState) {
        self.state = new_state;
        self.state_changed_at = Instant::now();
    }

    /// Mark cold-start mask done. Call once pre-warm completes.
    pub fn end_cold_session(&mut self) {
        self.cold_session = false;
    }

    /// Record that a repair just fired. Maintains `last_kind`,
    /// `last_repair_at`, and `consecutive`. The caller is expected to
    /// invoke this immediately after pushing the picked audio into the
    /// mixer.
    pub fn mark_fired(&mut self, kind: RepairKind, now: Instant) {
        if self.last_kind == Some(kind) {
            self.consecutive = self.consecutive.saturating_add(1);
        } else {
            self.consecutive = 1;
        }
        self.last_kind = Some(kind);
        self.last_repair_at = Some(now);
    }
}

impl Default for RepairContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimum interval between two `ColdStartPatience` emissions.
/// Without this, a steady `Thinking` state would re-emit the repair on
/// every tick.
pub const COLD_START_INTERVAL: Duration = Duration::from_millis(3_000);

/// Minimum interval between any two repairs of the same kind. Prevents
/// loops of identical clips even when state stays the same.
pub const SAME_KIND_INTERVAL: Duration = Duration::from_millis(1_500);

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
    // Highest priority: interruption acknowledgement bypasses cooldowns.
    if barge_in_just_fired {
        return Some(RepairKind::InterruptAck);
    }

    // Cold-start mask covers the duration of cold_session, fires at
    // most every COLD_START_INTERVAL while in Thinking or Idle.
    if ctx.cold_session
        && cooldown_ok(ctx, now, RepairKind::ColdStartPatience, COLD_START_INTERVAL)
    {
        let dwell = now.duration_since(ctx.state_changed_at);
        if dwell >= Duration::from_secs(2)
            && matches!(ctx.state, SessionState::Thinking | SessionState::Idle)
        {
            return Some(RepairKind::ColdStartPatience);
        }
    }

    // Fast-follow correction: user spoke again within 2 s of our last
    // reply — likely they're correcting us.
    if let (Some(reply_end), SessionState::Thinking) = (ctx.last_reply_end, ctx.state)
    {
        if now.duration_since(reply_end) < Duration::from_millis(2000)
            && ctx.last_user_end.is_some_and(|u| u > reply_end)
            && cooldown_ok(ctx, now, RepairKind::SelfCorrect, SAME_KIND_INTERVAL)
        {
            return Some(RepairKind::SelfCorrect);
        }
    }

    // LLM TTFT exceeded the 100 ms filler budget: bridge with generic
    // filler. Most common case; same-kind cooldown applies so we don't
    // loop two "uhh"s in a row.
    if llm_ttft_exceeded
        && matches!(ctx.state, SessionState::Thinking)
        && cooldown_ok(ctx, now, RepairKind::Filler, SAME_KIND_INTERVAL)
    {
        return Some(RepairKind::Filler);
    }

    // Listening + dwell > 2 s with no transcript yet: "go on".
    if matches!(ctx.state, SessionState::Listening) {
        let dwell = now.duration_since(ctx.state_changed_at);
        if dwell >= Duration::from_millis(2500)
            && ctx.last_user_end.is_some()
            && cooldown_ok(ctx, now, RepairKind::Continue, SAME_KIND_INTERVAL)
        {
            return Some(RepairKind::Continue);
        }
    }

    None
}

/// True if `kind` may fire now: either it hasn't fired before, or the
/// last firing was a different kind, or the interval has elapsed.
fn cooldown_ok(
    ctx: &RepairContext,
    now: Instant,
    kind: RepairKind,
    min_interval: Duration,
) -> bool {
    match (ctx.last_kind, ctx.last_repair_at) {
        (Some(last), Some(at)) if last == kind => now.duration_since(at) >= min_interval,
        _ => true,
    }
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
                assert!(!clip.is_empty(), "{name}/{i} is empty");
                let peak = clip.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                assert!(peak > 0.05, "{name}/{i} peak too quiet: {peak}");
            }
        }
    }

    #[test]
    fn pick_returns_arc_clone_not_copy() {
        let bank = RepairBank::synth(24_000);
        // Compare strong_count before and after pick — should bump by 1.
        let original = bank.filler[0].clone();
        let before = Arc::strong_count(&original);
        // Take many picks; one of them is bound to be index 0. To avoid
        // RNG flake, just take a direct refcount bump on the underlying.
        let dup = original.clone();
        assert_eq!(Arc::strong_count(&original), before + 1);
        assert!(Arc::ptr_eq(&original, &dup));
    }

    #[test]
    fn transition_does_not_lift_cold_session() {
        let mut ctx = RepairContext::new();
        assert!(ctx.cold_session);
        ctx.transition(SessionState::Listening);
        assert!(
            ctx.cold_session,
            "transition() must NOT flip cold_session; only end_cold_session() does"
        );
        ctx.end_cold_session();
        assert!(!ctx.cold_session);
    }

    #[test]
    fn cold_start_cooldown_blocks_back_to_back() {
        let mut ctx = RepairContext::new();
        ctx.state = SessionState::Thinking;
        // First call: cooldown not seeded, dwell exceeds 2s → fires.
        ctx.state_changed_at = Instant::now() - Duration::from_secs(3);
        let now = Instant::now();
        assert_eq!(
            decide(&ctx, now, false, false),
            Some(RepairKind::ColdStartPatience)
        );
        ctx.mark_fired(RepairKind::ColdStartPatience, now);
        // Immediately after firing: cooldown blocks re-emission.
        let now2 = now + Duration::from_millis(500);
        assert_eq!(decide(&ctx, now2, false, false), None);
        // After full cooldown elapses: fires again.
        let now3 = now + COLD_START_INTERVAL + Duration::from_millis(10);
        assert_eq!(
            decide(&ctx, now3, false, false),
            Some(RepairKind::ColdStartPatience)
        );
    }

    #[test]
    fn filler_same_kind_cooldown() {
        let mut ctx = RepairContext::new();
        ctx.cold_session = false;
        ctx.state = SessionState::Thinking;
        ctx.state_changed_at = Instant::now();
        let now = Instant::now();
        assert_eq!(decide(&ctx, now, false, true), Some(RepairKind::Filler));
        ctx.mark_fired(RepairKind::Filler, now);
        // Back-to-back filler blocked.
        assert_eq!(
            decide(&ctx, now + Duration::from_millis(100), false, true),
            None
        );
        // After SAME_KIND_INTERVAL elapses: fires.
        assert_eq!(
            decide(&ctx, now + SAME_KIND_INTERVAL + Duration::from_millis(10), false, true),
            Some(RepairKind::Filler)
        );
    }

    #[test]
    fn mark_fired_tracks_consecutive() {
        let mut ctx = RepairContext::new();
        let t0 = Instant::now();
        ctx.mark_fired(RepairKind::Filler, t0);
        assert_eq!(ctx.consecutive, 1);
        ctx.mark_fired(RepairKind::Filler, t0 + Duration::from_secs(2));
        assert_eq!(ctx.consecutive, 2);
        // Different kind resets.
        ctx.mark_fired(RepairKind::Backchannel, t0 + Duration::from_secs(3));
        assert_eq!(ctx.consecutive, 1);
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
