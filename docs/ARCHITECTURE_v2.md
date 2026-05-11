# Architecture v2 — distributed pipeline, mid-sentence repair, dialog repairs

Where v1 (shipped Phases 0–5 + LLM CUDA) lands a single linear pipe (mic →
VAD → ASR → LLM → TTS → speaker), v2 distributes work across hardware tiers
and adds the structural pieces that make a real conversational AI feel
human: parallel synthesis, mid-sentence repair, and contextual dialog
repairs that aren't just "uhh".

Patterns adopted from CosySim's existing engine (`engine/lmstudio/` and
`engine/pipeline/`). Where CosySim diverges from a voice-first design,
this doc notes the delta.

## Hardware loadout — who runs where

```
┌───────── Intel UHD iGPU (32 EU, USM zero-copy) ─────────┐
│  ASR decoder (Voxtral Q4, L0 hybrid path)               │
│  VHT2 + entropy + flatness on USM-shared buffers         │
│  Optional: draft model (Gemma 270M Q4) for spec decode   │
└──────────────────────────────────────────────────────────┘
                  ↑ Burn.to_data / .from_data (UMA zero-copy on Beast Canyon)
┌───────────── RTX 2060 (12 GB VRAM) ─────────────────────┐
│  ASR encoder (Voxtral Q4)                                │
│  Big LLM (Qwen2.5-0.5B Q4, candle CUDA)                  │
│  TTS backbone + codec (Voxtral Q4 TTS)                   │
│  ─── 5.6 GB resident, 6 GB headroom for KV + workspace ─ │
└──────────────────────────────────────────────────────────┘
                  ↑ PCIe ~1 µs
┌───────────── DRAM ──────────────────────────────────────┐
│  Page cache for mmap'd GGUFs                            │
│  Voice presets, tokenizers                              │
│  LLM prompt buffers, mixer ring buffers, ringbacks      │
└──────────────────────────────────────────────────────────┘
                  ↑ Optane M10 ~1 µs (50–100× faster than NVMe at QD1)
┌───────────── Optane M10 ────────────────────────────────┐
│  GGUF cold storage; mmap pages in on first-touch        │
│  KV-cache spill tier (future, for > 4 k contexts)       │
│  Multi-model swap pool (future router + deep model)     │
└──────────────────────────────────────────────────────────┘
                  ↑ NVMe ~50 µs
┌───────────── NVMe / Archive ────────────────────────────┐
│  Model backups, training data, transcripts              │
└──────────────────────────────────────────────────────────┘
```

The key principle: **no memcpy across tier boundaries when a pointer
suffices.** Burn's `to_data/from_data` on UMA architectures is already a
DMA-mapped pointer pass; the existing SP-SVM USM path in `src/l0/` keeps
CPU VHT2 and GPU Q4 matmul on the *same* DRAM pointer. v2 extends this
discipline to the new modules.

## New modules

### `assistant::shredder` — sentence-boundary token dispatcher

The single biggest TTS latency fix. Today TTS waits for the full LLM
reply before synthesizing — measured RTF 4.0 means a 3 s reply takes
12 s wall before first audio plays.

The shredder sits between `LlmEvent::Token` and the TTS task:

```
LlmEvent::Token stream  ──►  Shredder  ──►  multiple parallel TTS jobs
   (one piece per token)          │              │
                                  ▼              ▼
                          SentenceChunk     SentenceChunk
                          { text, idx,       { text, idx,
                            is_final }         is_final }
                                  │
                                  ▼
                          Mixer.voice_tx (chunks arrive in order;
                                          first one starts playback
                                          while later sentences are
                                          still synthesizing)
```

Boundaries: `.`, `!`, `?` followed by space, or `\n\n`. Soft fallback: any
clause break (`,`, `;`, `:`) after `min_chunk_chars` characters since the
last boundary, so a long run-on doesn't stall the speaker.

CosySim's equivalent (`pipeline/stream_watcher.py`) only watches for
*kill* signals; it doesn't fork the stream into chunks. The shredder is
new ground in our codebase, but the **token-ahead dispatch pattern**
(prepare downstream work before it's confirmed needed) is straight from
`pipeline/token_router.py::TokenAheadRouter`.

### `assistant::repair` — contextual dialog repairs

The 100 ms filler manager today picks from 4 generic "uhh / um / mmm"
clips at random. That's worse than silence in some contexts — what the
user actually wants is a contextual repair:

| Trigger                                | Repair token (`RepairKind`) | Example                          |
| -------------------------------------- | --------------------------- | -------------------------------- |
| LLM TTFT > 100 ms                      | `Filler` (uhh, mhm)         | bridge to first sentence         |
| Long pause in `Listening`              | `Continue` ("go on")        | invite to resume                 |
| Barge-in during reply                  | `InterruptAck`              | "sorry, you were saying"         |
| Fast-follow utterance after our reply  | `SelfCorrect`               | "oh wait, you mean…"             |
| Cold session, dwell > 2 s              | `ColdStartPatience`         | "hang on, getting set up"        |
| Quiet acknowledgement during long user turn | `Backchannel` ("mhm")  | acknowledge user is thinking     |

The session-start chirp is *not* a `RepairKind` — it stays in
`assistant::filler::play_connection` because it's a one-shot session
lifecycle cue, not a dialog repair.

Selection is a small state machine in `repair::decide()`, not a
classifier. Inputs: current `SessionState`, time since last state
change, time since last repair (per-kind cooldown), last user/reply
timestamps, whether barge-in just fired. `decide()` is pure — the
caller updates state via `RepairContext::mark_fired(kind, now)` after
actually pushing the picked audio into the mixer.

Repairs come from a tiny pre-rendered bank in `assets/repairs/` (or
synthesized at startup via the TTS like the current filler bank). Each
clip ≤ 800 ms so it fits inside any conversational gap.

### `assistant::routing` — token-ahead pre-warm + KV affinity

Both pieces live in one module since they share the "tier" vocabulary.

**`routing::TokenAheadDispatcher`** — CosySim's `TokenAheadRouter`
dispatches tool calls as soon as the intent classifier (Gemma 270M)
confidence-scores a `pre_warm` signal, ~50 ms into the stream. For
voice, equivalent pre-warms are:

- **TTS voice swap** (`PreWarm::Voice`): if the early tokens suggest a
  different speaker, pre-warm that voice's embedding.
- **Long-form** (`PreWarm::HighQualityTts`): if intent is "tell me a
  story", switch TTS to higher `euler_steps` for quality.
- **Short response** (`PreWarm::FastTts`): if reply starts with "yes",
  "no", "sure", "ok", switch TTS to faster `euler_steps`.

v2 ships the dispatcher skeleton with a small rule set; the larger
intent-classifier-driven version is later work.

**`routing::AgentAffinity`** — CosySim's most non-obvious routing
signal. A bounded LRU mapping each `voice_id` to the `Tier` (compute
location) where their KV cache is warm. Switching tiers means
re-prefilling the persona prefix; sticky affinity amortizes that.

For voice, the "agent" is the *voice preset*. If the user has been
talking to `casual_female` for 10 turns, the LLM's KV cache for that
character's persona is warm. Switching to `cheerful_male` mid-session
means recomputing the system-prompt prefix. `AgentAffinity::record(voice_id, tier)`
remembers last-used voice→tier and biases router decisions toward it.

## Mid-sentence repair — partial suffix surgery

CosySim's `KillSwitch.evaluate` returns whole-sentence kill+retry only.
For voice that's wrong: when the user barges in 800 ms into a 3 s reply,
the speaker has already played the first sentence. We don't regenerate
that sentence on resume — we **fork at the spoken boundary** and only
re-plan the suffix.

Required pieces:

1. **`AudioChunk` indexing.** Every chunk leaving the shredder is tagged
   with `(sentence_idx, char_offset_in_reply)`. The mixer reports back
   which `(sentence_idx, char_offset)` was last *actually emitted* to
   the speaker (post-buffering).

2. **Spoken-prefix anchoring.** On barge-in:
   - Mixer flushes the buffered (unplayed) tail.
   - Shredder records the spoken prefix as `committed: String`.
   - Orchestrator continues to `Listening`.

3. **Resume policy.** On the next LLM reply, the LLM's chat history
   shows the assistant's previous turn as `committed + "…"`. New
   generation continues from the natural break, not from scratch.

This is the structural fix to "throw away half the sentence instead of
the whole thing." v2 lands the indexing + anchoring; the LLM-side
resume policy is a one-line history rewrite once the indexing is real.

## Filler-on-connect — buying 10–20 s of cold-start mask

The "ambient tail + connection chirp" today is decorative. With the new
repair vocabulary, the *cold-start* sequence becomes:

```
t=0      connect chirp (250 ms)
t=250    ambient tail starts (loops, -30 dB)
t=400    "hey, give me just a sec" (repair: Greeting + ColdStart)
t=2000   "still loading models, hang on" (repair: ColdStart + Patience)
t=5000   "almost there" (repair: ColdStart + Patience)
t=10000  model load complete; pre-warm pass kicks off
t=12000  pre-warm complete; first prompt accepted
```

That buys 12 seconds of cold-start latency masked by *seemingly natural*
chatter, vs the current solution of 12 s of silence followed by a click.

## Compute optimizations the user called out

- **Pointers instead of memcpy.** Already done in the SP-SVM USM path
  for KV writes. Extend to the shredder ↔ TTS handoff: pass the
  tokenized chunk by reference, not by string clone. Internal note for
  the implementer: `SentenceChunk` holds `Arc<str>` not `String`.

- **Reduce math to sub/add.** The mixer's soft-clip is currently
  `x / (1 + |x|) * 1.052`. The division and absolute-value chain has a
  ~6-cycle dependency chain. An add/select alternative: for
  `|x| < 0.7`, identity; for `0.7 ≤ |x| < 0.95`, linear; for
  `|x| ≥ 0.95`, hard cap. Three branches + 2 ops vs 4 ops, but the
  branch predictor wins because real audio sits in the first bucket
  > 99 % of the time. Measure before committing.

- **CRT chunking.** Chinese-remainder-theorem chunking for the VHT2
  composite path: split N = 96 = 2⁵ · 3 into a 32-point radix-2 pass
  and a 3-point pass that share strides via CRT indexing. The existing
  `vht2_composite_f32` does this for VHT2 but the same technique
  applies to any block-decomposable kernel. Not implemented yet —
  flagged for the next perf pass.

## What's deferred (deliberately, with reasons)

- **Full speculative decoding** — candle 0.10 doesn't expose a verifier
  hook. Spec decoding via two-model coordination would mean rolling
  our own draft-verify loop, which is multi-day work. CosySim sidesteps
  this by delegating to LMStudio's built-in spec decode.

- **In-process LM Studio client.** CosySim's `lms_client.py` /
  `sdk_client.py` are real value, but they wrap a Python-friendly
  service. Embedding that in Rust would duplicate candle's role. Better
  long-term move: expose a CosySim-compatible HTTP endpoint from this
  binary so existing CosySim code can drive it.

- **Fine-tuned router model.** CosySim's `FinetunedRouter` requires a
  per-task adapter pool. Skipping until the routing surface is large
  enough to justify training.
