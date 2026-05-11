//! Sentence-boundary token stream chunker.
//!
//! Sits between the LLM's `LlmEvent::Token` stream and the TTS task.
//! Each incoming token piece is appended to a buffer; whenever the
//! buffer crosses a sentence (or clause) boundary, the completed chunk
//! is emitted downstream so TTS can start synthesizing while the LLM is
//! still generating the rest of the reply.
//!
//! Result: instead of waiting for the whole 60-token reply (~1.5 s at
//! 42 tok/s on CUDA) before TTS begins, the first phrase fires after
//! the first ~10 tokens (~250 ms). The TTS RTF of 4.0 then applies to
//! each sentence independently, and sentences synthesize in parallel.
//!
//! Each emitted [`SentenceChunk`] is indexed so that on barge-in the
//! orchestrator knows which sentences were spoken (and stay in chat
//! history) vs which were generated-but-not-played (and get dropped).

use std::sync::Arc;

/// One bounded slice of the LLM reply, ready for TTS dispatch.
#[derive(Debug, Clone)]
pub struct SentenceChunk {
    /// 0-based index in the reply.
    pub idx: u32,
    /// Character offset where this chunk starts in the full reply.
    pub char_offset: u32,
    /// The text. `Arc<str>` so multiple consumers (TTS + transcript
    /// history + TUI) share a single allocation.
    pub text: Arc<str>,
    /// True when this is the last chunk for the current LLM turn (the
    /// LLM emitted `LlmEvent::Done` and we flushed any partial buffer).
    pub is_final: bool,
    /// Reason this chunk was emitted; useful for TUI/telemetry.
    pub boundary: Boundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// `.`, `!`, `?` followed by whitespace or end.
    Sentence,
    /// `,`, `;`, `:` after `min_chunk_chars` since last boundary.
    Clause,
    /// `\n\n` paragraph break.
    Paragraph,
    /// LLM finished and we flushed remaining text.
    Final,
}

/// Tuning knobs.
#[derive(Debug, Clone)]
pub struct ShredderConfig {
    /// Below this character count, we don't emit even on clause
    /// boundaries — keep accumulating so TTS gets enough context for
    /// good prosody.
    pub min_chunk_chars: usize,
    /// Hard limit: if we've buffered this much without finding any
    /// boundary, force-emit at the next whitespace so the speaker
    /// doesn't stall.
    pub max_chunk_chars: usize,
}

impl Default for ShredderConfig {
    fn default() -> Self {
        Self {
            min_chunk_chars: 24,
            max_chunk_chars: 180,
        }
    }
}

/// Streaming shredder. Push tokens with [`push`](Self::push), call
/// [`flush`](Self::flush) when the LLM is done.
pub struct Shredder {
    cfg: ShredderConfig,
    buf: String,
    char_offset: u32,
    next_idx: u32,
    /// Characters in `buf` since the last boundary candidate (used to
    /// gate the clause-emit path).
    chars_since_break: usize,
}

impl Shredder {
    pub fn new(cfg: ShredderConfig) -> Self {
        Self {
            cfg,
            buf: String::with_capacity(256),
            char_offset: 0,
            next_idx: 0,
            chars_since_break: 0,
        }
    }

    /// Feed one token piece (typically the UTF-8-safe piece emitted by
    /// `TokenOutputStream::next_token`). Returns any sentence chunks
    /// that completed inside this token. May return zero or several.
    pub fn push(&mut self, piece: &str) -> Vec<SentenceChunk> {
        let mut out = Vec::new();
        for ch in piece.chars() {
            self.buf.push(ch);
            self.chars_since_break += 1;
            if let Some(boundary) = self.detect_boundary() {
                if let Some(chunk) = self.cut(boundary) {
                    out.push(chunk);
                }
            }
        }
        // Force-emit if we exceeded max_chunk_chars and the last char
        // was whitespace — gives TTS a workable break.
        if self.buf.len() >= self.cfg.max_chunk_chars
            && self.buf.chars().last().is_some_and(|c| c.is_whitespace())
        {
            if let Some(chunk) = self.cut(Boundary::Clause) {
                out.push(chunk);
            }
        }
        out
    }

    /// Mark end-of-turn: any remaining buffer is emitted as a final
    /// chunk (`is_final = true`).
    pub fn flush(&mut self) -> Option<SentenceChunk> {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            return None;
        }
        let text: Arc<str> = self.buf.trim().to_string().into();
        let chunk = SentenceChunk {
            idx: self.next_idx,
            char_offset: self.char_offset,
            text,
            is_final: true,
            boundary: Boundary::Final,
        };
        self.char_offset += self.buf.len() as u32;
        self.next_idx += 1;
        self.buf.clear();
        self.chars_since_break = 0;
        Some(chunk)
    }

    /// Look at the tail of `buf`. Returns the boundary type if we
    /// should cut, else None.
    fn detect_boundary(&self) -> Option<Boundary> {
        let buf = self.buf.as_bytes();
        let n = buf.len();
        if n < 2 {
            return None;
        }
        // Paragraph: \n\n
        if n >= 2 && buf[n - 1] == b'\n' && buf[n - 2] == b'\n' {
            return Some(Boundary::Paragraph);
        }
        // Sentence: [.?!] followed by whitespace (current char is the
        // whitespace, we're looking at the char before).
        let last = buf[n - 1];
        let prev = buf[n - 2];
        if (last == b' ' || last == b'\n' || last == b'\t')
            && (prev == b'.' || prev == b'?' || prev == b'!')
        {
            return Some(Boundary::Sentence);
        }
        // Clause: same shape with , ; : — only emit if we have enough
        // material in the buffer.
        if (last == b' ' || last == b'\n')
            && (prev == b',' || prev == b';' || prev == b':')
            && self.chars_since_break >= self.cfg.min_chunk_chars
        {
            return Some(Boundary::Clause);
        }
        None
    }

    /// Cut at the current buffer tail and return the chunk.
    /// `chars_since_break` and the buffer reset to 0.
    fn cut(&mut self, boundary: Boundary) -> Option<SentenceChunk> {
        let trimmed = self.buf.trim();
        if trimmed.is_empty() {
            self.buf.clear();
            self.chars_since_break = 0;
            return None;
        }
        let text: Arc<str> = trimmed.to_string().into();
        let chunk = SentenceChunk {
            idx: self.next_idx,
            char_offset: self.char_offset,
            text,
            is_final: false,
            boundary,
        };
        self.char_offset += self.buf.len() as u32;
        self.next_idx += 1;
        self.buf.clear();
        self.chars_since_break = 0;
        Some(chunk)
    }

    /// Reset to a clean slate. Called on barge-in to discard any
    /// in-flight buffer.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.char_offset = 0;
        self.next_idx = 0;
        self.chars_since_break = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(s: &mut Shredder, t: &str) -> Vec<SentenceChunk> {
        s.push(t)
    }

    #[test]
    fn empty_buffer_emits_nothing() {
        let mut s = Shredder::new(ShredderConfig::default());
        assert!(drain(&mut s, "Hello").is_empty());
    }

    #[test]
    fn sentence_period_emits_chunk() {
        let mut s = Shredder::new(ShredderConfig::default());
        assert!(drain(&mut s, "Hello there.").is_empty());
        let out = drain(&mut s, " ");
        assert_eq!(out.len(), 1);
        assert_eq!(&*out[0].text, "Hello there.");
        assert_eq!(out[0].idx, 0);
        assert_eq!(out[0].boundary, Boundary::Sentence);
    }

    #[test]
    fn two_sentences_two_chunks() {
        let mut s = Shredder::new(ShredderConfig::default());
        let mut all = Vec::new();
        for piece in ["Hi. ", "How are you?", " "] {
            all.extend(s.push(piece));
        }
        assert_eq!(all.len(), 2);
        assert_eq!(&*all[0].text, "Hi.");
        assert_eq!(&*all[1].text, "How are you?");
        // idx increments.
        assert_eq!(all[0].idx, 0);
        assert_eq!(all[1].idx, 1);
    }

    #[test]
    fn clause_only_after_min_chars() {
        let mut s = Shredder::new(ShredderConfig {
            min_chunk_chars: 24,
            max_chunk_chars: 180,
        });
        // "Hi, " is only 4 chars before the comma — too short. Should NOT cut.
        let out = s.push("Hi, ");
        assert!(out.is_empty(), "short clause should not cut");

        // Build up past 24 chars then a comma+space — SHOULD cut.
        let out2 = s.push("this is a longer prefix, ");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].boundary, Boundary::Clause);
    }

    #[test]
    fn flush_emits_remaining_and_marks_final() {
        let mut s = Shredder::new(ShredderConfig::default());
        s.push("Trailing without punctuation");
        let last = s.flush().expect("flush should emit");
        assert!(last.is_final);
        assert_eq!(&*last.text, "Trailing without punctuation");
        assert_eq!(last.boundary, Boundary::Final);
        assert!(s.flush().is_none(), "empty buf flush returns None");
    }

    #[test]
    fn char_offset_advances() {
        let mut s = Shredder::new(ShredderConfig::default());
        let a = s.push("First. ");
        let b = s.push("Second. ");
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].char_offset, 0);
        assert_eq!(b[0].char_offset, 7);
    }

    #[test]
    fn paragraph_break_emits() {
        let mut s = Shredder::new(ShredderConfig::default());
        let out = s.push("First paragraph\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].boundary, Boundary::Paragraph);
    }

    #[test]
    fn reset_clears_state() {
        let mut s = Shredder::new(ShredderConfig::default());
        s.push("In progress");
        s.reset();
        assert_eq!(s.buf.len(), 0);
        assert_eq!(s.next_idx, 0);
        assert_eq!(s.char_offset, 0);
    }

    #[test]
    fn arc_str_text_is_cheap_to_clone() {
        // Compile-time check: SentenceChunk: Clone via Arc.
        let mut s = Shredder::new(ShredderConfig::default());
        let mut out = s.push("Hello world. ");
        let original = out.remove(0);
        let cloned = original.clone();
        assert!(Arc::ptr_eq(&original.text, &cloned.text));
    }
}
