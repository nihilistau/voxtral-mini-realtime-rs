//! LLM adapter — local GGUF inference via candle (pure Rust).
//!
//! A dedicated OS thread owns the `ModelWeights` + tokenizer. The
//! orchestrator submits prompts via `std::sync::mpsc<String>`; the LLM
//! streams UTF-8 token pieces back via `tokio::sync::mpsc<LlmEvent>`.
//!
//! Cancellation: a new prompt arriving while we're decoding aborts the
//! current generation and starts over with the new one.
//!
//! Gated behind the `llm` cargo feature. Without it, the orchestrator
//! falls back to echo mode (transcript → TTS direct).
//!
//! Target model: Qwen2.5-0.5B-Instruct Q4_K_M GGUF (also works for any
//! Qwen2 family model). To use Gemma 3 or other architectures, swap the
//! `quantized_qwen2::ModelWeights` import for the architecture-specific
//! module and update the chat template / EOS token.

use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc as smpsc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;
use tokio::sync::mpsc as tmpsc;
use tracing::{debug, info, warn};

/// Configuration for the LLM thread.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub gguf_path: PathBuf,
    /// Local tokenizer.json path. If missing, auto-downloaded from
    /// `hf_repo`/tokenizer.json and cached here.
    pub tokenizer_path: PathBuf,
    /// HuggingFace repo for the tokenizer fallback. Default: Qwen2.5-0.5B-Instruct.
    pub hf_repo: String,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
    pub system_prompt: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::from(
                "D:/Files/Models/lmstudio-community/Qwen 2.5 coder 0.5b-1b-3b-14b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
            ),
            tokenizer_path: PathBuf::from("models/qwen2.5-0.5b-tokenizer.json"),
            hf_repo: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            max_new_tokens: 200,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            system_prompt:
                "You are a concise spoken assistant. Reply in one short sentence."
                    .to_string(),
        }
    }
}

/// Ensure the tokenizer is on disk. If `cfg.tokenizer_path` doesn't exist,
/// download it from `cfg.hf_repo`/tokenizer.json via hf-hub and copy to
/// the configured path so subsequent runs are offline.
fn ensure_tokenizer(cfg: &LlmConfig) -> Result<PathBuf> {
    if cfg.tokenizer_path.exists() {
        return Ok(cfg.tokenizer_path.clone());
    }
    info!(
        repo = %cfg.hf_repo,
        cache = %cfg.tokenizer_path.display(),
        "Tokenizer not local; downloading from HuggingFace"
    );
    let api = hf_hub::api::sync::Api::new().context("hf-hub api init")?;
    let repo = api.model(cfg.hf_repo.clone());
    let downloaded = repo
        .get("tokenizer.json")
        .with_context(|| format!("fetching {}/tokenizer.json", cfg.hf_repo))?;
    if let Some(parent) = cfg.tokenizer_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::copy(&downloaded, &cfg.tokenizer_path)
        .with_context(|| format!("cache tokenizer to {}", cfg.tokenizer_path.display()))?;
    Ok(cfg.tokenizer_path.clone())
}

/// Streaming events emitted by the LLM thread.
#[derive(Debug, Clone)]
pub enum LlmEvent {
    /// One decoded UTF-8 piece (partial word, full word, or punctuation).
    Token(String),
    /// Generation finished.
    Done {
        reason: DoneReason,
        n_tokens: u32,
        total_ms: u64,
        ttft_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoneReason {
    Eos,
    MaxTokens,
    Cancelled,
    Error,
}

/// Handle returned by [`spawn`]. Drop to terminate the LLM thread.
pub struct LlmHandle {
    pub prompt_tx: smpsc::Sender<String>,
    pub _thread: std::thread::JoinHandle<()>,
}

/// Spawn the LLM engine on a dedicated thread. Loads the model lazily on
/// first prompt; an idle handle holds no GPU/CPU resources.
pub fn spawn(cfg: LlmConfig, event_tx: tmpsc::UnboundedSender<LlmEvent>) -> Result<LlmHandle> {
    let (prompt_tx, prompt_rx) = smpsc::channel::<String>();
    let thread = std::thread::Builder::new()
        .name("voxtral-llm".into())
        .spawn(move || {
            if let Err(e) = run(cfg, prompt_rx, &event_tx) {
                warn!(?e, "LLM thread exited with error");
                let _ = event_tx.send(LlmEvent::Done {
                    reason: DoneReason::Error,
                    n_tokens: 0,
                    total_ms: 0,
                    ttft_ms: 0,
                });
            }
        })
        .context("spawning LLM thread")?;
    Ok(LlmHandle {
        prompt_tx,
        _thread: thread,
    })
}

fn run(
    cfg: LlmConfig,
    prompt_rx: smpsc::Receiver<String>,
    event_tx: &tmpsc::UnboundedSender<LlmEvent>,
) -> Result<()> {
    let device = Device::Cpu;

    info!(path = %cfg.gguf_path.display(), "Loading LLM GGUF");
    let mut file = File::open(&cfg.gguf_path)
        .with_context(|| format!("opening GGUF at {}", cfg.gguf_path.display()))?;
    let content = gguf_file::Content::read(&mut file).context("parsing GGUF metadata")?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)
        .context("constructing Qwen2 ModelWeights")?;
    drop(file);

    let tokenizer_path = ensure_tokenizer(&cfg)?;
    info!(path = %tokenizer_path.display(), "Loading tokenizer");
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("tokenizer load: {e}"))?;
    let mut tos = TokenOutputStream::new(tokenizer);

    let eos_id = tos
        .get_token("<|im_end|>")
        .or_else(|| tos.get_token("<|endoftext|>"))
        .ok_or_else(|| anyhow!("EOS token not found in tokenizer vocab"))?;
    info!(eos_id, "LLM ready; awaiting prompts");

    while let Ok(prompt) = prompt_rx.recv() {
        // Drain any backlog so we only run the most recent prompt.
        let mut latest = prompt;
        while let Ok(next) = prompt_rx.try_recv() {
            latest = next;
        }
        tos.clear();
        // Each turn starts with a fresh KV cache. Reloading from GGUF is
        // too expensive (~500 ms+); instead we reconstruct ModelWeights
        // only between turn boundaries when needed. For now, a single
        // long-lived model handles consecutive turns by re-prefilling the
        // full context each time. With max ~200 tokens that's <100 ms.
        if let Err(e) = generate(
            &mut model,
            &mut tos,
            &cfg,
            &device,
            eos_id,
            &latest,
            &prompt_rx,
            event_tx,
        ) {
            warn!(?e, "Generation error");
            let _ = event_tx.send(LlmEvent::Done {
                reason: DoneReason::Error,
                n_tokens: 0,
                total_ms: 0,
                ttft_ms: 0,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate(
    model: &mut ModelWeights,
    tos: &mut TokenOutputStream,
    cfg: &LlmConfig,
    device: &Device,
    eos_id: u32,
    user: &str,
    prompt_rx: &smpsc::Receiver<String>,
    event_tx: &tmpsc::UnboundedSender<LlmEvent>,
) -> Result<()> {
    let start = Instant::now();
    let mut ttft_ms = 0u64;

    let formatted = format_qwen_chat(&cfg.system_prompt, user);
    let enc = tos
        .tokenizer()
        .encode(formatted, true)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    debug!(prompt_tokens = prompt_ids.len(), "Prefilling");

    // Prefill the entire prompt as one forward pass at index 0.
    let input = Tensor::new(prompt_ids.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
    // Some versions return per-position logits; take the last row when 2D.
    let last_logits = if logits.dims().len() >= 2 {
        let last = logits.dim(0)? - 1;
        logits.get(last)?
    } else {
        logits
    };

    let sampling = if cfg.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: cfg.top_p,
            temperature: cfg.temperature,
        }
    };
    let mut sampler = LogitsProcessor::from_sampling(cfg.seed, sampling);

    let mut next_token = sampler.sample(&last_logits)?;
    let mut emitted = 0u32;
    let mut reason = DoneReason::MaxTokens;

    let emit = |s: String, ttft: &mut u64| {
        if *ttft == 0 {
            *ttft = start.elapsed().as_millis() as u64;
        }
        event_tx.send(LlmEvent::Token(s)).is_ok()
    };

    if next_token != eos_id {
        if let Some(piece) = tos.next_token(next_token)? {
            if !emit(piece, &mut ttft_ms) {
                reason = DoneReason::Cancelled;
            }
        }
        emitted += 1;
    } else {
        reason = DoneReason::Eos;
    }

    let mut idx_pos = prompt_ids.len();
    while emitted < cfg.max_new_tokens as u32 && reason == DoneReason::MaxTokens {
        if prompt_rx.try_recv().is_ok() {
            reason = DoneReason::Cancelled;
            break;
        }
        if next_token == eos_id {
            reason = DoneReason::Eos;
            break;
        }
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, idx_pos)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let last_logits = if logits.dims().len() >= 2 {
            let last = logits.dim(0)? - 1;
            logits.get(last)?
        } else {
            logits
        };
        next_token = sampler.sample(&last_logits)?;
        if next_token == eos_id {
            reason = DoneReason::Eos;
            break;
        }
        if let Some(piece) = tos.next_token(next_token)? {
            if !emit(piece, &mut ttft_ms) {
                reason = DoneReason::Cancelled;
                break;
            }
        }
        idx_pos += 1;
        emitted += 1;
    }
    if let Some(rest) = tos.decode_rest()? {
        let _ = emit(rest, &mut ttft_ms);
    }
    let total_ms = start.elapsed().as_millis() as u64;
    info!(
        ?reason,
        n_tokens = emitted,
        total_ms,
        ttft_ms,
        "LLM generation done"
    );
    let _ = event_tx.send(LlmEvent::Done {
        reason,
        n_tokens: emitted,
        total_ms,
        ttft_ms,
    });
    Ok(())
}

/// Qwen2.5-Instruct chat template:
/// `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`.
fn format_qwen_chat(system: &str, user: &str) -> String {
    let mut s = String::with_capacity(user.len() + system.len() + 96);
    if !system.is_empty() {
        s.push_str("<|im_start|>system\n");
        s.push_str(system);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(user);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    s
}

// ---------------------------------------------------------------------------
// TokenOutputStream — inlined from candle-examples (not a library crate).
// Buffers tokens until a UTF-8-safe boundary so streaming output is clean.
// ---------------------------------------------------------------------------

/// Streaming token decoder that handles split multi-byte codepoints.
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().is_some_and(|c| !c.is_alphanumeric()) {
            let new = text.split_at(prev_text.len()).1.to_string();
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(new))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            Ok(Some(text.split_at(prev_text.len()).1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn get_token(&self, s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(s).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_has_im_markers() {
        let s = format_qwen_chat("be concise", "hi");
        assert!(s.contains("<|im_start|>system"));
        assert!(s.contains("<|im_start|>user"));
        assert!(s.contains("<|im_start|>assistant"));
        assert!(s.contains("<|im_end|>"));
    }

    #[test]
    fn chat_template_skips_empty_system() {
        let s = format_qwen_chat("", "hello");
        assert!(!s.contains("<|im_start|>system"));
        assert!(s.contains("hello"));
    }
}
