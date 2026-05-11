//! Sesame-AI-style "call" TUI for the assistant.
//!
//! Layout:
//!
//! ```text
//! ┌── state pill + metrics ──────────────────────────────────────────────┐
//! │  [LISTENING]  00:23  •  H(X)=4.32 b  flatness=0.41  RMS=0.062        │
//! ├── mic waveform ──────────────────────┬── VHT2 power spectrum ────────┤
//! │           ▂▃▆█▆▃▂                    │   █▆▄▃▂▂▁▁▁▁▁▁▁▁▁▁           │
//! ├── transcript ─────────────────────────┴───────────────────────────────┤
//! │  you: hello there                                                    │
//! │  assistant: hello back                                               │
//! ├── q quit ────────────────────────────────────────────────────────────┤
//! ```
//!
//! Data flow:
//!
//! - The orchestrator broadcasts `VadFrame`s, transcripts, and session state.
//! - A small tokio task subscribes and updates a shared
//!   `Arc<Mutex<AssistantViewState>>` snapshot.
//! - This TUI runs on its own OS thread (ratatui + crossterm need a sync loop),
//!   reads the snapshot at ~30 fps, and renders.

use std::collections::VecDeque;
use std::io::{self, stdout};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::audio::RingBuffer;
use crate::tui::WaveformWidget;

/// One line in the transcript history pane.
#[derive(Debug, Clone)]
pub struct TranscriptLine {
    pub role: TranscriptRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptRole {
    User,
    Assistant,
    System,
}

/// Snapshot read by the TUI render loop. Producer-side updates lock briefly
/// to mutate; the TUI thread locks briefly to read.
#[derive(Debug, Clone)]
pub struct AssistantViewState {
    /// One of the labels from `SessionState`.
    pub state_label: &'static str,
    pub session_started: Instant,
    pub mic_buf: RingBuffer,
    /// Most recent VHT2 power spectrum (one frame).
    pub vht2_power: Vec<f32>,
    pub rms: f32,
    pub entropy: f32,
    pub flatness: f32,
    pub noise_floor: f32,
    /// Last N transcript lines (user + assistant).
    pub transcript: VecDeque<TranscriptLine>,
    /// TTFT for the last reply, in milliseconds.
    pub last_ttft_ms: Option<u64>,
    /// User-requested quit.
    pub should_quit: bool,
}

impl AssistantViewState {
    pub fn new(mic_window_secs: f32, mic_rate_hz: u32) -> Self {
        Self {
            state_label: "IDLE",
            session_started: Instant::now(),
            mic_buf: RingBuffer::from_duration_secs(mic_window_secs, mic_rate_hz),
            vht2_power: Vec::new(),
            rms: 0.0,
            entropy: 0.0,
            flatness: 0.0,
            noise_floor: 0.0,
            transcript: VecDeque::with_capacity(64),
            last_ttft_ms: None,
            should_quit: false,
        }
    }

    pub fn push_transcript(&mut self, role: TranscriptRole, text: String) {
        if self.transcript.len() >= 64 {
            self.transcript.pop_front();
        }
        self.transcript.push_back(TranscriptLine { role, text });
    }
}

pub type SharedAssistantViewState = Arc<Mutex<AssistantViewState>>;

/// Run the assistant TUI on the calling thread. Blocks until the user
/// presses q/Esc or `state.should_quit` is true.
pub fn run(state: SharedAssistantViewState) -> io::Result<()> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;

    let result = run_inner(&state);

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

fn run_inner(state: &SharedAssistantViewState) -> io::Result<()> {
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    loop {
        if let Ok(s) = state.lock() {
            if s.should_quit {
                break;
            }
        }
        terminal.draw(|frame| draw(frame, state))?;

        // ~30 fps
        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            if let Ok(mut s) = state.lock() {
                                s.should_quit = true;
                            }
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(())
}

fn draw(frame: &mut Frame, state: &SharedAssistantViewState) {
    let snapshot = match state.lock() {
        Ok(s) => s.clone(),
        Err(_) => return,
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Length(10), // mic + spectrum
            Constraint::Min(3),    // transcript
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

    draw_header(frame, rows[0], &snapshot);

    let split = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);
    draw_mic_wave(frame, split[0], &snapshot);
    draw_vht2_spectrum(frame, split[1], &snapshot);

    draw_transcript(frame, rows[2], &snapshot);
    draw_footer(frame, rows[3]);
}

fn draw_header(frame: &mut Frame, area: Rect, s: &AssistantViewState) {
    let (pill_bg, pill_fg) = match s.state_label {
        "LISTENING" => (Color::Rgb(60, 110, 180), Color::White),
        "THINKING" => (Color::Rgb(180, 140, 60), Color::Black),
        "SPEAKING" => (Color::Rgb(60, 160, 90), Color::Black),
        "INTERRUPTED" => (Color::Rgb(180, 60, 60), Color::White),
        _ => (Color::DarkGray, Color::White),
    };

    let elapsed = s.session_started.elapsed().as_secs();
    let mm = elapsed / 60;
    let ss = elapsed % 60;
    let ttft = match s.last_ttft_ms {
        Some(ms) => format!("TTFT={ms}ms"),
        None => "TTFT=—".to_string(),
    };

    let line = Line::from(vec![
        Span::raw(" "),
        Span::styled(
            format!(" {} ", s.state_label),
            Style::default().bg(pill_bg).fg(pill_fg).bold(),
        ),
        Span::raw(format!("  {mm:02}:{ss:02}  ")),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" H(X)={:.2} b ", s.entropy)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" flat={:.2} ", s.flatness)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" RMS={:.3} ", s.rms)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" nf={:.3} ", s.noise_floor)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!(" {ttft} ")),
    ]);
    let header = Paragraph::new(line).block(
        Block::default()
            .borders(Borders::BOTTOM)
            .style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(header, area);
}

fn draw_mic_wave(frame: &mut Frame, area: Rect, s: &AssistantViewState) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let peaks = s.mic_buf.snapshot_peaks(inner_width);
    let widget = WaveformWidget::new(&peaks)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" mic "),
        )
        .color(Color::Rgb(120, 180, 230));
    frame.render_widget(widget, area);
}

fn draw_vht2_spectrum(frame: &mut Frame, area: Rect, s: &AssistantViewState) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let peaks = if s.vht2_power.is_empty() {
        vec![0.0; inner_width]
    } else {
        // Normalize so the largest power becomes 1.0, then bucket into width.
        let mx = s.vht2_power.iter().cloned().fold(0.0f32, f32::max).max(1e-9);
        let n = s.vht2_power.len();
        (0..inner_width.max(1))
            .map(|i| {
                let start = i * n / inner_width.max(1);
                let end = ((i + 1) * n / inner_width.max(1)).min(n);
                if start >= end {
                    0.0
                } else {
                    let m = s.vht2_power[start..end]
                        .iter()
                        .cloned()
                        .fold(0.0f32, f32::max);
                    (m / mx).sqrt()
                }
            })
            .collect()
    };
    let widget = WaveformWidget::new(&peaks)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" VHT2 power spectrum "),
        )
        .color(Color::Rgb(207, 106, 76));
    frame.render_widget(widget, area);
}

fn draw_transcript(frame: &mut Frame, area: Rect, s: &AssistantViewState) {
    let mut lines: Vec<Line> = Vec::with_capacity(s.transcript.len());
    for entry in s.transcript.iter() {
        let (tag, tag_color) = match entry.role {
            TranscriptRole::User => ("you: ", Color::Rgb(120, 180, 230)),
            TranscriptRole::Assistant => ("assistant: ", Color::Rgb(60, 200, 130)),
            TranscriptRole::System => ("system: ", Color::DarkGray),
        };
        lines.push(Line::from(vec![
            Span::styled(tag, Style::default().fg(tag_color).bold()),
            Span::raw(entry.text.clone()),
        ]));
    }
    let para = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" transcript "),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(para, area);
}

fn draw_footer(frame: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" q", Style::default().fg(Color::Yellow)),
        Span::raw(" quit  •  "),
        Span::styled("Ctrl-C", Style::default().fg(Color::Yellow)),
        Span::raw(" force-shutdown"),
    ]))
    .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(footer, area);
}
