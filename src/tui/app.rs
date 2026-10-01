//! Pure UI state and event handling (no terminal access).

use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tokio::sync::oneshot;

use super::UiInfo;
use super::input::InputBuffer;
use crate::types::{AgentEvent, Approval, ToolMode, UserCommand};

pub const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// Ticks (~100ms each) a footer hint stays visible.
const HINT_TICKS: u64 = 30;

#[derive(Debug, Clone, PartialEq)]
pub enum ToolStatus {
    Running,
    Done { ok: bool, output: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    User(String),
    Assistant {
        text: String,
        done: bool,
    },
    Tool {
        id: String,
        title: String,
        status: ToolStatus,
    },
    Info(String),
    Error(String),
}

pub struct PendingApproval {
    pub title: String,
    pub reason: String,
    reply: Option<oneshot::Sender<Approval>>,
}

/// Side effects the event loop must perform after handling an input.
#[derive(Debug)]
pub enum Effect {
    Command(UserCommand),
    CancelTurn,
}

pub struct App {
    pub model: String,
    pub base_url: String,
    pub tool_mode: ToolMode,
    pub auto_approve: bool,
    pub entries: Vec<Entry>,
    pub input: InputBuffer,
    pub busy: bool,
    pub expand_tools: bool,
    /// `None` = follow the bottom; `Some(top_line)` = pinned view.
    pub scroll_top: Option<usize>,
    pub last_total: usize,
    pub last_height: usize,
    pub approvals: VecDeque<PendingApproval>,
    pub tick: u64,
    pub quit: bool,
    pub events_closed: bool,
    pub hint: Option<(String, u64)>,
    history: Vec<String>,
    hist_idx: Option<usize>,
    draft: String,
    effects: Vec<Effect>,
}

pub const HELP: &str = "\
Keys:
  Enter            send message
  Alt+Enter/Ctrl+J newline
  Esc              cancel the running turn
  PgUp/PgDn        scroll by page
  Ctrl+Up/Down     scroll by line (Shift+Up/Down too)
  End              jump to bottom (when input is empty)
  Ctrl+O           expand/collapse tool output
  Up/Down          input history (single-line input)
  Ctrl+U           clear input
  Ctrl+C           cancel turn / quit
  Ctrl+D           quit (empty input)
Commands:
  /help            this help
  /clear           clear conversation
  /model <id>      switch model
  /quit, /exit     quit";

impl App {
    pub fn new(info: &UiInfo) -> Self {
        let mut app = Self {
            model: info.model.clone(),
            base_url: info.base_url.clone(),
            tool_mode: info.tool_mode,
            auto_approve: info.auto_approve,
            entries: Vec::new(),
            input: InputBuffer::default(),
            busy: false,
            expand_tools: false,
            scroll_top: None,
            last_total: 0,
            last_height: 0,
            approvals: VecDeque::new(),
            tick: 0,
            quit: false,
            events_closed: false,
            hint: None,
            history: Vec::new(),
            hist_idx: None,
            draft: String::new(),
            effects: Vec::new(),
        };
        app.entries.push(Entry::Info(
            "Welcome to mistan. Type a request such as \"AI network status?\" or \
             \"保存しているファイル一覧を見せて\". Type /help for keys and commands."
                .into(),
        ));
        app
    }

    pub fn take_effects(&mut self) -> Vec<Effect> {
        std::mem::take(&mut self.effects)
    }

    pub fn spinner(&self) -> char {
        SPINNER[(self.tick as usize) % SPINNER.len()]
    }

    pub fn on_tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        if let Some((_, t)) = &self.hint
            && self.tick.wrapping_sub(*t) > HINT_TICKS
        {
            self.hint = None;
        }
    }

    fn set_hint(&mut self, s: impl Into<String>) {
        self.hint = Some((s.into(), self.tick));
    }

    /// Label for the footer while busy.
    pub fn busy_label(&self) -> String {
        let running = self.entries.iter().rev().find_map(|e| match e {
            Entry::Tool {
                title,
                status: ToolStatus::Running,
                ..
            } => Some(title.clone()),
            _ => None,
        });
        match running {
            Some(t) => format!("running {t}"),
            None => "thinking…".into(),
        }
    }

    // ---------------------------------------------------------------- agent

    pub fn on_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::TextDelta(t) => match self.entries.last_mut() {
                Some(Entry::Assistant { text, done: false }) => text.push_str(&t),
                _ => self.entries.push(Entry::Assistant {
                    text: t,
                    done: false,
                }),
            },
            AgentEvent::AssistantDone => self.close_assistant(),
            AgentEvent::ToolStart { id, title } => {
                self.close_assistant();
                self.entries.push(Entry::Tool {
                    id,
                    title,
                    status: ToolStatus::Running,
                });
            }
            AgentEvent::ToolEnd { id, ok, output } => {
                let found = self.entries.iter_mut().rev().find_map(|e| match e {
                    Entry::Tool { id: i, status, .. } if *i == id => Some(status),
                    _ => None,
                });
                match found {
                    Some(status) => *status = ToolStatus::Done { ok, output },
                    None => self.entries.push(Entry::Tool {
                        title: id.clone(),
                        id,
                        status: ToolStatus::Done { ok, output },
                    }),
                }
            }
            AgentEvent::ApprovalRequest {
                title,
                reason,
                reply,
            } => {
                self.approvals.push_back(PendingApproval {
                    title,
                    reason,
                    reply: Some(reply),
                });
            }
            AgentEvent::Info(m) => self.entries.push(Entry::Info(m)),
            AgentEvent::ToolModeChanged(mode) => self.tool_mode = mode,
            AgentEvent::Error(m) => self.entries.push(Entry::Error(m)),
            AgentEvent::TurnDone => {
                self.close_assistant();
                self.busy = false;
            }
        }
    }

    fn close_assistant(&mut self) {
        if let Some(Entry::Assistant { done, .. }) = self.entries.last_mut() {
            *done = true;
        }
    }

    pub fn on_events_closed(&mut self) {
        self.events_closed = true;
        self.busy = false;
        self.entries.push(Entry::Error(
            "Agent stopped unexpectedly. Press any key to quit.".into(),
        ));
    }

    // ------------------------------------------------------------ approvals

    pub fn current_approval(&self) -> Option<&PendingApproval> {
        self.approvals.front()
    }

    /// Answer the front approval request exactly once.
    pub fn answer_approval(&mut self, a: Approval) {
        let Some(mut p) = self.approvals.pop_front() else {
            return;
        };
        if let Some(tx) = p.reply.take() {
            let _ = tx.send(a);
        }
        if a == Approval::Always {
            self.auto_approve = true;
        }
    }

    // ---------------------------------------------------------------- input

    pub fn on_paste(&mut self, s: &str) {
        if !self.approvals.is_empty() || self.events_closed {
            return;
        }
        self.input.insert_str(s);
    }

    pub fn scroll_up(&mut self, n: usize) {
        let max = self.last_total.saturating_sub(self.last_height);
        let cur = self.scroll_top.unwrap_or(max).min(max);
        self.scroll_top = Some(cur.saturating_sub(n));
    }

    pub fn scroll_down(&mut self, n: usize) {
        let max = self.last_total.saturating_sub(self.last_height);
        if let Some(t) = self.scroll_top {
            let nt = t + n;
            self.scroll_top = if nt >= max { None } else { Some(nt) };
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if self.events_closed {
            self.quit = true;
            return;
        }
        if !self.approvals.is_empty() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    self.answer_approval(Approval::Yes)
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.answer_approval(Approval::No)
                }
                KeyCode::Char('a') | KeyCode::Char('A') => self.answer_approval(Approval::Always),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.answer_approval(Approval::No)
                }
                _ => {}
            }
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let page = self.last_height.saturating_sub(1).max(1);

        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy {
                    self.effects.push(Effect::CancelTurn);
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Char('d') if ctrl => {
                if self.input.is_empty() {
                    self.quit = true;
                }
            }
            KeyCode::Char('o') if ctrl => self.expand_tools = !self.expand_tools,
            KeyCode::Char('u') if ctrl => self.input.clear(),
            KeyCode::Char('j') if ctrl => self.input.insert_char('\n'),
            KeyCode::Enter if alt || shift || ctrl => self.input.insert_char('\n'),
            KeyCode::Enter => self.submit(),
            KeyCode::Esc => {
                if self.busy {
                    self.effects.push(Effect::CancelTurn);
                }
            }
            KeyCode::PageUp => self.scroll_up(page),
            KeyCode::PageDown => self.scroll_down(page),
            KeyCode::Up if ctrl || shift => self.scroll_up(1),
            KeyCode::Down if ctrl || shift => self.scroll_down(1),
            KeyCode::Up => {
                if self.input.is_multiline() {
                    self.input.up();
                } else {
                    self.history_prev();
                }
            }
            KeyCode::Down => {
                if self.input.is_multiline() {
                    self.input.down();
                } else {
                    self.history_next();
                }
            }
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => {
                if self.input.is_empty() {
                    self.scroll_top = None;
                } else {
                    self.input.end();
                }
            }
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Tab => self.input.insert_str("  "),
            KeyCode::Char(c) if !ctrl && !alt => self.input.insert_char(c),
            _ => {}
        }
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            None => {
                self.draft = self.input.text().to_string();
                self.history.len() - 1
            }
            Some(i) => i.saturating_sub(1),
        };
        self.hist_idx = Some(idx);
        let s = self.history[idx].clone();
        self.input.set(&s);
    }

    fn history_next(&mut self) {
        let Some(i) = self.hist_idx else {
            return;
        };
        if i + 1 >= self.history.len() {
            self.hist_idx = None;
            let d = std::mem::take(&mut self.draft);
            self.input.set(&d);
        } else {
            self.hist_idx = Some(i + 1);
            let s = self.history[i + 1].clone();
            self.input.set(&s);
        }
    }

    fn submit(&mut self) {
        let text = self.input.text().trim().to_string();
        if text.is_empty() {
            return;
        }
        if let Some(cmd) = text.strip_prefix('/') {
            self.input.clear();
            self.remember(&text);
            self.slash(cmd.trim());
            return;
        }
        if self.busy {
            self.set_hint("A turn is running. Wait for it to finish, or press Esc to cancel.");
            return;
        }
        self.input.clear();
        self.remember(&text);
        self.entries.push(Entry::User(text.clone()));
        self.busy = true;
        self.scroll_top = None;
        self.effects.push(Effect::Command(UserCommand::Send(text)));
    }

    fn remember(&mut self, text: &str) {
        if self.history.last().map(|s| s.as_str()) != Some(text) {
            self.history.push(text.to_string());
        }
        self.hist_idx = None;
        self.draft.clear();
    }

    fn slash(&mut self, cmd: &str) {
        let (name, arg) = match cmd.split_once(char::is_whitespace) {
            Some((n, a)) => (n, a.trim()),
            None => (cmd, ""),
        };
        match name {
            "help" | "?" => self.entries.push(Entry::Info(HELP.into())),
            "clear" => {
                if self.busy {
                    self.set_hint("Cannot /clear while a turn is running (Esc to cancel).");
                    return;
                }
                self.entries.clear();
                self.scroll_top = None;
                self.effects.push(Effect::Command(UserCommand::Clear));
            }
            "model" => {
                if arg.is_empty() {
                    let m = if self.model.is_empty() {
                        "(server default)"
                    } else {
                        &self.model
                    };
                    self.entries.push(Entry::Info(format!(
                        "Current model: {m}. Usage: /model <id>"
                    )));
                } else if self.busy {
                    self.set_hint("Cannot switch model while a turn is running.");
                } else {
                    self.model = arg.to_string();
                    self.effects
                        .push(Effect::Command(UserCommand::SetModel(arg.to_string())));
                    self.entries
                        .push(Entry::Info(format!("Model set to {arg}.")));
                }
            }
            "quit" | "exit" => self.quit = true,
            _ => self
                .entries
                .push(Entry::Info(format!("Unknown command: /{name}. Try /help."))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventState;

    fn info() -> UiInfo {
        UiInfo {
            model: String::new(),
            base_url: "http://x/v1".into(),
            tool_mode: ToolMode::Native,
            auto_approve: false,
        }
    }

    fn key(code: KeyCode, mods: KeyModifiers, kind: KeyEventKind) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: mods,
            kind,
            state: KeyEventState::NONE,
        }
    }
    fn press(app: &mut App, code: KeyCode) {
        app.on_key(key(code, KeyModifiers::NONE, KeyEventKind::Press));
    }

    #[test]
    fn runtime_tool_mode_updates_header_state() {
        let mut app = App::new(&info());
        app.on_agent_event(AgentEvent::ToolModeChanged(ToolMode::Prompt));
        assert_eq!(app.tool_mode, ToolMode::Prompt);
    }

    #[test]
    fn streaming_deltas_accumulate() {
        let mut a = App::new(&info());
        a.on_agent_event(AgentEvent::TextDelta("こん".into()));
        a.on_agent_event(AgentEvent::TextDelta("にちは".into()));
        a.on_agent_event(AgentEvent::AssistantDone);
        a.on_agent_event(AgentEvent::TextDelta("next".into()));
        let asst: Vec<_> = a
            .entries
            .iter()
            .filter(|e| matches!(e, Entry::Assistant { .. }))
            .collect();
        assert_eq!(asst.len(), 2);
        assert_eq!(
            asst[0],
            &Entry::Assistant {
                text: "こんにちは".into(),
                done: true
            }
        );
        assert_eq!(
            asst[1],
            &Entry::Assistant {
                text: "next".into(),
                done: false
            }
        );
    }

    #[test]
    fn tool_start_end_pair_by_id() {
        let mut a = App::new(&info());
        a.on_agent_event(AgentEvent::ToolStart {
            id: "1".into(),
            title: "mistl a".into(),
        });
        a.on_agent_event(AgentEvent::ToolStart {
            id: "2".into(),
            title: "mistl b".into(),
        });
        a.on_agent_event(AgentEvent::ToolEnd {
            id: "1".into(),
            ok: true,
            output: "x".into(),
        });
        let statuses: Vec<_> = a
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::Tool { id, status, .. } => Some((id.clone(), status.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            statuses[0],
            (
                "1".into(),
                ToolStatus::Done {
                    ok: true,
                    output: "x".into()
                }
            )
        );
        assert_eq!(statuses[1], ("2".into(), ToolStatus::Running));
        assert_eq!(a.busy_label(), "running mistl b");
    }

    #[test]
    fn approval_answers_exactly_once_and_queues() {
        let mut a = App::new(&info());
        let (t1, mut r1) = oneshot::channel();
        let (t2, mut r2) = oneshot::channel();
        a.on_agent_event(AgentEvent::ApprovalRequest {
            title: "one".into(),
            reason: "".into(),
            reply: t1,
        });
        a.on_agent_event(AgentEvent::ApprovalRequest {
            title: "two".into(),
            reason: "".into(),
            reply: t2,
        });
        // keys other than the modal ones are swallowed
        press(&mut a, KeyCode::Char('x'));
        assert!(a.input.is_empty());
        press(&mut a, KeyCode::Char('y'));
        assert_eq!(r1.try_recv().unwrap(), Approval::Yes);
        assert!(r2.try_recv().is_err());
        assert_eq!(a.current_approval().unwrap().title, "two");
        assert!(!a.auto_approve);
        press(&mut a, KeyCode::Char('a'));
        assert_eq!(r2.try_recv().unwrap(), Approval::Always);
        assert!(a.auto_approve);
        // nothing pending: extra keys do not panic or re-answer
        press(&mut a, KeyCode::Char('y'));
        assert_eq!(a.input.text(), "y");
    }

    #[test]
    fn send_busy_and_release_ignored() {
        let mut a = App::new(&info());
        a.on_key(key(
            KeyCode::Char('h'),
            KeyModifiers::NONE,
            KeyEventKind::Release,
        ));
        assert!(a.input.is_empty());
        a.on_paste("日本語\nline2");
        press(&mut a, KeyCode::Enter);
        assert!(a.busy);
        assert!(matches!(
            a.take_effects()[0],
            Effect::Command(UserCommand::Send(_))
        ));
        a.on_paste("again");
        press(&mut a, KeyCode::Enter);
        assert!(a.take_effects().is_empty());
        assert!(a.hint.is_some());
        a.on_agent_event(AgentEvent::TurnDone);
        assert!(!a.busy);
    }

    #[test]
    fn slash_commands() {
        let mut a = App::new(&info());
        a.on_paste("/model foo");
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.model, "foo");
        assert!(matches!(
            a.take_effects()[0],
            Effect::Command(UserCommand::SetModel(_))
        ));
        a.on_paste("/clear");
        press(&mut a, KeyCode::Enter);
        assert!(a.entries.is_empty());
        a.on_paste("/nope");
        press(&mut a, KeyCode::Enter);
        assert!(matches!(a.entries[0], Entry::Info(_)));
        a.on_paste("/quit");
        press(&mut a, KeyCode::Enter);
        assert!(a.quit);
    }
}
