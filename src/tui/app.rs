//! Pure UI state and event handling (no terminal access).

use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tokio::sync::oneshot;

use super::UiInfo;
use super::input::InputBuffer;
use crate::config::{Backend, LlmSettings, default_tool_mode, normalize_effort};
use crate::types::{AgentEvent, Approval, MistlOp, ToolMode, UserCommand};

pub const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// Ticks (~100ms each) a footer hint stays visible.
const HINT_TICKS: u64 = 30;

pub const EFFORTS: [&str; 7] = [
    "default", "none", "minimal", "low", "medium", "high", "xhigh",
];

/// Slash commands offered as completions, in display order. Bare commands
/// come before their argument forms so an exact match is selected first.
const COMMANDS: [(&str, &str); 11] = [
    ("/help", "show keys and commands"),
    ("/clear", "clear the conversation"),
    ("/model", "choose a model"),
    ("/models", "choose a model"),
    ("/effort", "choose reasoning effort"),
    ("/mistl", "show mistl installation info"),
    ("/mistl install", "download and install mistl"),
    ("/mistl start", "start mistl"),
    (
        "/settings",
        "edit backend, API, model, effort and tool mode",
    ),
    ("/quit", "quit"),
    ("/exit", "quit"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    pub text: String,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerKind {
    Model,
    Effort,
}

pub struct Picker {
    pub kind: PickerKind,
    pub items: Vec<String>,
    pub filter: String,
    /// Index in the filtered list.
    pub selected: usize,
}

impl Picker {
    fn new(kind: PickerKind, items: Vec<String>, current: &str) -> Self {
        let selected = items.iter().position(|s| s == current).unwrap_or(0);
        Self {
            kind,
            items,
            filter: String::new(),
            selected,
        }
    }

    pub fn filtered(&self) -> Vec<&str> {
        let filter = self.filter.to_lowercase();
        self.items
            .iter()
            .filter(|s| s.to_lowercase().contains(&filter))
            .map(String::as_str)
            .collect()
    }

    fn choice(&self) -> Option<String> {
        self.filtered()
            .get(self.selected)
            .map(|s| (*s).to_string())
            .or_else(|| (!self.filter.trim().is_empty()).then(|| self.filter.trim().to_string()))
    }

    fn edit(&mut self, text: &str) {
        self.filter.push_str(text);
        self.selected = 0;
    }

    fn on_key(&mut self, key: KeyEvent) {
        let max = self.filtered().len().saturating_sub(1);
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(max),
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(10),
            KeyCode::PageDown => self.selected = (self.selected + 10).min(max),
            KeyCode::Backspace => {
                self.filter.pop();
                self.selected = 0;
            }
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.edit(&c.to_string())
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickerTarget {
    Model,
    Form,
}

pub struct SettingsForm {
    pub settings: LlmSettings,
    /// A blank field preserves the stored key; never prefill it.
    pub api_key: String,
    pub row: usize,
    pub picker: Option<Picker>,
    pub hint: Option<String>,
}

impl SettingsForm {
    fn new(settings: LlmSettings) -> Self {
        Self {
            settings,
            api_key: String::new(),
            row: 0,
            picker: None,
            hint: None,
        }
    }

    fn value(&self) -> LlmSettings {
        let mut settings = self.settings.clone();
        if !self.api_key.is_empty() {
            settings.api_key = Some(self.api_key.clone());
        }
        settings
    }

    fn text_mut(&mut self) -> Option<&mut String> {
        match self.row {
            1 if self.settings.backend == Backend::Custom => Some(&mut self.settings.base_url),
            2 => Some(&mut self.api_key),
            3 => Some(&mut self.settings.model),
            _ => None,
        }
    }

    fn edit(&mut self, text: &str) {
        if let Some(field) = self.text_mut() {
            field.extend(text.chars().filter(|c| !c.is_control()));
            self.hint = None;
        }
    }

    fn change(&mut self, backwards: bool) {
        match self.row {
            0 => {
                self.settings.backend = match self.settings.backend {
                    Backend::Mistl => Backend::Custom,
                    Backend::Custom => Backend::Mistl,
                };
                self.settings.tool_mode = default_tool_mode(self.settings.backend);
            }
            4 => {
                let current = self
                    .settings
                    .reasoning_effort
                    .as_deref()
                    .unwrap_or("default");
                let i = EFFORTS.iter().position(|s| *s == current).unwrap_or(0);
                let next = if backwards {
                    (i + EFFORTS.len() - 1) % EFFORTS.len()
                } else {
                    (i + 1) % EFFORTS.len()
                };
                self.settings.reasoning_effort = normalize_effort(Some(EFFORTS[next].into()));
            }
            5 => {
                self.settings.tool_mode = match self.settings.tool_mode {
                    ToolMode::Native => ToolMode::Prompt,
                    ToolMode::Prompt => ToolMode::Native,
                }
            }
            _ => {}
        }
    }
}

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
    pub backend: Backend,
    pub reasoning_effort: Option<String>,
    pub mistl_found: bool,
    pub settings: LlmSettings,
    pub models: Vec<String>,
    pub fetching: bool,
    pub picker: Option<Picker>,
    pub form: Option<SettingsForm>,
    pending_picker: Option<PickerTarget>,
    startup_models: bool,
    /// Info operations also emit TurnDone, but never own the busy turn.
    pending_mistl_info: usize,
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
    /// Selected row of the slash-command completion popup.
    pub suggest_sel: usize,
    /// Popup dismissed with Esc; cleared by the next edit.
    suggest_hidden: bool,
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
  /                command completions: Up/Down select, Tab complete,
                   Enter run (or complete a partial command), Esc hide
Commands:
  /help            this help
  /clear           clear conversation
  /model [id]      choose or switch model (/models also opens the picker)
  /effort [level]  choose or set reasoning effort (default omits the field)
  /mistl [action]  show installation info; install or start mistl
  /settings        edit backend, API, model, effort and tool mode
  /quit, /exit     quit
Modal keys:
  Type/Up/Down/PgUp/PgDn  filter and navigate a picker; Enter chooses
  Up/Down/Tab     move between settings rows; Enter/Left/Right change
  Ctrl+S          save settings; Esc cancels a modal";

impl App {
    pub fn new(info: &UiInfo) -> Self {
        let mut app = Self {
            backend: info.backend,
            reasoning_effort: info.reasoning_effort.clone(),
            mistl_found: info.mistl_found,
            settings: info.settings.clone(),
            models: Vec::new(),
            fetching: false,
            picker: None,
            form: None,
            pending_picker: None,
            startup_models: false,
            pending_mistl_info: 0,
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
            suggest_sel: 0,
            suggest_hidden: false,
            effects: Vec::new(),
        };
        app.entries.push(Entry::Info(
            "Welcome to mistan. Type a request such as \"AI network status?\" or \
             \"保存しているファイル一覧を見せて\". Type /help for keys and commands."
                .into(),
        ));
        if !info.mistl_found {
            app.entries.push(Entry::Info(match info.backend {
                Backend::Mistl => "mistl was not found. Run /mistl install to download it, or /settings to use an OpenAI-compatible API.",
                Backend::Custom => "mistl was not found: chat only. Run /mistl install to enable the mistl tools.",
            }.into()));
        }
        if info.backend == Backend::Custom {
            app.startup_models = true;
            app.request_models(None, None);
        }
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
            AgentEvent::ToolModeChanged(mode) => {
                self.tool_mode = mode;
                self.settings.tool_mode = mode;
            }
            AgentEvent::SettingsApplied(settings) => {
                self.backend = settings.backend;
                self.model = settings.model.clone();
                self.base_url = match settings.backend {
                    Backend::Mistl => "AI network (mistl)".into(),
                    Backend::Custom => settings.base_url.clone(),
                };
                self.tool_mode = settings.tool_mode;
                self.reasoning_effort = settings.reasoning_effort.clone();
                self.settings = settings;
                self.models.clear();
            }
            AgentEvent::MistlAvailable(found) => self.mistl_found = found,
            AgentEvent::Models { models, error } => {
                self.fetching = false;
                let target = self.pending_picker.take();
                let startup = std::mem::take(&mut self.startup_models);
                if let Some(error) = error {
                    let message = format!("could not list models: {error}");
                    if target == Some(PickerTarget::Form) {
                        if let Some(form) = &mut self.form {
                            form.hint = Some(message);
                        }
                    } else if startup {
                        self.entries.push(Entry::Info(message));
                    } else {
                        self.entries.push(Entry::Error(message));
                    }
                } else {
                    self.models = models;
                    match target {
                        Some(PickerTarget::Model) => {
                            self.picker = Some(Picker::new(
                                PickerKind::Model,
                                self.models.clone(),
                                &self.model,
                            ));
                        }
                        Some(PickerTarget::Form) => {
                            if let Some(form) = &mut self.form {
                                form.hint = None;
                                form.picker = Some(Picker::new(
                                    PickerKind::Model,
                                    self.models.clone(),
                                    &form.settings.model,
                                ));
                            }
                        }
                        None if startup => self.entries.push(Entry::Info(format!(
                            "{} models available (/model to choose)",
                            self.models.len()
                        ))),
                        None => {}
                    }
                }
            }
            AgentEvent::Error(m) => self.entries.push(Entry::Error(m)),
            AgentEvent::TurnDone => {
                if self.pending_mistl_info > 0 {
                    self.pending_mistl_info -= 1;
                } else {
                    self.close_assistant();
                    self.busy = false;
                }
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
        self.fetching = false;
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
        let text: String = s.chars().filter(|c| !c.is_control()).collect();
        if let Some(form) = &mut self.form {
            if let Some(picker) = &mut form.picker {
                picker.edit(&text);
            } else {
                form.edit(&text);
            }
        } else if let Some(picker) = &mut self.picker {
            picker.edit(&text);
        } else {
            self.input.insert_str(s);
            self.suggest_sel = 0;
            self.suggest_hidden = false;
        }
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

        if self.form.is_some() {
            self.on_form_key(key);
            return;
        }
        if self.picker.is_some() {
            self.on_picker_key(key);
            return;
        }

        let before = self.input.text().to_string();
        if !self.on_suggest_key(key) {
            self.on_input_key(key);
        }
        if self.input.text() != before {
            self.suggest_sel = 0;
            self.suggest_hidden = false;
        }
    }

    fn on_input_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let page = self.last_height.saturating_sub(1).max(1);

        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy || self.fetching {
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
                if self.busy || self.fetching {
                    self.pending_picker = None;
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

    pub fn modal_open(&self) -> bool {
        self.current_approval().is_some() || self.picker.is_some() || self.form.is_some()
    }

    /// Slash-command completions for the current input; empty when the popup
    /// should not be shown.
    pub fn suggestions(&self) -> Vec<Suggestion> {
        let text = self.input.text();
        if self.suggest_hidden
            || self.hist_idx.is_some()
            || self.input.is_multiline()
            || !text.starts_with('/')
        {
            return Vec::new();
        }
        let mut all: Vec<Suggestion> = Vec::new();
        for (cmd, desc) in COMMANDS {
            all.push(Suggestion {
                text: cmd.into(),
                desc,
            });
            match cmd {
                "/models" => all.extend(self.models.iter().map(|m| Suggestion {
                    text: format!("/model {m}"),
                    desc: "switch to this model",
                })),
                "/effort" => all.extend(EFFORTS.iter().map(|e| Suggestion {
                    text: format!("/effort {e}"),
                    desc: "set reasoning effort",
                })),
                _ => {}
            }
        }
        let typed = text.to_lowercase();
        let found: Vec<Suggestion> = all
            .into_iter()
            .filter(|s| s.text.to_lowercase().starts_with(&typed))
            .collect();
        // A lone exact match has nothing left to suggest.
        if found.len() == 1 && found[0].text == text {
            return Vec::new();
        }
        found
    }

    /// Handle a key aimed at the completion popup. Returns false when the key
    /// should fall through to the normal input handling.
    fn on_suggest_key(&mut self, key: KeyEvent) -> bool {
        if !key.modifiers.is_empty() && key.code != KeyCode::BackTab {
            return false;
        }
        let list = self.suggestions();
        if list.is_empty() {
            return false;
        }
        let n = list.len();
        let sel = self.suggest_sel.min(n - 1);
        match key.code {
            KeyCode::Up | KeyCode::BackTab => self.suggest_sel = (sel + n - 1) % n,
            KeyCode::Down => self.suggest_sel = (sel + 1) % n,
            KeyCode::Tab => {
                let chosen = &list[sel].text;
                // Completing an already complete command steps into its
                // arguments, e.g. "/effort" -> "/effort ".
                let next = if chosen == self.input.text() {
                    format!("{chosen} ")
                } else {
                    chosen.clone()
                };
                self.input.set(&next);
                self.suggest_sel = 0;
            }
            KeyCode::Enter => {
                if list[sel].text == self.input.text() {
                    return false;
                }
                let chosen = list[sel].text.clone();
                self.input.set(&chosen);
                self.suggest_sel = 0;
                // A command that takes no further input runs right away;
                // otherwise leave it in the editor so arguments can be added.
                if !list
                    .iter()
                    .any(|s| s.text.starts_with(&format!("{chosen} ")))
                {
                    self.submit();
                }
            }
            KeyCode::Esc if !(self.busy || self.fetching) => self.suggest_hidden = true,
            _ => return false,
        }
        true
    }

    fn request_models(&mut self, settings: Option<LlmSettings>, target: Option<PickerTarget>) {
        if self.fetching {
            self.set_hint(
                "Already fetching models. Wait for it to finish, or press Esc to cancel.",
            );
            return;
        }
        self.fetching = true;
        self.pending_picker = target;
        self.effects
            .push(Effect::Command(UserCommand::ListModels(settings)));
    }

    fn set_model(&mut self, model: String) {
        self.model = model.clone();
        self.settings.model = model.clone();
        self.entries
            .push(Entry::Info(format!("Model set to {model}.")));
        self.effects
            .push(Effect::Command(UserCommand::SetModel(model)));
    }

    fn set_effort(&mut self, effort: &str) {
        let effort = normalize_effort(Some(effort.into()));
        self.reasoning_effort = effort.clone();
        self.settings.reasoning_effort = effort.clone();
        self.effects
            .push(Effect::Command(UserCommand::SetEffort(effort)));
    }

    fn on_picker_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Esc {
            self.picker = None;
        } else if key.code == KeyCode::Enter {
            if self.busy {
                self.set_hint("Cannot change settings while a turn is running.");
                return;
            }
            let choice = self.picker.as_ref().and_then(Picker::choice);
            if let Some(choice) = choice {
                let picker = self.picker.take().unwrap();
                match picker.kind {
                    PickerKind::Model => self.set_model(choice),
                    PickerKind::Effort => self.set_effort(&choice),
                }
            }
        } else if let Some(picker) = &mut self.picker {
            picker.on_key(key);
        }
    }

    fn on_form_key(&mut self, key: KeyEvent) {
        let form = self.form.as_mut().unwrap();
        if let Some(picker) = &mut form.picker {
            match key.code {
                KeyCode::Esc => form.picker = None,
                KeyCode::Enter => {
                    if let Some(choice) = picker.choice() {
                        form.settings.model = choice;
                        form.picker = None;
                    }
                }
                _ => picker.on_key(key),
            }
            return;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => {
                self.form = None;
                if self.pending_picker == Some(PickerTarget::Form) {
                    self.pending_picker = None;
                    self.effects.push(Effect::CancelTurn);
                }
            }
            KeyCode::Char('s') if ctrl => {
                if self.busy {
                    form.hint = Some("Cannot save settings while a turn is running.".into());
                } else {
                    let settings = form.value();
                    if self.pending_picker == Some(PickerTarget::Form) {
                        self.pending_picker = None;
                        self.effects.push(Effect::CancelTurn);
                    }
                    self.effects
                        .push(Effect::Command(UserCommand::Configure(settings)));
                    self.form = None;
                }
            }
            KeyCode::Up | KeyCode::BackTab => form.row = (form.row + 5) % 6,
            KeyCode::Down | KeyCode::Tab => form.row = (form.row + 1) % 6,
            KeyCode::Enter if form.row == 3 => {
                if self.busy {
                    form.hint = Some("Cannot list models while a turn is running.".into());
                } else if self.fetching {
                    form.hint = Some("Already fetching models.".into());
                } else {
                    let settings = form.value();
                    form.hint = Some("fetching models…".into());
                    self.request_models(Some(settings), Some(PickerTarget::Form));
                }
            }
            KeyCode::Left => form.change(true),
            KeyCode::Right | KeyCode::Enter => form.change(false),
            KeyCode::Backspace => {
                if let Some(text) = form.text_mut() {
                    text.pop();
                }
            }
            KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                form.edit(&c.to_string())
            }
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
        if self.busy || self.fetching {
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
        if self.busy && matches!(name, "model" | "models" | "effort" | "mistl" | "settings") {
            self.set_hint(
                "Cannot change settings or manage mistl while a turn is running (Esc to cancel).",
            );
            return;
        }
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
            "model" if !arg.is_empty() => self.set_model(arg.into()),
            "model" | "models" => self.request_models(None, Some(PickerTarget::Model)),
            "effort" if !arg.is_empty() => self.set_effort(arg),
            "effort" => {
                self.pending_picker = None;
                self.picker = Some(Picker::new(
                    PickerKind::Effort,
                    EFFORTS.iter().map(|s| (*s).into()).collect(),
                    self.reasoning_effort.as_deref().unwrap_or("default"),
                ))
            }
            "mistl" => {
                let op = match arg {
                    "" => MistlOp::Info,
                    "install" => MistlOp::Install,
                    "start" => MistlOp::Start,
                    _ => {
                        self.set_hint("Usage: /mistl [install|start]");
                        return;
                    }
                };
                self.busy = op != MistlOp::Info;
                if op == MistlOp::Info {
                    self.pending_mistl_info += 1;
                }
                self.effects.push(Effect::Command(UserCommand::Mistl(op)));
            }
            "settings" => {
                self.pending_picker = None;
                self.form = Some(SettingsForm::new(self.settings.clone()));
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
        UiInfo::from_config(&crate::config::Config::default())
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

    fn command(app: &mut App, text: &str) {
        app.on_paste(text);
        press(app, KeyCode::Enter);
    }

    fn save(app: &mut App) {
        app.on_key(key(
            KeyCode::Char('s'),
            KeyModifiers::CONTROL,
            KeyEventKind::Press,
        ));
    }

    #[test]
    fn new_slash_commands_and_busy_guard() {
        let mut app = App::new(&info());
        command(&mut app, "/mistl install");
        assert!(app.busy);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::Mistl(MistlOp::Install))]
        ));
        for cmd in [
            "/mistl",
            "/mistl start",
            "/model",
            "/model id",
            "/models",
            "/effort",
            "/effort high",
            "/settings",
        ] {
            command(&mut app, cmd);
            assert!(app.take_effects().is_empty(), "{cmd}");
            assert!(!app.modal_open());
            assert!(!app.fetching);
        }
        app.on_agent_event(AgentEvent::TurnDone);
        command(&mut app, "/mistl start");
        assert!(app.busy);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::Mistl(MistlOp::Start))]
        ));
        app.on_agent_event(AgentEvent::TurnDone);
        command(&mut app, "/mistl");
        assert!(!app.busy);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::Mistl(MistlOp::Info))]
        ));
        command(&mut app, "/effort high");
        assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(app.settings.reasoning_effort.as_deref(), Some("high"));
        assert!(
            matches!(app.take_effects().as_slice(), [Effect::Command(UserCommand::SetEffort(Some(e)))] if e == "high")
        );
        command(&mut app, "/effort default");
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::SetEffort(None))]
        ));
        command(&mut app, "/model");
        assert!(app.fetching);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::ListModels(None))]
        ));
        press(&mut app, KeyCode::Esc);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::CancelTurn]
        ));
        app.on_agent_event(AgentEvent::Models {
            models: vec![],
            error: Some("cancelled".into()),
        });
        assert!(!app.fetching);
        assert!(app.picker.is_none());
        command(&mut app, "/models");
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::ListModels(None))]
        ));
    }

    #[test]
    fn models_open_requested_picker_and_custom_choice() {
        let mut app = App::new(&info());
        command(&mut app, "/model");
        app.take_effects();
        app.on_agent_event(AgentEvent::Models {
            models: vec!["Alpha".into(), "beta".into(), "ALPINE".into()],
            error: None,
        });
        assert!(!app.fetching);
        assert_eq!(app.models.len(), 3);
        app.on_paste("aLp");
        assert_eq!(app.picker.as_ref().unwrap().filtered(), ["Alpha", "ALPINE"]);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.model, "ALPINE");
        assert!(
            matches!(app.take_effects().as_slice(), [Effect::Command(UserCommand::SetModel(m))] if m == "ALPINE")
        );
        command(&mut app, "/model");
        app.take_effects();
        app.on_agent_event(AgentEvent::Models {
            models: app.models.clone(),
            error: None,
        });
        app.on_paste("custom-model");
        assert!(app.picker.as_ref().unwrap().filtered().is_empty());
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.model, "custom-model");
        assert!(app.picker.is_none());
        assert!(app.input.is_empty());
    }

    #[test]
    fn picker_navigation_effort_and_cancel_swallow_keys() {
        let mut picker = Picker::new(
            PickerKind::Model,
            (0..25).map(|i| format!("m{i}")).collect(),
            "m12",
        );
        picker.on_key(key(
            KeyCode::PageDown,
            KeyModifiers::NONE,
            KeyEventKind::Press,
        ));
        assert_eq!(picker.selected, 22);
        picker.on_key(key(
            KeyCode::PageDown,
            KeyModifiers::NONE,
            KeyEventKind::Press,
        ));
        assert_eq!(picker.selected, 24);
        picker.on_key(key(
            KeyCode::PageUp,
            KeyModifiers::NONE,
            KeyEventKind::Press,
        ));
        assert_eq!(picker.selected, 14);
        let mut app = App::new(&info());
        command(&mut app, "/effort");
        app.on_paste("HIGH");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
        app.take_effects();
        command(&mut app, "/effort");
        app.on_paste("default");
        press(&mut app, KeyCode::Enter);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::SetEffort(None))]
        ));
        command(&mut app, "/effort");
        app.on_key(key(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
            KeyEventKind::Press,
        ));
        assert!(!app.quit);
        press(&mut app, KeyCode::Esc);
        assert!(app.take_effects().is_empty());
        assert!(!app.modal_open());
    }

    #[test]
    fn settings_edit_and_save_expected_configuration() {
        let mut app = App::new(&info());
        app.settings.api_key = Some("stored-test-value".into());
        command(&mut app, "/settings");
        assert!(app.form.as_ref().unwrap().api_key.is_empty());
        press(&mut app, KeyCode::Enter);
        assert_eq!(
            app.form.as_ref().unwrap().settings.tool_mode,
            ToolMode::Prompt
        );
        press(&mut app, KeyCode::Tab);
        app.on_paste("https://example.invalid/v1X");
        press(&mut app, KeyCode::Backspace);
        press(&mut app, KeyCode::Down);
        app.on_paste("new-test-value");
        press(&mut app, KeyCode::Down);
        app.on_paste("model-id");
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Left); // default -> xhigh
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Right);
        save(&mut app);
        let expected = LlmSettings {
            backend: Backend::Custom,
            base_url: "https://example.invalid/v1".into(),
            api_key: Some("new-test-value".into()),
            model: "model-id".into(),
            reasoning_effort: Some("xhigh".into()),
            tool_mode: ToolMode::Native,
        };
        assert!(
            matches!(app.take_effects().as_slice(), [Effect::Command(UserCommand::Configure(s))] if s == &expected)
        );
        assert!(app.form.is_none());
        assert_eq!(app.backend, Backend::Mistl); // Wait for SettingsApplied.
        app.on_agent_event(AgentEvent::SettingsApplied(expected.clone()));
        assert_eq!(app.settings, expected);
        assert_eq!(app.base_url, "https://example.invalid/v1");
        command(&mut app, "/settings");
        save(&mut app);
        assert!(
            matches!(app.take_effects().as_slice(), [Effect::Command(UserCommand::Configure(s))] if s.api_key.as_deref() == Some("new-test-value"))
        );
        let mut network = app.settings.clone();
        network.backend = Backend::Mistl;
        app.on_agent_event(AgentEvent::SettingsApplied(network));
        assert_eq!(app.base_url, "AI network (mistl)");
        app.on_agent_event(AgentEvent::MistlAvailable(false));
        assert!(!app.mistl_found);
    }

    #[test]
    fn form_models_probe_fills_form_without_applying() {
        let mut app = App::new(&info());
        command(&mut app, "/settings");
        app.form.as_mut().unwrap().row = 1;
        app.on_paste("ignored-url");
        assert!(app.form.as_ref().unwrap().settings.base_url.is_empty());
        app.form.as_mut().unwrap().row = 3;
        press(&mut app, KeyCode::Enter);
        assert!(app.fetching);
        assert!(
            matches!(app.take_effects().as_slice(), [Effect::Command(UserCommand::ListModels(Some(s)))] if s == &app.settings)
        );
        app.on_agent_event(AgentEvent::Models {
            models: vec!["chosen".into()],
            error: None,
        });
        assert!(app.picker.is_none());
        assert!(app.form.as_ref().unwrap().picker.is_some());
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.form.as_ref().unwrap().settings.model, "chosen");
        assert!(app.model.is_empty());
        press(&mut app, KeyCode::Enter);
        app.take_effects();
        app.on_agent_event(AgentEvent::Models {
            models: vec![],
            error: Some("offline".into()),
        });
        assert!(
            app.form
                .as_ref()
                .unwrap()
                .hint
                .as_deref()
                .unwrap()
                .contains("offline")
        );
        press(&mut app, KeyCode::Esc);
        assert!(app.form.is_none());
        assert!(app.take_effects().is_empty());
    }

    #[test]
    fn approval_priority_and_form_cancel_during_fetch() {
        let mut app = App::new(&info());
        command(&mut app, "/settings");
        let (reply, mut receive) = oneshot::channel();
        app.on_agent_event(AgentEvent::ApprovalRequest {
            title: "install".into(),
            reason: "".into(),
            reply,
        });
        app.on_paste("discarded");
        press(&mut app, KeyCode::Enter);
        assert_eq!(receive.try_recv().unwrap(), Approval::Yes);
        assert_eq!(app.form.as_ref().unwrap().settings.backend, Backend::Mistl);
        app.busy = true;
        save(&mut app);
        assert!(app.form.is_some());
        assert!(app.take_effects().is_empty());
        app.busy = false;
        app.form.as_mut().unwrap().row = 3;
        press(&mut app, KeyCode::Enter);
        app.take_effects();
        press(&mut app, KeyCode::Esc);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::CancelTurn]
        ));
        app.on_agent_event(AgentEvent::Models {
            models: vec!["late-result".into()],
            error: None,
        });
        assert!(!app.modal_open());
        assert!(!app.fetching);
    }

    #[test]
    fn startup_fetch_and_missing_mistl_notices() {
        let mut info = info();
        info.mistl_found = false;
        let app = App::new(&info);
        assert!(app.entries.contains(&Entry::Info("mistl was not found. Run /mistl install to download it, or /settings to use an OpenAI-compatible API.".into())));
        info.backend = Backend::Custom;
        info.settings.backend = Backend::Custom;
        let mut app = App::new(&info);
        assert!(app.entries.contains(&Entry::Info(
            "mistl was not found: chat only. Run /mistl install to enable the mistl tools.".into()
        )));
        assert!(app.fetching);
        assert!(matches!(
            app.take_effects().as_slice(),
            [Effect::Command(UserCommand::ListModels(None))]
        ));
        app.on_agent_event(AgentEvent::Models {
            models: vec!["one".into(), "two".into()],
            error: None,
        });
        assert_eq!(app.models.len(), 2);
        assert!(app.picker.is_none());
        assert!(!app.fetching);
        assert!(
            app.entries
                .contains(&Entry::Info("2 models available (/model to choose)".into()))
        );
        let mut app = App::new(&info);
        app.on_agent_event(AgentEvent::Models {
            models: vec![],
            error: Some("offline".into()),
        });
        assert!(
            app.entries
                .contains(&Entry::Info("could not list models: offline".into()))
        );
    }

    #[test]
    fn mistl_info_completion_does_not_release_a_later_turn() {
        let mut app = App::new(&info());
        command(&mut app, "/mistl");
        command(&mut app, "/mistl");
        command(&mut app, "question");
        assert!(app.busy);
        app.on_agent_event(AgentEvent::TurnDone);
        assert!(app.busy);
        app.on_agent_event(AgentEvent::TurnDone);
        assert!(app.busy);
        app.on_agent_event(AgentEvent::TurnDone);
        assert!(!app.busy);
    }

    #[test]
    fn saving_during_form_fetch_does_not_open_a_late_picker() {
        let mut app = App::new(&info());
        command(&mut app, "/settings");
        app.form.as_mut().unwrap().row = 3;
        press(&mut app, KeyCode::Enter);
        app.take_effects();
        save(&mut app);
        assert!(app.form.is_none());
        assert!(
            matches!(app.take_effects().as_slice(), [Effect::CancelTurn, Effect::Command(UserCommand::Configure(s))] if s == &app.settings)
        );
        app.on_agent_event(AgentEvent::Models {
            models: vec!["late".into()],
            error: None,
        });
        assert!(!app.modal_open());
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

    fn texts(a: &App) -> Vec<String> {
        a.suggestions().into_iter().map(|s| s.text).collect()
    }

    #[test]
    fn suggestions_filter_and_hide() {
        let mut a = App::new(&info());
        assert!(a.suggestions().is_empty());
        press(&mut a, KeyCode::Char('/'));
        assert_eq!(texts(&a).len(), COMMANDS.len() + EFFORTS.len());
        a.on_paste("MI");
        assert_eq!(texts(&a), ["/mistl", "/mistl install", "/mistl start"]);
        a.input.set("/clear");
        assert!(a.suggestions().is_empty(), "lone exact match");
        a.input.set("hello /");
        assert!(a.suggestions().is_empty());
        a.models = vec!["Alpha".into(), "beta".into()];
        a.input.set("/model a");
        assert_eq!(texts(&a), ["/model Alpha"]);
        a.input.set("/ef");
        press(&mut a, KeyCode::Esc);
        assert!(a.suggestions().is_empty());
        press(&mut a, KeyCode::Char('f'));
        assert!(!a.suggestions().is_empty(), "editing shows it again");
    }

    #[test]
    fn suggestions_select_complete_and_run() {
        let mut a = App::new(&info());
        a.on_paste("/mi");
        press(&mut a, KeyCode::Down);
        press(&mut a, KeyCode::Down);
        assert_eq!(a.suggest_sel, 2);
        press(&mut a, KeyCode::Up);
        press(&mut a, KeyCode::Tab);
        assert_eq!(a.input.text(), "/mistl install");
        assert!(a.take_effects().is_empty());
        // Enter completes a partial command and runs it when it takes no args.
        a.input.set("/q");
        press(&mut a, KeyCode::Enter);
        assert!(a.quit);

        let mut a = App::new(&info());
        a.on_paste("/ef");
        // "/effort" takes arguments: Enter completes but does not run.
        press(&mut a, KeyCode::Enter);
        assert_eq!(a.input.text(), "/effort");
        assert!(a.picker.is_none());
        // Tab on the complete command steps into its arguments.
        press(&mut a, KeyCode::Tab);
        assert_eq!(a.input.text(), "/effort ");
        press(&mut a, KeyCode::Up);
        press(&mut a, KeyCode::Enter);
        assert!(
            matches!(a.take_effects().as_slice(), [Effect::Command(UserCommand::SetEffort(Some(e)))] if e == "xhigh")
        );
        assert!(a.input.is_empty());

        // Exact "/effort" with more candidates still runs on Enter.
        a.on_paste("/effort");
        press(&mut a, KeyCode::Enter);
        assert!(a.picker.is_some());
    }

    #[test]
    fn suggestions_leave_history_and_scroll_alone() {
        let mut a = App::new(&info());
        a.on_paste("/help");
        press(&mut a, KeyCode::Enter);
        press(&mut a, KeyCode::Up);
        assert_eq!(a.input.text(), "/help");
        assert!(a.suggestions().is_empty(), "browsing history");
        press(&mut a, KeyCode::Down);
        a.on_paste("/");
        a.on_key(key(KeyCode::Up, KeyModifiers::SHIFT, KeyEventKind::Press));
        assert_eq!(a.suggest_sel, 0);
    }
}
