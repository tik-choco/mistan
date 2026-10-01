//! Terminal UI.

mod app;
mod input;
mod text;
mod ui;

use std::io::{Stdout, stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste, Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::agent::AgentHandle;
use crate::config::{self, Backend, Config, LlmSettings};

/// Static facts shown in the header / status bar.
#[derive(Debug, Clone)]
pub struct UiInfo {
    pub backend: Backend,
    pub reasoning_effort: Option<String>,
    pub mistl_found: bool,
    pub settings: LlmSettings,
    pub model: String,
    pub base_url: String,
    pub tool_mode: crate::types::ToolMode,
    pub auto_approve: bool,
}

impl UiInfo {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            backend: cfg.backend,
            reasoning_effort: cfg.reasoning_effort.clone(),
            mistl_found: config::mistl_available(&cfg.mistl_bin),
            settings: LlmSettings::from_config(cfg),
            model: cfg.model.clone(),
            base_url: match cfg.backend {
                Backend::Mistl => "AI network (mistl)".into(),
                Backend::Custom => cfg.base_url.clone().unwrap_or_default(),
            },
            tool_mode: cfg.tool_mode,
            auto_approve: cfg.auto_approve,
        }
    }
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(stdout(), DisableBracketedPaste, LeaveAlternateScreen);
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    if let Err(e) = execute!(stdout(), EnterAlternateScreen, EnableBracketedPaste) {
        restore_terminal();
        return Err(e.into());
    }
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        prev(info);
    }));
    Ok(Terminal::new(CrosstermBackend::new(stdout()))?)
}

pub async fn run(info: UiInfo, handle: AgentHandle) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = event_loop(&mut terminal, info, handle).await;
    restore_terminal();
    let _ = terminal.show_cursor();
    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    info: UiInfo,
    mut handle: AgentHandle,
) -> Result<()> {
    let mut app = app::App::new(&info);
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    while !app.quit {
        for eff in app.take_effects() {
            match eff {
                app::Effect::Command(c) => {
                    let _ = handle.commands.send(c);
                }
                app::Effect::CancelTurn => handle.cancel_turn(),
            }
        }
        terminal.draw(|f| ui::draw(f, &mut app))?;
        tokio::select! {
            ev = events.next() => match ev {
                Some(Ok(Event::Key(k))) => app.on_key(k),
                Some(Ok(Event::Paste(s))) => app.on_paste(&s),
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e.into()),
                None => break,
            },
            ev = handle.events.recv(), if !app.events_closed => match ev {
                Some(ev) => app.on_agent_event(ev),
                None => app.on_events_closed(),
            },
            _ = tick.tick() => app.on_tick(),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_backend_label() {
        let cfg = Config::default();
        let info = UiInfo::from_config(&cfg);
        assert_eq!(info.base_url, "AI network (mistl)");
        assert_eq!(info.backend, Backend::Mistl);
        assert_eq!(info.settings, LlmSettings::from_config(&cfg));
        assert_eq!(info.reasoning_effort, cfg.reasoning_effort);
        assert_eq!(info.mistl_found, config::mistl_available(&cfg.mistl_bin));
        let cfg = Config {
            backend: Backend::Custom,
            base_url: Some("https://example.invalid/v1".into()),
            reasoning_effort: Some("high".into()),
            ..cfg
        };
        assert_eq!(
            UiInfo::from_config(&cfg).base_url,
            "https://example.invalid/v1"
        );
        let info = UiInfo::from_config(&cfg);
        assert_eq!(info.backend, Backend::Custom);
        assert_eq!(info.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(info.settings, LlmSettings::from_config(&cfg));
    }
}
