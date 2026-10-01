//! mistan: a natural-language TUI agent that operates the mistl daemon CLI.

mod agent;
mod config;
mod llm;
mod mistl;
mod prompt;
mod tui;
mod types;

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::types::{AgentEvent, Approval, ToolMode, UserCommand};

#[derive(Parser)]
#[command(
    name = "mistan",
    version,
    about = "Talk to your mistl node in natural language"
)]
struct Cli {
    /// Config file (default: <config dir>/mistan/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Use a custom OpenAI-compatible endpoint instead of the mistl AI network
    #[arg(long, env = "MISTAN_BASE_URL")]
    base_url: Option<String>,
    /// Model id
    #[arg(long, short, env = "MISTAN_MODEL")]
    model: Option<String>,
    /// Tool calling style: native (OpenAI tools) or prompt (fenced blocks)
    #[arg(long, value_parser = parse_tool_mode)]
    tool_mode: Option<ToolMode>,
    /// mistl executable (name on PATH or full path)
    #[arg(long, env = "MISTAN_MISTL")]
    mistl: Option<String>,
    /// mistl instance name (`mistl --instance`)
    #[arg(long)]
    instance: Option<String>,
    /// Run state-changing mistl commands without asking
    #[arg(long, short = 'y')]
    yes: bool,
    /// Non-interactive: answer one prompt on stdout and exit (no TUI).
    /// State-changing commands are refused unless --yes is given.
    #[arg(long, short = 'p')]
    prompt: Option<String>,
}

fn parse_tool_mode(s: &str) -> Result<ToolMode, String> {
    match s {
        "native" => Ok(ToolMode::Native),
        "prompt" => Ok(ToolMode::Prompt),
        _ => Err("expected `native` or `prompt`".into()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = config::load(
        cli.config.as_deref(),
        config::Overrides {
            base_url: cli.base_url,
            model: cli.model,
            tool_mode: cli.tool_mode,
            mistl_bin: cli.mistl,
            mistl_instance: cli.instance,
            auto_approve: cli.yes,
        },
    )?;

    match cli.prompt {
        Some(prompt) => headless(cfg, prompt).await,
        None => {
            let info = tui::UiInfo::from_config(&cfg);
            let handle = agent::spawn(cfg)?;
            tui::run(info, handle).await
        }
    }
}

async fn headless(cfg: config::Config, prompt: String) -> Result<()> {
    let mut handle = agent::spawn(cfg)?;
    handle.commands.send(UserCommand::Send(prompt))?;
    let mut out = std::io::stdout();
    let mut failed = false;
    while let Some(ev) = handle.events.recv().await {
        match ev {
            AgentEvent::TextDelta(t) => {
                print!("{t}");
                out.flush()?;
            }
            AgentEvent::AssistantDone => println!(),
            AgentEvent::ToolStart { title, .. } => eprintln!("\n[run] {title}"),
            AgentEvent::ToolEnd { ok, output, .. } => {
                eprintln!(
                    "[{}] {}",
                    if ok { "ok" } else { "failed" },
                    first_line(&output)
                );
            }
            AgentEvent::ApprovalRequest { title, reply, .. } => {
                // auto_approve never reaches here; refuse without a terminal.
                eprintln!("[refused] {title} (needs approval; re-run with --yes)");
                let _ = reply.send(Approval::No);
            }
            AgentEvent::Info(m) => eprintln!("[info] {m}"),
            AgentEvent::ToolModeChanged(mode) => eprintln!(
                "[info] tool mode: {}",
                match mode {
                    ToolMode::Native => "native",
                    ToolMode::Prompt => "prompt",
                }
            ),
            AgentEvent::Error(m) => {
                eprintln!("[error] {m}");
                failed = true;
            }
            AgentEvent::TurnDone => break,
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}
