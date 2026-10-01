//! mistan: a natural-language TUI agent that operates the mistl daemon CLI.

mod agent;
mod config;
mod install;
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
    /// Reasoning effort sent as `reasoning_effort` (low, medium, high, ...)
    #[arg(long, env = "MISTAN_REASONING_EFFORT")]
    reasoning_effort: Option<String>,
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
    /// Print the model ids of the configured endpoint and exit
    #[arg(long)]
    list_models: bool,
    /// Download and install the latest mistl release, then exit
    #[arg(long)]
    install_mistl: bool,
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
    if cli.install_mistl {
        return install_mistl().await;
    }
    let cfg = config::load(
        cli.config.as_deref(),
        config::Overrides {
            base_url: cli.base_url,
            model: cli.model,
            tool_mode: cli.tool_mode,
            reasoning_effort: cli.reasoning_effort,
            mistl_bin: cli.mistl,
            mistl_instance: cli.instance,
            auto_approve: cli.yes,
        },
    )?;

    if cli.list_models {
        return list_models(cfg).await;
    }
    match cli.prompt {
        Some(prompt) => headless(cfg, prompt).await,
        None => {
            let info = tui::UiInfo::from_config(&cfg);
            let handle = agent::spawn(cfg)?;
            tui::run(info, handle).await
        }
    }
}

async fn install_mistl() -> Result<()> {
    let dest = install::install_path()
        .ok_or_else(|| anyhow::anyhow!("cannot determine the install location"))?;
    let release = install::latest_release().await?;
    eprintln!(
        "[info] downloading {} ({})",
        release.asset_name, release.tag
    );
    let path = install::install_release(&release, &dest).await?;
    println!("installed mistl {} at {}", release.tag, path.display());
    Ok(())
}

async fn list_models(cfg: config::Config) -> Result<()> {
    let mut handle = agent::spawn(cfg)?;
    handle.commands.send(UserCommand::ListModels(None))?;
    while let Some(ev) = handle.events.recv().await {
        match ev {
            AgentEvent::Models { models, error } => {
                if let Some(e) = error {
                    anyhow::bail!("{e}");
                }
                for m in models {
                    println!("{m}");
                }
                return Ok(());
            }
            AgentEvent::Info(m) => eprintln!("[info] {m}"),
            _ => {}
        }
    }
    anyhow::bail!("agent stopped before answering")
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
            AgentEvent::Models { .. }
            | AgentEvent::SettingsApplied(_)
            | AgentEvent::MistlAvailable(_) => {}
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
