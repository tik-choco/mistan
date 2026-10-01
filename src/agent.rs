//! Agent loop: model <-> mistl tool round-trips.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::config::{self, Backend, Config, LlmSettings};
use crate::install;
use crate::llm::{LlmClient, is_tools_unsupported};
use crate::mistl::{self, MistlRunner};
use crate::prompt;
use crate::types::{
    AgentEvent, Approval, Message, MistlOp, Role, Safety, ToolCall, ToolMode, ToolOutput,
    UserCommand,
};

pub struct AgentHandle {
    pub commands: mpsc::UnboundedSender<UserCommand>,
    pub events: mpsc::UnboundedReceiver<AgentEvent>,
    cancel: Arc<Mutex<CancellationToken>>,
}

impl AgentHandle {
    /// Abort the in-flight user turn (model stream and running tool).
    pub fn cancel_turn(&self) {
        self.cancel.lock().unwrap().cancel();
    }
}

struct Agent {
    cfg: Config,
    llm: LlmClient,
    runner: MistlRunner,
    messages: Vec<Message>,
    mode: ToolMode,
    catalog: String,
    auto_approve: bool,
    /// The LLM endpoint is known (always true for a custom backend; for the
    /// mistl backend it needs the daemon and `ai serve`).
    endpoint_ready: bool,
    /// mistl was found when the session started, so the mistl tools are on.
    /// Without it mistan is a plain chat client.
    mistl_ready: bool,
    tx: mpsc::UnboundedSender<AgentEvent>,
}

const NO_MISTL_HINT: &str = "mistl was not found. Run /mistl install (or `mistan --install-mistl`) \
to download it, or point mistan at it with --mistl / mistl_bin.";

/// Rewrite native tool history without retaining any tool-only wire fields.
fn prompt_history(messages: &[Message]) -> Vec<Message> {
    let mut history = Vec::new();
    let mut titles = HashMap::new();
    let mut results = Vec::new();
    for message in messages {
        if message.role == Role::Tool {
            let title = message
                .tool_call_id
                .as_ref()
                .and_then(|id| titles.get(id))
                .cloned()
                .unwrap_or_else(|| "mistl (unknown call)".into());
            let text = message.content.clone().unwrap_or_default();
            // Native history stores only text; successful runner output starts here.
            let ok = text.lines().next() == Some("exit code: 0");
            results.push((title, ToolOutput { text, ok }));
            continue;
        }
        if !results.is_empty() {
            history.push(Message::user(prompt::format_prompt_results(&results)));
            results.clear();
        }
        let mut content = message.content.clone().unwrap_or_default();
        if message.role == Role::Assistant {
            for call in &message.tool_calls {
                let (title, command) =
                    match mistl::parse_call(&call.function.name, &call.function.arguments) {
                        Ok(inv) => {
                            let title = mistl::title(&inv);
                            let command = match inv {
                                mistl::Invocation::Run(_) => title
                                    .strip_prefix("mistl")
                                    .unwrap()
                                    .trim_start()
                                    .to_string(),
                                mistl::Invocation::Help(path) => {
                                    mistl::title(&mistl::Invocation::Run(path))
                                        .strip_prefix("mistl")
                                        .unwrap()
                                        .trim_start()
                                        .to_string()
                                }
                            };
                            (title, command)
                        }
                        Err(e) => (
                            call.function.name.clone(),
                            format!(
                                "# invalid tool call: {e}\n# {} {}",
                                call.function.name, call.function.arguments
                            ),
                        ),
                    };
                titles.insert(call.id.clone(), title);
                let fence = if call.function.name == mistl::TOOL_HELP {
                    "mistl-help"
                } else {
                    "mistl"
                };
                if !content.is_empty() {
                    content.push_str("\n\n");
                }
                content.push_str(&format!("```{fence}\n{command}\n```"));
            }
        }
        history.push(Message {
            role: message.role,
            content: Some(content),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    if !results.is_empty() {
        history.push(Message::user(prompt::format_prompt_results(&results)));
    }
    history
}

/// Spawn the agent task on the current tokio runtime.
pub fn spawn(cfg: Config) -> Result<AgentHandle> {
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<UserCommand>();
    let (ev_tx, ev_rx) = mpsc::unbounded_channel::<AgentEvent>();
    let cancel = Arc::new(Mutex::new(CancellationToken::new()));

    let mut agent = Agent {
        llm: LlmClient::new(&cfg)?,
        runner: MistlRunner::new(&cfg),
        messages: Vec::new(),
        mode: cfg.tool_mode,
        catalog: String::new(),
        auto_approve: cfg.auto_approve,
        endpoint_ready: false,
        mistl_ready: false,
        tx: ev_tx,
        cfg,
    };
    let cancel_slot = cancel.clone();
    tokio::spawn(async move {
        while let Some(cmd) = cmd_rx.recv().await {
            // Every command gets a fresh token so Esc can interrupt slow ones.
            let token = CancellationToken::new();
            *cancel_slot.lock().unwrap() = token.clone();
            match cmd {
                UserCommand::Send(text) => {
                    agent.turn(text, &token).await;
                    let _ = agent.tx.send(AgentEvent::TurnDone);
                }
                UserCommand::Clear => agent.messages.truncate(1),
                UserCommand::SetModel(m) => {
                    agent.cfg.model = m.clone();
                    agent.llm.set_model(m.clone());
                    agent.send(AgentEvent::Info(format!("model: {m}")));
                }
                UserCommand::SetEffort(e) => {
                    let e = config::normalize_effort(e);
                    agent.cfg.reasoning_effort = e.clone();
                    agent.llm.set_reasoning_effort(e.clone());
                    agent.send(AgentEvent::Info(format!(
                        "reasoning effort: {}",
                        e.as_deref().unwrap_or("default")
                    )));
                }
                UserCommand::Configure(s) => agent.configure(s),
                UserCommand::ListModels(probe) => agent.list_models(probe, &token).await,
                UserCommand::Mistl(op) => {
                    agent.mistl_op(op, &token).await;
                    agent.send(AgentEvent::TurnDone);
                }
            }
        }
    });

    Ok(AgentHandle {
        commands: cmd_tx,
        events: ev_rx,
        cancel,
    })
}

impl Agent {
    fn send(&self, ev: AgentEvent) {
        let _ = self.tx.send(ev);
    }

    fn report(&self, e: anyhow::Error) {
        if e.to_string() == "cancelled" {
            self.send(AgentEvent::Info("cancelled".into()));
        } else {
            self.send(AgentEvent::Error(format!("{e:#}")));
        }
    }

    fn system_prompt(&self) -> String {
        if self.mistl_ready {
            prompt::system_prompt(self.mode, &self.catalog)
        } else {
            prompt::system_prompt_plain()
        }
    }

    /// Run `fut`, turning cancellation into the `"cancelled"` error.
    async fn cancellable<T>(
        token: &CancellationToken,
        fut: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        tokio::select! {
            _ = token.cancelled() => Err(anyhow::anyhow!("cancelled")),
            r = fut => r,
        }
    }

    /// Start the mistl daemon if it is down (best effort; reports a notice).
    async fn start_daemon(&self, token: &CancellationToken) {
        if !self.runner.is_available() {
            return;
        }
        match Self::cancellable(token, self.runner.ensure_daemon(token)).await {
            Ok(true) => self.send(AgentEvent::Info("started the mistl daemon".into())),
            Ok(false) => {}
            Err(e) if e.to_string() == "cancelled" => {}
            Err(e) => self.send(AgentEvent::Info(format!("{e:#}"))),
        }
    }

    /// Find the mistl local API: needs mistl, a running daemon, and `ai serve`.
    async fn discover_mistl_api(&self, token: &CancellationToken) -> Result<String> {
        if !self.runner.is_available() {
            anyhow::bail!(
                "the mistl AI network needs mistl, which was not found. Run /mistl install, or \
                 open /settings and use an OpenAI-compatible API instead."
            );
        }
        self.start_daemon(token).await;
        let listen = Self::cancellable(token, self.runner.serve_start(token)).await?;
        Ok(format!("http://{listen}/v1"))
    }

    async fn ensure_endpoint(&mut self, token: &CancellationToken) -> Result<()> {
        if self.endpoint_ready {
            return Ok(());
        }
        if self.cfg.backend == Backend::Mistl {
            let url = self.discover_mistl_api(token).await?;
            self.send(AgentEvent::Info(format!("AI network: via mistl ({url})")));
            self.llm.set_base_url(url);
        }
        self.endpoint_ready = true;
        Ok(())
    }

    /// Apply new LLM settings, save them, and start a fresh conversation.
    fn configure(&mut self, s: LlmSettings) {
        let mut cfg = self.cfg.clone();
        if let Err(e) = s.apply(&mut cfg) {
            self.send(AgentEvent::Error(format!("{e:#}")));
            return;
        }
        let llm = match LlmClient::new(&cfg) {
            Ok(l) => l,
            Err(e) => {
                self.send(AgentEvent::Error(format!("{e:#}")));
                return;
            }
        };
        let saved = match cfg.config_path.as_deref() {
            Some(path) => match config::save_settings(path, &s) {
                Ok(()) => format!("saved to {}", path.display()),
                Err(e) => format!("not saved: {e:#}"),
            },
            None => "not saved: no config path".into(),
        };
        self.mode = cfg.tool_mode;
        self.llm = llm;
        self.cfg = cfg;
        self.messages.clear();
        self.endpoint_ready = false;
        self.mistl_ready = false;
        self.send(AgentEvent::SettingsApplied(LlmSettings::from_config(
            &self.cfg,
        )));
        self.send(AgentEvent::Info(format!(
            "settings applied; new conversation ({saved})"
        )));
    }

    async fn list_models(&mut self, probe: Option<LlmSettings>, token: &CancellationToken) {
        let result = async {
            let settings = probe.unwrap_or_else(|| LlmSettings::from_config(&self.cfg));
            let mut cfg = self.cfg.clone();
            settings.apply(&mut cfg)?;
            let mut client = LlmClient::new(&cfg)?;
            if cfg.backend == Backend::Mistl {
                client.set_base_url(self.discover_mistl_api(token).await?);
            }
            client.list_models(token).await
        }
        .await;
        match result {
            Ok(models) => self.send(AgentEvent::Models {
                models,
                error: None,
            }),
            Err(e) => self.send(AgentEvent::Models {
                models: Vec::new(),
                error: Some(format!("{e:#}")),
            }),
        }
    }

    async fn mistl_op(&mut self, op: MistlOp, token: &CancellationToken) {
        match op {
            MistlOp::Info => {
                let found = self.runner.is_available();
                self.send(AgentEvent::MistlAvailable(found));
                self.send(AgentEvent::Info(if found {
                    format!("mistl: {}", self.runner.bin())
                } else {
                    NO_MISTL_HINT.to_string()
                }));
            }
            MistlOp::Start => {
                if !self.runner.is_available() {
                    self.send(AgentEvent::Error(NO_MISTL_HINT.into()));
                    return;
                }
                match Self::cancellable(token, self.runner.ensure_daemon(token)).await {
                    Ok(true) => self.send(AgentEvent::Info("started the mistl daemon".into())),
                    Ok(false) => self.send(AgentEvent::Info(
                        "the mistl daemon is already running".into(),
                    )),
                    Err(e) => self.report(e),
                }
            }
            MistlOp::Install => {
                if let Err(e) = self.install_mistl(token).await {
                    self.report(e);
                }
            }
        }
    }

    async fn install_mistl(&mut self, token: &CancellationToken) -> Result<()> {
        let dest = install::install_path()
            .ok_or_else(|| anyhow::anyhow!("cannot determine the install location"))?;
        self.send(AgentEvent::Info(
            "looking up the latest mistl release…".into(),
        ));
        let release = Self::cancellable(token, install::latest_release()).await?;

        let (reply, rx) = oneshot::channel();
        self.send(AgentEvent::ApprovalRequest {
            title: format!(
                "Download mistl {} from github.com/tik-choco/mistl and install it to {}?",
                release.tag,
                dest.display()
            ),
            reason: "Downloads an executable and verifies it against the release's \
                     SHA256SUMS.txt (the checksum list itself is not signature-checked)."
                .into(),
            reply,
        });
        let answer = tokio::select! {
            _ = token.cancelled() => Approval::No,
            a = rx => a.unwrap_or(Approval::No),
        };
        if answer == Approval::No {
            self.send(AgentEvent::Info("install declined".into()));
            return Ok(());
        }
        if answer == Approval::Always {
            // The UI already shows auto-approve; keep both sides in agreement.
            self.auto_approve = true;
        }

        self.send(AgentEvent::Info(format!(
            "downloading {}…",
            release.asset_name
        )));
        let path = Self::cancellable(token, install::install_release(&release, &dest)).await?;
        let path = path.to_string_lossy().into_owned();
        self.cfg.mistl_bin = path.clone();
        self.runner.set_bin(path.clone());
        if !self.mistl_ready {
            // The running conversation was chat-only; the next one gets the tools.
            self.messages.clear();
            self.endpoint_ready = false;
        }
        self.send(AgentEvent::MistlAvailable(true));
        self.send(AgentEvent::Info(format!(
            "installed mistl {} at {path}. The daemon starts automatically with your next message.",
            release.tag
        )));
        Ok(())
    }

    fn fallback_to_prompt(&mut self, err: &anyhow::Error) -> bool {
        if self.mode != ToolMode::Native || !is_tools_unsupported(err) {
            return false;
        }
        self.mode = ToolMode::Prompt;
        self.messages = prompt_history(&self.messages);
        self.messages[0] = Message::system(self.system_prompt());
        self.send(AgentEvent::Info(
            "the AI provider does not support tools; switched to prompt mode".into(),
        ));
        self.send(AgentEvent::ToolModeChanged(self.mode));
        true
    }

    async fn turn(&mut self, text: String, token: &CancellationToken) {
        if self.messages.is_empty() {
            if let Err(e) = self.ensure_endpoint(token).await {
                self.report(e);
                return;
            }
            self.mistl_ready = self.runner.is_available();
            if self.mistl_ready {
                if self.cfg.backend == Backend::Custom {
                    self.start_daemon(token).await;
                }
                let catalog = tokio::select! {
                    _ = token.cancelled() => {
                        self.send(AgentEvent::Info("cancelled".into()));
                        return;
                    }
                    c = self.runner.catalog() => c,
                };
                self.catalog = catalog;
            } else {
                self.send(AgentEvent::Info(format!(
                    "{NO_MISTL_HINT} Chatting without mistl tools."
                )));
            }
            let sys = self.system_prompt();
            self.messages.push(Message::system(sys));
        }
        self.messages.push(Message::user(text));

        for _ in 0..self.cfg.max_steps {
            let turn = loop {
                let specs = if self.mistl_ready && self.mode == ToolMode::Native {
                    mistl::tool_specs()
                } else {
                    Vec::new()
                };
                let tx = self.tx.clone();
                let res = self
                    .llm
                    .complete(
                        &self.messages,
                        &specs,
                        &mut |d| {
                            let _ = tx.send(AgentEvent::TextDelta(d.to_string()));
                        },
                        token,
                    )
                    .await;
                match res {
                    Ok(t) => break t,
                    Err(e) if !token.is_cancelled() && self.fallback_to_prompt(&e) => continue,
                    Err(e) => {
                        if token.is_cancelled() || e.to_string() == "cancelled" {
                            self.send(AgentEvent::Info("cancelled".into()));
                        } else {
                            self.send(AgentEvent::Error(format!("{e:#}")));
                        }
                        return;
                    }
                };
            };
            self.send(AgentEvent::AssistantDone);

            let native = self.mode == ToolMode::Native;
            let calls: Vec<ToolCall> = if !self.mistl_ready {
                self.messages.push(Message::assistant(turn.content.clone()));
                Vec::new()
            } else if native {
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: (!turn.content.is_empty()).then(|| turn.content.clone()),
                    tool_calls: turn.tool_calls.clone(),
                    tool_call_id: None,
                });
                turn.tool_calls
            } else {
                self.messages.push(Message::assistant(turn.content.clone()));
                prompt::parse_prompt_calls(&turn.content)
            };
            if calls.is_empty() {
                return;
            }

            // (call id, title, output); stops early if the turn is cancelled.
            let mut results: Vec<(String, String, ToolOutput)> = Vec::new();
            for call in &calls {
                let (title, out) = self.run_call(call, token).await;
                results.push((call.id.clone(), title, out));
                if token.is_cancelled() {
                    break;
                }
            }

            if native {
                // Every tool_call id needs a response, even if cancelled.
                for call in &calls {
                    let text = results
                        .iter()
                        .find(|(id, _, _)| *id == call.id)
                        .map(|(_, _, o)| o.text.clone())
                        .unwrap_or_else(|| "cancelled".into());
                    self.messages
                        .push(Message::tool_result(call.id.clone(), text));
                }
            } else {
                let pairs: Vec<(String, ToolOutput)> = results
                    .iter()
                    .map(|(_, t, o)| (t.clone(), o.clone()))
                    .collect();
                self.messages
                    .push(Message::user(prompt::format_prompt_results(&pairs)));
            }

            if token.is_cancelled() {
                self.send(AgentEvent::Info("cancelled".into()));
                return;
            }
        }
        self.send(AgentEvent::Info(format!(
            "step limit reached ({} model calls); send another message to continue",
            self.cfg.max_steps
        )));
    }

    async fn run_call(
        &mut self,
        call: &ToolCall,
        token: &CancellationToken,
    ) -> (String, ToolOutput) {
        let fail = |text: String| ToolOutput { text, ok: false };
        let inv = match mistl::parse_call(&call.function.name, &call.function.arguments) {
            Ok(i) => i,
            Err(e) => {
                return (call.function.name.clone(), fail(format!("error: {e:#}")));
            }
        };
        let title = mistl::title(&inv);
        match mistl::classify(&inv) {
            Safety::ReadOnly => {}
            Safety::Blocked(r) => return (title, fail(format!("refused: {r}"))),
            Safety::Mutating => {
                if !self.auto_approve {
                    let (reply, rx) = oneshot::channel();
                    self.send(AgentEvent::ApprovalRequest {
                        title: title.clone(),
                        reason: "changes state".into(),
                        reply,
                    });
                    let answer = tokio::select! {
                        _ = token.cancelled() => Approval::No,
                        a = rx => a.unwrap_or(Approval::No),
                    };
                    match answer {
                        Approval::Yes => {}
                        Approval::Always => self.auto_approve = true,
                        Approval::No => {
                            let text = if token.is_cancelled() {
                                "cancelled"
                            } else {
                                "the user declined to run this command"
                            };
                            return (title, fail(text.into()));
                        }
                    }
                }
            }
        }
        self.send(AgentEvent::ToolStart {
            id: call.id.clone(),
            title: title.clone(),
        });
        let out = self.runner.run(&inv, token).await;
        self.send(AgentEvent::ToolEnd {
            id: call.id.clone(),
            ok: out.ok,
            output: out.text.clone(),
        });
        (title, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionCall;

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    #[test]
    fn converts_native_history_to_prompt() {
        let calls = vec![
            call(
                "run",
                "mistl",
                r#"{"args":["mistl","ai","chat","hello \"world\"","C:\\tmp\\folder name"]}"#,
            ),
            call(
                "help",
                "mistl_help",
                r#"{"command":"mistl store folder-share --help"}"#,
            ),
            call("root", "mistl_help", "{}"),
        ];
        let input = vec![
            Message::system("system"),
            Message::user("question"),
            Message {
                role: Role::Assistant,
                content: Some("Checking.".into()),
                tool_calls: calls.clone(),
                tool_call_id: None,
            },
            Message::tool_result("run", "exit code: 0\n--- stdout ---\nhello"),
            Message::tool_result("help", "error: refused"),
            Message::tool_result("root", "cancelled"),
            Message::assistant("answer"),
            Message::user("next"),
        ];
        let converted = prompt_history(&input);
        assert_eq!(converted.len(), 6);
        assert_eq!(&converted[..2], &input[..2]);
        let content = converted[2].content.as_deref().unwrap();
        assert!(content.starts_with("Checking.\n\n```mistl\n"));
        assert!(content.contains("```mistl-help\nstore folder-share\n```"));
        assert!(content.ends_with("```mistl-help\n\n```"));
        // The parser ignores empty help blocks; the serialized history still keeps them.
        let reconstructed = prompt::parse_prompt_calls(content);
        for (original, rebuilt) in calls.iter().zip(&reconstructed) {
            assert_eq!(
                mistl::parse_call(&original.function.name, &original.function.arguments).unwrap(),
                mistl::parse_call(&rebuilt.function.name, &rebuilt.function.arguments).unwrap(),
            );
        }
        assert_eq!(reconstructed.len(), 2);
        let pairs = vec![
            (
                mistl::title(&mistl::parse_call("mistl", &calls[0].function.arguments).unwrap()),
                ToolOutput {
                    text: input[3].content.clone().unwrap(),
                    ok: true,
                },
            ),
            (
                "mistl store folder-share --help".into(),
                ToolOutput {
                    text: "error: refused".into(),
                    ok: false,
                },
            ),
            (
                "mistl --help".into(),
                ToolOutput {
                    text: "cancelled".into(),
                    ok: false,
                },
            ),
        ];
        assert_eq!(
            converted[3],
            Message::user(prompt::format_prompt_results(&pairs))
        );
        assert_eq!(&converted[4..], &input[6..]);
        for message in &converted {
            assert_ne!(message.role, Role::Tool);
            assert!(message.content.is_some());
            assert!(message.tool_calls.is_empty());
            assert!(message.tool_call_id.is_none());
        }
        assert_eq!(prompt_history(&converted), converted);
        assert_eq!(input[2].tool_calls, calls);
        let wire = serde_json::to_value(&converted).unwrap();
        for message in wire.as_array().unwrap() {
            assert!(message["content"].is_string());
            assert!(message.get("tool_calls").is_none());
            assert!(message.get("tool_call_id").is_none());
        }
    }

    #[test]
    fn conversion_handles_missing_content_and_invalid_calls() {
        let input = vec![
            Message {
                role: Role::Assistant,
                content: None,
                tool_calls: vec![call("bad", "mistl", "{bad")],
                tool_call_id: Some("stray".into()),
            },
            Message::tool_result("bad", "error: invalid arguments"),
            Message {
                role: Role::User,
                content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            Message {
                role: Role::Tool,
                content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
        ];
        let converted = prompt_history(&input);
        assert!(
            converted[0]
                .content
                .as_ref()
                .unwrap()
                .contains("```mistl\n# invalid tool call:")
        );
        assert!(
            converted[1]
                .content
                .as_ref()
                .unwrap()
                .contains("$ mistl\nerror: invalid arguments")
        );
        assert_eq!(converted[2], Message::user(""));
        assert!(
            converted[3]
                .content
                .as_ref()
                .unwrap()
                .contains("$ mistl (unknown call)")
        );
        assert!(prompt_history(&[]).is_empty());
        for message in converted {
            assert_ne!(message.role, Role::Tool);
            assert!(message.content.is_some());
            assert!(message.tool_calls.is_empty());
            assert!(message.tool_call_id.is_none());
        }
    }

    #[test]
    fn fallback_changes_mode_once_and_rebuilds_system_prompt() {
        let cfg = Config::default();
        let (tx, mut events) = mpsc::unbounded_channel();
        let mut agent = Agent {
            llm: LlmClient::new(&cfg).unwrap(),
            runner: MistlRunner::new(&cfg),
            mode: cfg.tool_mode,
            catalog: "CATALOG".into(),
            messages: vec![Message::system("native"), Message::user("question")],
            auto_approve: false,
            endpoint_ready: true,
            mistl_ready: true,
            tx,
            cfg,
        };
        assert!(!agent.fallback_to_prompt(&anyhow::anyhow!("tools_unsupported")));
        let err = anyhow::Error::new(crate::llm::HttpError {
            status: 400,
            body: r#"{"error":{"code":"tools_unsupported"}}"#.into(),
        });
        assert!(agent.fallback_to_prompt(&err));
        assert_eq!(agent.mode, ToolMode::Prompt);
        assert_eq!(
            agent.messages[0],
            Message::system(prompt::system_prompt(ToolMode::Prompt, "CATALOG"))
        );
        assert_eq!(agent.messages[1], Message::user("question"));
        assert!(matches!(events.try_recv().unwrap(), AgentEvent::Info(s)
            if s == "the AI provider does not support tools; switched to prompt mode"));
        assert!(matches!(
            events.try_recv().unwrap(),
            AgentEvent::ToolModeChanged(ToolMode::Prompt)
        ));
        agent.messages.truncate(1);
        assert!(!agent.fallback_to_prompt(&err));
        assert_eq!(agent.mode, ToolMode::Prompt);
        assert!(events.try_recv().is_err());
    }

    #[tokio::test]
    async fn startup_failure_leaves_history_empty_for_retry() {
        // A directory cannot be executed, so this never invokes mistl.
        let cfg = Config {
            mistl_bin: std::env::temp_dir().to_string_lossy().into_owned(),
            ..Config::default()
        };
        let (tx, mut events) = mpsc::unbounded_channel();
        let mut agent = Agent {
            llm: LlmClient::new(&cfg).unwrap(),
            runner: MistlRunner::new(&cfg),
            messages: Vec::new(),
            mode: cfg.tool_mode,
            catalog: String::new(),
            auto_approve: false,
            endpoint_ready: false,
            mistl_ready: false,
            tx,
            cfg,
        };
        for _ in 0..2 {
            agent
                .turn("question".into(), &CancellationToken::new())
                .await;
            assert!(agent.messages.is_empty());
            assert!(
                matches!(events.try_recv().unwrap(), AgentEvent::Error(reason)
                if reason.contains("needs mistl, which was not found"))
            );
            assert!(events.try_recv().is_err());
        }
    }
}
