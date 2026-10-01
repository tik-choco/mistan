//! The mistl tool: schema, safety classification, and subprocess runner.

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::types::{FunctionSpec, Safety, ToolOutput, ToolSpec};

pub const TOOL_MISTL: &str = "mistl";
pub const TOOL_HELP: &str = "mistl_help";

/// Output cap handed to the model (head + tail are kept).
const MAX_OUTPUT_BYTES: usize = 12 * 1024;
/// How long readers get to drain after the child exited.
const READER_GRACE: Duration = Duration::from_millis(300);

/// One parsed tool request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Invocation {
    /// `mistl <args...>`
    Run(Vec<String>),
    /// `mistl <path...> --help` (empty path = top level)
    Help(Vec<String>),
}

/// Native-mode tool definitions.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: TOOL_MISTL,
                description: "Run the mistl CLI with these arguments (do not include the leading \
                    \"mistl\"), e.g. [\"store\",\"ls\"], [\"ai\",\"status\"], \
                    [\"config\",\"set\",\"ai.default_preset_id\",\"default\"]. Many commands print \
                    JSON. Read-only queries run immediately; state-changing commands require \
                    user approval and may be declined."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Command-line arguments, one array element per argument"
                        }
                    },
                    "required": ["args"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: TOOL_HELP,
                description: "Show `mistl <command> --help`. Use it to learn the exact syntax of \
                    a subcommand before running it. Omit `command` for the top-level help."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Subcommand path, e.g. \"store folder-share\""
                        }
                    }
                }),
            },
        },
    ]
}

/// Parse a tool call (name + JSON arguments) into an invocation.
pub fn parse_call(name: &str, arguments: &str) -> Result<Invocation> {
    let trimmed = arguments.trim();
    let value: Value = if trimmed.is_empty() {
        json!({})
    } else {
        serde_json::from_str(trimmed).map_err(|e| anyhow!("invalid tool arguments JSON: {e}"))?
    };
    match name {
        TOOL_MISTL => {
            let mut args = words_from(value.get("args"))?;
            drop_leading_mistl(&mut args);
            Ok(Invocation::Run(args))
        }
        TOOL_HELP => {
            let mut path = words_from(value.get("command").or_else(|| value.get("args")))?;
            drop_leading_mistl(&mut path);
            if matches!(path.last().map(String::as_str), Some("--help" | "-h")) {
                path.pop();
            }
            Ok(Invocation::Help(path))
        }
        other => bail!("unknown tool: {other}"),
    }
}

/// Accepts an array of strings (numbers/bools stringified), a single string
/// (shell-split), or nothing.
fn words_from(v: Option<&Value>) -> Result<Vec<String>> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(split_args(s)),
        Some(Value::Array(items)) => items
            .iter()
            .map(|i| match i {
                Value::String(s) => Ok(s.clone()),
                Value::Number(n) => Ok(n.to_string()),
                Value::Bool(b) => Ok(b.to_string()),
                _ => Err(anyhow!("args must be an array of strings")),
            })
            .collect(),
        Some(_) => bail!("args must be an array of strings"),
    }
}

fn drop_leading_mistl(args: &mut Vec<String>) {
    if matches!(
        args.first().map(String::as_str),
        Some("mistl" | "mistl.exe")
    ) {
        args.remove(0);
    }
}

/// Small shell-style word splitter: whitespace separates words; single quotes
/// are literal; double quotes group; a backslash escapes a quote character
/// (otherwise it is kept literally so Windows paths survive).
pub(crate) fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some('\'') => {
                if c == '\'' {
                    quote = None;
                } else {
                    cur.push(c);
                }
            }
            Some(q) => {
                if c == q {
                    quote = None;
                } else if c == '\\' && matches!(chars.peek(), Some('"') | Some('\\')) {
                    // Inside double quotes `\\` and `\"` are escapes.
                    cur.push(chars.next().unwrap());
                } else {
                    cur.push(c);
                }
            }
            None => {
                if c.is_whitespace() {
                    if in_word {
                        out.push(std::mem::take(&mut cur));
                        in_word = false;
                    }
                } else if c == '"' || c == '\'' {
                    quote = Some(c);
                    in_word = true;
                } else if c == '\\' && matches!(chars.peek(), Some('"') | Some('\'')) {
                    cur.push(chars.next().unwrap());
                    in_word = true;
                } else {
                    cur.push(c);
                    in_word = true;
                }
            }
        }
    }
    if in_word {
        out.push(cur);
    }
    out
}

/// Subcommand paths (leading non-flag words) that are read-only. Matching is
/// by prefix, so trailing positionals (`ai chat "hi"`, `chat log room`) are fine.
const READ_ONLY: &[&[&str]] = &[
    &["status"],
    &["build-info"],
    &["daemon", "status"],
    &["network", "status"],
    &["profile", "show"],
    &["key", "list"],
    &["key", "did"],
    &["key", "delegations"],
    &["store", "ls"],
    &["store", "parse-link"],
    &["store", "sandbox", "ls"],
    &["store", "sandbox-list"],
    &["store", "folder-sync", "ls"],
    &["store", "folder-share", "ls"],
    &["store", "folder-access", "ls"],
    &["stream", "status"],
    &["chat", "rooms"],
    &["chat", "log"],
    &["ai", "status"],
    &["ai", "models"],
    &["ai", "chat"],
    &["sched", "ls"],
    &["sched", "logs"],
    &["sched", "next"],
    &["bot", "list"],
    &["bot", "logs"],
    &["bot", "items"],
    &["bot", "status"],
    &["tunnel", "status"],
    &["tunnel", "ls"],
    &["config", "show"],
    &["update", "check"],
    &["update", "status"],
    &["autostart", "status"],
];

/// Global flags mistan controls itself.
const FORBIDDEN_FLAGS: &[&str] = &["--instance", "--state-dir", "--no-tray"];

pub fn classify(inv: &Invocation) -> Safety {
    let args = match inv {
        Invocation::Help(_) => return Safety::ReadOnly,
        Invocation::Run(args) => args,
    };
    if args.is_empty() {
        return Safety::Blocked(
            "no arguments: bare `mistl` opens the dashboard (use `ui`, or ask for help)".into(),
        );
    }
    for a in args {
        let flag = a.split('=').next().unwrap_or(a);
        if FORBIDDEN_FLAGS.contains(&flag) {
            return Safety::Blocked(format!("{flag} is controlled by mistan and cannot be set"));
        }
    }
    if args
        .iter()
        .any(|a| matches!(a.as_str(), "--help" | "-h" | "--version" | "-V"))
    {
        return Safety::ReadOnly;
    }

    // Leading non-flag words; the store-level `--json` flag may sit anywhere.
    let mut words: Vec<&str> = Vec::new();
    for a in args {
        if a == "--json" {
            continue;
        }
        if a.starts_with('-') {
            break;
        }
        words.push(a.as_str());
    }
    let w = |i: usize| words.get(i).copied();

    if w(0) == Some("daemon") && w(1) == Some("run") {
        return Safety::Blocked(
            "`daemon run` runs in the foreground forever; use `daemon start`".into(),
        );
    }
    if w(0) == Some("tunnel") && w(1) == Some("tui") {
        return Safety::Blocked("`tunnel tui` is an interactive terminal UI".into());
    }
    if w(0) == Some("help") {
        return Safety::ReadOnly;
    }
    // `update` alone defaults to `update check`.
    if words == ["update"] {
        return Safety::ReadOnly;
    }
    if w(0) == Some("tunnel") && w(1) == Some("trust") {
        let revoking = args
            .iter()
            .any(|a| a == "--revoke" || a.starts_with("--revoke="));
        return if revoking {
            Safety::Mutating
        } else {
            Safety::ReadOnly
        };
    }
    if w(0) == Some("tunnel") && w(1) == Some("room") {
        let list = args.iter().any(|a| a == "--list");
        let new = args.iter().any(|a| a == "--new");
        return if list && !new && w(2).is_none() {
            Safety::ReadOnly
        } else {
            Safety::Mutating
        };
    }
    if READ_ONLY
        .iter()
        .any(|p| words.len() >= p.len() && p.iter().zip(&words).all(|(a, b)| a == b))
    {
        return Safety::ReadOnly;
    }
    Safety::Mutating
}

/// Human-readable one-liner, e.g. `mistl store ls`.
pub fn title(inv: &Invocation) -> String {
    let (words, help) = match inv {
        Invocation::Run(a) => (a.as_slice(), false),
        Invocation::Help(p) => (p.as_slice(), true),
    };
    let mut s = String::from("mistl");
    for w in words {
        s.push(' ');
        s.push_str(&quote_word(w));
    }
    if help {
        s.push_str(" --help");
    }
    s
}

fn quote_word(w: &str) -> String {
    if !w.is_empty()
        && !w
            .chars()
            .any(|c| c.is_whitespace() || c == '"' || c == '\'')
    {
        return w.to_string();
    }
    format!("\"{}\"", w.replace('\\', "\\\\").replace('"', "\\\""))
}

pub struct MistlRunner {
    bin: String,
    prefix: Vec<String>,
    timeout: Duration,
}

impl MistlRunner {
    pub fn new(cfg: &Config) -> Self {
        // `--no-tray` is a global flag: the agent never spawns tray icons.
        let mut prefix = vec!["--no-tray".to_string()];
        if let Some(inst) = cfg.mistl_instance.as_deref().filter(|i| !i.is_empty()) {
            prefix.push("--instance".into());
            prefix.push(inst.to_string());
        }
        Self {
            bin: cfg.mistl_bin.clone(),
            prefix,
            timeout: Duration::from_secs(cfg.mistl_timeout_secs.max(1)),
        }
    }

    pub fn bin(&self) -> &str {
        &self.bin
    }

    pub fn set_bin(&mut self, bin: String) {
        self.bin = bin;
    }

    /// Whether the configured mistl executable exists.
    pub fn is_available(&self) -> bool {
        crate::config::mistl_available(&self.bin)
    }

    /// Make sure the mistl daemon is running, starting it in the background if
    /// not. Returns `true` when this call started it.
    pub async fn ensure_daemon(&self, cancel: &CancellationToken) -> Result<bool> {
        let status = self
            .exec(
                &["daemon".to_string(), "status".to_string()],
                self.timeout,
                cancel,
            )
            .await;
        if status.ok {
            return Ok(false);
        }
        // `daemon status` exits non-zero when it is down; starting it is
        // idempotent enough (a racing start reports "already running").
        let start = self
            .exec(
                &["daemon".to_string(), "start".to_string()],
                self.timeout,
                cancel,
            )
            .await;
        if start.ok || start.text.contains("already running") {
            Ok(start.ok)
        } else {
            bail!("mistl daemon start failed: {}", start.text)
        }
    }

    pub async fn run(&self, inv: &Invocation, cancel: &CancellationToken) -> ToolOutput {
        let args: Vec<String> = match inv {
            Invocation::Run(a) => a.clone(),
            Invocation::Help(p) => p.iter().cloned().chain(["--help".to_string()]).collect(),
        };
        self.exec(&args, self.timeout, cancel).await
    }

    pub async fn serve_start(&self, cancel: &CancellationToken) -> Result<String> {
        let args = ["ai", "serve", "start"].map(String::from);
        let out = self.exec(&args, self.timeout, cancel).await;
        if !out.ok {
            bail!("mistl ai serve start failed: {}", out.text);
        }
        let stdout = extract_stdout(&out.text)
            .ok_or_else(|| anyhow!("mistl ai serve start returned no stdout"))?;
        extract_listen(&stdout)
    }

    /// Top-level `mistl --help` text for the system prompt (cached by the
    /// caller); a short error note if mistl cannot be started.
    pub async fn catalog(&self) -> String {
        let out = self
            .exec(
                &["--help".to_string()],
                Duration::from_secs(10),
                &CancellationToken::new(),
            )
            .await;
        match extract_stdout(&out.text) {
            Some(s) if out.ok && !s.trim().is_empty() => s,
            _ => format!(
                "(mistl --help unavailable: {})",
                out.text
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("unknown error")
            ),
        }
    }

    async fn exec(
        &self,
        args: &[String],
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> ToolOutput {
        let mut cmd = Command::new(&self.bin);
        cmd.args(&self.prefix)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("NO_COLOR", "1")
            .env("RUST_LOG", "warn")
            .kill_on_drop(true);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return ToolOutput {
                    text: format!(
                        "failed to start `{}`: {e}. Install mistl with `/mistl install` \
                         (or `mistan --install-mistl`), or point mistan at it with \
                         `--mistl <path>`, `mistl_bin` in config.toml, or the MISTAN_MISTL \
                         environment variable.",
                        self.bin
                    ),
                    ok: false,
                };
            }
        };

        let out_buf = Arc::new(Mutex::new(Vec::new()));
        let err_buf = Arc::new(Mutex::new(Vec::new()));
        let mut readers = Vec::new();
        if let Some(s) = child.stdout.take() {
            readers.push(tokio::spawn(pump(s, out_buf.clone())));
        }
        if let Some(s) = child.stderr.take() {
            readers.push(tokio::spawn(pump(s, err_buf.clone())));
        }

        enum End {
            Exit(std::io::Result<std::process::ExitStatus>),
            Timeout,
            Cancelled,
        }
        let end = tokio::select! {
            r = child.wait() => End::Exit(r),
            _ = tokio::time::sleep(timeout) => End::Timeout,
            _ = cancel.cancelled() => End::Cancelled,
        };
        if !matches!(end, End::Exit(_)) {
            let _ = child.kill().await;
        }
        // A background daemon may hold the pipes open: do not wait for EOF.
        let _ = tokio::time::timeout(READER_GRACE, async {
            for r in &mut readers {
                let _ = r.await;
            }
        })
        .await;
        for r in &readers {
            r.abort();
        }

        let stdout = take_text(&out_buf);
        let stderr = take_text(&err_buf);
        let (code_line, note, ok) = match end {
            End::Exit(Ok(st)) => (
                format!(
                    "exit code: {}",
                    st.code().map_or("none".to_string(), |c| c.to_string())
                ),
                None,
                st.success(),
            ),
            End::Exit(Err(e)) => (
                "exit code: unknown".into(),
                Some(format!("wait failed: {e}")),
                false,
            ),
            End::Timeout => (
                "exit code: none".into(),
                Some(format!(
                    "timed out after {}s; process killed",
                    timeout.as_secs()
                )),
                false,
            ),
            End::Cancelled => (
                "exit code: none".into(),
                Some("cancelled by user; process killed".into()),
                false,
            ),
        };
        let mut text = code_line;
        if !stdout.trim().is_empty() {
            text.push_str("\n--- stdout ---\n");
            text.push_str(stdout.trim_end());
        }
        if !stderr.trim().is_empty() {
            text.push_str("\n--- stderr ---\n");
            text.push_str(stderr.trim_end());
        }
        if let Some(n) = note {
            text.push_str("\n[");
            text.push_str(&n);
            text.push(']');
        }
        ToolOutput {
            text: truncate_middle(&text, MAX_OUTPUT_BYTES),
            ok,
        }
    }
}

fn extract_listen(stdout: &str) -> Result<String> {
    let object = stdout
        .find('{')
        .zip(stdout.rfind('}'))
        .filter(|(start, end)| start <= end)
        .map(|(start, end)| &stdout[start..=end])
        .ok_or_else(|| anyhow!("mistl ai serve start returned no JSON object"))?;
    let value: Value = serde_json::from_str(object)
        .map_err(|e| anyhow!("invalid mistl ai serve start JSON: {e}"))?;
    value
        .get("listen")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(String::from)
        .ok_or_else(|| anyhow!("mistl ai serve start JSON is missing a nonempty string `listen`"))
}

async fn pump<R: tokio::io::AsyncRead + Unpin>(mut r: R, buf: Arc<Mutex<Vec<u8>>>) {
    let mut chunk = [0u8; 4096];
    loop {
        match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if let Ok(mut b) = buf.lock() {
                    b.extend_from_slice(&chunk[..n]);
                }
            }
        }
    }
}

fn take_text(buf: &Arc<Mutex<Vec<u8>>>) -> String {
    let bytes = buf.lock().map(|b| b.clone()).unwrap_or_default();
    strip_ansi(&String::from_utf8_lossy(&bytes))
}

/// Pull the stdout section out of the formatted output text.
fn extract_stdout(text: &str) -> Option<String> {
    let start = text.find("--- stdout ---\n")? + "--- stdout ---\n".len();
    let rest = &text[start..];
    let end = rest.find("\n--- stderr ---").unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// Remove ANSI escape sequences (CSI, OSC, and two-byte escapes).
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('[') => {
                for n in it.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&n) {
                        break;
                    }
                }
            }
            Some(']') => {
                while let Some(n) = it.next() {
                    if n == '\x07' {
                        break;
                    }
                    if n == '\x1b' {
                        it.next(); // ST: ESC \
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Keep the head and tail of `s` within `max` bytes, marking the cut.
pub(crate) fn truncate_middle(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let half = max / 2;
    let mut head_end = half;
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len() - half;
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!(
        "{}\n... [{} bytes truncated] ...\n{}",
        &s[..head_end],
        tail_start - head_end,
        &s[tail_start..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_listen_json() {
        assert_eq!(
            extract_listen("starting API\n{\"listen\":\"127.0.0.1:8123\"}\nready").unwrap(),
            "127.0.0.1:8123"
        );
        assert_eq!(
            extract_listen(r#"{"serving":true,"already_running":true,"listen":"127.0.0.1:6478"}"#)
                .unwrap(),
            "127.0.0.1:6478"
        );
        assert_eq!(
            extract_listen(r#"{"listen":"[::1]:8123"}"#).unwrap(),
            "[::1]:8123"
        );
        for invalid in [
            "no JSON",
            "}{",
            "{broken}",
            "{}",
            r#"{"listen":null}"#,
            r#"{"listen":42}"#,
            r#"{"listen":" "}"#,
        ] {
            assert!(extract_listen(invalid).is_err(), "{invalid}");
        }
    }

    fn run(a: &[&str]) -> Invocation {
        Invocation::Run(a.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn parse_array_and_string_args() {
        assert_eq!(
            parse_call("mistl", r#"{"args":["store","ls"]}"#).unwrap(),
            run(&["store", "ls"])
        );
        assert_eq!(
            parse_call("mistl", r#"{"args":"mistl ai chat 'hello world'"}"#).unwrap(),
            run(&["ai", "chat", "hello world"])
        );
        assert_eq!(
            parse_call("mistl", r#"{"args":["mistl","status"]}"#).unwrap(),
            run(&["status"])
        );
        assert_eq!(
            parse_call("mistl", r#"{"args":["tunnel","approve",3]}"#).unwrap(),
            run(&["tunnel", "approve", "3"])
        );
    }

    #[test]
    fn parse_help_and_errors() {
        assert_eq!(
            parse_call("mistl_help", "").unwrap(),
            Invocation::Help(vec![])
        );
        assert_eq!(
            parse_call("mistl_help", r#"{"command":"store folder-share --help"}"#).unwrap(),
            Invocation::Help(vec!["store".into(), "folder-share".into()])
        );
        assert!(parse_call("nope", "{}").is_err());
        assert!(parse_call("mistl", "{bad").is_err());
        assert!(parse_call("mistl", r#"{"args":[{"a":1}]}"#).is_err());
    }

    #[test]
    fn classify_samples() {
        assert!(matches!(classify(&run(&[])), Safety::Blocked(_)));
        assert!(matches!(
            classify(&run(&["--instance", "x", "status"])),
            Safety::Blocked(_)
        ));
        assert!(matches!(
            classify(&run(&["--state-dir=x", "status"])),
            Safety::Blocked(_)
        ));
        assert!(matches!(
            classify(&run(&["status", "--no-tray"])),
            Safety::Blocked(_)
        ));
        assert!(matches!(
            classify(&run(&["daemon", "run"])),
            Safety::Blocked(_)
        ));
        assert!(matches!(
            classify(&run(&["tunnel", "tui"])),
            Safety::Blocked(_)
        ));
        assert_eq!(classify(&run(&["store", "ls"])), Safety::ReadOnly);
        assert_eq!(classify(&run(&["store", "--json", "ls"])), Safety::ReadOnly);
        assert_eq!(classify(&run(&["store", "ls", "--json"])), Safety::ReadOnly);
        assert_eq!(
            classify(&run(&["store", "sandbox", "ls"])),
            Safety::ReadOnly
        );
        assert_eq!(
            classify(&run(&["store", "folder-share", "ls"])),
            Safety::ReadOnly
        );
        assert_eq!(classify(&run(&["ai", "chat", "hi"])), Safety::ReadOnly);
        assert_eq!(
            classify(&run(&["chat", "log", "room1", "-l", "5"])),
            Safety::ReadOnly
        );
        assert_eq!(classify(&run(&["update"])), Safety::ReadOnly);
        assert_eq!(classify(&run(&["tunnel", "trust"])), Safety::ReadOnly);
        assert_eq!(
            classify(&run(&["tunnel", "room", "--list"])),
            Safety::ReadOnly
        );
        assert_eq!(
            classify(&run(&["store", "put", "--help"])),
            Safety::ReadOnly
        );
        assert_eq!(classify(&Invocation::Help(vec![])), Safety::ReadOnly);
        assert_eq!(classify(&run(&["store", "put", "ls"])), Safety::Mutating);
        assert_eq!(
            classify(&run(&["config", "set", "a.b", "1"])),
            Safety::Mutating
        );
        assert_eq!(classify(&run(&["daemon", "start"])), Safety::Mutating);
        assert_eq!(classify(&run(&["ui"])), Safety::Mutating);
        assert_eq!(
            classify(&run(&["tunnel", "trust", "--revoke", "a@b"])),
            Safety::Mutating
        );
        assert_eq!(classify(&run(&["tunnel", "room"])), Safety::Mutating);
        assert_eq!(classify(&run(&["update", "apply"])), Safety::Mutating);
        assert_eq!(classify(&run(&["sched", "rm", "job-1"])), Safety::Mutating);
    }

    #[test]
    fn title_quoting() {
        assert_eq!(title(&run(&["store", "ls"])), "mistl store ls");
        assert_eq!(
            title(&run(&["ai", "chat", "a b"])),
            r#"mistl ai chat "a b""#
        );
        assert_eq!(
            title(&Invocation::Help(vec!["store".into()])),
            "mistl store --help"
        );
    }

    #[test]
    fn split_shell_words() {
        assert_eq!(
            split_args(r#"a "b c" 'd e' f\"g"#),
            vec!["a", "b c", "d e", "f\"g"]
        );
        assert_eq!(split_args("  x   y "), vec!["x", "y"]);
        assert_eq!(split_args(r#"a "" b"#), vec!["a", "", "b"]);
        assert_eq!(split_args(r"C:\tmp\x"), vec![r"C:\tmp\x"]);
        assert!(split_args("   ").is_empty());
    }

    #[test]
    fn ansi_and_truncation() {
        assert_eq!(
            strip_ansi("\x1b[31mred\x1b[0m ok \x1b]0;t\x07!"),
            "red ok !"
        );
        let s = "a".repeat(100);
        assert_eq!(truncate_middle(&s, 200), s);
        let t = truncate_middle(&"é".repeat(100), 50);
        assert!(t.contains("bytes truncated"));
        assert!(t.len() < 120);
    }
}
