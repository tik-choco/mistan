//! System prompt and prompt-mode tool protocol.

use serde_json::json;

use crate::mistl::{TOOL_HELP, TOOL_MISTL, split_args};
use crate::types::{FunctionCall, ToolCall, ToolMode, ToolOutput};

/// System prompt when no mistl executable is available: plain chat, no tools.
pub fn system_prompt_plain() -> String {
    "You are mistan, a terminal assistant for the tik-choco ecosystem. The mistl CLI is not \
installed on this machine, so you cannot run commands or inspect the user's mistl node. Answer \
in the user's language, keep answers short, and never claim to have run anything. If the user \
wants you to operate mistl, tell them to run `/mistl install` (or `mistan --install-mistl`) \
and then start a new message.\n"
        .into()
}

pub fn system_prompt(mode: ToolMode, catalog: &str) -> String {
    let mut s = String::from(
        "You are mistan, an assistant that operates the user's local mistl node by calling the \
mistl CLI. mistl is the P2P daemon of the tik-choco ecosystem: identity/DID, content store, \
stream, chat relay, tunnel, AI network, scheduler, and bot pipelines.\n\
\n\
Rules:\n\
- Answer in the user's language.\n\
- Use mistl help before guessing the syntax of an unfamiliar subcommand.\n\
- Prefer read-only queries to inspect state before changing anything.\n\
- State-changing commands need the user's approval and may be declined; respect a refusal and do not retry the same command.\n\
- Never invent command output; base answers on actual results.\n\
- Keep answers short. Many commands print JSON; summarize it for humans instead of pasting it.\n\
- Never print or request secrets. Use a passphrase only if the user provides it.\n",
    );
    match mode {
        ToolMode::Native => {
            s.push_str(
                "\nTools: `mistl` runs the CLI with an args array (no leading \"mistl\"); \
`mistl_help` shows `mistl <command> --help`.\n",
            );
        }
        ToolMode::Prompt => {
            s.push_str(
                "\nTool protocol:\n\
To run a mistl command, output a fenced block (one command per block, shell-style quoting, no leading \"mistl\"):\n\
```mistl\n\
store ls\n\
```\n\
To read help for a subcommand:\n\
```mistl-help\n\
store folder-share\n\
```\n\
After emitting blocks, STOP and wait. The results arrive in the next user message starting with `[mistl results]`. \
When no tool is needed, answer normally without any such blocks.\n",
            );
        }
    }
    s.push_str("\n## mistl command overview (`mistl --help`)\n");
    s.push_str(catalog.trim());
    s.push('\n');
    s
}

/// Extract prompt-mode tool calls from a finished assistant message.
pub fn parse_prompt_calls(text: &str) -> Vec<ToolCall> {
    #[derive(PartialEq)]
    enum Kind {
        Mistl,
        Help,
        Shell,
        Other,
    }
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut kind: Option<Kind> = None;

    let mut push = |name: &str, arguments: String| {
        calls.push(ToolCall {
            id: format!("p{}", calls.len() + 1),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments,
            },
        });
    };

    for raw in text.lines() {
        let line = raw.trim();
        match &kind {
            None => {
                if let Some(info) = line.strip_prefix("```") {
                    let lang = info.trim().to_ascii_lowercase();
                    kind = Some(match lang.as_str() {
                        "mistl" => Kind::Mistl,
                        "mistl-help" => Kind::Help,
                        "sh" | "bash" | "shell" | "console" => Kind::Shell,
                        _ => Kind::Other,
                    });
                }
            }
            Some(k) => {
                if line.starts_with("```") {
                    kind = None;
                    continue;
                }
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                match k {
                    Kind::Mistl => {
                        let l = line.strip_prefix("$ ").unwrap_or(line);
                        push(TOOL_MISTL, json!({ "args": split_args(l) }).to_string());
                    }
                    Kind::Help => {
                        push(TOOL_HELP, json!({ "command": line }).to_string());
                    }
                    Kind::Shell => {
                        let l = line.strip_prefix("$ ").unwrap_or(line);
                        if l.starts_with("mistl ") {
                            push(TOOL_MISTL, json!({ "args": split_args(l) }).to_string());
                        }
                    }
                    Kind::Other => {}
                }
            }
        }
    }
    calls
}

/// Build the user-role message that feeds prompt-mode results back. Each entry
/// is `(title, output)`.
pub fn format_prompt_results(results: &[(String, ToolOutput)]) -> String {
    let mut s = String::from("[mistl results]\n");
    for (title, out) in results {
        s.push_str("$ ");
        s.push_str(title);
        s.push('\n');
        s.push_str(out.text.trim_end());
        s.push('\n');
        s.push_str(if out.ok {
            "status: ok\n"
        } else {
            "status: failed\n"
        });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mistl::{Invocation, parse_call};

    fn inv(c: &ToolCall) -> Invocation {
        parse_call(&c.function.name, &c.function.arguments).unwrap()
    }

    #[test]
    fn parses_mistl_blocks() {
        let text = "Let me look.\n```mistl\nstore ls\n# comment\n\nai chat \"hi there\"\n```\nand\n```mistl-help\nstore folder-share\n```";
        let calls = parse_prompt_calls(text);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].id, "p1");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(
            inv(&calls[0]),
            Invocation::Run(vec!["store".into(), "ls".into()])
        );
        assert_eq!(
            inv(&calls[1]),
            Invocation::Run(vec!["ai".into(), "chat".into(), "hi there".into()])
        );
        assert_eq!(
            inv(&calls[2]),
            Invocation::Help(vec!["store".into(), "folder-share".into()])
        );
    }

    #[test]
    fn shell_blocks_only_when_mistl() {
        let text = "```bash\nls -la\n$ mistl status\nmistl ai models\n```\n```python\nmistl x\n```";
        let calls = parse_prompt_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(inv(&calls[0]), Invocation::Run(vec!["status".into()]));
        assert_eq!(
            inv(&calls[1]),
            Invocation::Run(vec!["ai".into(), "models".into()])
        );
    }

    #[test]
    fn no_blocks() {
        assert!(parse_prompt_calls("plain answer").is_empty());
    }

    #[test]
    fn results_format() {
        let r = vec![
            (
                "mistl store ls".to_string(),
                ToolOutput {
                    text: "exit code: 0\n".into(),
                    ok: true,
                },
            ),
            (
                "mistl x".to_string(),
                ToolOutput {
                    text: "boom".into(),
                    ok: false,
                },
            ),
        ];
        let s = format_prompt_results(&r);
        assert!(s.starts_with("[mistl results]\n$ mistl store ls\nexit code: 0\nstatus: ok\n"));
        assert!(s.ends_with("$ mistl x\nboom\nstatus: failed\n"));
    }

    #[test]
    fn prompt_modes() {
        let p = system_prompt(ToolMode::Prompt, "CATALOG");
        assert!(
            p.contains("```mistl-help") && p.contains("CATALOG") && p.contains("[mistl results]")
        );
        let n = system_prompt(ToolMode::Native, "CATALOG");
        assert!(!n.contains("```mistl") && n.contains("CATALOG"));
    }
}
