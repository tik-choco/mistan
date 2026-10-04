//! OpenAI-compatible streaming chat client.

use anyhow::{Result, anyhow, bail};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::config::{Backend, Config};
use crate::types::{AssistantTurn, FunctionCall, Message, ToolCall, ToolSpec};

#[derive(Debug)]
pub struct HttpError {
    pub status: u16,
    pub body: String,
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snippet: String = self.body.chars().take(500).collect();
        write!(f, "HTTP {}: {snippet}", self.status)
    }
}

impl std::error::Error for HttpError {}

pub fn is_tools_unsupported(err: &anyhow::Error) -> bool {
    err.downcast_ref::<HttpError>()
        .is_some_and(|e| e.status == 400 && e.body.contains("tools_unsupported"))
}

fn http_error(backend: Backend, status: u16, body: String) -> anyhow::Error {
    let err = anyhow::Error::new(HttpError { status, body });
    if backend == Backend::Mistl && status == 502 {
        let message = format!(
            "{err}; no AI provider reachable on the mistl AI network — check `mistl ai status`"
        );
        err.context(message)
    } else {
        err
    }
}

pub struct LlmClient {
    http: reqwest::Client,
    base_url: Option<String>,
    backend: Backend,
    model: String,
    api_key: Option<String>,
    reasoning_effort: Option<String>,
    unavailable: bool,
}

impl LlmClient {
    pub fn new(cfg: &Config) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()?;
        Ok(Self {
            http,
            base_url: if cfg.backend == Backend::Custom {
                cfg.base_url.clone()
            } else {
                None
            },
            backend: cfg.backend,
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            reasoning_effort: cfg.reasoning_effort.clone(),
            unavailable: cfg.backend == Backend::Custom
                && cfg.default_ref.as_ref().is_some_and(|r| {
                    !cfg.providers
                        .iter()
                        .any(|p| p.id == r.provider_id && p.enabled)
                }),
        })
    }

    pub fn set_base_url(&mut self, url: String) {
        self.base_url = Some(url.trim_end_matches('/').to_string());
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn set_reasoning_effort(&mut self, effort: Option<String>) {
        self.reasoning_effort = effort;
    }

    /// `GET <base_url>/models` -> sorted, de-duplicated model ids.
    pub async fn list_models(&self, cancel: &CancellationToken) -> Result<Vec<String>> {
        if self.unavailable {
            bail!("{}", crate::config::text::get("unavailable"));
        }
        let base_url = self.base_url.as_deref().filter(|url| !url.is_empty()).ok_or_else(
            || anyhow!("LLM base URL is not set; initialize the mistl AI network or configure a custom endpoint"),
        )?;
        let mut req = self
            .http
            .get(format!("{base_url}/models"))
            .timeout(Duration::from_secs(20));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = tokio::select! {
            _ = cancel.cancelled() => bail!("cancelled"),
            r = req.send() => r.map_err(|e| anyhow!("request failed: {}", e.without_url()))?,
        };
        let status = resp.status();
        let bytes = tokio::select! {
            _ = cancel.cancelled() => bail!("cancelled"),
            b = resp.bytes() => b.map_err(|e| anyhow!("read failed: {}", e.without_url()))?,
        };
        if !status.is_success() {
            return Err(http_error(
                self.backend,
                status.as_u16(),
                String::from_utf8_lossy(&bytes).into_owned(),
            ));
        }
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
            let snippet: String = String::from_utf8_lossy(&bytes).chars().take(200).collect();
            anyhow!("unparsable model list ({e}): {snippet}")
        })?;
        Ok(parse_models(&v))
    }

    /// Stream one completion. `tools` empty -> no `tools` field sent.
    /// Calls `on_delta` with each content delta. Returns early with an
    /// error if `cancel` fires.
    pub async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        on_delta: &mut (dyn FnMut(&str) + Send),
        cancel: &CancellationToken,
    ) -> Result<AssistantTurn> {
        if self.unavailable {
            bail!("{}", crate::config::text::get("unavailable"));
        }
        let base_url = self.base_url.as_deref().filter(|url| !url.is_empty())
            .ok_or_else(|| anyhow!("LLM base URL is not set; initialize the mistl AI network or configure a custom endpoint"))?;
        let mut body = json!({ "messages": messages, "stream": true });
        if !self.model.is_empty() {
            body["model"] = json!(self.model);
        }
        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools)?;
        }
        if let Some(e) = &self.reasoning_effort {
            body["reasoning_effort"] = json!(e);
        }

        let mut req = self
            .http
            .post(format!("{base_url}/chat/completions"))
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = tokio::select! {
            _ = cancel.cancelled() => bail!("cancelled"),
            r = req.send() => r.map_err(|e| anyhow!("request failed: {}", e.without_url()))?,
        };

        let status = resp.status();
        if !status.is_success() {
            let text = tokio::select! {
                _ = cancel.cancelled() => bail!("cancelled"),
                t = resp.text() => t.unwrap_or_default(),
            };
            return Err(http_error(self.backend, status.as_u16(), text));
        }

        let is_sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.to_ascii_lowercase().contains("text/event-stream"));

        if !is_sse {
            let bytes = tokio::select! {
                _ = cancel.cancelled() => bail!("cancelled"),
                b = resp.bytes() => b.map_err(|e| anyhow!("read failed: {}", e.without_url()))?,
            };
            return parse_full_response(&bytes, on_delta);
        }

        let mut parser = SseParser::default();
        let mut stream = resp.bytes_stream();
        loop {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => bail!("cancelled"),
                c = stream.next() => c,
            };
            match chunk {
                None => break,
                Some(Err(e)) => bail!("stream error: {}", e.without_url()),
                Some(Ok(bytes)) => {
                    parser.feed(&bytes, on_delta);
                    if parser.done {
                        break;
                    }
                }
            }
        }
        Ok(parser.finish(on_delta))
    }
}

/// Model ids out of an OpenAI `{"data":[{"id":..}]}` list (also accepts
/// `{"models":[..]}` and plain string entries).
fn parse_models(v: &Value) -> Vec<String> {
    let items = v
        .get("data")
        .or_else(|| v.get("models"))
        .or(Some(v))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut ids: Vec<String> = items
        .iter()
        .filter_map(|m| match m {
            Value::String(s) => Some(s.as_str()),
            _ => ["id", "name", "model"]
                .iter()
                .find_map(|k| m.get(*k).and_then(Value::as_str)),
        })
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// Non-streaming `chat.completion` fallback.
fn parse_full_response(
    bytes: &[u8],
    on_delta: &mut (dyn FnMut(&str) + Send),
) -> Result<AssistantTurn> {
    let v: Value = serde_json::from_slice(bytes).map_err(|e| {
        let snippet: String = String::from_utf8_lossy(bytes).chars().take(200).collect();
        anyhow!("unparsable response ({e}): {snippet}")
    })?;
    let msg = &v["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("").to_string();
    if !content.is_empty() {
        on_delta(&content);
    }
    let mut parts: Vec<PartialCall> = Vec::new();
    if let Some(calls) = msg["tool_calls"].as_array() {
        for (i, c) in calls.iter().enumerate() {
            parts.push(PartialCall {
                index: i,
                id: c["id"].as_str().unwrap_or("").to_string(),
                name: c["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: match &c["function"]["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                },
            });
        }
    }
    Ok(AssistantTurn {
        content,
        tool_calls: build_calls(parts),
    })
}

#[derive(Debug, Default, Clone)]
struct PartialCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

fn build_calls(mut parts: Vec<PartialCall>) -> Vec<ToolCall> {
    parts.sort_by_key(|p| p.index);
    parts
        .into_iter()
        .enumerate()
        .map(|(n, p)| ToolCall {
            id: if p.id.is_empty() {
                format!("call_{n}")
            } else {
                p.id
            },
            kind: "function".into(),
            function: FunctionCall {
                name: p.name,
                arguments: if p.arguments.trim().is_empty() {
                    "{}".into()
                } else {
                    p.arguments
                },
            },
        })
        .collect()
}

/// Pure SSE accumulator: feed raw bytes, get content deltas and merged tool calls.
#[derive(Default)]
struct SseParser {
    buf: Vec<u8>,
    content: String,
    calls: Vec<PartialCall>,
    done: bool,
}

impl SseParser {
    fn feed(&mut self, bytes: &[u8], on_delta: &mut (dyn FnMut(&str) + Send)) {
        self.buf.extend_from_slice(bytes);
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            self.line(&line, on_delta);
            if self.done {
                return;
            }
        }
    }

    fn finish(mut self, on_delta: &mut (dyn FnMut(&str) + Send)) -> AssistantTurn {
        if !self.buf.is_empty() && !self.done {
            let line = std::mem::take(&mut self.buf);
            self.line(&line, on_delta);
        }
        AssistantTurn {
            content: self.content,
            tool_calls: build_calls(self.calls),
        }
    }

    fn line(&mut self, raw: &[u8], on_delta: &mut (dyn FnMut(&str) + Send)) {
        let line = String::from_utf8_lossy(raw);
        let line = line.trim_end_matches(['\r', '\n']);
        let Some(data) = line.strip_prefix("data:") else {
            return; // comments, event:, blank lines
        };
        let data = data.trim();
        if data == "[DONE]" {
            self.done = true;
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let delta = &v["choices"][0]["delta"];
        if let Some(text) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            self.content.push_str(text);
            on_delta(text);
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for (pos, c) in calls.iter().enumerate() {
                let index = c["index"].as_u64().map(|i| i as usize).unwrap_or(pos);
                let slot = match self.calls.iter().position(|p| p.index == index) {
                    Some(i) => i,
                    None => {
                        self.calls.push(PartialCall {
                            index,
                            ..Default::default()
                        });
                        self.calls.len() - 1
                    }
                };
                let p = &mut self.calls[slot];
                if let Some(id) = c["id"].as_str().filter(|s| !s.is_empty()) {
                    p.id = id.to_string();
                }
                if let Some(name) = c["function"]["name"].as_str() {
                    p.name.push_str(name);
                }
                if let Some(args) = c["function"]["arguments"].as_str() {
                    p.arguments.push_str(args);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_list_shapes() {
        let openai = json!({"object":"list","data":[{"id":"b"},{"id":"a"},{"id":"b"},{"id":" "}]});
        assert_eq!(parse_models(&openai), ["a", "b"]);
        let alt = json!({"models":[{"name":"x"},"y",{"model":"z"},{"other":1}]});
        assert_eq!(parse_models(&alt), ["x", "y", "z"]);
        assert_eq!(parse_models(&json!(["m"])), ["m"]);
        assert!(parse_models(&json!({"error":"nope"})).is_empty());
    }

    #[tokio::test]
    async fn list_models_needs_base_url() {
        let client = LlmClient::new(&Config::default()).unwrap();
        let err = client
            .list_models(&CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("LLM base URL is not set"));
    }

    #[tokio::test]
    async fn missing_base_url() {
        let client = LlmClient::new(&Config::default()).unwrap();
        let err = client
            .complete(&[], &[], &mut |_| {}, &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("LLM base URL is not set"));
    }

    #[tokio::test]
    async fn disabled_default_is_not_replaced_or_fetched() {
        let cfg = Config {
            backend: Backend::Custom,
            default_ref: Some(crate::config::ModelRef {
                provider_id: "disabled".into(),
                model: "raw".into(),
            }),
            providers: vec![crate::config::Provider {
                id: "other".into(),
                base_url: "https://example.invalid/v1".into(),
                ..Default::default()
            }],
            ..Config::default()
        };
        let client = LlmClient::new(&cfg).unwrap();
        let cancel = CancellationToken::new();
        assert_eq!(
            client.list_models(&cancel).await.unwrap_err().to_string(),
            crate::config::text::get("unavailable")
        );
        assert_eq!(
            client
                .complete(&[], &[], &mut |_| {}, &cancel)
                .await
                .unwrap_err()
                .to_string(),
            crate::config::text::get("unavailable")
        );
    }

    #[tokio::test]
    async fn request_sends_raw_model_and_effort_without_temperature() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let body = loop {
                let mut chunk = [0; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(start) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..start]).to_ascii_lowercase();
                    let length: usize = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap();
                    if bytes.len() >= start + 4 + length {
                        break serde_json::from_slice::<Value>(
                            &bytes[start + 4..start + 4 + length],
                        )
                        .unwrap();
                    }
                }
            };
            let reply = r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await.unwrap();
            body
        });
        let cfg = Config {
            backend: Backend::Custom,
            base_url: Some(format!("http://{address}/v1")),
            model: "raw-model".into(),
            reasoning_effort: Some("high".into()),
            ..Config::default()
        };
        let client = LlmClient::new(&cfg).unwrap();
        client
            .complete(
                &[Message::user("hello")],
                &[],
                &mut |_| {},
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let body = server.await.unwrap();
        assert_eq!(body["model"], "raw-model");
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn typed_http_errors_and_provider_hint() {
        let body = "upstream backend error (see the mistl daemon log)";
        let err = http_error(Backend::Mistl, 502, body.into());
        let typed = err.downcast_ref::<HttpError>().unwrap();
        assert_eq!(typed.status, 502);
        assert_eq!(typed.body, body);
        let text = err.to_string();
        assert!(text.contains("HTTP 502"));
        assert!(text.contains(body));
        assert!(text.contains("no AI provider reachable"));
        assert!(text.contains("mistl ai status"));
        assert!(!is_tools_unsupported(&err));
        let custom = http_error(Backend::Custom, 502, body.into());
        assert!(!custom.to_string().contains("mistl ai status"));
    }

    #[test]
    fn tools_unsupported_requires_typed_400_and_full_body() {
        let body = format!(
            "{}{}",
            "x".repeat(600),
            r#"{"error":{"code":"tools_unsupported"}}"#
        );
        let err = http_error(Backend::Mistl, 400, body.clone()).context("completion failed");
        assert!(is_tools_unsupported(&err));
        assert_eq!(err.downcast_ref::<HttpError>().unwrap().body, body);
        assert!(!is_tools_unsupported(&http_error(
            Backend::Mistl,
            502,
            body
        )));
        assert!(!is_tools_unsupported(&http_error(
            Backend::Mistl,
            400,
            "invalid request".into()
        )));
        assert!(!is_tools_unsupported(&anyhow!(
            "HTTP 400: tools_unsupported"
        )));
    }

    fn run(chunks: &[&[u8]]) -> (AssistantTurn, Vec<String>) {
        let mut deltas = Vec::new();
        let mut p = SseParser::default();
        let mut cb = |d: &str| deltas.push(d.to_string());
        for c in chunks {
            p.feed(c, &mut cb);
        }
        let t = p.finish(&mut cb);
        (t, deltas)
    }

    #[test]
    fn split_lines_and_content() {
        let (t, d) = run(&[
            b": comment\n\nevent: x\ndata: {\"choices\":[{\"delta\":{\"con",
            b"tent\":\"Hel\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\",\"reasoning_content\":\"zzz\"}}]}\n",
            b"data: [DONE]\n",
        ]);
        assert_eq!(t.content, "Hello");
        assert_eq!(d, vec!["Hel", "lo"]);
        assert!(t.tool_calls.is_empty());
    }

    #[test]
    fn split_utf8_multibyte() {
        let line = "data: {\"choices\":[{\"delta\":{\"content\":\"\u{3042}\"}}]}\n".as_bytes();
        let cut = line.len() - 12;
        let (t, _) = run(&[&line[..cut], &line[cut..]]);
        assert_eq!(t.content, "\u{3042}");
    }

    #[test]
    fn tool_call_merge() {
        let (t, _) = run(&[
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"abc\",\"function\":{\"name\":\"mistl\",\"arguments\":\"\"}}]}}]}\n",
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"args\\\":\"}}]}}]}\n",
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"[\\\"ls\\\"]}\"}},{\"index\":1,\"function\":{\"name\":\"mistl_help\"}}]}}]}\n",
            b"data: [DONE]\n",
        ]);
        assert_eq!(t.tool_calls.len(), 2);
        assert_eq!(t.tool_calls[0].id, "abc");
        assert_eq!(t.tool_calls[0].function.name, "mistl");
        assert_eq!(t.tool_calls[0].function.arguments, "{\"args\":[\"ls\"]}");
        assert_eq!(t.tool_calls[1].id, "call_1");
        assert_eq!(t.tool_calls[1].function.arguments, "{}");
    }

    #[test]
    fn no_trailing_newline_before_finish() {
        let (t, _) = run(&[b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}"]);
        assert_eq!(t.content, "x");
    }

    #[test]
    fn full_json_fallback() {
        let body = br#"{"choices":[{"message":{"content":"hi","tool_calls":[{"id":"t1","type":"function","function":{"name":"mistl","arguments":"{}"}}]}}]}"#;
        let mut got = String::new();
        let t = parse_full_response(body, &mut |d| got.push_str(d)).unwrap();
        assert_eq!(got, "hi");
        assert_eq!(t.tool_calls[0].id, "t1");
    }
}
