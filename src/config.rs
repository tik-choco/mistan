//! Configuration: built-in defaults < `config.toml` < environment < CLI flags.
//!
//! The config file lives in the per-user config dir (never in the repo):
//! `%APPDATA%\mistan\config.toml` / `~/.config/mistan/config.toml`.
//! The API key is best supplied via `MISTAN_API_KEY` (or `OPENAI_API_KEY`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::types::ToolMode;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Mistl,
    Custom,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub backend: Backend,
    /// Custom OpenAI-compatible base URL, e.g. `https://api.openai.com/v1`.
    pub base_url: Option<String>,
    /// Model id; empty means "let the server choose" (field omitted).
    pub model: String,
    pub api_key: Option<String>,
    pub tool_mode: ToolMode,
    /// OpenAI `reasoning_effort` (`low`, `medium`, `high`, ...); omitted when unset.
    pub reasoning_effort: Option<String>,
    pub temperature: Option<f32>,
    /// mistl executable (name on PATH or a full path).
    pub mistl_bin: String,
    /// Passed as `mistl --instance <name>` to target a specific instance.
    pub mistl_instance: Option<String>,
    /// Per-invocation timeout for mistl commands.
    pub mistl_timeout_secs: u64,
    /// Max model round-trips per user message.
    pub max_steps: usize,
    /// Run mutating mistl commands without asking.
    pub auto_approve: bool,
    /// Where `/settings` saves changes (the `--config` file or the default path).
    pub config_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Mistl,
            base_url: None,
            model: String::new(),
            api_key: None,
            // Native tools on the AI network, with session fallback to prompt mode.
            tool_mode: ToolMode::Native,
            reasoning_effort: None,
            temperature: None,
            mistl_bin: "mistl".into(),
            mistl_instance: None,
            mistl_timeout_secs: 60,
            max_steps: 12,
            auto_approve: false,
            config_path: None,
        }
    }
}

/// `config.toml` shape; every field optional.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    backend: Option<Backend>,
    base_url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    tool_mode: Option<ToolMode>,
    reasoning_effort: Option<String>,
    temperature: Option<f32>,
    mistl_bin: Option<String>,
    mistl_instance: Option<String>,
    mistl_timeout_secs: Option<u64>,
    max_steps: Option<usize>,
    auto_approve: Option<bool>,
}

/// Overrides from the command line (already merged with env by clap).
#[derive(Debug, Default)]
pub struct Overrides {
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub tool_mode: Option<ToolMode>,
    pub reasoning_effort: Option<String>,
    pub mistl_bin: Option<String>,
    pub mistl_instance: Option<String>,
    pub auto_approve: bool,
}

pub fn default_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("mistan").join("config.toml"))
}

pub fn load(explicit: Option<&Path>, ov: Overrides) -> Result<Config> {
    let mut cfg = Config::default();
    let mut file_backend = None;
    let mut tool_mode = ov.tool_mode;

    let path = explicit.map(Path::to_path_buf).or_else(default_path);
    cfg.config_path = path.clone();
    if let Some(path) = path.filter(|p| explicit.is_some() || p.exists()) {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let file: FileConfig =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        file_backend = file.backend;
        tool_mode = tool_mode.or(file.tool_mode);
        apply_file(&mut cfg, file);
    }

    if let Some(key) = env_nonempty("MISTAN_API_KEY").or_else(|| env_nonempty("OPENAI_API_KEY")) {
        cfg.api_key = Some(key);
    }

    if let Some(v) = ov.base_url.or_else(|| env_nonempty("MISTAN_BASE_URL")) {
        cfg.base_url = Some(v);
    }
    if let Some(v) = ov.model {
        cfg.model = v;
    }
    if let Some(v) = ov.tool_mode {
        cfg.tool_mode = v;
    }
    if let Some(v) = ov
        .reasoning_effort
        .or_else(|| env_nonempty("MISTAN_REASONING_EFFORT"))
    {
        cfg.reasoning_effort = Some(v);
    }
    cfg.reasoning_effort = normalize_effort(cfg.reasoning_effort.take());
    if let Some(v) = ov.mistl_bin {
        cfg.mistl_bin = v;
    }
    if let Some(v) = ov.mistl_instance {
        cfg.mistl_instance = Some(v);
    }
    cfg.auto_approve |= ov.auto_approve;

    select_backend(&mut cfg, file_backend, tool_mode)?;
    cfg.mistl_bin = resolve_mistl_bin(&cfg.mistl_bin);
    Ok(cfg)
}

fn select_backend(
    cfg: &mut Config,
    explicit: Option<Backend>,
    tool_mode: Option<ToolMode>,
) -> Result<()> {
    cfg.backend = explicit.unwrap_or_else(|| {
        if cfg.base_url.is_some() {
            Backend::Custom
        } else {
            Backend::Mistl
        }
    });
    cfg.base_url = cfg
        .base_url
        .take()
        .map(|url| url.trim().trim_end_matches('/').to_string());
    match cfg.backend {
        Backend::Custom if cfg.base_url.as_deref().is_none_or(str::is_empty) => {
            bail!("custom backend requires base_url (config, MISTAN_BASE_URL, or --base-url)");
        }
        _ => {}
    }
    cfg.tool_mode = tool_mode.unwrap_or(default_tool_mode(cfg.backend));
    Ok(())
}

fn apply_file(cfg: &mut Config, f: FileConfig) {
    macro_rules! take {
        ($($field:ident),*) => { $( if let Some(v) = f.$field { cfg.$field = v; } )* };
    }
    take!(
        model,
        tool_mode,
        mistl_bin,
        mistl_timeout_secs,
        max_steps,
        auto_approve
    );
    if f.base_url.is_some() {
        cfg.base_url = f.base_url;
    }
    if f.api_key.as_deref().is_some_and(|k| !k.is_empty()) {
        cfg.api_key = f.api_key;
    }
    if f.reasoning_effort.is_some() {
        cfg.reasoning_effort = f.reasoning_effort;
    }
    if f.temperature.is_some() {
        cfg.temperature = f.temperature;
    }
    if f.mistl_instance.is_some() {
        cfg.mistl_instance = f.mistl_instance;
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Normalize a user-supplied reasoning effort: trimmed, lowercase; empty or
/// `default` means "do not send the field".
pub fn normalize_effort(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty() && s != "default")
}

/// A bare `mistl` that is not on PATH falls back to the per-user install
/// location (`mistl install` on Windows, `~/.local/bin` elsewhere).
pub fn resolve_mistl_bin(bin: &str) -> String {
    if bin != "mistl" || on_path(bin) {
        return bin.to_string();
    }
    if let Some(installed) = crate::install::install_path().filter(|p| p.is_file()) {
        return installed.to_string_lossy().into_owned();
    }
    bin.to_string()
}

/// Whether `bin` (a path or a name on PATH) points at an existing file.
pub fn mistl_available(bin: &str) -> bool {
    let p = Path::new(bin);
    if p.is_absolute() || p.components().count() > 1 {
        p.is_file()
    } else {
        on_path(bin)
    }
}

/// The LLM-related settings the user can change at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmSettings {
    pub backend: Backend,
    /// Custom OpenAI-compatible base URL (kept even while the backend is mistl).
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub tool_mode: ToolMode,
}

impl LlmSettings {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            backend: cfg.backend,
            base_url: cfg.base_url.clone().unwrap_or_default(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            reasoning_effort: cfg.reasoning_effort.clone(),
            tool_mode: cfg.tool_mode,
        }
    }

    /// Validate and apply to `cfg`.
    pub fn apply(&self, cfg: &mut Config) -> Result<()> {
        let url = self.base_url.trim().trim_end_matches('/').to_string();
        if self.backend == Backend::Custom && url.is_empty() {
            bail!("the OpenAI-compatible API needs a base URL");
        }
        cfg.backend = self.backend;
        cfg.base_url = (!url.is_empty()).then_some(url);
        cfg.api_key = self.api_key.clone().filter(|k| !k.trim().is_empty());
        cfg.model = self.model.trim().to_string();
        cfg.reasoning_effort = normalize_effort(self.reasoning_effort.clone());
        cfg.tool_mode = self.tool_mode;
        Ok(())
    }
}

/// Default tool mode for a backend when the user has not chosen one.
pub fn default_tool_mode(backend: Backend) -> ToolMode {
    match backend {
        Backend::Mistl => ToolMode::Native,
        Backend::Custom => ToolMode::Prompt,
    }
}

/// Persist `s` into the config file (created if missing), keeping unrelated
/// keys. Comments in an existing file are not preserved. The file is written
/// owner-only on Unix because it may hold an API key.
pub fn save_settings(path: &Path, s: &LlmSettings) -> Result<()> {
    let mut table: toml::Table = match std::fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => toml::Table::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let backend = match s.backend {
        Backend::Mistl => "mistl",
        Backend::Custom => "custom",
    };
    let tool_mode = match s.tool_mode {
        ToolMode::Native => "native",
        ToolMode::Prompt => "prompt",
    };
    table.insert("backend".into(), backend.into());
    table.insert("tool_mode".into(), tool_mode.into());
    table.insert("model".into(), s.model.trim().into());
    let url = s.base_url.trim().trim_end_matches('/');
    if url.is_empty() {
        table.remove("base_url");
    } else {
        table.insert("base_url".into(), url.into());
    }
    match normalize_effort(s.reasoning_effort.clone()) {
        Some(e) => table.insert("reasoning_effort".into(), e.into()),
        None => table.remove("reasoning_effort"),
    };
    // An empty key never erases a stored one by accident; clearing it is done
    // by editing the file.
    // A key that came from the environment is never copied into the file.
    let env_key = env_nonempty("MISTAN_API_KEY").or_else(|| env_nonempty("OPENAI_API_KEY"));
    if let Some(k) = s
        .api_key
        .as_deref()
        .filter(|k| !k.trim().is_empty() && env_key.as_deref() != Some(*k))
    {
        table.insert("api_key".into(), k.into());
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let text = toml::to_string_pretty(&table)?;
    let tmp = path.with_extension("toml.tmp");
    write_private(&tmp, text.as_bytes())?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    // Start from a fresh file so an old, more permissive one is never reused.
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("removing {}", path.display())),
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts
        .open(path)
        .with_context(|| format!("writing {}", path.display()))?;
    f.write_all(bytes)?;
    Ok(())
}

fn on_path(bin: &str) -> bool {
    let exts: &[&str] = if cfg!(windows) { &["", ".exe"] } else { &[""] };
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| {
            exts.iter()
                .any(|ext| dir.join(format!("{bin}{ext}")).is_file())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_selection() {
        let mut cfg = Config::default();
        select_backend(&mut cfg, None, None).unwrap();
        assert_eq!(cfg.backend, Backend::Mistl);
        assert_eq!(cfg.tool_mode, ToolMode::Native);
        assert!(cfg.base_url.is_none());

        cfg.base_url = Some("https://example.invalid/v1/".into());
        select_backend(&mut cfg, None, None).unwrap();
        assert_eq!(cfg.backend, Backend::Custom);
        assert_eq!(cfg.base_url.as_deref(), Some("https://example.invalid/v1"));
        assert_eq!(cfg.tool_mode, ToolMode::Prompt);

        select_backend(&mut cfg, Some(Backend::Mistl), None).unwrap();
        assert_eq!(cfg.backend, Backend::Mistl);
        assert_eq!(cfg.tool_mode, ToolMode::Native);

        for backend in [Backend::Mistl, Backend::Custom] {
            for mode in [ToolMode::Prompt, ToolMode::Native] {
                select_backend(&mut cfg, Some(backend), Some(mode)).unwrap();
                assert_eq!(cfg.tool_mode, mode);
            }
        }
    }

    #[test]
    fn custom_requires_url() {
        for url in [None, Some("   ".into())] {
            let mut cfg = Config {
                base_url: url,
                ..Config::default()
            };
            assert!(
                select_backend(&mut cfg, Some(Backend::Custom), None)
                    .unwrap_err()
                    .to_string()
                    .contains("custom backend requires base_url")
            );
        }
    }

    #[test]
    fn file_backend_wins_over_url() {
        for (text, expected) in [
            ("backend = 'mistl'", Backend::Mistl),
            ("backend = 'mistl'\ntool_mode = 'prompt'", Backend::Mistl),
            ("base_url = 'https://example.invalid/v1'", Backend::Custom),
            (
                "backend = 'mistl'\nbase_url = 'https://example.invalid/v1'\ntool_mode = 'native'",
                Backend::Mistl,
            ),
            (
                "backend = 'custom'\nbase_url = 'https://example.invalid/v1'",
                Backend::Custom,
            ),
        ] {
            let file: FileConfig = toml::from_str(text).unwrap();
            let explicit = file.backend;
            let tool_mode = file.tool_mode;
            let mut cfg = Config::default();
            apply_file(&mut cfg, file);
            select_backend(&mut cfg, explicit, tool_mode).unwrap();
            assert_eq!(cfg.backend, expected);
            let default_mode = match expected {
                Backend::Mistl => ToolMode::Native,
                Backend::Custom => ToolMode::Prompt,
            };
            assert_eq!(cfg.tool_mode, tool_mode.unwrap_or(default_mode));
            for flag_mode in [ToolMode::Native, ToolMode::Prompt] {
                select_backend(&mut cfg, explicit, Some(flag_mode).or(tool_mode)).unwrap();
                assert_eq!(cfg.tool_mode, flag_mode);
            }
        }
    }

    #[test]
    fn save_settings_round_trip_keeps_other_keys() {
        let dir = std::env::temp_dir().join(format!("mistan-cfg-test-{}", std::process::id()));
        let path = dir.join("config.toml");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "max_steps = 5\n").unwrap();
        let s = LlmSettings {
            backend: Backend::Custom,
            base_url: "https://example.invalid/v1/".into(),
            api_key: Some("file-key-not-from-env".into()),
            model: " m1 ".into(),
            reasoning_effort: Some("High".into()),
            tool_mode: ToolMode::Native,
        };
        save_settings(&path, &s).unwrap();
        let file: FileConfig = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(file.max_steps, Some(5));
        assert_eq!(file.backend, Some(Backend::Custom));
        assert_eq!(file.base_url.as_deref(), Some("https://example.invalid/v1"));
        assert_eq!(file.model.as_deref(), Some("m1"));
        assert_eq!(file.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(file.api_key.as_deref(), Some("file-key-not-from-env"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn effort_normalization_and_apply_validation() {
        assert_eq!(normalize_effort(Some(" Default ".into())), None);
        assert_eq!(normalize_effort(Some(" LOW ".into())), Some("low".into()));
        let mut cfg = Config::default();
        let mut s = LlmSettings::from_config(&cfg);
        s.backend = Backend::Custom;
        assert!(s.apply(&mut cfg).is_err());
        s.base_url = "http://x/v1/".into();
        s.apply(&mut cfg).unwrap();
        assert_eq!(cfg.base_url.as_deref(), Some("http://x/v1"));
    }
}
