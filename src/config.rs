//! Configuration: built-in defaults < `config.toml` < environment < CLI flags.
//!
//! The config file lives in the per-user config dir (never in the repo):
//! `%APPDATA%\mistan\config.toml` / `~/.config/mistan/config.toml`.
//! The API key is best supplied via `MISTAN_API_KEY` (or `OPENAI_API_KEY`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::types::ToolMode;

#[path = "llm_text.rs"]
pub mod text;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Mistl,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRef {
    pub provider_id: String,
    pub model: String,
}

/// Local HTTP connections. Room membership and sharing belong to mistl.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Provider {
    pub id: String,
    pub label: String,
    pub base_url: String,
    pub api_key: String,
    pub enabled: bool,
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models_fetched_at: Option<String>,
}

impl Default for Provider {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            enabled: true,
            models: Vec::new(),
            models_fetched_at: None,
        }
    }
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
    pub providers: Vec<Provider>,
    pub default_ref: Option<ModelRef>,
    /// Raw model id sent to mistl's API; empty lets mistl choose its default.
    pub network_model: String,
    /// mistl executable (name on PATH or a full path).
    pub mistl_bin: String,
    /// Passed as `mistl --instance <name>` to target a specific instance.
    pub mistl_instance: Option<String>,
    /// Per-invocation timeout for mistl commands.
    pub mistl_timeout_secs: u64,
    /// `just` executable (name on PATH or a full path) for justfile recipes.
    pub just_bin: String,
    /// Timeout for foreground recipes and shell commands run by the agent
    /// (user `!` commands have none; Esc cancels them).
    pub command_timeout_secs: u64,
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
            providers: Vec::new(),
            default_ref: None,
            network_model: String::new(),
            mistl_bin: "mistl".into(),
            mistl_instance: None,
            mistl_timeout_secs: 60,
            just_bin: "just".into(),
            command_timeout_secs: 600,
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
    providers: Option<Vec<Provider>>,
    default_ref: Option<ModelRef>,
    network_model: Option<String>,
    mistl_bin: Option<String>,
    mistl_instance: Option<String>,
    mistl_timeout_secs: Option<u64>,
    just_bin: Option<String>,
    command_timeout_secs: Option<u64>,
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
        let _: FileConfig =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let mut table: toml::Table =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if migrate_file(&mut table)? {
            // Persist before applying environment/CLI overrides or keys.
            write_table(&path, &table)?;
        }
        let file: FileConfig = table
            .try_into()
            .with_context(|| format!("parsing {}", path.display()))?;
        file_backend = file.backend;
        tool_mode = tool_mode.or(file.tool_mode);
        apply_file(&mut cfg, file);
    }

    cfg.backend = file_backend.unwrap_or(if cfg.default_ref.is_some() {
        Backend::Custom
    } else {
        Backend::Mistl
    });
    cfg.resolve_default()?;

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
    cfg.capture_selection();
    cfg.mistl_bin = resolve_mistl_bin(&cfg.mistl_bin);
    Ok(cfg)
}

fn select_backend(
    cfg: &mut Config,
    explicit: Option<Backend>,
    tool_mode: Option<ToolMode>,
) -> Result<()> {
    cfg.backend = explicit.unwrap_or_else(|| {
        if cfg.base_url.is_some() || cfg.default_ref.is_some() {
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
        Backend::Custom
            if cfg.base_url.as_deref().is_none_or(str::is_empty) && cfg.default_ref.is_none() =>
        {
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
        just_bin,
        command_timeout_secs,
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
    // Legacy temperature is read only; never sent or saved.
    let _ = f.temperature;
    cfg.providers = f.providers.unwrap_or_default();
    cfg.default_ref = f.default_ref;
    cfg.network_model = f.network_model.unwrap_or_default();
    if f.mistl_instance.is_some() {
        cfg.mistl_instance = f.mistl_instance;
    }
}

impl Config {
    fn resolve_default(&mut self) -> Result<()> {
        if self.backend == Backend::Mistl {
            self.model = self.network_model.clone();
            return Ok(());
        }
        if let Some(reference) = &self.default_ref {
            self.model = reference.model.clone();
            let Some(p) = self
                .providers
                .iter()
                .find(|p| p.id == reference.provider_id && p.enabled)
            else {
                self.base_url = None;
                self.api_key = None;
                return Ok(());
            };
            if p.base_url.starts_with("mist-network://") {
                bail!("{}", text::get("rooms_in_mistl"));
            }
            self.base_url = Some(p.base_url.clone());
            self.api_key = (!p.api_key.is_empty()).then(|| p.api_key.clone());
            self.model = reference.model.clone();
        }
        Ok(())
    }

    /// Record the selected model without replacing a stored provider reference.
    pub fn set_model(&mut self, model: String) {
        self.model = model;
        if self.backend == Backend::Mistl {
            self.network_model = self.model.clone();
        } else if let Some(reference) = &mut self.default_ref {
            reference.model = self.model.clone();
        }
    }

    fn capture_selection(&mut self) {
        if self.backend == Backend::Mistl {
            self.network_model = self.model.clone();
            return;
        }
        let id = self.default_ref.as_ref().map(|r| r.provider_id.clone());
        if let Some(p) = self
            .providers
            .iter_mut()
            .find(|p| Some(&p.id) == id.as_ref())
        {
            if let Some(url) = &self.base_url {
                p.base_url = url.clone();
            }
        } else {
            let id = unique_provider_id(
                &self.providers,
                self.default_ref.as_ref().map(|r| r.provider_id.as_str()),
            );
            self.providers.push(Provider {
                id: id.clone(),
                label: "HTTP".into(),
                base_url: self.base_url.clone().unwrap_or_default(),
                ..Provider::default()
            });
            self.default_ref = Some(ModelRef {
                provider_id: id,
                model: self.model.clone(),
            });
        }
        if let Some(reference) = &mut self.default_ref {
            reference.model = self.model.clone();
        }
    }
}

pub fn unique_provider_id(providers: &[Provider], reserved: Option<&str>) -> String {
    (1..)
        .map(|i| format!("http-{i}"))
        .find(|id| reserved != Some(id.as_str()) && providers.iter().all(|p| p.id != *id))
        .unwrap()
}

/// Upgrade only the old flat connection; never infer a default from array order.
fn migrate_file(table: &mut toml::Table) -> Result<bool> {
    let before = table.clone();
    if !table.contains_key("providers") {
        let url = table
            .get("base_url")
            .and_then(toml::Value::as_str)
            .unwrap_or("")
            .trim();
        let model = table
            .get("model")
            .and_then(toml::Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let custom = table
            .get("backend")
            .and_then(toml::Value::as_str)
            .map_or(!url.is_empty(), |b| b == "custom");
        let mut providers = Vec::new();
        if !url.is_empty() {
            providers.push(Provider {
                id: "http-1".into(),
                label: "HTTP".into(),
                base_url: url.trim_end_matches('/').into(),
                api_key: table
                    .get("api_key")
                    .and_then(toml::Value::as_str)
                    .unwrap_or("")
                    .into(),
                models: (!model.is_empty() && custom)
                    .then(|| model.to_string())
                    .into_iter()
                    .collect(),
                ..Provider::default()
            });
        }
        if custom && !providers.is_empty() && !table.contains_key("default_ref") {
            table.insert(
                "default_ref".into(),
                toml::Value::try_from(ModelRef {
                    provider_id: "http-1".into(),
                    model: model.clone(),
                })?,
            );
        }
        if !custom && !model.is_empty() && !table.contains_key("network_model") {
            table.insert("network_model".into(), model.into());
        }
        table.insert("providers".into(), toml::Value::try_from(providers)?);
    }
    for key in ["base_url", "api_key", "model", "temperature"] {
        table.remove(key);
    }
    Ok(*table != before)
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
    pub providers: Vec<Provider>,
    pub default_ref: Option<ModelRef>,
    pub network_model: String,
}

impl LlmSettings {
    pub fn from_config(cfg: &Config) -> Self {
        let provider = cfg
            .default_ref
            .as_ref()
            .and_then(|r| cfg.providers.iter().find(|p| p.id == r.provider_id));
        Self {
            backend: cfg.backend,
            base_url: cfg
                .base_url
                .clone()
                .or_else(|| provider.map(|p| p.base_url.clone()))
                .unwrap_or_default(),
            api_key: cfg.api_key.clone().or_else(|| {
                provider
                    .filter(|p| !p.api_key.is_empty())
                    .map(|p| p.api_key.clone())
            }),
            model: cfg.model.clone(),
            reasoning_effort: cfg.reasoning_effort.clone(),
            tool_mode: cfg.tool_mode,
            providers: cfg.providers.clone(),
            default_ref: cfg.default_ref.clone(),
            network_model: cfg.network_model.clone(),
        }
    }

    /// Validate and apply to `cfg`.
    pub fn apply(&self, cfg: &mut Config) -> Result<()> {
        let url = self.base_url.trim().trim_end_matches('/').to_string();
        if self.backend == Backend::Custom && url.is_empty() && self.default_ref.is_none() {
            bail!("the OpenAI-compatible API needs a base URL");
        }
        cfg.backend = self.backend;
        cfg.base_url = (!url.is_empty()).then_some(url);
        cfg.api_key = self.api_key.clone().filter(|k| !k.trim().is_empty());
        cfg.model = self.model.trim().to_string();
        cfg.reasoning_effort = normalize_effort(self.reasoning_effort.clone());
        cfg.tool_mode = self.tool_mode;
        let canonical = self.canonical()?;
        cfg.providers = canonical.providers;
        cfg.default_ref = canonical.default_ref;
        cfg.network_model = canonical.network_model;
        cfg.resolve_default()?;
        // Keys from the environment are runtime-only.
        if let Some(key) = env_nonempty("MISTAN_API_KEY").or_else(|| env_nonempty("OPENAI_API_KEY"))
        {
            cfg.api_key = Some(key);
        }
        Ok(())
    }

    pub fn selected_provider(&self) -> Option<&Provider> {
        let id = &self.default_ref.as_ref()?.provider_id;
        self.providers.iter().find(|p| &p.id == id)
    }

    /// Commit the form's fields to its selected connection only.
    pub fn canonical(&self) -> Result<Self> {
        let mut s = self.clone();
        let env_key = env_nonempty("MISTAN_API_KEY").or_else(|| env_nonempty("OPENAI_API_KEY"));
        if s.backend == Backend::Mistl {
            s.network_model = s.model.trim().into();
            return Ok(s);
        }
        if s.default_ref.is_none() {
            let id = unique_provider_id(&s.providers, None);
            s.providers.push(Provider {
                id: id.clone(),
                label: "HTTP".into(),
                ..Provider::default()
            });
            s.default_ref = Some(ModelRef {
                provider_id: id,
                model: s.model.trim().into(),
            });
        }
        let reference = s.default_ref.as_mut().unwrap();
        reference.model = s.model.trim().into();
        if let Some(p) = s
            .providers
            .iter_mut()
            .find(|p| p.id == reference.provider_id)
        {
            if !s.base_url.trim().is_empty() {
                p.base_url = s.base_url.trim().trim_end_matches('/').into();
            }
            if let Some(key) = s
                .api_key
                .as_deref()
                .filter(|k| env_key.as_deref() != Some(*k))
            {
                p.api_key = key.into();
            }
            if p.base_url.starts_with("mist-network://") {
                bail!("{}", text::get("rooms_in_mistl"));
            }
            if p.enabled
                && reqwest::Url::parse(&p.base_url).ok().is_none_or(|u| {
                    !matches!(u.scheme(), "http" | "https") || u.host_str().is_none()
                })
            {
                bail!("{}", text::get("invalid_url"));
            }
        }
        Ok(s)
    }

    pub fn select_provider(&mut self, id: &str) {
        if let Some(p) = self.providers.iter().find(|p| p.id == id) {
            self.backend = Backend::Custom;
            self.base_url = p.base_url.clone();
            self.api_key = (!p.api_key.is_empty()).then(|| p.api_key.clone());
            let model = self
                .default_ref
                .as_ref()
                .filter(|r| r.provider_id == id)
                .map(|r| r.model.clone())
                .unwrap_or_default();
            self.model = model.clone();
            self.default_ref = Some(ModelRef {
                provider_id: id.into(),
                model,
            });
        }
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
    let s = s.canonical()?;
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
    for key in ["base_url", "api_key", "model", "temperature"] {
        table.remove(key);
    }
    table.insert("providers".into(), toml::Value::try_from(&s.providers)?);
    match &s.default_ref {
        Some(r) => {
            table.insert("default_ref".into(), toml::Value::try_from(r)?);
        }
        None => {
            table.remove("default_ref");
        }
    }
    table.insert("network_model".into(), s.network_model.as_str().into());
    match normalize_effort(s.reasoning_effort.clone()) {
        Some(e) => table.insert("reasoning_effort".into(), e.into()),
        None => table.remove("reasoning_effort"),
    };
    write_table(path, &table)
}

fn write_table(path: &Path, table: &toml::Table) -> Result<()> {
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
            ..LlmSettings::from_config(&Config::default())
        };
        save_settings(&path, &s).unwrap();
        let file: FileConfig = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(file.max_steps, Some(5));
        assert_eq!(file.backend, Some(Backend::Custom));
        let provider = &file.providers.as_ref().unwrap()[0];
        assert_eq!(provider.base_url, "https://example.invalid/v1");
        assert_eq!(file.default_ref.as_ref().unwrap().model, "m1");
        assert_eq!(file.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(provider.api_key, "file-key-not-from-env");
        assert!(file.base_url.is_none() && file.model.is_none() && file.api_key.is_none());
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

    #[test]
    fn flat_config_migration_is_idempotent_and_preserves_options() {
        let mut table: toml::Table = toml::from_str("base_url = 'https://example.invalid/v1/'\napi_key = 'legacy-test-key'\nmodel = 'raw-model'\ntemperature = 0.2\nreasoning_effort = 'high'\nmax_steps = 7\n").unwrap();
        assert!(migrate_file(&mut table).unwrap());
        assert!(!migrate_file(&mut table).unwrap());
        let file: FileConfig = table.clone().try_into().unwrap();
        let p = &file.providers.unwrap()[0];
        assert_eq!(p.id, "http-1");
        assert_eq!(p.api_key, "legacy-test-key");
        assert_eq!(p.models, ["raw-model"]);
        assert!(p.enabled);
        assert_eq!(
            file.default_ref.unwrap(),
            ModelRef {
                provider_id: p.id.clone(),
                model: "raw-model".into()
            }
        );
        assert_eq!(file.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(file.max_steps, Some(7));
        for key in ["base_url", "api_key", "model", "temperature"] {
            assert!(!table.contains_key(key));
        }
        assert!(
            !table.contains_key("backend"),
            "keep inferred backend overridable by environment"
        );
    }

    #[test]
    fn network_migration_keeps_raw_model_and_explicit_backend() {
        let mut table: toml::Table = toml::from_str("backend = 'mistl'\nbase_url = 'https://example.invalid/v1'\nmodel = 'network-model'\ntemperature = 0.9\n").unwrap();
        assert!(migrate_file(&mut table).unwrap());
        let file: FileConfig = table.try_into().unwrap();
        assert_eq!(file.backend, Some(Backend::Mistl));
        assert_eq!(file.network_model.as_deref(), Some("network-model"));
        assert!(file.default_ref.is_none());
        assert_eq!(file.providers.unwrap().len(), 1);
    }

    #[test]
    fn new_config_keeps_missing_and_disabled_references() {
        for enabled in [true, false] {
            let mut cfg = Config::default();
            let mut s = LlmSettings::from_config(&cfg);
            s.backend = Backend::Custom;
            s.providers = vec![Provider {
                id: "p".into(),
                base_url: "https://example.invalid/v1".into(),
                enabled,
                ..Provider::default()
            }];
            s.default_ref = Some(ModelRef {
                provider_id: "p".into(),
                model: "raw".into(),
            });
            s.model = "raw".into();
            s.apply(&mut cfg).unwrap();
            assert_eq!(cfg.default_ref, s.default_ref);
            assert_eq!(cfg.base_url.is_some(), enabled);
            s.providers.clear();
            s.apply(&mut cfg).unwrap();
            assert_eq!(cfg.default_ref, s.default_ref);
            assert!(cfg.base_url.is_none());
        }
    }

    #[test]
    fn load_migrates_file_once_before_session_overrides() {
        let dir =
            std::env::temp_dir().join(format!("mistan-migration-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "base_url = 'https://example.invalid/v1'\nmodel = 'stored'\ntemperature = 0.4\n",
        )
        .unwrap();
        let cfg = load(
            Some(&path),
            Overrides {
                model: Some("session".into()),
                ..Overrides::default()
            },
        )
        .unwrap();
        assert_eq!(cfg.backend, Backend::Custom);
        assert_eq!(cfg.model, "session");
        let saved = std::fs::read_to_string(&path).unwrap();
        let file: FileConfig = toml::from_str(&saved).unwrap();
        assert_eq!(file.default_ref.unwrap().model, "stored");
        load(Some(&path), Overrides::default()).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), saved);
        std::fs::write(&path, "unknown = true\ntemperature = 0.4\n").unwrap();
        let invalid = std::fs::read_to_string(&path).unwrap();
        assert!(load(Some(&path), Overrides::default()).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), invalid);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
