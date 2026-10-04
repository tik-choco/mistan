# mistan

**mistan** is a terminal UI (TUI) AI agent for [mistl](https://github.com/tik-choco/mistl),
the tik-choco ecosystem daemon. Ask in natural language — "is the AI network up?",
"list my stored files", "share this folder" — and mistan runs the matching `mistl`
CLI commands for you, then summarizes the results.

- Uses the **mistl AI network** by default through its local API, started automatically
  with `mistl ai serve start`. No API key is needed; custom OpenAI-compatible endpoints are optional
- Two tool-calling styles: **native** (OpenAI `tools`) and **prompt** (fenced
  `mistl` blocks). The mistl backend defaults to native and automatically falls
  back to prompt mode when the network provider does not support tools
- **Safe by default**: read-only queries (`status`, `store ls`, `config show`, …) run
  directly. Commands that change state ask first (`y` yes / `n` no / `a` always this
  session). Interactive or never-ending commands (`daemon run`, `tunnel tui`, …) are refused
- **Project aware**: the directory you start mistan in is its workspace. When it has a
  `justfile`, the model sees its recipes and runs them (`just build`, `just test`, …),
  including long-running ones such as dev servers in the background. It can also list,
  find, grep and read files in the workspace (e.g. "where does this build to?") and run
  shell commands. Recipes and shell commands always ask first
- `!<command>` runs a shell command yourself (no approval); its output is shared with the
  model on your next message
- Streaming output (including live command output), Japanese/CJK-aware wrapping, cancel with `Esc`

## Requirements

- Rust (edition 2024)
- `just` is optional; with it on `PATH` (or `just_bin` set), justfile recipes become tools.
- `mistl` is optional. mistan finds it on `PATH`, at the per-user install location, or via
  `--mistl <path>`. Without it mistan still works with an OpenAI-compatible API, using
  only the workspace tools (no mistl tools). `/mistl install` (or `mistan --install-mistl`)
  downloads the latest release from GitHub, verifies it against the release's
  `SHA256SUMS.txt`, and installs it per user (the checksum list is not signature-checked).
  When mistl is present but its daemon is down, mistan starts it automatically.

## Build

```console
$ cargo build --release
$ ./target/release/mistan
```

## Usage

```console
$ mistan                                  # TUI, uses the mistl AI network
$ mistan --base-url https://example.invalid/v1 --model model-id --tool-mode native
$ mistan -p "Is the AI network up?"         # one-shot, no TUI (state changes refused unless --yes)
$ mistan --list-models                      # print the endpoint's model ids
$ mistan --install-mistl                    # download and install mistl
```

mistan starts the local API automatically on the first turn and discovers its address
from the selected mistl instance. The network needs a reachable AI provider, for example
a node running `mistl ai provide start`. If none is reachable, check `mistl ai status`.

| Key | Action |
| --- | --- |
| `Enter` | Send |
| `Alt+Enter` / `Ctrl+J` | New line |
| `Esc` | Cancel the running turn |
| `PgUp` / `PgDn` | Scroll |
| `Ctrl+O` | Expand/collapse tool output |
| `Ctrl+C` | Cancel, or quit when idle |

### Workspace tools

| Tool | What it does | Approval |
| --- | --- | --- |
| `just` | Run a justfile recipe (`background: true` for servers/apps) | Yes |
| `shell` | Run a shell command in the workspace | Yes |
| `list_dir`, `find_files`, `grep`, `read_file` | Inspect files inside the workspace (including gitignored build output on request) | No |
| `process_list`, `process_output`, `process_stop` | Inspect or stop background processes started this session | Stop only |

File tools cannot reach outside the startup directory. Commands get no stdin, so
interactive programs should run in the background. Background processes are stopped when
mistan exits.

Commands:

Typing `/` opens a completion list of commands (and their arguments, such as effort levels
and fetched model IDs). Use `Up`/`Down` to select, `Tab` to complete, `Enter` to run, and
`Esc` to hide it.

| Command | Action |
| --- | --- |
| `/model`, `/models` | Fetch the selected connection's model list (`GET <base_url>/models`) and pick one; type to filter. `/model <id>` sets a raw model id directly |
| `/effort [level]` | Pick or set `reasoning_effort` (`default`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh`) for this session |
| `/settings` | Edit backend, HTTP connections, default model, reasoning effort, and tool mode. Ctrl+N adds a connection, Left/Right on Connection selects one, Enabled toggles it, Ctrl+D twice deletes it. Ctrl+S saves and starts a new conversation |
| `/mistl [install\|start]` | Show where mistl is, install it, or start its daemon |
| `/just` | List the workspace's justfile recipes |
| `!<command>` | Run a shell command in the workspace (PowerShell on Windows, `sh` elsewhere); `!` also completes `!just <recipe>`. Esc stops it |
| `/clear`, `/help`, `/quit` | As named |

`reasoning_effort` is sent to OpenAI-compatible APIs. On the mistl AI network the provider
chooses it, so the value is ignored there. `/settings` writes the API key in plain text to
`config.toml` (owner-only on Unix); prefer `MISTAN_API_KEY`, which is never copied into the file.
Saving rewrites `config.toml` without preserving comments.

## Configuration

Settings come from built-in defaults, then the config file, then environment
variables, then command-line flags. The config file lives outside the repository:

- Windows: `%APPDATA%\mistan\config.toml`
- Linux: `~/.config/mistan/config.toml`
- macOS: `~/Library/Application Support/mistan/config.toml`

See [`config.example.toml`](config.example.toml) for every field.

HTTP connections are stored in `providers` with `id`, `label`, `base_url`, `api_key`,
`enabled` (default true), and optional `models`/`models_fetched_at` cache. A custom
backend selects `default_ref = { provider_id, model }`. Disabling or deleting a
connection keeps the reference and reports an error until it is usable or you
explicitly select another connection. Array order never determines the default.

Setting `default_ref`, `MISTAN_BASE_URL`, or `--base-url` selects a custom
endpoint unless the file explicitly sets `backend`. `backend = "mistl"` always uses
the network, using native tool calling when the provider advertises `"tools"`.
mistl providers advertise this capability; providers without it trigger an automatic
fallback to prompt mode for the rest of the session. Set `tool_mode = "prompt"`
(or `--tool-mode prompt`) to force prompt mode. `backend = "custom"` requires a URL
and defaults to prompt mode.
For a custom endpoint, add an HTTP provider, set `default_ref`, and use
`tool_mode = "native"` when it supports tools; supply its key through `MISTAN_API_KEY`.

On first load, the old flat `base_url`/`api_key`/`model` settings are migrated to
a provider and reference (or `network_model` for the mistl backend). The migration
is saved once, keeps other options, and runs before environment/CLI overrides.
Old `temperature` is removed; chat requests never send it. There are no presets.
Saving or migration rewrites the file without preserving comments.

Room providers and sharing stay in mistl's config: `ai.providers` supports
`mist-network://<room>`, `enabled`, `models`, `provide`, and `shared`; defaults and
voice use `ai.default_ref`, `ai.tts`, and `ai.stt`. `ai.status` includes `rooms[]`.
mistan forwards CLI results without imposing a config/status schema, and its
OpenAI API client uses raw model ids, so older mistl versions with `ai serve start`
and `/v1/models` remain usable. New Room configuration requires a mistl build
implementing the provider/room contract; mistan does not edit daemon config itself.

New connection controls support English, Japanese, and Chinese; select their
language with `MISTAN_LANGUAGE=en`, `ja`, or `zh` (otherwise the process locale).

| Environment variable | Meaning |
| --- | --- |
| `MISTAN_API_KEY` (or `OPENAI_API_KEY`) | Bearer token for the chat endpoint |
| `MISTAN_BASE_URL` | OpenAI-compatible base URL |
| `MISTAN_MODEL` | Model id |
| `MISTAN_REASONING_EFFORT` | `reasoning_effort` value |
| `MISTAN_MISTL` | mistl executable |

Use environment variables for API keys. Never commit them; `.gitignore` already
excludes `config.toml`, `.env*`, and key files.

## License

[MPL-2.0](LICENSE)
