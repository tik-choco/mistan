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
- Streaming output, Japanese/CJK-aware wrapping, cancel with `Esc`

## Requirements

- Rust (edition 2024)
- `mistl` is optional. mistan finds it on `PATH`, at the per-user install location, or via
  `--mistl <path>`. Without it mistan still works as a plain chat client for an
  OpenAI-compatible API (no mistl tools). `/mistl install` (or `mistan --install-mistl`)
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
$ mistan --base-url https://api.openai.com/v1 --model gpt-4.1-mini --tool-mode native
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

Commands:

Typing `/` opens a completion list of commands (and their arguments, such as effort levels
and fetched model IDs). Use `Up`/`Down` to select, `Tab` to complete, `Enter` to run, and
`Esc` to hide it.

| Command | Action |
| --- | --- |
| `/model`, `/models` | Fetch the model list (`GET <base_url>/models`) and pick one; type to filter, Enter on unmatched text uses it as a custom id. `/model <id>` sets it directly |
| `/effort [level]` | Pick or set `reasoning_effort` (`default`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh`) for this session |
| `/settings` | Form for backend (mistl AI network / OpenAI-compatible API), base URL, API key, model, reasoning effort, tool mode. Ctrl+S saves to `config.toml` and starts a new conversation |
| `/mistl [install\|start]` | Show where mistl is, install it, or start its daemon |
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

Setting `base_url` in the file, `MISTAN_BASE_URL`, or `--base-url` selects a custom
endpoint unless the file explicitly sets `backend`. `backend = "mistl"` always uses
the network, using native tool calling when the provider advertises `"tools"`.
mistl providers advertise this capability; providers without it trigger an automatic
fallback to prompt mode for the rest of the session. Set `tool_mode = "prompt"`
(or `--tool-mode prompt`) to force prompt mode. `backend = "custom"` requires a URL
and defaults to prompt mode.
For a custom endpoint, set `base_url`, `model`, and `tool_mode = "native"` when it
supports tools; supply its key through `MISTAN_API_KEY`.

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
