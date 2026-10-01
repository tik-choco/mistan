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
- A `mistl` binary on `PATH`, at the per-user install location (`mistl install`), or
  given with `--mistl <path>`

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

Commands: `/help`, `/clear`, `/model <id>`, `/quit`.

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
| `MISTAN_MISTL` | mistl executable |

Use environment variables for API keys. Never commit them; `.gitignore` already
excludes `config.toml`, `.env*`, and key files.

## License

[MPL-2.0](LICENSE)
