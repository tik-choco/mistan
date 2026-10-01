# mistan task runner — `just <recipe>` (run `just` to list)
#
# Requires: `just` (cargo install just) and a Rust toolchain. mistan drives a
# `mistl` binary at runtime; set MISTAN_MISTL or pass `--mistl <path>` when it
# is not on PATH.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List available recipes
default:
    @just --list

# Type-check
check:
    cargo check --all-targets

# Build (debug)
build:
    cargo build

# Build an optimized binary into target/release
release:
    cargo build --release

# Run the unit tests (optionally filtered, e.g. `just test prompt`)
test filter="":
    cargo test {{filter}}

# Clippy with warnings as errors
lint:
    cargo clippy --all-targets -- -D warnings

# Format the crate
fmt:
    cargo fmt

# Start the TUI (extra args are passed through, e.g. `just run --instance dev`)
run *args:
    cargo run -- {{args}}

# One-shot, non-interactive prompt (no TUI), e.g. `just ask "AI network status?"`
ask prompt *args:
    cargo run -- {{args}} -p "{{prompt}}"

# Run the TUI against a sibling ../mistl debug build (its isolated dev instance)
[windows]
run-dev *args:
    cargo run -- --mistl ../mistl/target/debug/mistl.exe {{args}}

[unix]
run-dev *args:
    cargo run -- --mistl ../mistl/target/debug/mistl {{args}}

# Install mistan into ~/.cargo/bin
install:
    cargo install --path . --locked

# Remove build output
clean:
    cargo clean
