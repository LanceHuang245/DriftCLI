<p align="right">
  <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <h1 align="center">DriftCLI</h1>
  <p align="center">
    A blazing-fast terminal AI coding agent built in Rust.<br/>
    Bring your own key. Ship with your own model.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platform">
</p>

---

DriftCLI is a terminal AI coding agent that reads, writes, edits, and searches your codebase. It streams responses in real-time, executes tools on your behalf, and compresses context automatically — all from your terminal.

**BYOK** — DriftCLI is a pure client. No proxy, no markup, no vendor lock-in. Use your own API keys with Anthropic, OpenAI, Google, Groq, Ollama, or any OpenAI-compatible endpoint.

## Features

- **Submit/Event streaming architecture** — the agent loop never blocks the TUI. Every token, tool call, and status change is a typed event pushed to subscribers.
- **60 fps terminal UI** — Ratatui immediate-mode rendering. Syntax-highlighted diffs, streaming markdown, fuzzy file browsing, collapsible reasoning blocks.
- **Multi-provider LLM** — Anthropic, OpenAI, Google Gemini, Groq, Ollama, and any OpenAI-compatible endpoint. Automatic fallback across providers.
- **11 built-in tools** — `bash`, `read`, `write`, `edit` (similar diff), `grep` (ripgrep library), `glob`, `task` (sub-agents), `web_fetch`, `web_search`, `todowrite`
- **MCP** — first-class Model Context Protocol support via `rmcp`. Connect any MCP server and use its tools/resources directly.
- **WASM plugins** — extend DriftCLI with sandboxed WASM plugins. Write tools in Rust, Go, C, or AssemblyScript. Restricted capability model.
- **Sub-agents** — spawn isolated agent tasks (`explore`, `general`, `build`) to parallelize work. Depth-1 limit. Results summarized back.
- **Auto-compaction** — four-stage context compression pipeline keeps you under the token budget: truncate → drop old turns → auto-summarize → emergency compact.
- **Prompt caching** — stable prefixes (system prompt, tool definitions) marked for reuse. ~90% input cost reduction on cache hits.
- **Append-only transcripts** — every session is a JSONL file. Crash-safe, auditable, replayable, forkable.
- **Cross-platform** — Linux, macOS, Windows as first-class citizens. Correct signal handling, shell detection, path normalization on all three.
- **Permission system** — `deny > ask > allow` with per-tool, per-pattern rules. Safe tools auto-approved. Sensitive output redacted from transcripts.

## Installation

### From source

```bash
git clone https://github.com/user/drift.git
cd drift
cargo build --release
cp target/release/drift ~/.local/bin/
```

### GitHub Releases

Prebuilt static binaries for Linux (musl/glibc × x86_64/arm64), macOS (x86_64/arm64), and Windows (x86_64) on the [Releases page](https://github.com/user/drift/releases).

## Quick start

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Ask something
drift "Fix the authentication bug in src/auth.rs"

# List sessions
drift --list-sessions

# Resume a session
drift --continue

# Use a specific model
drift --model claude-opus-4-5 "Design the new API layer"

# Skip permission prompts (trust the workspace)
drift --no-permissions "run all tests"

# Activate a skill
drift --skill code-review "Review my recent changes"

# Generate project config
drift init
```

### Configuration

```bash
# Global config
~/.config/drift/config.toml

# Project config (overrides global)
.drift/config.toml

# Global agent instructions (optional)
~/.config/drift/AGENTS.md

# Project agent instructions
.drift/AGENTS.md
```

Example `config.toml`:

```toml
[agent]
model = "claude-sonnet-4-5"
max_iterations = 50
subagent_max_concurrent = 6

[llm.providers.anthropic]
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-4-5-20250101", "claude-opus-4-5-20250101"]

[llm.providers.openai]
api_key = "${OPENAI_API_KEY}"
models = ["gpt-4o"]

[mcp]
enabled = true

[[mcp.servers]]
id = "filesystem"
command = "npx"
args = ["-y", "@anthropic/mcp-server-filesystem", "/home/user/projects"]
transport = "stdio"
auto_start = true
```

## TUI shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Interrupt current agent operation |
| `Ctrl+D` | Quit (when idle) |
| `Ctrl+L` | Refresh screen |
| `Ctrl+O` | Toggle sidebar (files / sessions / tools) |
| `Ctrl+S` | Switch session |
| `Ctrl+N` | New session |
| `Ctrl+P` | File browser |
| `Ctrl+Shift+S` | List available skills |
| `Ctrl+K` | Manual context compaction |
| `Tab` | Autocomplete (in prompt bar) |
| `Up/Down` | Command history |
| `Enter` | Submit input |
| `Shift+Enter` | Insert newline |

## Architecture

```
┌──────────────────────────────────┐
│         Terminal (Crossterm)      │
│  ┌────────────────────────────┐  │
│  │   TUI (Ratatui 60fps)       │  │
│  └──────────────┬─────────────┘  │
│                 │ EventBus        │
│  ┌──────────────▼─────────────┐  │
│  │     Agent Core (Tokio)      │  │
│  │  submit → stream → execute  │  │
│  └──┬───────┬────────┬────────┘  │
│     │       │        │            │
│  ┌──▼──┐ ┌──▼──┐ ┌───▼───────┐   │
│  │ LLM │ │Tools│ │MCP/WASM   │   │
│  │ req │ │grep │ │Plugins    │   │
│  └─────┘ └─────┘ └───────────┘   │
└──────────────────────────────────┘
```
