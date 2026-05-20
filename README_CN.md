<p align="right">
  <a href="README.md">English</a>
</p>

<p align="center">
  <h1 align="center">DriftCLI</h1>
  <p align="center">
    基于 Rust 构建的高性能终端 AI 编程 Agent。<br/>
    自带密钥，自选模型——无代理、无锁定。
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platform">
</p>

---

DriftCLI 是一个终端 AI 编程 Agent，能够读取、写入、编辑和搜索你的代码库。它实时流式输出响应，替代你执行工具，自动压缩上下文——一切都在终端中完成。

**BYOK**——DriftCLI 是纯客户端。无代理、无加价、无厂商锁定。使用你自己的 API 密钥接入 Anthropic、OpenAI、Google、Groq、Ollama 或任何 OpenAI 兼容端点。

## 为什么选择 DriftCLI

| | DriftCLI | Claude Code | CodexCLI | OpenCode/Kilo |
|---|---|---|---|---|
| **语言** | Rust | TypeScript | Rust | TypeScript |
| **启动速度** | < 5ms | ~200ms | < 5ms | ~200ms |
| **二进制大小** | ~5 MB | — | ~50 MB | ~15 MB |
| **运行时依赖** | 无（静态链接） | Bun | 无（静态链接） | Bun |
| **BYOK** | 支持 | 仅 Anthropic | 仅 OpenAI | 支持 |
| **MCP** | rmcp（原生） | 部分支持 | rmcp（原生） | MCP SDK |
| **插件** | WASM (wasmtime) | — | — | WASM |
| **子 Agent** | Tokio task 隔离 | 支持 | 支持 | 支持 |
| **沙箱** | bubblewrap / Seatbelt（可选） | — | bubblewrap | — |
| **许可协议** | MIT | 闭源 | Apache 2.0 | MIT |

## 特性

- **Submit/Event 流式架构**——Agent 循环永不阻塞 TUI。每个 token、工具调用、状态变更都是带类型的事件，推送给订阅者。
- **60fps 终端 UI**——Ratatui 即时模式渲染。语法高亮 diff、流式 Markdown、模糊文件浏览、可折叠推理块。
- **多 LLM 提供商**——Anthropic、OpenAI、Google Gemini、Groq、Ollama 以及任何 OpenAI 兼容端点。提供商间自动 fallback。
- **11 个内置工具**——`bash`、`read`、`write`、`edit`（similar diff）、`grep`（ripgrep 库）、`glob`、`task`（子 Agent）、`web_fetch`、`web_search`、`todowrite`
- **MCP**——通过 `rmcp` 提供一等 MCP 协议支持。连接任意 MCP Server 并直接使用其工具/资源。
- **WASM 插件**——使用沙箱化 WASM 插件扩展 DriftCLI。用 Rust、Go、C 或 AssemblyScript 编写工具。受限能力模型。
- **子 Agent**——派生隔离的 Agent 任务（`explore`、`general`、`build`）以并行化工作。深度上限 1，结果摘要回传。
- **自动压缩**——四阶段上下文压缩管道，确保不超出 token 预算：截断 → 裁剪旧轮次 → 自动摘要 → 紧急压缩。
- **Prompt 缓存**——稳定前缀（系统 prompt、工具定义）标记为可重用。缓存命中时输入成本降低约 90%。
- **仅追加转录**——每次会话一个 JSONL 文件。Crash 安全、可审计、可回放、可分支。
- **跨平台**——Linux、macOS、Windows 一等公民。三大平台信号处理、Shell 检测、路径规范化齐全。
- **权限系统**——`deny > ask > allow`，支持按工具、按模式设置规则。安全工具自动批准。敏感输出从转录中脱敏。

## 安装

### 源码编译

```bash
git clone https://github.com/user/drift.git
cd drift
cargo build --release
cp target/release/drift ~/.local/bin/
```

### GitHub Releases

在 [Releases 页面](https://github.com/user/drift/releases) 下载预构建的静态二进制文件，支持 Linux (musl/glibc × x86_64/arm64)、macOS (x86_64/arm64) 和 Windows (x86_64)。

## 快速开始

```bash
# 设置 API 密钥
export ANTHROPIC_API_KEY="sk-ant-..."

# 发起询问
drift "修复 src/auth.rs 中的认证 bug"

# 列出历史会话
drift --list-sessions

# 恢复一个会话
drift --continue

# 指定模型
drift --model claude-opus-4-5 "设计新的 API 层"

# 跳过权限确认（信任工作区）
drift --no-permissions "运行所有测试"

# 激活 Skill
drift --skill code-review "审查最近的改动"

# 生成项目配置
drift init
```

### 配置

```bash
# 全局配置
~/.config/drift/config.toml

# 项目配置（覆盖全局）
.drift/config.toml

# 全局 Agent 指令（可选）
~/.config/drift/AGENTS.md

# 项目 Agent 指令
.drift/AGENTS.md
```

示例 `config.toml`：

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

## TUI 快捷键

| 按键 | 功能 |
|------|------|
| `Ctrl+C` | 中断当前 Agent 操作 |
| `Ctrl+D` | 退出（空闲状态下） |
| `Ctrl+L` | 刷新屏幕 |
| `Ctrl+O` | 切换侧边栏（文件 / 会话 / 工具） |
| `Ctrl+S` | 切换会话 |
| `Ctrl+N` | 新建会话 |
| `Ctrl+P` | 文件浏览器 |
| `Ctrl+Shift+S` | 列出可用 Skills |
| `Ctrl+K` | 手动压缩上下文 |
| `Tab` | 自动补全（输入栏） |
| `↑/↓` | 命令历史 |
| `Enter` | 提交输入 |
| `Shift+Enter` | 输入换行 |

## 架构

```
┌──────────────────────────────────┐
│           终端 (Crossterm)        │
│  ┌────────────────────────────┐  │
│  │   TUI (Ratatui 60fps)       │  │
│  └──────────────┬─────────────┘  │
│                 │ EventBus        │
│  ┌──────────────▼─────────────┐  │
│  │     Agent 核心 (Tokio)      │  │
│  │  submit → stream → execute  │  │
│  └──┬───────┬────────┬────────┘  │
│     │       │        │            │
│  ┌──▼──┐ ┌──▼──┐ ┌───▼───────┐   │
│  │ LLM │ │工具 │ │MCP/WASM   │   │
│  │ 请求│ │目录 │ │插件       │   │
│  └─────┘ └─────┘ └───────────┘   │
└──────────────────────────────────┘
```
