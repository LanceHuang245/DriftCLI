use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod load;
mod mcp;
mod persist;
mod provider;

// ---------- Core Config ----------

// A named provider entry pairing a user-chosen label with the LLM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderEntry {
    pub name: String,
    #[serde(flatten)]
    pub config: LlmConfig,
}

// AppConfig: top-level configuration combining agent behaviour, active provider, and a named provider map.
// Re-export security config for convenience
pub use drift_security::SecurityConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub agent: AgentConfig,
    #[serde(default)]
    pub active_provider: String,
    #[serde(default)]
    pub providers: HashMap<String, ProviderEntry>,
    /// Security / permission configuration
    #[serde(default)]
    pub security: SecurityConfig,
    /// MCP server configuration.
    #[serde(default)]
    pub mcp: McpConfig,
    // Legacy migration marker
    #[serde(skip)]
    #[allow(dead_code)]
    migrated: bool,
}

/// Top-level MCP configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            servers: Vec::new(),
        }
    }
}

/// Configuration for one MCP server process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub id: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    pub transport: McpTransport,
    #[serde(default = "default_true")]
    pub auto_start: bool,
}

/// Supported MCP transports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
}

// AgentConfig: tunable parameters for the agent's behaviour — model, iteration cap, temperature, thinking budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub thinking_budget: Option<usize>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default = "default_auto_compaction")]
    pub auto_compaction: bool,
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: f64,
    #[serde(default = "default_compaction_target")]
    pub compaction_target: f64,
}

// LlmConfig: tagged enum selecting the active LLM provider — one variant per backend family.
/// The LLM provider configuration — only one active provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum LlmConfig {
    // Anthropic Claude models via the Anthropic Messages API.
    Anthropic {
        api_key: String,
        model: String,
        #[serde(default = "default_anthropic_base_url")]
        base_url: String,
        #[serde(default)]
        reasoning_effort: Option<String>,
    },
    // OpenAI or any OpenAI-compatible endpoint (vLLM, LocalAI, etc.) via the chat completions API.
    OpenAiCompatible {
        api_key: String,
        model: String,
        #[serde(default = "default_openai_compat_base_url")]
        base_url: String,
        #[serde(default)]
        supports_thinking: bool,
    },
}

// ---------- Defaults ----------

fn default_true() -> bool {
    true
}

fn default_model() -> String {
    "claude-sonnet-4-5-20250101".into()
}
fn default_max_iterations() -> usize {
    50
}

fn default_auto_compaction() -> bool {
    true
}

fn default_compaction_threshold() -> f64 {
    0.75
}

fn default_compaction_target() -> f64 {
    0.4
}
fn default_anthropic_base_url() -> String {
    "https://api.anthropic.com/v1".into()
}
fn default_openai_compat_base_url() -> String {
    "https://api.openai.com/v1".into()
}

// ---------- Config Loading ----------

// ConfigError: exhaustive error set covering I/O, parse failures, and missing configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read config: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("Global config directory not found")]
    NoHomeDirectory,
    #[error("No LLM provider configured. Set [llm] in your config file.")]
    NoLlmProvider,
    #[error("API key not set for provider. Set api_key in config or DRIFT_API_KEY env var.")]
    MissingApiKey,
    #[error("Invalid permission mode '{0}'. Expected ask, allow, or deny.")]
    InvalidPermissionMode(String),
    #[error("Security profile not found: {0}")]
    UnknownSecurityProfile(String),
    #[error("Invalid MCP configuration: {0}")]
    InvalidMcp(String),
    #[error("Failed to serialize config: {0}")]
    Serialize(#[from] toml::ser::Error),
}

#[cfg(test)]
mod tests;
