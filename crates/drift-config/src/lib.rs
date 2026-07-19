use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

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

// Wrapper used to serialize the security configuration under the [security] TOML table.
#[derive(Serialize)]
struct SecurityConfigWrapper<'a> {
    security: &'a SecurityConfig,
}

#[derive(Serialize)]
struct McpConfigWrapper<'a> {
    mcp: &'a McpConfig,
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

impl McpConfig {
    /// Validates identifiers and process fields before MCP startup.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut server_ids = HashSet::with_capacity(self.servers.len());
        for server in &self.servers {
            if server.id.is_empty()
                || server.id.len() > 56
                || !server.id.chars().enumerate().all(|(index, ch)| {
                    (index == 0 && ch.is_ascii_alphanumeric())
                        || (index > 0 && (ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')))
                })
            {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has an invalid or overly long id",
                    server.id
                )));
            }
            if !server_ids.insert(&server.id) {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has a duplicate id",
                    server.id
                )));
            }
            if server.command.trim().is_empty() {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has an empty command",
                    server.id
                )));
            }
            if server.env.keys().any(String::is_empty) {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has an empty environment key",
                    server.id
                )));
            }
        }
        Ok(())
    }
}

impl AppConfig {
    // Load config with layered merge: CLI args → env vars → project .drift/ → global → hardcoded defaults.
    /// Load config with merge: CLI args > env vars > .drift/config.toml > ~/.config/drift/config.toml > defaults
    pub fn load(cli_model: Option<&str>, cli_api_key: Option<&str>) -> Result<Self, ConfigError> {
        let mut config = Self::load_defaults_with_files()?;
        config.apply_env_overrides();
        config.apply_cli_overrides(cli_model, cli_api_key);
        config.mcp.validate()?;
        Ok(config)
    }

    // Load configuration with project and explicit-file layers before env and CLI overrides.
    pub fn load_for_workspace(
        cwd: &Path,
        explicit_config: Option<&Path>,
        cli_model: Option<&str>,
        cli_api_key: Option<&str>,
    ) -> Result<Self, ConfigError> {
        let mut config = Self::load_defaults_with_files()?;
        config.apply_project_override(cwd)?;
        if let Some(path) = explicit_config {
            config = Self::merge_toml_file(config, path)?;
        }
        config.apply_env_overrides();
        config.apply_cli_overrides(cli_model, cli_api_key);
        config.mcp.validate()?;
        Ok(config)
    }

    // Creates the global config directory and writes a default template if no config exists yet.
    /// Generate a default config and write it to disk
    pub fn init_global() -> Result<PathBuf, ConfigError> {
        let dir = Self::global_config_dir().ok_or(ConfigError::NoHomeDirectory)?;
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        if !path.exists() {
            let default = Self::default_template();
            std::fs::write(&path, default)?;
        }
        Ok(path)
    }

    // Creates the project-level .drift/ directory and writes a default config template.
    /// Init project-level config
    pub fn init_project(cwd: &Path) -> Result<PathBuf, ConfigError> {
        let dir = cwd.join(".drift");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        if !path.exists() {
            let default = Self::default_template();
            std::fs::write(&path, default)?;
        }
        Ok(path)
    }

    // Returns the platform-aware global config directory (e.g. ~/.config/drift on Linux).
    /// Global config directory (platform-aware via `directories` crate)
    pub fn global_config_dir() -> Option<PathBuf> {
        directories::ProjectDirs::from("com", "driftcli", "DriftCLI")
            .map(|d| d.config_dir().to_path_buf())
    }

    // Returns the path to the global config.toml file.
    /// Global config file path
    pub fn global_config_path() -> Option<PathBuf> {
        Self::global_config_dir().map(|d| d.join("config.toml"))
    }

    // Returns the path to the project-level .drift/config.toml file.
    /// Project config file path
    pub fn project_config_path(cwd: &Path) -> PathBuf {
        cwd.join(".drift").join("config.toml")
    }

    // Returns the LlmConfig of the currently active provider, or the first provider if active is unset.
    pub fn active_llm_config(&self) -> Option<&LlmConfig> {
        self.providers
            .get(&self.active_provider)
            .map(|e| &e.config)
            .or_else(|| self.providers.values().next().map(|e| &e.config))
    }

    // Returns a mutable reference to the active provider's config.
    pub fn active_llm_config_mut(&mut self) -> Option<&mut LlmConfig> {
        self.providers
            .get_mut(&self.active_provider)
            .map(|e| &mut e.config)
    }

    // Returns a vector of all provider names for the /provider picker.
    pub fn list_provider_names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    // Returns the LlmConfig for a specific provider by name.
    pub fn get_provider_config(&self, name: &str) -> Option<&LlmConfig> {
        self.providers.get(name).map(|e| &e.config)
    }

    // Adds or updates a named provider. Sets it as active if it's the only one or explicitly requested.
    pub fn save_provider(&mut self, name: String, config: LlmConfig) {
        let is_first = self.providers.is_empty();
        self.providers.insert(
            name.clone(),
            ProviderEntry {
                name: name.clone(),
                config,
            },
        );
        if is_first || self.active_provider.is_empty() {
            self.active_provider = name;
        }
    }

    // Removes a named provider. If the active provider was removed, switches to the first remaining.
    pub fn remove_provider(&mut self, name: &str) {
        self.providers.remove(name);
        if self.active_provider == name {
            self.active_provider = self.providers.keys().next().cloned().unwrap_or_default();
        }
    }

    // Switches to the given provider name (must exist).
    pub fn activate_provider(&mut self, name: &str) -> Result<(), ConfigError> {
        if !self.providers.contains_key(name) {
            return Err(ConfigError::NoLlmProvider);
        }
        self.active_provider = name.to_string();
        Ok(())
    }

    // Override the selected security profile's approval policy from the CLI.
    pub fn apply_permission_mode(
        &mut self,
        profile_name: &str,
        mode: &str,
    ) -> Result<(), ConfigError> {
        let policy = match mode.to_ascii_lowercase().as_str() {
            "ask" => drift_security::ApprovalPolicy::OnRequest,
            "allow" => drift_security::ApprovalPolicy::Never,
            "deny" => drift_security::ApprovalPolicy::Deny,
            _ => return Err(ConfigError::InvalidPermissionMode(mode.to_string())),
        };
        let profile = self
            .security
            .profiles
            .get_mut(profile_name)
            .ok_or_else(|| ConfigError::UnknownSecurityProfile(profile_name.to_string()))?;
        profile.approval_policy = policy;
        Ok(())
    }

    // Builds a config starting from hardcoded defaults, then merges global config on top.
    fn load_defaults_with_files() -> Result<Self, ConfigError> {
        let mut providers = HashMap::new();
        let default_name = "default".to_string();
        providers.insert(
            default_name.clone(),
            ProviderEntry {
                name: default_name.clone(),
                config: LlmConfig::Anthropic {
                    api_key: String::new(),
                    model: default_model(),
                    base_url: default_anthropic_base_url(),
                    reasoning_effort: None,
                },
            },
        );
        let mut config = Self {
            agent: AgentConfig {
                model: default_model(),
                max_iterations: default_max_iterations(),
                temperature: None,
                thinking_budget: None,
                reasoning_effort: None,
                auto_compaction: default_auto_compaction(),
                compaction_threshold: default_compaction_threshold(),
                compaction_target: default_compaction_target(),
            },
            active_provider: default_name,
            providers,
            security: SecurityConfig::default(),
            mcp: McpConfig::default(),
            migrated: false,
        };

        // Layer 1: global config
        if let Some(global_path) = Self::global_config_path()
            && global_path.exists()
        {
            config = Self::merge_toml_file(config, &global_path)?;
        }

        // Layer 2: project config (overrides global)
        // NOTE: We don't know cwd at this point; caller should call apply_project_override after
        Ok(config)
    }

    // Merges a project-level .drift/config.toml on top of the currently loaded config.
    /// Apply project-level config override (call after determining workspace)
    pub fn apply_project_override(&mut self, cwd: &Path) -> Result<(), ConfigError> {
        let project_path = Self::project_config_path(cwd);
        if project_path.exists() {
            let partial: toml::Value = toml::from_str(&std::fs::read_to_string(&project_path)?)?;
            Self::merge_toml_value(self, &partial)?;
        }
        Ok(())
    }

    // Overrides API key and model from DRIFT_API_KEY / DRIFT_MODEL environment variables.
    fn apply_env_overrides(&mut self) {
        if let Ok(key) = std::env::var("DRIFT_API_KEY")
            && let Some(config) = self.active_llm_config_mut()
        {
            match config {
                LlmConfig::Anthropic { api_key, .. }
                | LlmConfig::OpenAiCompatible { api_key, .. } => {
                    *api_key = key;
                }
            }
        }
        if let Ok(model) = std::env::var("DRIFT_MODEL") {
            self.agent.model = model;
        }
    }

    // Applies --model and --api-key CLI flags as the highest-priority overrides.
    fn apply_cli_overrides(&mut self, cli_model: Option<&str>, cli_api_key: Option<&str>) {
        if let Some(m) = cli_model {
            self.agent.model = m.to_string();
        }
        if let Some(k) = cli_api_key
            && let Some(config) = self.active_llm_config_mut()
        {
            match config {
                LlmConfig::Anthropic { api_key, .. }
                | LlmConfig::OpenAiCompatible { api_key, .. } => {
                    *api_key = k.to_string();
                }
            }
        }
    }

    // Reads a TOML file and merges its contents into a base config via merge_toml_value.
    fn merge_toml_file(mut base: Self, path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let overlay: toml::Value = toml::from_str(&content)?;
        Self::merge_toml_value(&mut base, &overlay)?;
        Ok(base)
    }

    // Merges a TOML value overlay into an AppConfig: selectively overwrites agent, providers, and active_provider fields.
    fn merge_toml_value(config: &mut Self, overlay: &toml::Value) -> Result<(), ConfigError> {
        // Merge [agent] section
        if let Some(agent) = overlay.get("agent") {
            if let Some(m) = agent.get("model").and_then(|v| v.as_str()) {
                config.agent.model = m.to_string();
            }
            if let Some(m) = agent.get("max_iterations").and_then(|v| v.as_integer()) {
                config.agent.max_iterations = m as usize;
            }
            if let Some(t) = agent.get("temperature").and_then(|v| v.as_float()) {
                config.agent.temperature = Some(t);
            }
            if let Some(t) = agent.get("thinking_budget").and_then(|v| v.as_integer()) {
                config.agent.thinking_budget = Some(t as usize);
            }
            if let Some(e) = agent.get("reasoning_effort").and_then(|v| v.as_str()) {
                config.agent.reasoning_effort = Some(e.to_string());
            }
            if let Some(enabled) = agent.get("auto_compaction").and_then(|v| v.as_bool()) {
                config.agent.auto_compaction = enabled;
            }
            if let Some(threshold) = agent.get("compaction_threshold").and_then(|v| v.as_float()) {
                config.agent.compaction_threshold = threshold;
            }
            if let Some(target) = agent.get("compaction_target").and_then(|v| v.as_float()) {
                config.agent.compaction_target = target;
            }
        }

        // Merge active_provider
        if let Some(ap) = overlay.get("active_provider").and_then(|v| v.as_str()) {
            config.active_provider = ap.to_string();
        }

        // Merge [[providers]] array
        if let Some(providers_array) = overlay.get("providers").and_then(|v| v.as_array()) {
            for entry_value in providers_array {
                if let (Some(name), llm) = (
                    entry_value.get("name").and_then(|v| v.as_str()),
                    entry_value,
                ) {
                    let name = name.to_string();
                    if let Some(provider) = llm.get("provider").and_then(|v| v.as_str()) {
                        let provider_config = match provider {
                            "anthropic" => LlmConfig::Anthropic {
                                api_key: llm
                                    .get("api_key")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .into(),
                                model: llm
                                    .get("model")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(&default_model())
                                    .into(),
                                base_url: llm
                                    .get("base_url")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(&default_anthropic_base_url())
                                    .into(),
                                reasoning_effort: llm
                                    .get("reasoning_effort")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string()),
                            },
                            "openai_compatible" | "openai-compatible" => {
                                LlmConfig::OpenAiCompatible {
                                    api_key: llm
                                        .get("api_key")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .into(),
                                    model: llm
                                        .get("model")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("gpt-4o")
                                        .into(),
                                    base_url: llm
                                        .get("base_url")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or(&default_openai_compat_base_url())
                                        .into(),
                                    supports_thinking: llm
                                        .get("supports_thinking")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false),
                                }
                            }
                            _ => continue,
                        };
                        config.providers.insert(
                            name.clone(),
                            ProviderEntry {
                                name,
                                config: provider_config,
                            },
                        );
                    }
                }
            }
        }

        // Merge MCP settings and replace complete server entries by id.
        if let Some(mcp) = overlay.get("mcp").and_then(|v| v.as_table()) {
            if let Some(enabled) = mcp.get("enabled").and_then(|v| v.as_bool()) {
                config.mcp.enabled = enabled;
            }
            if let Some(servers) = mcp.get("servers").and_then(|v| v.as_array()) {
                for value in servers {
                    let server: McpServerConfig = value.clone().try_into()?;
                    if let Some(existing) = config
                        .mcp
                        .servers
                        .iter_mut()
                        .find(|existing| existing.id == server.id)
                    {
                        *existing = server;
                    } else {
                        config.mcp.servers.push(server);
                    }
                }
            }
        }

        // Backward compat: handle old [llm] format
        if let Some(llm) = overlay.get("llm")
            && config.providers.is_empty()
            && let Some(provider) = llm.get("provider").and_then(|v| v.as_str())
        {
            let config_llm = match provider {
                "anthropic" => Some(LlmConfig::Anthropic {
                    api_key: llm
                        .get("api_key")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    model: llm
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&default_model())
                        .into(),
                    base_url: llm
                        .get("base_url")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&default_anthropic_base_url())
                        .into(),
                    reasoning_effort: llm
                        .get("reasoning_effort")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                }),
                "openai_compatible" | "openai-compatible" => Some(LlmConfig::OpenAiCompatible {
                    api_key: llm
                        .get("api_key")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    model: llm
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("gpt-4o")
                        .into(),
                    base_url: llm
                        .get("base_url")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&default_openai_compat_base_url())
                        .into(),
                    supports_thinking: llm
                        .get("supports_thinking")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                }),
                _ => None,
            };
            if let Some(config_llm) = config_llm {
                let name = "default".to_string();
                config.providers.insert(
                    name.clone(),
                    ProviderEntry {
                        name: name.clone(),
                        config: config_llm,
                    },
                );
                if config.active_provider.is_empty() {
                    config.active_provider = name;
                }
            }
        }

        // Merge security settings, profiles, and network domain restrictions.
        if let Some(security) = overlay.get("security").and_then(|v| v.as_table()) {
            if let Some(default_profile) = security.get("default_profile").and_then(|v| v.as_str())
            {
                config.security.default_profile = default_profile.to_string();
            }
            if let Some(enabled) = security.get("enabled").and_then(|v| v.as_bool()) {
                config.security.enabled = enabled;
            }
            if let Some(allowed_domains) =
                security.get("allowed_domains").and_then(|v| v.as_array())
            {
                config.security.allowed_domains = allowed_domains
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
            if let Some(blocked_domains) =
                security.get("blocked_domains").and_then(|v| v.as_array())
            {
                config.security.blocked_domains = blocked_domains
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
            if let Some(profiles) = security.get("profiles").and_then(|v| v.as_table()) {
                for (name, value) in profiles {
                    let mut profile_value = value.clone();
                    if let Some(table) = profile_value.as_table_mut() {
                        table.insert("name".into(), toml::Value::String(name.clone()));
                    }
                    let profile: drift_security::SecurityProfile = profile_value.try_into()?;
                    config.security.profiles.insert(name.clone(), profile);
                }
            }
        }

        Ok(())
    }

    // Returns a default TOML config template string used for initializing new config files.
    fn default_template() -> String {
        let security = toml::to_string(&SecurityConfigWrapper {
            security: &SecurityConfig::default(),
        })
        .unwrap_or_default();
        format!(
            r###"# DriftCLI Configuration
active_provider = "default"

[agent]
model = "claude-sonnet-4-5-20250101"
max_iterations = 50
auto_compaction = true
compaction_threshold = 0.75
compaction_target = 0.4

[mcp]
enabled = true

# [[mcp.servers]]
# id = "example"
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
# transport = "stdio"
# auto_start = true

[[providers]]
name = "default"
provider = "anthropic"
model = "claude-sonnet-4-5-20250101"
api_key = ""
base_url = "https://api.anthropic.com/v1"
{}"###,
            security
        )
    }

    // Builds a display string summarizing the active provider, model, endpoint, and masked API key.
    /// Extract a display summary for the /connect command
    pub fn connection_summary(&self) -> String {
        if let Some(config) = self.active_llm_config() {
            let (provider_name, api_key, model, base_url) = match config {
                LlmConfig::Anthropic {
                    api_key,
                    model,
                    base_url,
                    ..
                } => (
                    "Anthropic",
                    api_key.as_str(),
                    model.as_str(),
                    base_url.as_str(),
                ),
                LlmConfig::OpenAiCompatible {
                    api_key,
                    model,
                    base_url,
                    ..
                } => (
                    "OpenAI Compatible",
                    api_key.as_str(),
                    model.as_str(),
                    base_url.as_str(),
                ),
            };
            let key_masked = if api_key.is_empty() {
                "(not set)".to_string()
            } else if api_key.len() <= 8 {
                "***".to_string()
            } else {
                format!("{}...{}", &api_key[..4], &api_key[api_key.len() - 4..])
            };
            format!(
                "Provider: {}\nModel: {}\nEndpoint: {}\nAPI Key: {}",
                provider_name, model, base_url, key_masked
            )
        } else {
            "No providers configured.".into()
        }
    }

    // Serializes the current config and writes it to .drift/config.toml in the given project directory.
    /// Write the current config to the project .drift/config.toml
    pub fn save_to_project(&self, cwd: &Path) -> Result<(), ConfigError> {
        let dir = cwd.join(".drift");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        let toml_str = self.to_toml_string()?;
        std::fs::write(&path, toml_str)?;
        Ok(())
    }

    // Converts the current AppConfig into a human-readable TOML string with inline comments.
    fn to_toml_string(&self) -> Result<String, ConfigError> {
        let mut providers_str = String::new();
        for entry in self.providers.values() {
            let name = &entry.name;
            let inner = match &entry.config {
                LlmConfig::Anthropic {
                    api_key,
                    model,
                    base_url,
                    reasoning_effort,
                } => {
                    let mut s = format!(
                        "provider = \"anthropic\"\nmodel = \"{}\"\napi_key = \"{}\"\nbase_url = \"{}\"",
                        model, api_key, base_url
                    );
                    if let Some(e) = reasoning_effort {
                        s.push_str(&format!("\nreasoning_effort = \"{}\"", e));
                    }
                    s
                }
                LlmConfig::OpenAiCompatible {
                    api_key,
                    model,
                    base_url,
                    supports_thinking,
                } => {
                    format!(
                        "provider = \"openai_compatible\"\nmodel = \"{}\"\napi_key = \"{}\"\nbase_url = \"{}\"\nsupports_thinking = {}",
                        model, api_key, base_url, supports_thinking
                    )
                }
            };
            providers_str.push_str(&format!(
                "[[providers]]\nname = \"{}\"\n{}\n\n",
                name, inner
            ));
        }
        let mcp = toml::to_string(&McpConfigWrapper { mcp: &self.mcp })?;
        let security = toml::to_string(&SecurityConfigWrapper {
            security: &self.security,
        })?;
        Ok(format!(
            "# DriftCLI Configuration\n\nactive_provider = \"{}\"\n\n[agent]\nmodel = \"{}\"\nmax_iterations = {}\nauto_compaction = {}\ncompaction_threshold = {}\ncompaction_target = {}\n\n{}\n{}",
            self.active_provider,
            self.agent.model,
            self.agent.max_iterations,
            self.agent.auto_compaction,
            self.agent.compaction_threshold,
            self.agent.compaction_target,
            mcp,
            providers_str,
        ) + &security)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_default_config() {
        let tmpl = AppConfig::default_template();
        assert!(tmpl.contains("[agent]"));
        assert!(tmpl.contains("[[providers]]"));
        assert!(tmpl.contains("active_provider"));
        assert!(tmpl.contains("auto_compaction = true"));
        assert!(tmpl.contains("compaction_threshold = 0.75"));
        assert!(tmpl.contains("compaction_target = 0.4"));
        toml::from_str::<toml::Value>(&tmpl).expect("default template must be valid TOML");
    }

    #[test]
    fn test_compaction_config_merge() {
        let mut config = AppConfig::load_defaults_with_files().unwrap();
        let overlay: toml::Value = toml::from_str(
            "[agent]\nauto_compaction = false\ncompaction_threshold = 0.8\ncompaction_target = 0.3\n",
        )
        .unwrap();

        AppConfig::merge_toml_value(&mut config, &overlay).unwrap();

        assert!(!config.agent.auto_compaction);
        assert_eq!(config.agent.compaction_threshold, 0.8);
        assert_eq!(config.agent.compaction_target, 0.3);
    }

    #[test]
    fn test_connection_summary() {
        let mut providers = HashMap::new();
        providers.insert(
            "test".into(),
            ProviderEntry {
                name: "test".into(),
                config: LlmConfig::Anthropic {
                    api_key: "sk-ant-test1234".into(),
                    model: "test-model".into(),
                    base_url: "https://api.anthropic.com/v1".into(),
                    reasoning_effort: None,
                },
            },
        );
        let config = AppConfig {
            agent: AgentConfig {
                model: "test-model".into(),
                max_iterations: 50,
                temperature: None,
                thinking_budget: None,
                reasoning_effort: None,
                auto_compaction: true,
                compaction_threshold: 0.75,
                compaction_target: 0.4,
            },
            active_provider: "test".into(),
            providers,
            security: SecurityConfig::default(),
            mcp: McpConfig::default(),
            migrated: false,
        };
        let summary = config.connection_summary();
        assert!(summary.contains("Anthropic"));
        assert!(summary.contains("sk-a...1234"));
    }

    #[test]
    fn test_security_config_merge_and_save() {
        let mut config = AppConfig {
            agent: AgentConfig {
                model: default_model(),
                max_iterations: 50,
                temperature: None,
                thinking_budget: None,
                reasoning_effort: None,
                auto_compaction: true,
                compaction_threshold: 0.75,
                compaction_target: 0.4,
            },
            active_provider: String::new(),
            providers: HashMap::new(),
            security: SecurityConfig::default(),
            mcp: McpConfig::default(),
            migrated: false,
        };
        let overlay: toml::Value = toml::from_str(
            r#"
[security]
enabled = false
default_profile = "audit"
allowed_domains = ["example.com"]
blocked_domains = ["blocked.example.com"]

[security.profiles.audit]
approval_policy = "deny"
sandbox_mode = "read-only"
"#,
        )
        .unwrap();

        AppConfig::merge_toml_value(&mut config, &overlay).unwrap();
        assert!(!config.security.enabled);
        assert_eq!(config.security.default_profile, "audit");
        assert_eq!(config.security.allowed_domains, vec!["example.com"]);
        assert_eq!(config.security.blocked_domains, vec!["blocked.example.com"]);
        assert_eq!(
            config.security.profiles["audit"].approval_policy,
            drift_security::ApprovalPolicy::Deny
        );

        config.apply_permission_mode("audit", "allow").unwrap();
        assert_eq!(
            config.security.profiles["audit"].approval_policy,
            drift_security::ApprovalPolicy::Never
        );

        let serialized = config.to_toml_string().unwrap();
        let value: toml::Value = toml::from_str(&serialized).unwrap();
        assert_eq!(value["security"]["enabled"].as_bool(), Some(false));
        assert_eq!(
            value["security"]["profiles"]["audit"]["approval_policy"].as_str(),
            Some("never")
        );
    }

    #[test]
    fn test_mcp_defaults_and_template() {
        let config = McpConfig::default();
        assert!(config.enabled);
        assert!(config.servers.is_empty());
        let template = AppConfig::default_template();
        assert!(template.contains("[mcp]"));
        assert!(template.contains("# [[mcp.servers]]"));
    }

    #[test]
    fn test_mcp_parse_validation_and_layered_merge() {
        let server = |id: &str| McpServerConfig {
            id: id.into(),
            command: "server".into(),
            args: Vec::new(),
            env: HashMap::new(),
            transport: McpTransport::Stdio,
            auto_start: true,
        };
        let mut config = AppConfig::load_defaults_with_files().unwrap();
        let first: toml::Value = toml::from_str(
            r#"
[mcp]
enabled = true
[[mcp.servers]]
id = "alpha"
command = "first"
transport = "stdio"
[[mcp.servers]]
id = "keep"
command = "keep"
transport = "stdio"
"#,
        )
        .unwrap();
        let second: toml::Value = toml::from_str(
            r#"
[mcp]
[[mcp.servers]]
id = "alpha"
command = "second"
args = ["--flag"]
transport = "stdio"
"#,
        )
        .unwrap();
        AppConfig::merge_toml_value(&mut config, &first).unwrap();
        AppConfig::merge_toml_value(&mut config, &second).unwrap();
        assert_eq!(config.mcp.servers.len(), 2);
        assert_eq!(config.mcp.servers[0].command, "second");
        assert_eq!(config.mcp.servers[0].args, vec!["--flag"]);
        assert_eq!(config.mcp.servers[1].id, "keep");
        config.mcp.validate().unwrap();

        for id in ["bad id", "bad.id", &"a".repeat(57)] {
            let invalid = McpConfig {
                enabled: true,
                servers: vec![server(id)],
            };
            assert!(matches!(
                invalid.validate(),
                Err(ConfigError::InvalidMcp(_))
            ));
        }
        let duplicate = server("duplicate");
        let duplicates = McpConfig {
            enabled: true,
            servers: vec![duplicate.clone(), duplicate],
        };
        assert!(matches!(
            duplicates.validate(),
            Err(ConfigError::InvalidMcp(_))
        ));
        let invalid_transport = toml::from_str::<McpConfig>(
            r#"[[servers]]
id = "bad-transport"
command = "server"
transport = "sse"
"#,
        );
        assert!(invalid_transport.is_err());
        let empty_command = McpConfig {
            enabled: true,
            servers: vec![McpServerConfig {
                id: "empty-command".into(),
                command: "  ".into(),
                args: Vec::new(),
                env: HashMap::new(),
                transport: McpTransport::Stdio,
                auto_start: true,
            }],
        };
        assert!(matches!(
            empty_command.validate(),
            Err(ConfigError::InvalidMcp(_))
        ));
        let empty_env_key = McpConfig {
            enabled: true,
            servers: vec![McpServerConfig {
                id: "empty-env".into(),
                command: "server".into(),
                args: Vec::new(),
                env: [(String::new(), "value".into())].into_iter().collect(),
                transport: McpTransport::Stdio,
                auto_start: true,
            }],
        };
        assert!(matches!(
            empty_env_key.validate(),
            Err(ConfigError::InvalidMcp(_))
        ));
    }

    #[test]
    fn test_mcp_serialization_round_trip() {
        let mut config = AppConfig::load_defaults_with_files().unwrap();
        let overlay: toml::Value = toml::from_str(
            r#"
[mcp]
enabled = false
[[mcp.servers]]
id = "roundtrip"
command = "server"
env = { TOKEN = "${TOKEN}" }
transport = "stdio"
auto_start = false
"#,
        )
        .unwrap();
        AppConfig::merge_toml_value(&mut config, &overlay).unwrap();
        let serialized = config.to_toml_string().unwrap();
        let parsed: toml::Value = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed["mcp"]["enabled"].as_bool(), Some(false));
        assert_eq!(
            parsed["mcp"]["servers"][0]["id"].as_str(),
            Some("roundtrip")
        );
        assert_eq!(
            parsed["mcp"]["servers"][0]["auto_start"].as_bool(),
            Some(false)
        );
    }
}
