use super::{AppConfig, ConfigError, LlmConfig, McpConfig, SecurityConfig};
use serde::Serialize;
use std::path::{Path, PathBuf};

// Wrapper used to serialize security settings under the expected TOML table.
#[derive(Serialize)]
struct SecurityConfigWrapper<'a> {
    security: &'a SecurityConfig,
}

// Wrapper used to serialize MCP settings under the expected TOML table.
#[derive(Serialize)]
struct McpConfigWrapper<'a> {
    mcp: &'a McpConfig,
}

impl AppConfig {
    // Creates the global config directory and writes a default template if needed.
    pub fn init_global() -> Result<PathBuf, ConfigError> {
        let dir = Self::global_config_dir().ok_or(ConfigError::NoHomeDirectory)?;
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        if !path.exists() {
            std::fs::write(&path, Self::default_template())?;
        }
        Ok(path)
    }

    // Creates the project config directory and writes a default template if needed.
    pub fn init_project(cwd: &Path) -> Result<PathBuf, ConfigError> {
        let dir = cwd.join(".drift");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        if !path.exists() {
            std::fs::write(&path, Self::default_template())?;
        }
        Ok(path)
    }

    // Returns the platform-aware global config directory.
    pub fn global_config_dir() -> Option<PathBuf> {
        directories::ProjectDirs::from("com", "driftcli", "DriftCLI")
            .map(|dirs| dirs.config_dir().to_path_buf())
    }

    // Returns the path to the global config file.
    pub fn global_config_path() -> Option<PathBuf> {
        Self::global_config_dir().map(|dir| dir.join("config.toml"))
    }

    // Returns the path to the project config file.
    pub fn project_config_path(cwd: &Path) -> PathBuf {
        cwd.join(".drift").join("config.toml")
    }

    // Returns the default TOML config template.
    pub(crate) fn default_template() -> String {
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

    // Builds a display string for the active provider without exposing the API key.
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

    // Serializes the current config into the project config file.
    pub fn save_to_project(&self, cwd: &Path) -> Result<(), ConfigError> {
        let dir = cwd.join(".drift");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        std::fs::write(&path, self.to_toml_string()?)?;
        Ok(())
    }

    // Converts the current config into the persisted TOML representation.
    pub(crate) fn to_toml_string(&self) -> Result<String, ConfigError> {
        let mut providers_str = String::new();
        for entry in self.providers.values() {
            let inner = match &entry.config {
                LlmConfig::Anthropic {
                    api_key,
                    model,
                    base_url,
                    reasoning_effort,
                } => {
                    let mut value = format!(
                        "provider = \"anthropic\"\nmodel = \"{}\"\napi_key = \"{}\"\nbase_url = \"{}\"",
                        model, api_key, base_url
                    );
                    if let Some(effort) = reasoning_effort {
                        value.push_str(&format!("\nreasoning_effort = \"{}\"", effort));
                    }
                    value
                }
                LlmConfig::OpenAiCompatible {
                    api_key,
                    model,
                    base_url,
                    supports_thinking,
                } => format!(
                    "provider = \"openai_compatible\"\nmodel = \"{}\"\napi_key = \"{}\"\nbase_url = \"{}\"\nsupports_thinking = {}",
                    model, api_key, base_url, supports_thinking
                ),
            };
            providers_str.push_str(&format!(
                "[[providers]]\nname = \"{}\"\n{}\n\n",
                entry.name, inner
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
