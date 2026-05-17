use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------- Core Config ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub agent: AgentConfig,
    pub llm: LlmConfig,
}

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
}

/// The LLM provider configuration — only one active provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum LlmConfig {
    Anthropic {
        api_key: String,
        model: String,
        #[serde(default = "default_anthropic_base_url")]
        base_url: String,
    },
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

fn default_model() -> String {
    "claude-sonnet-4-5-20250101".into()
}
fn default_max_iterations() -> usize {
    50
}
fn default_anthropic_base_url() -> String {
    "https://api.anthropic.com/v1".into()
}
fn default_openai_compat_base_url() -> String {
    "https://api.openai.com/v1".into()
}

// ---------- Config Loading ----------

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
}

impl AppConfig {
    /// Load config with merge: CLI args > env vars > .drift/config.toml > ~/.config/drift/config.toml > defaults
    pub fn load(cli_model: Option<&str>, cli_api_key: Option<&str>) -> Result<Self, ConfigError> {
        let mut config = Self::load_defaults_with_files()?;
        config.apply_env_overrides();
        config.apply_cli_overrides(cli_model, cli_api_key);
        Ok(config)
    }

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

    /// Init project-level config
    pub fn init_project(cwd: &PathBuf) -> Result<PathBuf, ConfigError> {
        let dir = cwd.join(".drift");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        if !path.exists() {
            let default = Self::default_template();
            std::fs::write(&path, default)?;
        }
        Ok(path)
    }

    /// Global config directory (platform-aware via `directories` crate)
    pub fn global_config_dir() -> Option<PathBuf> {
        directories::ProjectDirs::from("com", "driftcli", "DriftCLI")
            .map(|d| d.config_dir().to_path_buf())
    }

    /// Global config file path
    pub fn global_config_path() -> Option<PathBuf> {
        Self::global_config_dir().map(|d| d.join("config.toml"))
    }

    /// Project config file path
    pub fn project_config_path(cwd: &PathBuf) -> PathBuf {
        cwd.join(".drift").join("config.toml")
    }

    fn load_defaults_with_files() -> Result<Self, ConfigError> {
        // Start with hardcoded defaults
        let mut config = Self {
            agent: AgentConfig {
                model: default_model(),
                max_iterations: default_max_iterations(),
                temperature: None,
                thinking_budget: None,
            },
            llm: LlmConfig::Anthropic {
                api_key: String::new(),
                model: default_model(),
                base_url: default_anthropic_base_url(),
            },
        };

        // Layer 1: global config
        if let Some(global_path) = Self::global_config_path() {
            if global_path.exists() {
                config = Self::merge_toml_file(config, &global_path)?;
            }
        }

        // Layer 2: project config (overrides global)
        // NOTE: We don't know cwd at this point; caller should call apply_project_override after
        Ok(config)
    }

    /// Apply project-level config override (call after determining workspace)
    pub fn apply_project_override(&mut self, cwd: &PathBuf) -> Result<(), ConfigError> {
        let project_path = Self::project_config_path(cwd);
        if project_path.exists() {
            let partial: toml::Value =
                toml::from_str(&std::fs::read_to_string(&project_path)?)?;
            Self::merge_toml_value(self, &partial);
        }
        Ok(())
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(key) = std::env::var("DRIFT_API_KEY") {
            match &mut self.llm {
                LlmConfig::Anthropic { api_key, .. }
                | LlmConfig::OpenAiCompatible { api_key, .. } => *api_key = key,
            }
        }
        if let Ok(model) = std::env::var("DRIFT_MODEL") {
            self.agent.model = model;
        }
    }

    fn apply_cli_overrides(&mut self, cli_model: Option<&str>, cli_api_key: Option<&str>) {
        if let Some(m) = cli_model {
            self.agent.model = m.to_string();
        }
        if let Some(k) = cli_api_key {
            match &mut self.llm {
                LlmConfig::Anthropic { api_key, .. }
                | LlmConfig::OpenAiCompatible { api_key, .. } => *api_key = k.to_string(),
            }
        }
    }

    fn merge_toml_file(mut base: Self, path: &PathBuf) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let overlay: toml::Value = toml::from_str(&content)?;
        Self::merge_toml_value(&mut base, &overlay);
        Ok(base)
    }

    fn merge_toml_value(config: &mut Self, overlay: &toml::Value) {
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
        }
        if let Some(llm) = overlay.get("llm") {
            if let Some(provider) = llm.get("provider").and_then(|v| v.as_str()) {
                match provider {
                    "anthropic" => {
                        let api_key = llm
                            .get("api_key")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let model = llm
                            .get("model")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&default_model())
                            .to_string();
                        let base_url = llm
                            .get("base_url")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&default_anthropic_base_url())
                            .to_string();
                        config.llm = LlmConfig::Anthropic {
                            api_key,
                            model,
                            base_url,
                        };
                    }
                    "openai_compatible" | "openai-compatible" => {
                        let api_key = llm
                            .get("api_key")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let model = llm
                            .get("model")
                            .and_then(|v| v.as_str())
                            .unwrap_or("gpt-4o")
                            .to_string();
                        let base_url = llm
                            .get("base_url")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&default_openai_compat_base_url())
                            .to_string();
                        let supports_thinking = llm
                            .get("supports_thinking")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        config.llm = LlmConfig::OpenAiCompatible {
                            api_key,
                            model,
                            base_url,
                            supports_thinking,
                        };
                    }
                    _ => {}
                }
            }
        }
    }

    fn default_template() -> String {
        r###"# DriftCLI Configuration
# See dev-docs/11-configuration.md for all options

[agent]
model = "claude-sonnet-4-5-20250101"
max_iterations = 50

# Set your LLM provider:
#   provider = "anthropic"   — Anthropic Claude
#   provider = "openai_compatible"  — OpenAI or any OpenAI-compatible endpoint

[llm]
provider = "anthropic"
model = "claude-sonnet-4-5-20250101"
api_key = ""
base_url = "https://api.anthropic.com/v1"
"###
        .to_string()
    }

    /// Extract a display summary for the /connect command
    pub fn connection_summary(&self) -> String {
        let (provider_name, api_key, model, base_url) = match &self.llm {
            LlmConfig::Anthropic {
                api_key,
                model,
                base_url,
            } => ("Anthropic", api_key.as_str(), model.as_str(), base_url.as_str()),
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
            format!(
                "{}...{}",
                &api_key[..4],
                &api_key[api_key.len() - 4..]
            )
        };
        format!(
            "Provider: {}\nModel: {}\nEndpoint: {}\nAPI Key: {}",
            provider_name, model, base_url, key_masked
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        // Test that load with no files and no env works
        // We can't fully test load() without filesystem, but test the template
        let tmpl = AppConfig::default_template();
        assert!(tmpl.contains("[agent]"));
        assert!(tmpl.contains("[llm]"));
    }

    #[test]
    fn test_connection_summary() {
        let config = AppConfig {
            agent: AgentConfig {
                model: "test-model".into(),
                max_iterations: 50,
                temperature: None,
                thinking_budget: None,
            },
            llm: LlmConfig::Anthropic {
                api_key: "sk-ant-test1234".into(),
                model: "test-model".into(),
                base_url: "https://api.anthropic.com/v1".into(),
            },
        };
        let summary = config.connection_summary();
        assert!(summary.contains("Anthropic"));
        assert!(summary.contains("sk-a...1234"));
    }
}
