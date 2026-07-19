use super::*;
use std::collections::HashMap;
use std::path::Path;

impl AppConfig {
    // Loads global configuration before applying environment and CLI overrides.
    pub fn load(cli_model: Option<&str>, cli_api_key: Option<&str>) -> Result<Self, ConfigError> {
        let mut config = Self::load_defaults_with_files()?;
        config.apply_env_overrides();
        config.apply_cli_overrides(cli_model, cli_api_key);
        config.mcp.validate()?;
        Ok(config)
    }

    // Loads workspace and explicit configuration before runtime overrides.
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

    // Builds hardcoded defaults and overlays the global config file when present.
    pub(crate) fn load_defaults_with_files() -> Result<Self, ConfigError> {
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

        if let Some(global_path) = Self::global_config_path()
            && global_path.exists()
        {
            config = Self::merge_toml_file(config, &global_path)?;
        }
        Ok(config)
    }

    // Merges the workspace config into the current layered configuration.
    pub fn apply_project_override(&mut self, cwd: &Path) -> Result<(), ConfigError> {
        let project_path = Self::project_config_path(cwd);
        if project_path.exists() {
            let partial: toml::Value = toml::from_str(&std::fs::read_to_string(&project_path)?)?;
            Self::merge_toml_value(self, &partial)?;
        }
        Ok(())
    }

    // Applies environment values without changing the persisted configuration source.
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

    // Applies CLI values as the highest-priority runtime overrides.
    fn apply_cli_overrides(&mut self, cli_model: Option<&str>, cli_api_key: Option<&str>) {
        if let Some(model) = cli_model {
            self.agent.model = model.to_string();
        }
        if let Some(key) = cli_api_key
            && let Some(config) = self.active_llm_config_mut()
        {
            match config {
                LlmConfig::Anthropic { api_key, .. }
                | LlmConfig::OpenAiCompatible { api_key, .. } => {
                    *api_key = key.to_string();
                }
            }
        }
    }

    // Reads and merges one TOML configuration layer.
    fn merge_toml_file(mut base: Self, path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let overlay: toml::Value = toml::from_str(&content)?;
        Self::merge_toml_value(&mut base, &overlay)?;
        Ok(base)
    }

    // Merges supported TOML sections into an existing configuration.
    pub(crate) fn merge_toml_value(
        config: &mut Self,
        overlay: &toml::Value,
    ) -> Result<(), ConfigError> {
        if let Some(agent) = overlay.get("agent") {
            if let Some(model) = agent.get("model").and_then(|value| value.as_str()) {
                config.agent.model = model.to_string();
            }
            if let Some(iterations) = agent
                .get("max_iterations")
                .and_then(|value| value.as_integer())
            {
                config.agent.max_iterations = iterations as usize;
            }
            if let Some(temperature) = agent.get("temperature").and_then(|value| value.as_float()) {
                config.agent.temperature = Some(temperature);
            }
            if let Some(budget) = agent
                .get("thinking_budget")
                .and_then(|value| value.as_integer())
            {
                config.agent.thinking_budget = Some(budget as usize);
            }
            if let Some(effort) = agent
                .get("reasoning_effort")
                .and_then(|value| value.as_str())
            {
                config.agent.reasoning_effort = Some(effort.to_string());
            }
            if let Some(enabled) = agent
                .get("auto_compaction")
                .and_then(|value| value.as_bool())
            {
                config.agent.auto_compaction = enabled;
            }
            if let Some(threshold) = agent
                .get("compaction_threshold")
                .and_then(|value| value.as_float())
            {
                config.agent.compaction_threshold = threshold;
            }
            if let Some(target) = agent
                .get("compaction_target")
                .and_then(|value| value.as_float())
            {
                config.agent.compaction_target = target;
            }
        }

        if let Some(active) = overlay
            .get("active_provider")
            .and_then(|value| value.as_str())
        {
            config.active_provider = active.to_string();
        }

        if let Some(providers) = overlay.get("providers").and_then(|value| value.as_array()) {
            for entry in providers {
                let Some(name) = entry.get("name").and_then(|value| value.as_str()) else {
                    continue;
                };
                let Some(provider) = entry.get("provider").and_then(|value| value.as_str()) else {
                    continue;
                };
                let provider_config = match provider {
                    "anthropic" => LlmConfig::Anthropic {
                        api_key: entry
                            .get("api_key")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .into(),
                        model: entry
                            .get("model")
                            .and_then(|value| value.as_str())
                            .unwrap_or(&default_model())
                            .into(),
                        base_url: entry
                            .get("base_url")
                            .and_then(|value| value.as_str())
                            .unwrap_or(&default_anthropic_base_url())
                            .into(),
                        reasoning_effort: entry
                            .get("reasoning_effort")
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                    },
                    "openai_compatible" | "openai-compatible" => LlmConfig::OpenAiCompatible {
                        api_key: entry
                            .get("api_key")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .into(),
                        model: entry
                            .get("model")
                            .and_then(|value| value.as_str())
                            .unwrap_or("gpt-4o")
                            .into(),
                        base_url: entry
                            .get("base_url")
                            .and_then(|value| value.as_str())
                            .unwrap_or(&default_openai_compat_base_url())
                            .into(),
                        supports_thinking: entry
                            .get("supports_thinking")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                    },
                    _ => continue,
                };
                config.providers.insert(
                    name.to_string(),
                    ProviderEntry {
                        name: name.to_string(),
                        config: provider_config,
                    },
                );
            }
        }

        if let Some(mcp) = overlay.get("mcp").and_then(|value| value.as_table()) {
            if let Some(enabled) = mcp.get("enabled").and_then(|value| value.as_bool()) {
                config.mcp.enabled = enabled;
            }
            if let Some(servers) = mcp.get("servers").and_then(|value| value.as_array()) {
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

        if let Some(llm) = overlay.get("llm")
            && config.providers.is_empty()
            && let Some(provider) = llm.get("provider").and_then(|value| value.as_str())
        {
            let legacy_config = match provider {
                "anthropic" => Some(LlmConfig::Anthropic {
                    api_key: llm
                        .get("api_key")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .into(),
                    model: llm
                        .get("model")
                        .and_then(|value| value.as_str())
                        .unwrap_or(&default_model())
                        .into(),
                    base_url: llm
                        .get("base_url")
                        .and_then(|value| value.as_str())
                        .unwrap_or(&default_anthropic_base_url())
                        .into(),
                    reasoning_effort: llm
                        .get("reasoning_effort")
                        .and_then(|value| value.as_str())
                        .map(str::to_string),
                }),
                "openai_compatible" | "openai-compatible" => Some(LlmConfig::OpenAiCompatible {
                    api_key: llm
                        .get("api_key")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .into(),
                    model: llm
                        .get("model")
                        .and_then(|value| value.as_str())
                        .unwrap_or("gpt-4o")
                        .into(),
                    base_url: llm
                        .get("base_url")
                        .and_then(|value| value.as_str())
                        .unwrap_or(&default_openai_compat_base_url())
                        .into(),
                    supports_thinking: llm
                        .get("supports_thinking")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false),
                }),
                _ => None,
            };
            if let Some(config_llm) = legacy_config {
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

        if let Some(security) = overlay.get("security").and_then(|value| value.as_table()) {
            if let Some(default_profile) = security
                .get("default_profile")
                .and_then(|value| value.as_str())
            {
                config.security.default_profile = default_profile.to_string();
            }
            if let Some(enabled) = security.get("enabled").and_then(|value| value.as_bool()) {
                config.security.enabled = enabled;
            }
            if let Some(domains) = security
                .get("allowed_domains")
                .and_then(|value| value.as_array())
            {
                config.security.allowed_domains = domains
                    .iter()
                    .filter_map(|value| value.as_str().map(String::from))
                    .collect();
            }
            if let Some(domains) = security
                .get("blocked_domains")
                .and_then(|value| value.as_array())
            {
                config.security.blocked_domains = domains
                    .iter()
                    .filter_map(|value| value.as_str().map(String::from))
                    .collect();
            }
            if let Some(profiles) = security.get("profiles").and_then(|value| value.as_table()) {
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
}
