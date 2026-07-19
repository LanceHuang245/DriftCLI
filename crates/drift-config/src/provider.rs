use super::{AppConfig, ConfigError, LlmConfig, ProviderEntry};

impl AppConfig {
    // Returns the LlmConfig of the currently active provider, or the first provider if active is unset.
    pub fn active_llm_config(&self) -> Option<&LlmConfig> {
        self.providers
            .get(&self.active_provider)
            .map(|entry| &entry.config)
            .or_else(|| self.providers.values().next().map(|entry| &entry.config))
    }

    // Returns a mutable reference to the active provider's config.
    pub fn active_llm_config_mut(&mut self) -> Option<&mut LlmConfig> {
        self.providers
            .get_mut(&self.active_provider)
            .map(|entry| &mut entry.config)
    }

    // Returns a vector of all provider names for the /provider picker.
    pub fn list_provider_names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    // Returns the LlmConfig for a specific provider by name.
    pub fn get_provider_config(&self, name: &str) -> Option<&LlmConfig> {
        self.providers.get(name).map(|entry| &entry.config)
    }

    // Adds or updates a named provider and selects the first provider automatically.
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

    // Removes a named provider and selects a remaining provider when necessary.
    pub fn remove_provider(&mut self, name: &str) {
        self.providers.remove(name);
        if self.active_provider == name {
            self.active_provider = self.providers.keys().next().cloned().unwrap_or_default();
        }
    }

    // Switches to the given provider name when it exists.
    pub fn activate_provider(&mut self, name: &str) -> Result<(), ConfigError> {
        if !self.providers.contains_key(name) {
            return Err(ConfigError::NoLlmProvider);
        }
        self.active_provider = name.to_string();
        Ok(())
    }

    // Overrides the selected security profile's approval policy from the CLI.
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
}
