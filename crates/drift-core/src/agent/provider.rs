use super::*;

impl Agent {
    // Returns a human-readable summary of the current connection.
    pub fn connection_summary(&self) -> String {
        self.config.connection_summary()
    }

    // Provider ID string.
    pub fn provider_id(&self) -> &str {
        self.llm.provider_id()
    }

    // Currently configured model name string.
    pub fn model_name(&self) -> &str {
        self.llm.model_name()
    }

    // Clones the broadcast sender for external consumers.
    pub fn event_sender(&self) -> broadcast::Sender<EventMsg> {
        self.event_tx.clone()
    }

    // Reconfigure: swaps the LLM provider at runtime and persists to disk.
    pub async fn reconfigure(&mut self, llm_config: LlmConfig) -> Result<(), LlmError> {
        let model = match &llm_config {
            LlmConfig::Anthropic { model, .. } => model.clone(),
            LlmConfig::OpenAiCompatible { model, .. } => model.clone(),
        };
        self.config.agent.model = model;
        let name = self.config.active_provider.clone();
        self.save_provider(name, llm_config).await
    }

    // Save a named provider config and activate it.
    pub async fn save_provider(&mut self, name: String, config: LlmConfig) -> Result<(), LlmError> {
        self.config.save_provider(name.clone(), config);
        let llm_config = self
            .config
            .active_llm_config()
            .ok_or(LlmError::Config("No provider config".into()))?;
        self.llm = create_provider(llm_config)?;
        self.context.set_context_window(self.llm.context_window());
        self.config
            .save_to_project(&self.cwd)
            .map_err(|e| LlmError::Config(e.to_string()))?;
        let _ = self.event_tx.send(EventMsg::ProviderSwitched {
            name: self.config.active_provider.clone(),
            model: self.llm.model_name().to_string(),
        });
        info!(
            provider = %self.llm.provider_id(),
            model = %self.llm.model_name(),
            "Provider saved and activated"
        );
        Ok(())
    }

    // Switch to an existing named provider.
    pub async fn activate_provider(&mut self, name: &str) -> Result<(), LlmError> {
        self.config
            .activate_provider(name)
            .map_err(|e| LlmError::Config(e.to_string()))?;
        let llm_config = self
            .config
            .active_llm_config()
            .ok_or(LlmError::Config("No config".into()))?;
        self.llm = create_provider(llm_config)?;
        self.context.set_context_window(self.llm.context_window());
        self.config
            .save_to_project(&self.cwd)
            .map_err(|e| LlmError::Config(e.to_string()))?;
        let _ = self.event_tx.send(EventMsg::ProviderSwitched {
            name: name.to_string(),
            model: self.llm.model_name().to_string(),
        });
        info!("Switched to provider {}", name);
        Ok(())
    }

    // Remove a named provider.
    pub async fn remove_provider(&mut self, name: &str) -> Result<(), LlmError> {
        self.config.remove_provider(name);
        if let Some(config) = self.config.active_llm_config() {
            self.llm = create_provider(config)?;
        }
        self.config
            .save_to_project(&self.cwd)
            .map_err(|e| LlmError::Config(e.to_string()))?;
        info!("Provider '{}' removed", name);
        Ok(())
    }

    // Returns all provider names.
    pub fn list_providers(&self) -> Vec<String> {
        self.config.list_provider_names()
    }

    // Returns the full config for a specific provider.
    pub fn get_provider_config(&self, name: &str) -> Option<LlmConfig> {
        self.config.get_provider_config(name).cloned()
    }

    // Static: queries the provider's API for available model IDs.
    pub async fn fetch_models(
        provider: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<Vec<ModelInfo>, LlmError> {
        match provider {
            "Anthropic" => fetch_anthropic_models(api_key, base_url).await,
            "OpenAI Compatible" => {
                let ids = fetch_openai_compat_models(api_key, base_url).await?;
                Ok(ids
                    .into_iter()
                    .map(|id| ModelInfo {
                        id,
                        effort_levels: vec![],
                    })
                    .collect())
            }
            _ => Err(LlmError::Config(format!("Unknown provider: {}", provider))),
        }
    }
}
