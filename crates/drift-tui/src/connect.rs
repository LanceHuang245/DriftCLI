use drift_config::LlmConfig;
use drift_llm::ModelInfo;

// Form state for the /connect settings screen. Holds provider, URL, key, model, name, and list UX state.
#[derive(Debug, Clone)]
pub struct ConnectForm {
    // User-chosen label for this provider configuration.
    pub provider_name: String,
    pub provider_type: ProviderType,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub selected_field: usize,
    pub show_model_list: bool,
    pub model_list: Vec<ModelInfo>,
    pub model_list_index: usize,
    pub fetching_models: bool,
    pub status_message: String,
}

// Tracks which LLM provider protocol to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderType {
    Anthropic,
    OpenAiCompatible,
}

impl ProviderType {
    // Human-readable label for the current provider variant.
    fn label(&self) -> &str {
        match self {
            ProviderType::Anthropic => "Anthropic",
            ProviderType::OpenAiCompatible => "OpenAI Compatible",
        }
    }
}

impl ConnectForm {
    // Build a ConnectForm from an existing LlmConfig, preserving current settings.
    pub fn from_config(config: &LlmConfig) -> Self {
        Self::from_entry("default", config)
    }

    // Build a ConnectForm from an explicit provider name and LlmConfig.
    pub fn from_entry(name: &str, config: &LlmConfig) -> Self {
        match config {
            LlmConfig::Anthropic {
                api_key,
                model,
                base_url,
                reasoning_effort,
            } => Self {
                provider_name: name.to_string(),
                provider_type: ProviderType::Anthropic,
                base_url: base_url.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
                reasoning_effort: reasoning_effort.clone(),
                selected_field: 0,
                show_model_list: false,
                model_list: Vec::new(),
                model_list_index: 0,
                fetching_models: false,
                status_message: String::new(),
            },
            LlmConfig::OpenAiCompatible {
                api_key,
                model,
                base_url,
                ..
            } => Self {
                provider_name: name.to_string(),
                provider_type: ProviderType::OpenAiCompatible,
                base_url: base_url.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
                reasoning_effort: None,
                selected_field: 0,
                show_model_list: false,
                model_list: Vec::new(),
                model_list_index: 0,
                fetching_models: false,
                status_message: String::new(),
            },
        }
    }

    // Return the display name of the current provider.
    pub fn provider_label(&self) -> &str {
        self.provider_type.label()
    }

    // Return the provider_name field value.
    pub fn to_provider_name(&self) -> String {
        self.provider_name.clone()
    }

    // Advance selection to the next form field (wrap around). 7 fields: name, provider, url, key, model, save, cancel.
    pub fn next_field(&mut self) {
        self.selected_field = (self.selected_field + 1) % 7;
        self.show_model_list = false;
    }

    // Move selection to the previous form field (wrap around).
    pub fn previous_field(&mut self) {
        if self.selected_field == 0 {
            self.selected_field = 6;
        } else {
            self.selected_field -= 1;
        }
        self.show_model_list = false;
    }

    // Toggle provider type when on field 1 (provider) with Left arrow.
    pub fn on_left(&mut self) {
        if self.selected_field == 1 {
            self.provider_type = match self.provider_type {
                ProviderType::Anthropic => ProviderType::OpenAiCompatible,
                ProviderType::OpenAiCompatible => ProviderType::Anthropic,
            };
        }
    }

    // Toggle provider type when on field 1 (provider) with Right arrow.
    pub fn on_right(&mut self) {
        if self.selected_field == 1 {
            self.provider_type = match self.provider_type {
                ProviderType::Anthropic => ProviderType::OpenAiCompatible,
                ProviderType::OpenAiCompatible => ProviderType::Anthropic,
            };
        }
    }

    // Append a typed character to the currently selected text field.
    pub fn on_char(&mut self, c: char) {
        match self.selected_field {
            0 => self.provider_name.push(c),
            2 => self.base_url.push(c),
            3 => self.api_key.push(c),
            4 => self.model.push(c),
            _ => {}
        }
    }

    // Remove the last character from the currently selected text field.
    pub fn on_backspace(&mut self) {
        match self.selected_field {
            0 => {
                self.provider_name.pop();
            }
            2 => {
                self.base_url.pop();
            }
            3 => {
                self.api_key.pop();
            }
            4 => {
                self.model.pop();
            }
            _ => {}
        }
    }

    // Pick the currently highlighted model from the dropdown list.
    pub fn select_model(&mut self) {
        if let Some(selected) = self.model_list.get(self.model_list_index) {
            self.model = selected.id.clone();
            self.show_model_list = false;
            self.status_message = format!("Selected: {}", selected.id);
        }
    }

    // Convert the form state into a full LlmConfig for reconfiguration.
    pub fn to_llm_config(&self) -> LlmConfig {
        match self.provider_type {
            ProviderType::Anthropic => LlmConfig::Anthropic {
                api_key: self.api_key.clone(),
                model: self.model.clone(),
                base_url: self.base_url.clone(),
                reasoning_effort: self.reasoning_effort.clone(),
            },
            ProviderType::OpenAiCompatible => LlmConfig::OpenAiCompatible {
                api_key: self.api_key.clone(),
                model: self.model.clone(),
                base_url: self.base_url.clone(),
                supports_thinking: false,
            },
        }
    }
}
