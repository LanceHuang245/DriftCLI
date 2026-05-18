use drift_config::LlmConfig;

#[derive(Debug, Clone)]
pub struct ConnectForm {
    pub provider_type: ProviderType,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub selected_field: usize,
    pub show_model_list: bool,
    pub model_list: Vec<String>,
    pub model_list_index: usize,
    pub fetching_models: bool,
    pub status_message: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderType {
    Anthropic,
    OpenAiCompatible,
}

impl ProviderType {
    fn label(&self) -> &str {
        match self {
            ProviderType::Anthropic => "Anthropic",
            ProviderType::OpenAiCompatible => "OpenAI Compatible",
        }
    }
}

impl ConnectForm {
    pub fn from_config(config: &LlmConfig) -> Self {
        match config {
            LlmConfig::Anthropic {
                api_key,
                model,
                base_url,
            } => Self {
                provider_type: ProviderType::Anthropic,
                base_url: base_url.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
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
                provider_type: ProviderType::OpenAiCompatible,
                base_url: base_url.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
                selected_field: 0,
                show_model_list: false,
                model_list: Vec::new(),
                model_list_index: 0,
                fetching_models: false,
                status_message: String::new(),
            },
        }
    }

    pub fn provider_label(&self) -> &str {
        self.provider_type.label()
    }

    pub fn next_field(&mut self) {
        self.selected_field = (self.selected_field + 1) % 6;
        self.show_model_list = false;
    }

    pub fn previous_field(&mut self) {
        if self.selected_field == 0 {
            self.selected_field = 5;
        } else {
            self.selected_field -= 1;
        }
        self.show_model_list = false;
    }

    pub fn on_left(&mut self) {
        if self.selected_field == 0 {
            self.provider_type = match self.provider_type {
                ProviderType::Anthropic => ProviderType::OpenAiCompatible,
                ProviderType::OpenAiCompatible => ProviderType::Anthropic,
            };
        }
    }

    pub fn on_right(&mut self) {
        if self.selected_field == 0 {
            self.provider_type = match self.provider_type {
                ProviderType::Anthropic => ProviderType::OpenAiCompatible,
                ProviderType::OpenAiCompatible => ProviderType::Anthropic,
            };
        }
    }

    pub fn on_char(&mut self, c: char) {
        match self.selected_field {
            1 => self.base_url.push(c),
            2 => self.api_key.push(c),
            3 => self.model.push(c),
            _ => {}
        }
    }

    pub fn on_backspace(&mut self) {
        match self.selected_field {
            1 => {
                self.base_url.pop();
            }
            2 => {
                self.api_key.pop();
            }
            3 => {
                self.model.pop();
            }
            _ => {}
        }
    }

    pub fn select_model(&mut self) {
        if let Some(selected) = self.model_list.get(self.model_list_index) {
            self.model = selected.clone();
            self.show_model_list = false;
            self.status_message = format!("Selected: {}", selected);
        }
    }

    pub fn to_llm_config(&self) -> LlmConfig {
        match self.provider_type {
            ProviderType::Anthropic => LlmConfig::Anthropic {
                api_key: self.api_key.clone(),
                model: self.model.clone(),
                base_url: self.base_url.clone(),
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
