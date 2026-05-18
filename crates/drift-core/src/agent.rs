use std::path::PathBuf;

use crate::event::{AgentState, EventMsg};
use drift_config::{AppConfig, LlmConfig};
use drift_llm::{create_provider, fetch_anthropic_models, fetch_openai_compat_models, LlmError, LlmMessage, LlmProvider, ModelInfo};
use tokio::sync::broadcast;
use tracing::info;

// Agent: orchestrates a chat session — holds config, LLM provider, event bus, and message history.
pub struct Agent {
    config: AppConfig,
    llm: Box<dyn LlmProvider>,
    event_tx: broadcast::Sender<EventMsg>,
    messages: Vec<LlmMessage>,
    cwd: PathBuf,
}

impl Agent {
    // Create a new agent: builds the LLM provider, opens a broadcast channel, and logs the connection.
    pub fn new(config: AppConfig, cwd: PathBuf) -> Result<Self, LlmError> {
        let llm = create_provider(&config.llm)?;
        let (event_tx, _) = broadcast::channel(256);

        info!(
            provider = %llm.provider_id(),
            model = %llm.model_name(),
            "Agent created"
        );

        Ok(Self {
            config,
            llm,
            event_tx,
            messages: Vec::new(),
            cwd,
        })
    }

    // Subscribe returns a new broadcast receiver for consuming agent events in the TUI bridge.
    /// Get a receiver for agent events (for the TUI to subscribe to).
    pub fn subscribe(&self) -> broadcast::Receiver<EventMsg> {
        self.event_tx.subscribe()
    }

    // Submit: sends user input to the LLM, streams chunks as events, and appends the reply to history.
    /// Submit user input and stream the response.
    /// Returns immediately; events are sent via the broadcast channel.
    pub async fn submit(&mut self, user_input: String) {
        let _ = self
            .event_tx
            .send(EventMsg::AgentState(AgentState::Thinking));

        // Add user message to history
        self.messages.push(LlmMessage::user(user_input));

        // Get system prompt from config or use default
        let system_prompt = self.get_system_prompt();

        // Stream from LLM
        match self
            .llm
            .stream_chat(
                self.messages.clone(),
                system_prompt,
                self.config.agent.temperature,
                Some(4096),
            )
            .await
        {
            Ok(mut stream) => {
                let mut full_response = String::new();
                let mut full_reasoning = String::new();
                let mut streaming = false;

                loop {
                    match stream.next().await {
                        Some(Ok(drift_llm::LlmChunk::TextDelta(text))) => {
                            if !streaming {
                                if !full_reasoning.is_empty() {
                                    let _ = self.event_tx.send(EventMsg::Reasoning(full_reasoning.clone()));
                                }
                                let _ = self.event_tx.send(EventMsg::AgentState(
                                    AgentState::Generating(String::new()),
                                ));
                                streaming = true;
                            }
                            full_response.push_str(&text);
                            let _ = self.event_tx.send(EventMsg::Token(text));
                        }
                        Some(Ok(drift_llm::LlmChunk::ReasoningDelta(text))) => {
                            full_reasoning.push_str(&text);
                            let _ = self.event_tx.send(EventMsg::Reasoning(text));
                        }
                        Some(Ok(drift_llm::LlmChunk::Done)) => break,
                        Some(Err(e)) => {
                            let _ = self.event_tx.send(EventMsg::Error {
                                message: e.to_string(),
                                recoverable: true,
                            });
                            return;
                        }
                        None => break,
                    }
                }

                // Add assistant response to history
                if !full_response.is_empty() {
                    self.messages.push(LlmMessage::assistant(full_response));
                }
            }
            Err(e) => {
                let _ = self.event_tx.send(EventMsg::Error {
                    message: format!("LLM error: {}", e),
                    recoverable: matches!(e, LlmError::Stream(_)),
                });
            }
        }

        let _ = self.event_tx.send(EventMsg::AgentState(AgentState::Idle));
        let _ = self.event_tx.send(EventMsg::Done);
    }

    // Returns a human-readable summary of the current connection (provider, model, endpoint, key).
    /// Get the connection summary (for /connect display)
    pub fn connection_summary(&self) -> String {
        self.config.connection_summary()
    }

    // Returns a static identifier string for the current LLM provider (e.g. "Anthropic").
    /// Provider ID
    pub fn provider_id(&self) -> &str {
        self.llm.provider_id()
    }

    // Returns the currently configured model name string.
    /// Model name
    pub fn model_name(&self) -> &str {
        self.llm.model_name()
    }

    // Clones the broadcast sender so external consumers can emit events through the agent's channel.
    /// Get the event sender for this agent
    pub fn event_sender(&self) -> broadcast::Sender<EventMsg> {
        self.event_tx.clone()
    }

    // Reconfigure: swaps the LLM provider at runtime (e.g. from /connect) and persists the change to disk.
    /// Reconfigure the LLM provider at runtime and persist to disk
    pub async fn reconfigure(&mut self, llm_config: LlmConfig) -> Result<(), LlmError> {
        let model = match &llm_config {
            LlmConfig::Anthropic { model, .. } => model.clone(),
            LlmConfig::OpenAiCompatible { model, .. } => model.clone(),
        };
        self.config.agent.model = model;
        self.config.llm = llm_config;
        self.llm = create_provider(&self.config.llm)?;
        if let Err(e) = self.config.save_to_project(&self.cwd) {
            let _ = self.event_tx.send(EventMsg::Error {
                message: format!("Failed to save config: {}", e),
                recoverable: true,
            });
        }
        info!(
            provider = %self.llm.provider_id(),
            model = %self.llm.model_name(),
            "Agent reconfigured and config saved"
        );
        Ok(())
    }

    // Static method: queries the provider's API for available model IDs without needing an Agent instance.
    /// Fetch available models from the given provider params
    pub async fn fetch_models(
        provider: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<Vec<ModelInfo>, LlmError> {
        match provider {
            "Anthropic" => fetch_anthropic_models(api_key, base_url).await,
            "OpenAI Compatible" => {
                let ids = fetch_openai_compat_models(api_key, base_url).await?;
                Ok(ids.into_iter().map(|id| ModelInfo { id, effort_levels: vec![] }).collect())
            }
            _ => Err(LlmError::Config(format!(
                "Unknown provider: {}",
                provider
            ))),
        }
    }

    // Builds the system prompt injected at the start of every conversation turn.
    fn get_system_prompt(&self) -> Option<String> {
        Some(format!(
            "You are DriftCLI, a helpful AI coding assistant running in the terminal.\n\
             You are powered by {} (model: {}).\n\
             Answer concisely and help with software engineering tasks.",
            self.llm.provider_id(),
            self.llm.model_name(),
        ))
    }
}
