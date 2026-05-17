use crate::event::{AgentState, EventMsg};
use drift_config::AppConfig;
use drift_llm::{create_provider, LlmError, LlmMessage, LlmProvider};
use tokio::sync::broadcast;
use tracing::info;

pub struct Agent {
    config: AppConfig,
    llm: Box<dyn LlmProvider>,
    event_tx: broadcast::Sender<EventMsg>,
    messages: Vec<LlmMessage>,
}

impl Agent {
    /// Create a new Agent from the loaded configuration.
    pub fn new(config: AppConfig) -> Result<Self, LlmError> {
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
        })
    }

    /// Get a receiver for agent events (for the TUI to subscribe to).
    pub fn subscribe(&self) -> broadcast::Receiver<EventMsg> {
        self.event_tx.subscribe()
    }

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
                let mut streaming = false;

                loop {
                    match stream.next().await {
                        Some(Ok(drift_llm::LlmChunk::TextDelta(text))) => {
                            if !streaming {
                                let _ = self.event_tx.send(EventMsg::AgentState(
                                    AgentState::Generating(String::new()),
                                ));
                                streaming = true;
                            }
                            full_response.push_str(&text);
                            let _ = self.event_tx.send(EventMsg::Token(text));
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

    /// Get the connection summary (for /connect display)
    pub fn connection_summary(&self) -> String {
        self.config.connection_summary()
    }

    /// Provider ID
    pub fn provider_id(&self) -> &str {
        self.llm.provider_id()
    }

    /// Model name
    pub fn model_name(&self) -> &str {
        self.llm.model_name()
    }

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
