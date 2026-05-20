use std::collections::HashMap;
use std::path::PathBuf;

use crate::event::{AgentState, EventMsg};
use drift_config::{AppConfig, LlmConfig};
use drift_llm::{
    create_provider, fetch_anthropic_models, fetch_openai_compat_models, LlmChunk, LlmError,
    LlmMessage, LlmProvider, ModelInfo,
};
use drift_tools::{ToolContext, ToolRegistry};
use tokio::sync::broadcast;
use tracing::{info, warn};

// Track state of a tool call being accumulated from streaming chunks.
struct ActiveToolCall {
    id: String,
    name: String,
    args: Vec<String>,
}

impl ActiveToolCall {
    fn args_string(&self) -> String {
        self.args.join("")
    }
}

// Agent: orchestrates a chat session with tool calling — holds config, LLM provider,
// tool registry, event bus, message history, and working directory.
pub struct Agent {
    config: AppConfig,
    llm: Box<dyn LlmProvider>,
    tool_registry: ToolRegistry,
    event_tx: broadcast::Sender<EventMsg>,
    messages: Vec<LlmMessage>,
    cwd: PathBuf,
}

impl Agent {
    // Create a new agent: builds the LLM provider and tool registry, opens a broadcast channel.
    pub fn new(
        config: AppConfig,
        cwd: PathBuf,
        tool_registry: ToolRegistry,
    ) -> Result<Self, LlmError> {
        let llm = create_provider(config.active_llm_config().unwrap())?;
        let (event_tx, _) = broadcast::channel(256);

        info!(
            provider = %llm.provider_id(),
            model = %llm.model_name(),
            "Agent created"
        );

        Ok(Self {
            config,
            llm,
            tool_registry,
            event_tx,
            messages: Vec::new(),
            cwd,
        })
    }

    // Subscribe returns a new broadcast receiver for consuming agent events in the TUI bridge.
    pub fn subscribe(&self) -> broadcast::Receiver<EventMsg> {
        self.event_tx.subscribe()
    }

    // Submit: sends user input to the LLM, handles tool calls in a loop,
    // streams chunks as events, and appends the reply to history.
    pub async fn submit(&mut self, user_input: String) {
        let _ = self
            .event_tx
            .send(EventMsg::AgentState(AgentState::Thinking));

        // Add user message to history
        self.messages.push(LlmMessage::user(user_input));

        let system_prompt = self.get_system_prompt();
        let max_iterations = self.config.agent.max_iterations;

        // Tool calling loop: iterate until LLM stops requesting tools or max reached
        for iteration in 0..=max_iterations {
            // Collect tool definitions for the LLM
            let tool_defs = self.tool_registry.definitions().await;
            let tools_json: Vec<serde_json::Value> = tool_defs
                .iter()
                .map(|d| {
                    serde_json::json!({
                        "name": d.name,
                        "description": d.description,
                        "input_schema": d.input_schema,
                    })
                })
                .collect();
            let tools = if tools_json.is_empty() {
                None
            } else {
                tracing::info!(tool_count = tools_json.len(), "sending tools to LLM");
                Some(tools_json)
            };

            // Stream from LLM
            let stream_result = self
                .llm
                .stream_chat(
                    self.messages.clone(),
                    system_prompt.clone(),
                    self.config.agent.temperature,
                    Some(4096),
                    tools,
                )
                .await;

            let mut stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    let _ = self.event_tx.send(EventMsg::Error {
                        message: format!("LLM error: {}", e),
                        recoverable: matches!(e, LlmError::Stream(_)),
                    });
                    break;
                }
            };

            // Track accumulated state for this turn
            let mut full_response = String::new();
            let mut full_reasoning = String::new();
            let mut streaming = false;
            // Map call_id -> ActiveToolCall for correlating chunks
            let mut active_tool_calls: HashMap<String, ActiveToolCall> = HashMap::new();
            // Completed tool calls ready for execution (preserve order)
            let mut completed_tool_calls: Vec<ActiveToolCall> = Vec::new();

            // Process stream chunks
            loop {
                match stream.next().await {
                    Some(Ok(LlmChunk::TextDelta(text))) => {
                        if !streaming {
                            if !full_reasoning.is_empty() {
                                let _ = self
                                    .event_tx
                                    .send(EventMsg::Reasoning(full_reasoning.clone()));
                            }
                            let _ = self.event_tx.send(EventMsg::AgentState(
                                AgentState::Generating(String::new()),
                            ));
                            streaming = true;
                        }
                        full_response.push_str(&text);
                        let _ = self.event_tx.send(EventMsg::Token(text));
                    }
                    Some(Ok(LlmChunk::ReasoningDelta(text))) => {
                        full_reasoning.push_str(&text);
                        let _ = self.event_tx.send(EventMsg::Reasoning(text));
                    }
                    Some(Ok(LlmChunk::ToolCallStart { id, name })) => {
                        let _ = self.event_tx.send(EventMsg::ToolCallStart {
                            id: id.clone(),
                            name: name.clone(),
                        });
                        active_tool_calls.insert(
                            id.clone(),
                            ActiveToolCall {
                                id,
                                name,
                                args: Vec::new(),
                            },
                        );
                    }
                    Some(Ok(LlmChunk::ToolCallArgs { id, delta })) => {
                        // Some providers (DeepSeek) omit the id in subsequent
                        // tool-call deltas; fall back to the first active call.
                        let effective_id = if id.is_empty() {
                            active_tool_calls.keys().next().cloned().unwrap_or_default()
                        } else {
                            id
                        };
                        let _ = self.event_tx.send(EventMsg::ToolCallArgs {
                            id: effective_id.clone(),
                            delta: delta.clone(),
                        });
                        if let Some(tc) = active_tool_calls.get_mut(&effective_id) {
                            tc.args.push(delta);
                        }
                    }
                    Some(Ok(LlmChunk::ToolCallEnd { id })) => {
                        let _ = self
                            .event_tx
                            .send(EventMsg::ToolCallEnd { id: id.clone() });
                        if let Some(tc) = active_tool_calls.remove(&id) {
                            completed_tool_calls.push(tc);
                        }
                    }
                    Some(Ok(LlmChunk::Done)) => {
                        // Drain remaining active tool calls — some providers
                        // (OpenAI compat) don't emit ToolCallEnd, so everything
                        // left in active_tool_calls is a complete tool call.
                        for (_, tc) in active_tool_calls.drain() {
                            completed_tool_calls.push(tc);
                        }
                        break;
                    }
                    Some(Err(e)) => {
                        let _ = self.event_tx.send(EventMsg::Error {
                            message: e.to_string(),
                            recoverable: true,
                        });
                        break;
                    }
                    None => break,
                }
            }

            // If no tool calls were completed, this is a text-only response — finalize
            if completed_tool_calls.is_empty() {
                if !full_response.is_empty() {
                    self.messages.push(LlmMessage::assistant(full_response));
                }
                break;
            }

            // Build assistant message with unified ContentParts (provider-agnostic).
            let mut content_parts: Vec<drift_llm::ContentPart> = Vec::new();
            if !full_reasoning.is_empty() {
                content_parts.push(drift_llm::ContentPart::Reasoning(std::mem::take(
                    &mut full_reasoning,
                )));
            }
            if !full_response.is_empty() {
                content_parts.push(drift_llm::ContentPart::Text(std::mem::take(
                    &mut full_response,
                )));
            }
            for tc in &completed_tool_calls {
                content_parts.push(drift_llm::ContentPart::ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.args_string(),
                });
            }
            self.messages.push(LlmMessage {
                role: "assistant".into(),
                content: content_parts,
            });

            // Execute each tool call sequentially
            let mut tool_result_parts: Vec<drift_llm::ContentPart> = Vec::new();
            for tc in &completed_tool_calls {
                let args: serde_json::Value =
                    serde_json::from_str(&tc.args_string()).unwrap_or_default();

                let _ = self.event_tx.send(EventMsg::ToolExecStart {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                });

                let ctx = ToolContext {
                    session_id: uuid::Uuid::new_v4(),
                    working_dir: self.cwd.clone(),
                    tool_call_id: tc.id.clone(),
                };

                let result = self
                    .tool_registry
                    .execute(&tc.name, args, &ctx)
                    .await;

                match result {
                    Ok(r) => {
                        let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            success: r.success,
                            error: r.error.clone(),
                        });
                        let result_content = if r.success {
                            r.content
                        } else {
                            match &r.error {
                                Some(err) if !r.content.is_empty() => {
                                    format!("{} — Error: {}", r.content, err)
                                }
                                Some(err) => format!("Error: {}", err),
                                None => r.content,
                            }
                        };
                        tool_result_parts.push(drift_llm::ContentPart::ToolResult {
                            tool_use_id: tc.id.clone(),
                            content: result_content,
                            is_error: !r.success,
                        });
                    }
                    Err(e) => {
                        let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            success: false,
                            error: Some(e.to_string()),
                        });
                        tool_result_parts.push(drift_llm::ContentPart::ToolResult {
                            tool_use_id: tc.id.clone(),
                            content: format!("Error: {}", e),
                            is_error: true,
                        });
                    }
                }
            }

            // Add tool results as a user message (provider-agnostic).
            self.messages.push(LlmMessage {
                role: "user".into(),
                content: tool_result_parts,
            });

            // Warn if we're approaching the iteration limit
            if iteration >= max_iterations.saturating_sub(2) {
                warn!(
                    "Tool calling loop iteration {}/{}",
                    iteration, max_iterations
                );
            }
        }

        // Finalize
        let _ = self.event_tx.send(EventMsg::AgentState(AgentState::Idle));
        let _ = self.event_tx.send(EventMsg::Done);
    }

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
        let llm_config = self.config
            .active_llm_config()
            .ok_or(LlmError::Config("No provider config".into()))?;
        self.llm = create_provider(llm_config)?;
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
        let llm_config = self.config
            .active_llm_config()
            .ok_or(LlmError::Config("No config".into()))?;
        self.llm = create_provider(llm_config)?;
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

    // Builds the system prompt for every conversation turn.
    fn get_system_prompt(&self) -> Option<String> {
        Some(
            "You are DriftCLI, a terminal-based AI coding agent with direct access to tools.\n\
             You can read files, edit code, run shell commands, search code, fetch web pages, and manage tasks.\n\
             \n\
             CRITICAL RULES:\n\
             1. When the user asks you to do something, CALL THE TOOL immediately. Do NOT narrate your plan.\n\
             2. Do NOT say things like \"Let me check\", \"I'll try\", \"I need to\" — just invoke the tool.\n\
             3. Your first response to any file/action request should be a tool call, NOT a text description.\n\
             4. Only output text AFTER you have tool results to summarize.\n\
             5. For simple greetings or questions requiring no file access, respond directly.\n\
             \n\
             Be concise. Act, don't talk.".into(),
        )
    }
}
