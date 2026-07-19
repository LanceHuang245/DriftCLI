use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::context::ContextManager;
use crate::event::{AgentState, EventMsg};
use drift_config::{AppConfig, LlmConfig};
use drift_llm::{
    ContentPart, LlmChunk, LlmError, LlmMessage, LlmProvider, ModelInfo, create_provider,
    fetch_anthropic_models, fetch_openai_compat_models,
};
use drift_security::{PermissionDecision, PermissionEngine, ProcessSandbox, SecurityConfig};
use drift_tools::{ToolContext, ToolRegistry};
use tokio::sync::broadcast;
use tracing::{info, warn};

// Track state of a tool call being accumulated from streaming chunks.
struct ActiveToolCall {
    id: String,
    name: String,
    args: Vec<String>,
}

/// Convert provider messages into storage DTOs without adding drift-llm to storage.
fn to_persisted_messages(messages: &[LlmMessage]) -> Vec<drift_storage::PersistedMessage> {
    messages
        .iter()
        .map(|message| drift_storage::PersistedMessage {
            role: message.role.clone(),
            content: message
                .content
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => {
                        drift_storage::PersistedContentPart::Text(text.clone())
                    }
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => drift_storage::PersistedContentPart::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                    ContentPart::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => drift_storage::PersistedContentPart::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        is_error: *is_error,
                    },
                    ContentPart::Reasoning(text) => {
                        drift_storage::PersistedContentPart::Reasoning(text.clone())
                    }
                })
                .collect(),
        })
        .collect()
}

/// Apply one redaction boundary to every event before transcript persistence.
fn redact_session_event(event: &mut drift_storage::SessionEvent) -> Result<(), serde_json::Error> {
    let mut value = serde_json::to_value(&*event)?;
    drift_security::SensitiveDataFilter::filter_json(&mut value);
    *event = serde_json::from_value(value)?;
    Ok(())
}

/// Restore provider messages from the dependency-free storage representation.
fn from_persisted_messages(messages: &[drift_storage::PersistedMessage]) -> Vec<LlmMessage> {
    messages
        .iter()
        .map(|message| LlmMessage {
            role: message.role.clone(),
            content: message
                .content
                .iter()
                .map(|part| match part {
                    drift_storage::PersistedContentPart::Text(text) => {
                        ContentPart::Text(text.clone())
                    }
                    drift_storage::PersistedContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => ContentPart::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                    drift_storage::PersistedContentPart::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => ContentPart::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        is_error: *is_error,
                    },
                    drift_storage::PersistedContentPart::Reasoning(text) => {
                        ContentPart::Reasoning(text.clone())
                    }
                })
                .collect(),
        })
        .collect()
}

/// Rebuild messages while treating the newest compaction snapshot as a replay boundary.
fn replay_history(events: &[drift_storage::SessionEvent]) -> (Vec<LlmMessage>, Option<String>) {
    let mut messages: Vec<LlmMessage> = Vec::new();
    let mut summary = None;

    for event in events {
        match event {
            drift_storage::SessionEvent::Message {
                role,
                content,
                reasoning,
            } => {
                let mut content_parts = Vec::new();
                if let Some(r) = reasoning
                    && !r.is_empty()
                {
                    content_parts.push(ContentPart::Reasoning(r.clone()));
                }
                if !content.is_empty() {
                    content_parts.push(ContentPart::Text(content.clone()));
                }
                messages.push(LlmMessage {
                    role: role.clone(),
                    content: content_parts,
                });
            }
            drift_storage::SessionEvent::ToolCall {
                call_id,
                name,
                args,
            } => {
                if let Some(last) = messages.last_mut()
                    && last.role == "assistant"
                {
                    last.content.push(ContentPart::ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        arguments: match args {
                            serde_json::Value::String(raw) => raw.clone(),
                            value => value.to_string(),
                        },
                    });
                }
            }
            drift_storage::SessionEvent::ToolResult {
                call_id,
                name: _,
                success,
                content,
                error,
            } => {
                let result_content = if *success {
                    content.clone()
                } else {
                    match error {
                        Some(err) if !content.is_empty() => format!("{} — Error: {}", content, err),
                        Some(err) => format!("Error: {}", err),
                        None => content.clone(),
                    }
                };

                let mut needs_new_user_msg = true;
                if let Some(last) = messages.last_mut()
                    && last.role == "user"
                {
                    let only_results = last
                        .content
                        .iter()
                        .all(|part| matches!(part, ContentPart::ToolResult { .. }));
                    if only_results && !last.content.is_empty() {
                        last.content.push(ContentPart::ToolResult {
                            tool_use_id: call_id.clone(),
                            content: result_content.clone(),
                            is_error: !success,
                        });
                        needs_new_user_msg = false;
                    }
                }

                if needs_new_user_msg {
                    messages.push(LlmMessage {
                        role: "user".into(),
                        content: vec![ContentPart::ToolResult {
                            tool_use_id: call_id.clone(),
                            content: result_content,
                            is_error: !success,
                        }],
                    });
                }
            }
            // A persisted snapshot discards all earlier events from the active context.
            drift_storage::SessionEvent::ContextCompacted {
                summary: compacted_summary,
                messages: compacted_messages,
                saved_tokens: _,
            } => {
                messages = from_persisted_messages(compacted_messages);
                summary = compacted_summary.clone();
            }
        }
    }

    (messages, summary)
}

impl ActiveToolCall {
    fn args_string(&self) -> String {
        self.args.join("")
    }
}

// Receive only the decision correlated with the active permission request.
async fn receive_permission_response(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, drift_security::PermissionResponse)>,
    expected_request_id: &str,
) -> Option<drift_security::PermissionResponse> {
    while let Some((request_id, response)) = rx.recv().await {
        if request_id == expected_request_id {
            return Some(response);
        }

        tracing::debug!(
            request_id,
            expected_request_id,
            "Ignoring stale permission response"
        );
    }

    None
}

// Agent: orchestrates a chat session with tool calling — holds config, LLM provider,
// tool registry, event bus, message history, and working directory.
pub struct Agent {
    config: AppConfig,
    llm: Box<dyn LlmProvider>,
    tool_registry: std::sync::Arc<ToolRegistry>,
    /// Permission engine for tool call approval.
    permission_engine: PermissionEngine,
    /// Channel the bridge task writes user permission responses into.
    permission_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<(String, drift_security::PermissionResponse)>>,
    event_tx: broadcast::Sender<EventMsg>,
    context: ContextManager,
    cwd: PathBuf,
    session_id: uuid::Uuid,
    session_store: std::sync::Arc<drift_storage::SessionStore>,
    file_access: std::sync::Arc<drift_security::FileAccessGuard>,
    network: std::sync::Arc<drift_security::NetworkGuard>,
    process_sandbox: std::sync::Arc<ProcessSandbox>,
}

impl Agent {
    // Create a new agent: builds the LLM provider, tool registry, and permission engine.
    pub fn new(
        config: AppConfig,
        cwd: PathBuf,
        tool_registry: std::sync::Arc<ToolRegistry>,
        session_id: uuid::Uuid,
        session_store: std::sync::Arc<drift_storage::SessionStore>,
        security_config: &SecurityConfig,
        security_profile: &str,
    ) -> Result<Self, LlmError> {
        let llm = create_provider(config.active_llm_config().unwrap())?;
        let (event_tx, _) = broadcast::channel(256);
        let permission_engine = PermissionEngine::new(security_config, security_profile);
        let file_access = std::sync::Arc::new(
            permission_engine
                .file_access_guard(&cwd)
                .map_err(|error| LlmError::Config(format!("file access guard: {:?}", error)))?,
        );
        let network = std::sync::Arc::new(permission_engine.network_guard());
        let process_sandbox = std::sync::Arc::new(
            ProcessSandbox::new(permission_engine.sandbox_mode(), &cwd)
                .map_err(|error| LlmError::Config(format!("process sandbox: {error}")))?,
        );
        let context = ContextManager::for_workspace(
            llm.context_window(),
            config.agent.compaction_threshold,
            config.agent.compaction_target,
            &cwd,
        );

        info!(
            provider = %llm.provider_id(),
            model = %llm.model_name(),
            security_profile = permission_engine.profile_name(),
            approval_policy = ?permission_engine.approval_policy(),
            sandbox_mode = ?permission_engine.sandbox_mode(),
            "Agent created"
        );

        Ok(Self {
            config,
            llm,
            tool_registry,
            event_tx,
            context,
            cwd,
            session_id,
            session_store,
            file_access,
            network,
            process_sandbox,
            permission_engine,
            permission_rx: None,
        })
    }

    // Set up the channel through which the TUI bridge sends permission responses back to the agent loop.
    pub fn set_permission_channel(
        &mut self,
        rx: tokio::sync::mpsc::UnboundedReceiver<(String, drift_security::PermissionResponse)>,
    ) {
        self.permission_rx = Some(rx);
    }

    // Set the full conversation history from a reconstructed LlmMessage list (used when resuming a session).
    pub fn set_messages(&mut self, messages: Vec<LlmMessage>) {
        self.context.set_messages(messages);
    }

    // Retrieve active session ID.
    pub fn session_id(&self) -> uuid::Uuid {
        self.session_id
    }

    /// Share the immutable process boundary with MCP child-process management.
    pub fn process_sandbox(&self) -> std::sync::Arc<ProcessSandbox> {
        self.process_sandbox.clone()
    }

    // Switch the active session and rebuild the LLM history from its transcript.
    pub fn switch_session(
        &mut self,
        session_id: uuid::Uuid,
        events: &[drift_storage::SessionEvent],
    ) {
        self.session_id = session_id;
        self.reconstruct_history(events);
    }

    // Reconstruct the core messages history list from a vector of storage events.
    pub fn reconstruct_history(&mut self, events: &[drift_storage::SessionEvent]) {
        let (messages, summary) = replay_history(events);
        self.context.set_compacted_state(messages, summary);
    }

    // Subscribe returns a new broadcast receiver for consuming agent events in the TUI bridge.
    pub fn subscribe(&self) -> broadcast::Receiver<EventMsg> {
        self.event_tx.subscribe()
    }

    /// Redact all transcript payloads immediately before writing them to disk.
    fn append_session_event(
        &self,
        mut event: drift_storage::SessionEvent,
    ) -> Result<(), drift_storage::StorageError> {
        redact_session_event(&mut event)?;
        self.session_store.append_event(self.session_id, &event)
    }

    // Submit: sends user input to the LLM, handles tool calls in a loop,
    // streams chunks as events, and appends the reply to history.
    pub async fn submit(&mut self, user_input: String) {
        let _ = self
            .event_tx
            .send(EventMsg::AgentState(AgentState::Thinking));

        // Add user message to history
        self.context
            .push_message(LlmMessage::user(user_input.clone()));

        // Write user message to SessionStore
        let _ = self.append_session_event(drift_storage::SessionEvent::Message {
            role: "user".to_string(),
            content: user_input,
            reasoning: None,
        });

        let max_iterations = self.config.agent.max_iterations;
        // Suppress Done after any failure while still returning the agent to Idle.
        let mut turn_failed = false;

        // Tool calling loop: iterate until LLM stops requesting tools or max reached
        for iteration in 0..max_iterations {
            // Collect tool definitions for the LLM
            let tool_defs = self.tool_registry.definitions().await;
            let should_compact =
                self.config.agent.auto_compaction && self.context.needs_compaction(&tool_defs);
            if should_compact {
                let _ = self.event_tx.send(EventMsg::ContextCompacting);
            }
            // Prepare a candidate context without mutating the committed conversation.
            let built_context = match self
                .context
                .build_context(
                    &tool_defs,
                    self.config.agent.auto_compaction,
                    Some(self.llm.as_ref()),
                )
                .await
            {
                Ok(context) => context,
                Err(error) => {
                    turn_failed = true;
                    let _ = self.event_tx.send(EventMsg::Error {
                        message: format!("Context error: {}", error),
                        recoverable: true,
                    });
                    break;
                }
            };
            // Persist the candidate before committing it or sending it to the provider.
            if let Some(snapshot) = built_context.compaction.as_ref() {
                let event = drift_storage::SessionEvent::ContextCompacted {
                    summary: snapshot.summary.clone(),
                    messages: to_persisted_messages(&snapshot.messages),
                    saved_tokens: built_context.saved_tokens,
                };
                if let Err(error) = self.append_session_event(event) {
                    turn_failed = true;
                    let _ = self.event_tx.send(EventMsg::Error {
                        message: format!("Context compaction persistence error: {}", error),
                        recoverable: false,
                    });
                    break;
                }
                self.context.apply_compaction(snapshot);
                let summary = snapshot
                    .summary
                    .clone()
                    .unwrap_or_else(|| "Local context compaction completed".into());
                let _ = self.event_tx.send(EventMsg::ContextCompacted {
                    summary,
                    saved_tokens: built_context.saved_tokens,
                });
            }
            if !tool_defs.is_empty() {
                tracing::info!(tool_count = tool_defs.len(), "sending tools to LLM");
            }

            // Stream from LLM
            let stream_result = self
                .llm
                .stream_chat(
                    built_context.messages,
                    built_context.system_prompt,
                    self.config.agent.temperature,
                    Some(4096),
                    built_context.tools,
                )
                .await;

            let mut stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    turn_failed = true;
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
            let mut reasoning_start: Option<Instant> = None;
            let mut reasoning_complete_emitted = false;
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
                            if !full_reasoning.is_empty()
                                && !reasoning_complete_emitted
                                && let Some(start) = reasoning_start
                            {
                                let duration_ms = start.elapsed().as_millis() as u64;
                                let _ = self
                                    .event_tx
                                    .send(EventMsg::ReasoningComplete { duration_ms });
                                reasoning_complete_emitted = true;
                            }
                            let _ = self
                                .event_tx
                                .send(EventMsg::AgentState(AgentState::Generating(String::new())));
                            streaming = true;
                        }
                        full_response.push_str(&text);
                        let _ = self.event_tx.send(EventMsg::Token(text));
                    }
                    Some(Ok(LlmChunk::ReasoningDelta(text))) => {
                        full_reasoning.push_str(&text);
                        if reasoning_start.is_none() {
                            reasoning_start = Some(Instant::now());
                        }
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
                        let _ = self.event_tx.send(EventMsg::ToolCallEnd { id: id.clone() });
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
                        turn_failed = true;
                        let _ = self.event_tx.send(EventMsg::Error {
                            message: e.to_string(),
                            recoverable: true,
                        });
                        break;
                    }
                    None => break,
                }
            }

            // Emit ReasoningComplete for tool-call iterations that had
            // reasoning but no TextDelta (so the flag was never set).
            if !full_reasoning.is_empty()
                && !reasoning_complete_emitted
                && let Some(start) = reasoning_start
            {
                let duration_ms = start.elapsed().as_millis() as u64;
                let _ = self
                    .event_tx
                    .send(EventMsg::ReasoningComplete { duration_ms });
            }

            // If no tool calls were completed, this is a text-only response — finalize
            if completed_tool_calls.is_empty() {
                if !full_response.is_empty() {
                    self.context
                        .push_message(LlmMessage::assistant(full_response.clone()));
                    // Write assistant message to SessionStore
                    let reasoning_opt = if !full_reasoning.is_empty() {
                        Some(full_reasoning)
                    } else {
                        None
                    };
                    let _ = self.append_session_event(drift_storage::SessionEvent::Message {
                        role: "assistant".to_string(),
                        content: full_response,
                        reasoning: reasoning_opt,
                    });
                }
                break;
            }

            // Build assistant message with unified ContentParts (provider-agnostic).
            let mut content_parts: Vec<drift_llm::ContentPart> = Vec::new();
            let mut has_reasoning = false;
            let mut has_text = false;

            if !full_reasoning.is_empty() {
                content_parts.push(drift_llm::ContentPart::Reasoning(full_reasoning.clone()));
                has_reasoning = true;
            }
            if !full_response.is_empty() {
                content_parts.push(drift_llm::ContentPart::Text(full_response.clone()));
                has_text = true;
            }
            for tc in &completed_tool_calls {
                content_parts.push(drift_llm::ContentPart::ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.args_string(),
                });
            }
            self.context.push_message(LlmMessage {
                role: "assistant".into(),
                content: content_parts,
            });

            // Write assistant message and tool calls to SessionStore
            let _ = self.append_session_event(drift_storage::SessionEvent::Message {
                role: "assistant".to_string(),
                content: if has_text {
                    full_response
                } else {
                    String::new()
                },
                reasoning: if has_reasoning {
                    Some(full_reasoning)
                } else {
                    None
                },
            });

            for tc in &completed_tool_calls {
                let raw_args = tc.args_string();
                let args_val = match serde_json::from_str::<serde_json::Value>(&raw_args) {
                    Ok(value) => value,
                    Err(_) => serde_json::Value::String(raw_args),
                };
                let _ = self.append_session_event(drift_storage::SessionEvent::ToolCall {
                    call_id: tc.id.clone(),
                    name: tc.name.clone(),
                    args: args_val,
                });
            }

            // Execute each tool call sequentially
            let mut tool_result_parts: Vec<drift_llm::ContentPart> = Vec::new();
            for tc in &completed_tool_calls {
                let _ = self.event_tx.send(EventMsg::ToolExecStart {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                });

                let raw_args = tc.args_string();
                let args = match serde_json::from_str::<serde_json::Value>(&raw_args) {
                    Ok(value) => value,
                    Err(error) => {
                        // Return malformed tool arguments to the LLM instead of executing with an empty object.
                        let message =
                            format!("Invalid JSON arguments for tool '{}': {}", tc.name, error);
                        let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            success: false,
                            error: Some(message.clone()),
                        });
                        tool_result_parts.push(drift_llm::ContentPart::ToolResult {
                            tool_use_id: tc.id.clone(),
                            content: message.clone(),
                            is_error: true,
                        });
                        let _ =
                            self.append_session_event(drift_storage::SessionEvent::ToolResult {
                                call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                success: false,
                                content: String::new(),
                                error: Some(message),
                            });
                        continue;
                    }
                };

                let ctx = ToolContext {
                    session_id: self.session_id,
                    working_dir: self.cwd.clone(),
                    tool_call_id: tc.id.clone(),
                    file_access: self.file_access.clone(),
                    network: self.network.clone(),
                    process_sandbox: self.process_sandbox.clone(),
                };

                // ── Permission check ──
                let permission_decision = self
                    .permission_engine
                    .check_tool_permission(&tc.name, &args);
                let result: Result<drift_tools::ToolResult, drift_tools::ToolError> =
                    match permission_decision {
                        PermissionDecision::Allowed { .. } => {
                            // Proceed to execute
                            self.tool_registry.execute(&tc.name, args, &ctx).await
                        }
                        PermissionDecision::Denied { reason } => {
                            let _ = self.event_tx.send(EventMsg::ToolExecStart {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                            });
                            let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                success: false,
                                error: Some(reason.clone()),
                            });
                            Err(drift_tools::ToolError::PermissionDenied(reason))
                        }
                        PermissionDecision::AskUser { request } => {
                            // Send permission request to TUI
                            let _ = self.event_tx.send(EventMsg::PermissionRequest {
                                request_id: request.request_id.clone(),
                                tool_name: request.tool_name.clone(),
                                args_summary: request.args_summary.clone(),
                                reason: request.reason.clone(),
                                risk_level: format!("{:?}", request.risk_level),
                            });

                            // Wait for the response matching this request within one total timeout.
                            let response = match &mut self.permission_rx {
                                Some(rx) => tokio::time::timeout(
                                    std::time::Duration::from_secs(120),
                                    receive_permission_response(rx, &request.request_id),
                                )
                                .await
                                .ok()
                                .flatten(),
                                None => {
                                    // No channel configured — deny by default
                                    tracing::warn!(
                                        "Permission channel not set, denying tool call by default"
                                    );
                                    None
                                }
                            };

                            match response {
                                Some(drift_security::PermissionResponse::Allow) => {
                                    let _ = self.event_tx.send(EventMsg::PermissionResolved {
                                        request_id: request.request_id,
                                        allowed: true,
                                    });
                                    self.tool_registry.execute(&tc.name, args, &ctx).await
                                }
                                Some(drift_security::PermissionResponse::AllowAlways) => {
                                    let _ = self.event_tx.send(EventMsg::PermissionResolved {
                                        request_id: request.request_id,
                                        allowed: true,
                                    });
                                    // Record session-persistent allow rule
                                    let pattern = drift_security::DoomLoopTracker::fingerprint(
                                        &tc.name, &args,
                                    );
                                    self.permission_engine.add_session_rule(
                                        &tc.name,
                                        &pattern,
                                        drift_security::types::PermissionAction::Allow,
                                    );
                                    self.tool_registry.execute(&tc.name, args, &ctx).await
                                }
                                Some(drift_security::PermissionResponse::Deny) => {
                                    let reason = "User denied permission".to_string();
                                    let _ = self.event_tx.send(EventMsg::PermissionResolved {
                                        request_id: request.request_id,
                                        allowed: false,
                                    });
                                    let _ = self.event_tx.send(EventMsg::ToolExecStart {
                                        id: tc.id.clone(),
                                        name: tc.name.clone(),
                                    });
                                    let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                                        id: tc.id.clone(),
                                        name: tc.name.clone(),
                                        success: false,
                                        error: Some(reason.clone()),
                                    });
                                    Err(drift_tools::ToolError::PermissionDenied(reason))
                                }
                                Some(drift_security::PermissionResponse::DenyAlways) => {
                                    let reason = "User denied permission".to_string();
                                    let _ = self.event_tx.send(EventMsg::PermissionResolved {
                                        request_id: request.request_id,
                                        allowed: false,
                                    });
                                    // Record session-persistent deny rule
                                    let pattern = drift_security::DoomLoopTracker::fingerprint(
                                        &tc.name, &args,
                                    );
                                    self.permission_engine.add_session_rule(
                                        &tc.name,
                                        &pattern,
                                        drift_security::types::PermissionAction::Deny,
                                    );
                                    let _ = self.event_tx.send(EventMsg::ToolExecStart {
                                        id: tc.id.clone(),
                                        name: tc.name.clone(),
                                    });
                                    let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                                        id: tc.id.clone(),
                                        name: tc.name.clone(),
                                        success: false,
                                        error: Some(reason.clone()),
                                    });
                                    Err(drift_tools::ToolError::PermissionDenied(reason))
                                }
                                None => {
                                    // Timeout or no response — deny
                                    let reason = "Permission request timed out".to_string();
                                    let _ = self.event_tx.send(EventMsg::PermissionResolved {
                                        request_id: request.request_id,
                                        allowed: false,
                                    });
                                    let _ = self.event_tx.send(EventMsg::ToolExecStart {
                                        id: tc.id.clone(),
                                        name: tc.name.clone(),
                                    });
                                    let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                                        id: tc.id.clone(),
                                        name: tc.name.clone(),
                                        success: false,
                                        error: Some(reason.clone()),
                                    });
                                    Err(drift_tools::ToolError::PermissionDenied(reason))
                                }
                            }
                        }
                    };

                match result {
                    Ok(r) => {
                        let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            success: r.success,
                            error: r.error.clone(),
                        });
                        let result_content = if r.success {
                            r.content.clone()
                        } else {
                            match &r.error {
                                Some(err) if !r.content.is_empty() => {
                                    format!("{} — Error: {}", r.content, err)
                                }
                                Some(err) => format!("Error: {}", err),
                                None => r.content.clone(),
                            }
                        };
                        tool_result_parts.push(drift_llm::ContentPart::ToolResult {
                            tool_use_id: tc.id.clone(),
                            content: result_content.clone(),
                            is_error: !r.success,
                        });

                        // Write ToolResult to SessionStore
                        let _ =
                            self.append_session_event(drift_storage::SessionEvent::ToolResult {
                                call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                success: r.success,
                                content: r.content,
                                error: r.error,
                            });
                    }
                    Err(e) => {
                        let _ = self.event_tx.send(EventMsg::ToolExecEnd {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            success: false,
                            error: Some(e.to_string()),
                        });
                        let err_str = e.to_string();
                        tool_result_parts.push(drift_llm::ContentPart::ToolResult {
                            tool_use_id: tc.id.clone(),
                            content: format!("Error: {}", err_str),
                            is_error: true,
                        });

                        // Write ToolResult (Failure) to SessionStore
                        let _ =
                            self.append_session_event(drift_storage::SessionEvent::ToolResult {
                                call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                success: false,
                                error: Some(err_str),
                                content: String::new(), // No content for permission-denied errors
                            });
                    }
                }
            }

            // Add tool results as a user message (provider-agnostic).
            self.context.push_message(LlmMessage {
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
        if !turn_failed {
            let _ = self.event_tx.send(EventMsg::Done);
        }
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
#[cfg(test)]
mod tests {
    use super::*;

    // Build a snapshot message containing every persisted content variant.
    fn compacted_message() -> drift_storage::PersistedMessage {
        drift_storage::PersistedMessage {
            role: "assistant".into(),
            content: vec![
                drift_storage::PersistedContentPart::Text("answer".into()),
                drift_storage::PersistedContentPart::ToolCall {
                    id: "call-1".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
                drift_storage::PersistedContentPart::ToolResult {
                    tool_use_id: "call-1".into(),
                    content: "result".into(),
                    is_error: false,
                },
                drift_storage::PersistedContentPart::Reasoning("thought".into()),
            ],
        }
    }

    // Verify old events disappear while post-snapshot events remain active.
    #[test]
    fn replay_history_replaces_events_at_compaction_boundary() {
        let events = vec![
            drift_storage::SessionEvent::Message {
                role: "user".into(),
                content: "deleted old request".into(),
                reasoning: None,
            },
            drift_storage::SessionEvent::ContextCompacted {
                summary: Some("## summary ##".into()),
                messages: vec![compacted_message()],
                saved_tokens: 12,
            },
            drift_storage::SessionEvent::Message {
                role: "user".into(),
                content: "new request".into(),
                reasoning: None,
            },
            drift_storage::SessionEvent::ToolResult {
                call_id: "call-2".into(),
                name: "read".into(),
                success: true,
                content: "new result".into(),
                error: None,
            },
        ];

        let (messages, summary) = replay_history(&events);
        assert_eq!(summary.as_deref(), Some("## summary ##"));
        assert_eq!(messages.len(), 3);
        assert!(!messages.iter().any(|message| {
            message.content.iter().any(
                |part| matches!(part, ContentPart::Text(text) if text == "deleted old request"),
            )
        }));
        assert!(matches!(
            &messages[0].content[1],
            ContentPart::ToolCall { id, .. } if id == "call-1"
        ));
        assert!(matches!(
            &messages[2].content[0],
            ContentPart::ToolResult { tool_use_id, .. } if tool_use_id == "call-2"
        ));
    }

    // Verify transcripts without compaction retain their original replay behavior.
    #[test]
    fn replay_history_preserves_legacy_transcript_semantics() {
        let events = vec![
            drift_storage::SessionEvent::Message {
                role: "user".into(),
                content: "request".into(),
                reasoning: None,
            },
            drift_storage::SessionEvent::Message {
                role: "assistant".into(),
                content: "calling".into(),
                reasoning: Some("thinking".into()),
            },
            drift_storage::SessionEvent::ToolCall {
                call_id: "call-1".into(),
                name: "read".into(),
                args: serde_json::json!({"path": "x"}),
            },
            drift_storage::SessionEvent::ToolResult {
                call_id: "call-1".into(),
                name: "read".into(),
                success: false,
                content: String::new(),
                error: Some("denied".into()),
            },
        ];

        let (messages, summary) = replay_history(&events);
        assert!(summary.is_none());
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[1].content[0], ContentPart::Reasoning(_)));
        assert!(matches!(
            messages[1].content[2],
            ContentPart::ToolCall { .. }
        ));
        assert!(matches!(
            messages[2].content[0],
            ContentPart::ToolResult { is_error: true, .. }
        ));
    }

    // Verify conversion preserves IDs, flags, arguments, text, and reasoning.
    #[test]
    fn persisted_message_conversion_preserves_all_content_parts() {
        let source = vec![LlmMessage {
            role: "assistant".into(),
            content: vec![
                ContentPart::Text("answer".into()),
                ContentPart::ToolCall {
                    id: "call-1".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
                ContentPart::ToolResult {
                    tool_use_id: "call-1".into(),
                    content: "result".into(),
                    is_error: true,
                },
                ContentPart::Reasoning("thought".into()),
            ],
        }];
        let restored = from_persisted_messages(&to_persisted_messages(&source));
        assert_eq!(restored[0].role, source[0].role);
        assert_eq!(restored[0].content.len(), 4);
        assert!(matches!(
            restored[0].content[2],
            ContentPart::ToolResult { is_error: true, .. }
        ));
    }

    // Verify every payload written to a transcript crosses the same redaction boundary.
    #[test]
    fn transcript_redaction_covers_all_persisted_payloads() {
        let prefixed_secret = "sk-proj-transcript-secret";
        let plain_secret = "plain-tool-token";
        let mut events = vec![
            drift_storage::SessionEvent::Message {
                role: "assistant".into(),
                content: prefixed_secret.into(),
                reasoning: Some(prefixed_secret.into()),
            },
            drift_storage::SessionEvent::ToolCall {
                call_id: "call-1".into(),
                name: "bash".into(),
                args: serde_json::json!({ "token": plain_secret }),
            },
            drift_storage::SessionEvent::ToolResult {
                call_id: "call-1".into(),
                name: "bash".into(),
                success: false,
                content: prefixed_secret.into(),
                error: Some(prefixed_secret.into()),
            },
            drift_storage::SessionEvent::ContextCompacted {
                summary: Some(prefixed_secret.into()),
                messages: vec![drift_storage::PersistedMessage {
                    role: "assistant".into(),
                    content: vec![
                        drift_storage::PersistedContentPart::Text(prefixed_secret.into()),
                        drift_storage::PersistedContentPart::ToolCall {
                            id: "call-1".into(),
                            name: "bash".into(),
                            arguments: prefixed_secret.into(),
                        },
                        drift_storage::PersistedContentPart::ToolResult {
                            tool_use_id: "call-1".into(),
                            content: prefixed_secret.into(),
                            is_error: false,
                        },
                        drift_storage::PersistedContentPart::Reasoning(prefixed_secret.into()),
                    ],
                }],
                saved_tokens: 1,
            },
        ];

        for event in &mut events {
            redact_session_event(event).unwrap();
        }

        let transcript = serde_json::to_string(&events).unwrap();
        assert!(!transcript.contains(prefixed_secret));
        assert!(!transcript.contains(plain_secret));
    }
    // Empty stream fixture used by the provider implementation under test.
    struct EmptyStream;

    impl tokio_stream::Stream for EmptyStream {
        type Item = Result<drift_llm::LlmChunk, drift_llm::LlmError>;

        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(None)
        }
    }

    // Provider fixture that fails during summary generation without network access.
    struct FailingSummaryProvider;

    #[async_trait::async_trait]
    impl LlmProvider for FailingSummaryProvider {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_name(&self) -> &str {
            "fake"
        }

        fn context_window(&self) -> usize {
            100
        }

        async fn stream_chat(
            &self,
            _messages: Vec<LlmMessage>,
            _system_prompt: Option<String>,
            _temperature: Option<f64>,
            _max_output_tokens: Option<usize>,
            _tools: Option<Vec<serde_json::Value>>,
        ) -> Result<drift_llm::LlmResponseStream, drift_llm::LlmError> {
            Ok(drift_llm::LlmResponseStream::new(EmptyStream))
        }

        async fn chat(
            &self,
            _messages: Vec<LlmMessage>,
            _system_prompt: Option<String>,
        ) -> Result<String, drift_llm::LlmError> {
            Err(drift_llm::LlmError::Stream("fake summary failure".into()))
        }
    }

    // Verify failed summaries emit no Done event and no persisted snapshot.
    #[tokio::test]
    async fn failed_summary_does_not_persist_compaction_or_emit_done() {
        let root = std::env::temp_dir().join(format!("drift-agent-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut config =
            AppConfig::load_for_workspace(&root, None, None, None).expect("test config");
        config.agent.auto_compaction = true;
        config.agent.compaction_threshold = 0.1;
        config.agent.compaction_target = 0.05;
        let permission_engine = PermissionEngine::new(&config.security, "default");
        let file_access = std::sync::Arc::new(permission_engine.file_access_guard(&root).unwrap());
        let network = std::sync::Arc::new(permission_engine.network_guard());
        // Match production construction so the test uses the same process boundary.
        let process_sandbox = std::sync::Arc::new(
            ProcessSandbox::new(permission_engine.sandbox_mode(), &root).unwrap(),
        );
        let session_store =
            std::sync::Arc::new(drift_storage::SessionStore::new(root.join("store")).unwrap());
        let (session_id, _) = session_store
            .create(root.to_string_lossy().as_ref(), "fake")
            .unwrap();
        let (event_tx, _) = broadcast::channel(32);
        let mut agent = Agent {
            config,
            llm: Box::new(FailingSummaryProvider),
            tool_registry: std::sync::Arc::new(ToolRegistry::new()),
            permission_engine,
            permission_rx: None,
            event_tx,
            context: ContextManager::new(100, 0.1, 0.05),
            cwd: root.clone(),
            session_id,
            session_store: session_store.clone(),
            file_access,
            network,
            process_sandbox,
        };
        agent.set_messages(vec![LlmMessage::user("old request ".repeat(1000))]);
        let mut events = agent.subscribe();

        agent.submit("new request".into()).await;

        let mut emitted_done = false;
        while let Ok(event) = events.try_recv() {
            if matches!(event, EventMsg::Done) {
                emitted_done = true;
            }
        }
        assert!(!emitted_done);
        let stored = session_store.read_events(session_id).unwrap();
        assert!(!stored.iter().any(|event| {
            matches!(event, drift_storage::SessionEvent::ContextCompacted { .. })
        }));

        std::fs::remove_dir_all(root).ok();
    }
    // Verify an applied snapshot survives JSONL persistence and replay.
    #[test]
    fn persisted_snapshot_replays_after_apply_compaction() {
        let root = std::env::temp_dir().join(format!("drift-snapshot-{}", uuid::Uuid::new_v4()));
        let store = drift_storage::SessionStore::new(root.join("store")).unwrap();
        let (session_id, _) = store
            .create(root.to_string_lossy().as_ref(), "fake")
            .unwrap();
        let snapshot = crate::context::CompactionSnapshot {
            messages: vec![LlmMessage::user("initial request")],
            summary: Some("## summary ##".into()),
        };
        let mut manager = ContextManager::new(1000, 0.75, 0.4);
        manager.apply_compaction(&snapshot);
        let event = drift_storage::SessionEvent::ContextCompacted {
            summary: snapshot.summary.clone(),
            messages: to_persisted_messages(&snapshot.messages),
            saved_tokens: 42,
        };
        store.append_event(session_id, &event).unwrap();

        let events = store.read_events(session_id).unwrap();
        let (messages, summary) = replay_history(&events);
        assert_eq!(messages.len(), 1);
        assert_eq!(summary.as_deref(), Some("## summary ##"));
        assert_eq!(messages[0].role, "user");

        std::fs::remove_dir_all(root).ok();
    }

    // Verify a stale permission reply cannot authorize a later request.
    #[tokio::test]
    async fn permission_response_ignores_stale_request_ids() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send((
            "expired-request".to_string(),
            drift_security::PermissionResponse::Allow,
        ))
        .unwrap();
        tx.send((
            "current-request".to_string(),
            drift_security::PermissionResponse::Deny,
        ))
        .unwrap();

        let response = receive_permission_response(&mut rx, "current-request").await;

        assert!(matches!(
            response,
            Some(drift_security::PermissionResponse::Deny)
        ));
    }
}
