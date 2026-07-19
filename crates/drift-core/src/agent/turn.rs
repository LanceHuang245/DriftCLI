use super::history::to_persisted_messages;
use super::*;

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

// Receive only the decision correlated with the active permission request.
pub(super) async fn receive_permission_response(
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

impl Agent {
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
            // Preserve tool-call start order independently from HashMap iteration.
            let mut active_tool_call_order: Vec<String> = Vec::new();
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
                        if !active_tool_calls.contains_key(&id) {
                            active_tool_call_order.push(id.clone());
                            active_tool_calls.insert(
                                id.clone(),
                                ActiveToolCall {
                                    id,
                                    name,
                                    args: Vec::new(),
                                },
                            );
                        }
                    }
                    Some(Ok(LlmChunk::ToolCallArgs { id, delta })) => {
                        let _ = self.event_tx.send(EventMsg::ToolCallArgs {
                            id: id.clone(),
                            delta: delta.clone(),
                        });
                        if let Some(tc) = active_tool_calls.get_mut(&id) {
                            tc.args.push(delta);
                        } else {
                            turn_failed = true;
                            let _ = self.event_tx.send(EventMsg::Error {
                                message: format!("Received arguments for unknown tool call {id}"),
                                recoverable: true,
                            });
                            break;
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
                        // omit ToolCallEnd, so finish them in their start order.
                        for id in active_tool_call_order {
                            if let Some(tool_call) = active_tool_calls.remove(&id) {
                                completed_tool_calls.push(tool_call);
                            }
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
                    None => {
                        turn_failed = true;
                        let _ = self.event_tx.send(EventMsg::Error {
                            message: "LLM stream ended before a completion event".to_string(),
                            recoverable: true,
                        });
                        break;
                    }
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
}
