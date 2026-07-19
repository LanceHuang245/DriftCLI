use drift_core::EventMsg;
use drift_tui::{AppEvent, ChatMessage};
use tokio::sync::{broadcast, mpsc};

// Helper function to translate persistent storage SessionEvents into TUI ChatMessages
pub(crate) fn translate_events_to_chat_messages(
    events: &[drift_storage::SessionEvent],
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    for event in events {
        match event {
            drift_storage::SessionEvent::Message {
                role,
                content,
                reasoning,
            } => {
                messages.push(ChatMessage {
                    role: role.clone(),
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    thinking: false,
                    reasoning_duration_ms: None,
                    reasoning_collapsed: true,
                    thinking_tools: Vec::new(),
                });
            }
            drift_storage::SessionEvent::ToolCall {
                call_id: _,
                name,
                args,
            } => {
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" {
                        last.thinking_tools
                            .push(format!("> Calling tool: {} with {}", name, args));
                    }
                }
            }
            drift_storage::SessionEvent::ToolResult {
                call_id: _,
                name,
                success,
                content: _,
                error,
            } => {
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" {
                        last.thinking_tools.push(format!(
                            "> Tool {} {}",
                            name,
                            if *success { "completed" } else { "failed" }
                        ));
                        if let Some(err) = error {
                            last.thinking_tools.push(format!("  Error: {}", err));
                        }
                    }
                }
            }
            // Compaction snapshots are internal context state, not chat messages.
            drift_storage::SessionEvent::ContextCompacted { .. } => {}
        }
    }

    messages
}

// Spawns the one-way adapter from core events to TUI events.
pub(crate) fn spawn_event_bridge(
    mut core_rx: broadcast::Receiver<EventMsg>,
    tui_tx: mpsc::UnboundedSender<AppEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match core_rx.recv().await {
                Ok(EventMsg::Token(text)) => {
                    let _ = tui_tx.send(AppEvent::Token(text));
                }
                Ok(EventMsg::Reasoning(text)) => {
                    let _ = tui_tx.send(AppEvent::Reasoning(text));
                }
                Ok(EventMsg::ReasoningComplete { duration_ms }) => {
                    let _ = tui_tx.send(AppEvent::ReasoningComplete { duration_ms });
                }
                Ok(EventMsg::AgentState(state)) => {
                    let status = match state {
                        drift_core::AgentState::Idle => "Idle",
                        drift_core::AgentState::Thinking => "Thinking...",
                        drift_core::AgentState::Generating(_) => "Generating...",
                        drift_core::AgentState::Error(e) => {
                            let _ = tui_tx.send(AppEvent::Error(e.clone()));
                            "Error"
                        }
                    };
                    let _ = tui_tx.send(AppEvent::AgentStatus(status.to_string()));
                }
                Ok(EventMsg::Error { message, .. }) => {
                    let _ = tui_tx.send(AppEvent::Error(message));
                }
                Ok(EventMsg::Done) => {
                    let _ = tui_tx.send(AppEvent::Done);
                }
                Ok(EventMsg::Interrupted) => {
                    let _ = tui_tx.send(AppEvent::Interrupted);
                }
                Ok(EventMsg::ModelList(models)) => {
                    let _ = tui_tx.send(AppEvent::ModelList(models));
                }
                Ok(EventMsg::ProviderList(names)) => {
                    let _ = tui_tx.send(AppEvent::ProviderList(names));
                }
                Ok(EventMsg::ProviderSwitched { name, model }) => {
                    let _ = tui_tx.send(AppEvent::ProviderSwitched { name, model });
                }
                Ok(EventMsg::ProviderConfig { name, config }) => {
                    let _ = tui_tx.send(AppEvent::ProviderConfig { name, config });
                }
                Ok(EventMsg::ContextCompacting) => {
                    let _ = tui_tx.send(AppEvent::AgentStatus("Compacting context...".to_string()));
                }
                Ok(EventMsg::ContextCompacted { saved_tokens, .. }) => {
                    let _ = tui_tx.send(AppEvent::AgentStatus(format!(
                        "Context compacted (-{} tokens)",
                        saved_tokens
                    )));
                }
                Ok(EventMsg::ToolCallStart { id, name }) => {
                    let _ = tui_tx.send(AppEvent::ToolCallStart { id, name });
                }
                Ok(EventMsg::ToolCallArgs { id, delta }) => {
                    let _ = tui_tx.send(AppEvent::ToolCallArgs { id, delta });
                }
                Ok(EventMsg::ToolCallEnd { id }) => {
                    let _ = tui_tx.send(AppEvent::ToolCallEnd { id });
                }
                Ok(EventMsg::ToolExecStart { id, name }) => {
                    let _ = tui_tx.send(AppEvent::ToolExecStart { id, name });
                }
                Ok(EventMsg::ToolExecEnd {
                    id, name, success, ..
                }) => {
                    let _ = tui_tx.send(AppEvent::ToolExecEnd { id, name, success });
                }
                Ok(EventMsg::SessionList(meta_list)) => {
                    let _ = tui_tx.send(AppEvent::SessionList(meta_list));
                }
                Ok(EventMsg::SessionLoaded { session_id, events }) => {
                    let _ = tui_tx.send(AppEvent::SessionLoaded {
                        session_id,
                        messages: translate_events_to_chat_messages(&events),
                    });
                }
                Ok(EventMsg::PermissionRequest {
                    request_id,
                    tool_name,
                    args_summary,
                    reason,
                    ..
                }) => {
                    let _ = tui_tx.send(AppEvent::PermissionRequest {
                        request_id,
                        tool_name,
                        args_summary,
                        reason,
                    });
                }
                Ok(EventMsg::PermissionResolved {
                    request_id,
                    allowed,
                }) => {
                    let _ = tui_tx.send(AppEvent::PermissionResolved {
                        request_id,
                        allowed,
                    });
                }
                _ => {}
            }
        }
    })
}
