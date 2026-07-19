use super::*;

/// Convert provider messages into storage DTOs without adding drift-llm to storage.
pub(super) fn to_persisted_messages(
    messages: &[LlmMessage],
) -> Vec<drift_storage::PersistedMessage> {
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
pub(super) fn redact_session_event(
    event: &mut drift_storage::SessionEvent,
) -> Result<(), serde_json::Error> {
    let mut value = serde_json::to_value(&*event)?;
    drift_security::SensitiveDataFilter::filter_json(&mut value);
    *event = serde_json::from_value(value)?;
    Ok(())
}

/// Restore provider messages from the dependency-free storage representation.
pub(super) fn from_persisted_messages(
    messages: &[drift_storage::PersistedMessage],
) -> Vec<LlmMessage> {
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
pub(super) fn replay_history(
    events: &[drift_storage::SessionEvent],
) -> (Vec<LlmMessage>, Option<String>) {
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
