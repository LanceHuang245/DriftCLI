use super::history::{
    from_persisted_messages, redact_session_event, replay_history, to_persisted_messages,
};
use super::turn::receive_permission_response;
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
        message
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Text(text) if text == "deleted old request"))
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
    let mut config = AppConfig::load_for_workspace(&root, None, None, None).expect("test config");
    config.agent.auto_compaction = true;
    config.agent.compaction_threshold = 0.1;
    config.agent.compaction_target = 0.05;
    let permission_engine = PermissionEngine::new(&config.security, "default");
    let file_access = std::sync::Arc::new(permission_engine.file_access_guard(&root).unwrap());
    let network = std::sync::Arc::new(permission_engine.network_guard());
    // Match production construction so the test uses the same process boundary.
    let process_sandbox =
        std::sync::Arc::new(ProcessSandbox::new(permission_engine.sandbox_mode(), &root).unwrap());
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
    assert!(
        !stored
            .iter()
            .any(|event| { matches!(event, drift_storage::SessionEvent::ContextCompacted { .. }) })
    );

    std::fs::remove_dir_all(root).ok();
}

// Verify provider selection is restored from project config after a restart.
#[tokio::test]
async fn activate_provider_persists_selection() {
    let root = std::env::temp_dir().join(format!("drift-provider-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let mut config = AppConfig::load_for_workspace(&root, None, None, None).unwrap();
    config.save_provider(
        "alternate".into(),
        LlmConfig::OpenAiCompatible {
            api_key: String::new(),
            model: "alternate-model".into(),
            base_url: "https://example.com/v1".into(),
            supports_thinking: false,
        },
    );
    let security = config.security.clone();
    let session_store =
        std::sync::Arc::new(drift_storage::SessionStore::new(root.join("store")).unwrap());
    let (session_id, _) = session_store
        .create(root.to_string_lossy().as_ref(), "test-model")
        .unwrap();
    let mut agent = Agent::new(
        config,
        root.clone(),
        std::sync::Arc::new(ToolRegistry::new()),
        session_id,
        session_store,
        &security,
        "default",
    )
    .unwrap();

    agent.activate_provider("alternate").await.unwrap();

    let restored = AppConfig::load_for_workspace(&root, None, None, None).unwrap();
    assert_eq!(restored.active_provider, "alternate");
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
