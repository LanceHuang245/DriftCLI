use super::*;

#[test]
fn test_create_and_list_sessions() {
    let dir = std::env::temp_dir().join(format!("drift-test-{}", Uuid::new_v4()));
    let store = SessionStore::new(dir.clone()).unwrap();

    let (id1, _) = store.create("/tmp/test1", "claude-3").unwrap();
    let (id2, _) = store.create("/tmp/test2", "gpt-4").unwrap();

    let sessions = store.list().unwrap();
    assert_eq!(sessions.len(), 2);

    let session_ids: Vec<String> = sessions.iter().map(|s| s.session_id.clone()).collect();
    assert!(session_ids.contains(&id1.to_string()));
    assert!(session_ids.contains(&id2.to_string()));

    // Append an event
    store
        .append_event(
            id1,
            &SessionEvent::Message {
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning: None,
            },
        )
        .unwrap();

    // Read events back
    let events = store.read_events(id1).unwrap();
    assert_eq!(events.len(), 1);

    std::fs::remove_dir_all(&dir).ok();
}

// Verify every persisted content variant survives JSONL serialization.
#[test]
fn test_context_compacted_round_trip() {
    let dir = std::env::temp_dir().join(format!("drift-test-{}", Uuid::new_v4()));
    let store = SessionStore::new(dir.clone()).unwrap();
    let (session_id, _) = store.create("/tmp/test", "test-model").unwrap();

    let expected_messages = vec![PersistedMessage {
        role: "assistant".to_string(),
        content: vec![
            PersistedContentPart::Text("visible response".to_string()),
            PersistedContentPart::ToolCall {
                id: "call-1".to_string(),
                name: "search".to_string(),
                arguments: r#"{\"query\":\"rust\"}"#.to_string(),
            },
            PersistedContentPart::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: "search results".to_string(),
                is_error: false,
            },
            PersistedContentPart::Reasoning("internal reasoning".to_string()),
        ],
    }];
    let event = SessionEvent::ContextCompacted {
        summary: Some("conversation summary".to_string()),
        messages: expected_messages.clone(),
        saved_tokens: 321,
    };

    store.append_event(session_id, &event).unwrap();
    let events = store.read_events(session_id).unwrap();
    assert_eq!(events.len(), 1);
    let [
        SessionEvent::ContextCompacted {
            summary,
            messages,
            saved_tokens,
        },
    ] = events.as_slice()
    else {
        panic!("expected one context compaction event");
    };
    assert_eq!(summary.as_deref(), Some("conversation summary"));
    assert_eq!(messages, &expected_messages);
    assert_eq!(*saved_tokens, 321);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_chrono_now_format() {
    let ts = chrono_now();
    // Should match ISO 8601 basic format
    assert!(ts.ends_with("Z"));
    assert!(ts.contains("T"));
    assert_eq!(ts.len(), 20); // YYYY-MM-DDThh:mm:ssZ
}
