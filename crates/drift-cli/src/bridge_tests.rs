use super::*;

#[test]
fn session_history_restores_thinking_without_tool_messages() {
    let events = vec![
        drift_storage::SessionEvent::Message {
            role: "assistant".into(),
            content: String::new(),
            reasoning: Some("first thought".into()),
        },
        drift_storage::SessionEvent::ToolCall {
            call_id: "call-1".into(),
            name: "bash".into(),
            args: Default::default(),
        },
        drift_storage::SessionEvent::ToolResult {
            call_id: "call-1".into(),
            name: "bash".into(),
            success: true,
            content: "workspace".into(),
            error: None,
        },
        drift_storage::SessionEvent::Message {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning: Some("second thought".into()),
        },
    ];

    let messages = translate_events_to_chat_messages(&events);

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].reasoning.as_deref(), Some("first thought"));
    assert_eq!(messages[1].reasoning.as_deref(), Some("second thought"));
    assert_eq!(messages[2].content, "answer");
    assert!(
        messages
            .iter()
            .all(|message| !message.content.contains("bash"))
    );
}
