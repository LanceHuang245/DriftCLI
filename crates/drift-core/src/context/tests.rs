use super::*;

fn tool_result(content: &str) -> LlmMessage {
    LlmMessage {
        role: "user".into(),
        content: vec![ContentPart::ToolResult {
            tool_use_id: "call-1".into(),
            content: content.into(),
            is_error: false,
        }],
    }
}

fn tool_call() -> LlmMessage {
    LlmMessage {
        role: "assistant".into(),
        content: vec![ContentPart::ToolCall {
            id: "call-1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }],
    }
}

#[tokio::test]
async fn truncates_large_tool_results() {
    let mut manager = ContextManager::new(128_000, 0.75, 0.4);
    manager.set_messages(vec![tool_result(&"x".repeat(20_000))]);

    let built = manager.build_context(&[], true, None).await.unwrap();
    let ContentPart::ToolResult { content, .. } = &built.messages[0].content[0] else {
        panic!("expected tool result");
    };

    assert!(content.contains("[tool output truncated]"));
    assert!(built.compacted);
    assert!(built.compaction.is_some());
}

#[tokio::test]
async fn drops_complete_old_turns_only_with_summary_provider() {
    let mut manager = ContextManager::new(200, 0.75, 0.4);
    manager.set_messages(vec![
        LlmMessage::user("old user input ".repeat(10)),
        tool_call(),
        tool_result("old result"),
        LlmMessage::user("new user input ".repeat(10)),
    ]);

    let result = manager.build_context(&[], true, None).await;
    assert!(matches!(result, Err(ContextError::SummaryUnavailable)));
}

#[tokio::test]
async fn keeps_system_prompt_and_tool_definitions_in_context() {
    let manager = ContextManager::new(128_000, 0.75, 0.4);
    let built = manager
        .build_context(
            &[ToolDefinition {
                name: "read".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            false,
            None,
        )
        .await
        .unwrap();

    assert!(built.system_prompt.unwrap().contains("DriftCLI"));
    assert_eq!(built.tools.unwrap().len(), 1);
}

#[tokio::test]
async fn does_not_drop_history_before_threshold() {
    let mut manager = ContextManager::new(128_000, 0.75, 0.4);
    manager.set_messages(vec![
        LlmMessage::user("keep this"),
        LlmMessage::user("and this"),
    ]);

    let built = manager.build_context(&[], true, None).await.unwrap();

    assert!(!built.compacted);
    assert!(built.compaction.is_none());
    assert_eq!(built.messages.len(), 2);
}

// Verify workspace instructions precede global instructions in the final prompt.
#[tokio::test]
async fn loads_workspace_instructions_in_project_first_order() {
    let root = std::env::temp_dir().join(format!("drift-context-{}", uuid::Uuid::new_v4()));
    let global = root.join("global");
    std::fs::create_dir_all(root.join(".drift")).unwrap();
    std::fs::create_dir_all(&global).unwrap();
    std::fs::write(root.join(".drift/AGENTS.md"), "dot drift").unwrap();
    std::fs::write(root.join("AGENTS.md"), "root").unwrap();
    std::fs::write(global.join("AGENTS.md"), "global").unwrap();

    let blocks = load_instruction_blocks(&root, Some(&global));
    assert_eq!(blocks.len(), 3);
    assert!(blocks[0].contains("source=\".drift/AGENTS.md\""));
    assert!(blocks[1].contains("source=\"AGENTS.md\""));
    assert!(blocks[2].contains("source=\"global/AGENTS.md\""));
    assert!(blocks[0].contains("dot drift"));
    assert!(blocks[1].contains("root"));
    assert!(blocks[2].contains("global"));
    let manager = ContextManager::with_instruction_blocks(128_000, 0.75, 0.4, blocks);
    let prompt = manager
        .build_context(&[], false, None)
        .await
        .unwrap()
        .system_prompt
        .unwrap();
    assert!(prompt.find("dot drift").unwrap() < prompt.find("root").unwrap());
    assert!(prompt.find("root").unwrap() < prompt.find("global").unwrap());

    std::fs::remove_dir_all(root).ok();
}
