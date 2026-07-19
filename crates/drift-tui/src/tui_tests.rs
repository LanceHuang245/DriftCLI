use super::*;

#[test]
fn mcp_status_updates_status_bar() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);
    app.handle_app_event(AppEvent::McpStatus {
        server_id: "fixture".into(),
        status: "connected (1 tools)".into(),
    });
    assert_eq!(app.status_text, "MCP fixture: connected (1 tools)");
}

#[test]
fn status_bar_displays_full_session_id() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);
    app.session_id = uuid::Uuid::parse_str("12345678-1234-5678-9abc-def012345678").unwrap();
    let backend = ratatui::backend::TestBackend::new(160, 10);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| app.render(frame)).unwrap();

    // Read the rendered status row to verify the user-visible session identifier.
    let buffer = terminal.backend().buffer();
    let status_row = (0..buffer.area.width)
        .filter_map(|x| buffer.cell((x, buffer.area.height - 1)))
        .map(|cell| cell.symbol())
        .collect::<Vec<_>>()
        .concat();
    assert!(
        status_row.contains("Session: 12345678-1234-5678-9abc-def012345678"),
        "rendered status row: {status_row:?}"
    );
}

#[test]
fn provider_picker_enter_switches_selected_provider() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);
    app.providers = vec!["default".into(), "alternate".into()];
    app.provider_selected = 1;
    app.mode = TuiMode::ProviderPicker;

    app.handle_provider_key(KeyCode::Enter);

    assert!(matches!(
        cmd_rx.try_recv(),
        Ok(TuiCommand::SetActiveProvider(name)) if name == "alternate"
    ));
    assert_eq!(app.mode, TuiMode::Normal);
}

#[test]
fn provider_picker_edit_key_opens_selected_config() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);
    app.providers = vec!["default".into(), "alternate".into()];
    app.provider_selected = 1;
    app.mode = TuiMode::ProviderPicker;

    app.handle_provider_key(KeyCode::Char('e'));

    assert!(matches!(
        cmd_rx.try_recv(),
        Ok(TuiCommand::GetProviderConfig(name)) if name == "alternate"
    ));
}

#[test]
fn permission_prompt_displays_risk_and_response_keys() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);
    app.handle_app_event(AppEvent::PermissionRequest {
        request_id: "perm-1".into(),
        tool_name: "bash".into(),
        args_summary: "git push origin main".into(),
        reason: "Command changes remote state".into(),
        risk_level: "High".into(),
    });
    let backend = ratatui::backend::TestBackend::new(80, 18);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| app.render(frame)).unwrap();

    // Flatten the frame so the complete user-visible permission instructions are asserted.
    let buffer = terminal.backend().buffer();
    let rendered = (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .filter_map(|x| buffer.cell((x, y)))
                .map(|cell| cell.symbol())
                .collect::<Vec<_>>()
                .concat()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("Permission Required — bash"));
    assert!(rendered.contains("Risk: High"));
    assert!(rendered.contains("[y] Allow once"));
    assert!(rendered.contains("[Y] Always allow"));
    assert!(rendered.contains("[n] Deny once"));
    assert!(rendered.contains("[N] Always deny"));
    assert!(rendered.contains("[Esc] Cancel"));
}

#[test]
fn tool_iteration_creates_independent_thinking_blocks() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);

    app.handle_app_event(AppEvent::Reasoning("first thought".into()));
    app.handle_app_event(AppEvent::ReasoningComplete { duration_ms: 120 });
    app.handle_app_event(AppEvent::ToolCallStart {
        name: "bash".into(),
    });
    app.handle_app_event(AppEvent::ToolExecStart {
        name: "bash".into(),
    });
    app.handle_app_event(AppEvent::ToolExecEnd);

    assert_eq!(app.status_text, "Thinking...");
    assert!(app.current_reasoning.is_empty());
    assert!(app.reasoning_start_time.is_none());

    app.handle_app_event(AppEvent::Reasoning("second thought".into()));
    assert!(app.reasoning_start_time.is_some());
    app.handle_app_event(AppEvent::ReasoningComplete { duration_ms: 240 });

    let thinking: Vec<_> = app
        .messages
        .iter()
        .filter_map(|message| message.reasoning.as_deref())
        .collect();
    assert_eq!(thinking, ["first thought", "second thought"]);
    assert_eq!(app.messages[0].reasoning_duration_ms, Some(120));
    assert_eq!(app.messages[1].reasoning_duration_ms, Some(240));
}

#[test]
fn starting_next_turn_keeps_completed_thinking_history() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);

    app.begin_user_turn("first question".into());
    app.handle_app_event(AppEvent::Reasoning("preserved thought".into()));
    app.handle_app_event(AppEvent::ReasoningComplete { duration_ms: 50 });
    app.handle_app_event(AppEvent::Token("first answer".into()));
    app.handle_app_event(AppEvent::Done);
    app.begin_user_turn("second question".into());

    assert!(app.messages.iter().any(|message| {
        message.reasoning.as_deref() == Some("preserved thought")
            && message.reasoning_duration_ms == Some(50)
    }));
}

#[test]
fn tool_calls_never_render_in_chat_history() {
    let config = LlmConfig::Anthropic {
        api_key: String::new(),
        model: "test-model".into(),
        base_url: "https://example.com".into(),
        reasoning_effort: None,
    };
    let (_event_tx, event_rx) = mpsc::unbounded_channel();
    let (cmd_tx, _cmd_rx) = mpsc::unbounded_channel();
    let mut app = TuiApp::new(&config, event_rx, cmd_tx);
    app.handle_app_event(AppEvent::Reasoning("inspect workspace".into()));
    app.handle_app_event(AppEvent::ReasoningComplete { duration_ms: 10 });
    app.handle_app_event(AppEvent::ToolCallStart {
        name: "secret-tool-name".into(),
    });
    app.handle_app_event(AppEvent::ToolExecStart {
        name: "secret-tool-name".into(),
    });
    app.handle_app_event(AppEvent::ToolExecEnd);
    let backend = ratatui::backend::TestBackend::new(100, 16);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| app.render(frame)).unwrap();

    let buffer = terminal.backend().buffer();
    let rendered = (0..buffer.area.height)
        .flat_map(|y| {
            (0..buffer.area.width)
                .filter_map(move |x| buffer.cell((x, y)))
                .map(|cell| cell.symbol())
        })
        .collect::<Vec<_>>()
        .concat();
    assert!(!rendered.contains("secret-tool-name"));
    assert!(rendered.contains("Thinking..."));
}
