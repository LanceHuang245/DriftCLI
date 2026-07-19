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
