mod connect;
mod components;
mod input;
mod selection;

use crossterm::{
    event::{self, EnableMouseCapture, DisableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use drift_config::LlmConfig;
use drift_llm::ModelInfo;
use ratatui::{
    layout::{Constraint, Direction, Layout, Position},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use selection::SelectionState;
use std::io::{self, stdout};
use tokio::sync::mpsc;

// Events sent from the backend to the TUI via async channel.
#[derive(Debug, Clone)]
pub enum AppEvent {
    Token(String),
    Reasoning(String),
    AgentStatus(String),
    Error(String),
    Done,
    ModelList(Vec<ModelInfo>),
    ProviderList(Vec<String>),
    ProviderSwitched { name: String, model: String },
    // Full config for a specific provider loaded from the backend.
    ProviderConfig { name: String, config: LlmConfig },
    // Tool call started with id and name
    ToolCallStart { id: String, name: String },
    // Tool call arguments streaming
    ToolCallArgs { id: String, delta: String },
    // Tool call completed
    ToolCallEnd { id: String },
    // Tool execution started
    ToolExecStart { id: String, name: String },
    // Tool execution finished with result summary
    ToolExecEnd { id: String, name: String, success: bool },
}

// Commands sent from the TUI to the backend (chat, fetch models, reconfigure, provider management).
#[derive(Debug, Clone)]
pub enum TuiCommand {
    Chat(String),
    FetchModels {
        provider: String,
        base_url: String,
        api_key: String,
    },
    Reconfigure(LlmConfig),
    // Save or update a named provider configuration.
    SaveProvider { name: String, config: LlmConfig },
    // Switch to a different configured provider.
    SetActiveProvider(String),
    // Request the list of configured provider names.
    GetProviders,
    // Delete a named provider from the configuration.
    DeleteProvider(String),
    // Get the full config for a named provider (used for "Modify" in the provider picker).
    GetProviderConfig(String),
}

// Central application state for the DriftCLI TUI.
pub struct TuiApp {
    messages: Vec<ChatMessage>,
    current_response: String,
    current_reasoning: String,
    input_buffer: String,
    cursor_position: usize,
    status_text: String,
    model_name: String,
    mode: TuiMode,
    connect_form: connect::ConnectForm,
    variant_options: Vec<String>,
    variant_selected: usize,
    // Multi-provider support: configured providers, selection index, and active name.
    providers: Vec<String>,
    provider_selected: usize,
    provider_action_selected: usize,
    provider_name: String,
    event_rx: mpsc::UnboundedReceiver<AppEvent>,
    cmd_tx: mpsc::UnboundedSender<TuiCommand>,
    should_quit: bool,
    history: Vec<String>,
    history_index: Option<usize>,
    chat_scroll_offset: usize,
    selection: SelectionState,
    chat_area: ratatui::layout::Rect,
}

// Which screen/overlay the TUI is currently displaying.
#[derive(Debug, Clone, PartialEq)]
pub enum TuiMode {
    Normal,
    ConnectSettings,
    VariantPicker,
    // Provider switcher overlay (list configured providers with delete option)
    ProviderPicker,
    // Sub-menu after selecting a provider in the provider picker: Apply / Modify
    ProviderAction { provider_name: String },
}

// A single chat message with optional reasoning/thinking content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
    pub thinking: bool,
}

impl TuiApp {
    // Create a new TuiApp from the current LLM config and async channels.
    pub fn new(
        llm_config: &LlmConfig,
        event_rx: mpsc::UnboundedReceiver<AppEvent>,
        cmd_tx: mpsc::UnboundedSender<TuiCommand>,
    ) -> Self {
        let model_name = match llm_config {
            LlmConfig::Anthropic { model, .. } => model.clone(),
            LlmConfig::OpenAiCompatible { model, .. } => model.clone(),
        };
        Self {
            messages: Vec::new(),
            current_response: String::new(),
            current_reasoning: String::new(),
            input_buffer: String::new(),
            cursor_position: 0,
            status_text: "Idle".into(),
            model_name,
            mode: TuiMode::Normal,
            connect_form: connect::ConnectForm::from_config(llm_config),
            variant_options: Vec::new(),
            variant_selected: 0,
            providers: Vec::new(),
            provider_selected: 0,
            provider_action_selected: 0,
            provider_name: "default".to_string(),
            event_rx,
            cmd_tx,
            should_quit: false,
            history: Vec::new(),
            history_index: None,
            chat_scroll_offset: 0,
            selection: SelectionState::new(),
            chat_area: ratatui::layout::Rect::new(0, 0, 80, 24),
        }
    }

    // Enter raw mode, start the main loop, then restore the terminal on exit.
    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = ratatui::backend::CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.main_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        result
    }

    // Main event loop: draws the frame, processes async events, and handles key input.
    fn main_loop(
        &mut self,
        terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    ) -> anyhow::Result<()> {
        while !self.should_quit {
            // Draw the current frame. Also finalize any pending selection copy.
            terminal.draw(|f| {
                self.render(f);
                self.selection.finalize_copy(f.buffer_mut());
            })?;

            // If the selection was finalized, copy the extracted text.
            if let Some(text) = self.selection.take_copy_text() {
                match Self::copy_to_clipboard(&text) {
                    Ok(()) => {
                        self.status_text = "Copied to clipboard".into();
                    }
                    Err(e) => {
                        self.status_text = format!("Copy failed: {e}");
                    }
                }
            }

            // Drain any pending async events from the backend.
            if let Ok(event) = self.event_rx.try_recv() {
                self.handle_app_event(event);
            }

            // Poll for input events (non-blocking with 16 ms timeout).
            if event::poll(std::time::Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        match self.mode {
                            TuiMode::Normal => {
                                match key.code {
                                    KeyCode::Up => {
                                        if self.history_index.is_none() && !self.input_buffer.is_empty() {
                                            self.history.push(self.input_buffer.clone());
                                        }
                                        if !self.history.is_empty() {
                                            let idx = self.history_index.map_or(self.history.len(), |i| i);
                                            let new_idx = idx.saturating_sub(1);
                                            self.history_index = Some(new_idx);
                                            self.input_buffer = self.history[new_idx].clone();
                                            self.cursor_position = self.input_buffer.len();
                                        }
                                    }
                                    KeyCode::Down => {
                                        if let Some(idx) = self.history_index {
                                            let new_idx = idx + 1;
                                            if new_idx >= self.history.len() {
                                                self.history_index = None;
                                                self.input_buffer.clear();
                                            } else {
                                                self.history_index = Some(new_idx);
                                                self.input_buffer = self.history[new_idx].clone();
                                            }
                                            self.cursor_position = self.input_buffer.len();
                                        }
                                    }
                                    // PgUp / PgDn scroll the chat area by half the viewport.
                                    KeyCode::PageUp => {
                                        self.chat_scroll_offset += 10;
                                    }
                                    KeyCode::PageDown => {
                                        self.chat_scroll_offset = self.chat_scroll_offset.saturating_sub(10);
                                    }
                                    _ => {
                                        self.history_index = None;
                                        let action = input::process_key(
                                            key.code,
                                            key.modifiers,
                                            &mut self.input_buffer,
                                            &mut self.cursor_position,
                                        );
                                        match action {
                                            input::InputAction::Submit(text) => {
                                                if !text.trim().is_empty() {
                                                    if text.starts_with('/') {
                                                        self.handle_command(&text);
                                                    } else {
                                                        if self.history.last() != Some(&text) {
                                                            self.history.push(text.clone());
                                                        }
                                                        self.messages.push(ChatMessage {
                                                            role: "user".into(),
                                                            content: text.clone(),
                                                            reasoning: None,
                                                            thinking: false,
                                                        });
                                                        self.current_response.clear();
                                                        self.current_reasoning.clear();
                                                        self.chat_scroll_offset = 0;
                                                        self.selection.clear();
                                                        self.status_text = "Waiting...".into();
                                                        let _ = self.cmd_tx.send(TuiCommand::Chat(text));
                                                    }
                                                    self.input_buffer.clear();
                                                    self.cursor_position = 0;
                                                }
                                            }
                                            input::InputAction::Quit => self.should_quit = true,
                                            input::InputAction::ToggleConnectInfo => {}
                                        }
                                    }
                                }
                            }
                            TuiMode::ConnectSettings => { self.handle_connect_key(key.code); }
                            TuiMode::VariantPicker => { self.handle_variant_key(key.code); }
                            TuiMode::ProviderPicker => { self.handle_provider_key(key.code); }
                            TuiMode::ProviderAction { .. } => { self.handle_provider_action_key(key.code); }
                        }
                    }
                    Event::Mouse(mouse) => {
                        match mouse.kind {
                            MouseEventKind::ScrollUp => {
                                match self.mode {
                                    TuiMode::Normal => { self.chat_scroll_offset += 3; }
                                    TuiMode::ConnectSettings if self.connect_form.show_model_list => {
                                        self.connect_form.model_list_index =
                                            self.connect_form.model_list_index.saturating_sub(1);
                                    }
                                    TuiMode::VariantPicker => {
                                        self.variant_selected = self.variant_selected.saturating_sub(1);
                                    }
                                    TuiMode::ProviderPicker => {
                                        self.provider_selected = self.provider_selected.saturating_sub(1);
                                    }
                                    TuiMode::ProviderAction { .. } => {
                                        self.provider_action_selected = self.provider_action_selected.saturating_sub(1);
                                    }
                                    _ => {}
                                }
                                // Scroll clears any active selection.
                                self.selection.clear();
                            }
                            MouseEventKind::ScrollDown => {
                                match self.mode {
                                    TuiMode::Normal => {
                                        self.chat_scroll_offset = self.chat_scroll_offset.saturating_sub(3);
                                    }
                                    TuiMode::ConnectSettings if self.connect_form.show_model_list => {
                                        if self.connect_form.model_list_index + 1 < self.connect_form.model_list.len() {
                                            self.connect_form.model_list_index += 1;
                                        }
                                    }
                                    TuiMode::VariantPicker => {
                                        if self.variant_selected + 1 < self.variant_options.len() {
                                            self.variant_selected += 1;
                                        }
                                    }
                                    TuiMode::ProviderPicker => {
                                        if self.provider_selected <= self.providers.len() {
                                            self.provider_selected += 1;
                                        }
                                    }
                                    TuiMode::ProviderAction { .. } => {
                                        if self.provider_action_selected < 1 {
                                            self.provider_action_selected += 1;
                                        }
                                    }
                                    _ => {}
                                }
                                self.selection.clear();
                            }
                            MouseEventKind::Down(_)
                            | MouseEventKind::Drag(_)
                            | MouseEventKind::Up(_) => {
                                if self.mode == TuiMode::Normal {
                                    self.selection.handle_mouse_event(&mouse, self.chat_area);
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    // Parse and execute slash commands like /connect, /variant, /provider, /clear, /quit.
    fn handle_command(&mut self, cmd: &str) {
        let cmd = cmd.trim();
        match cmd {
            // Enter connection settings form.
"/connect" => {
    self.connect_form = connect::ConnectForm::new();
    self.mode = TuiMode::ConnectSettings;
}
            // Open the variant / reasoning-effort picker for the current model.
            "/variant" => {
                let model = &self.connect_form.model;
                let from_caps: Vec<String> = self
                    .connect_form
                    .model_list
                    .iter()
                    .find(|m| m.id == *model)
                    .map(|m| m.effort_levels.clone())
                    .unwrap_or_default();
                let mut efforts = vec!["(none)".to_string()];
                if from_caps.is_empty() {
                    efforts.extend(
                        ["low", "medium", "high", "max"]
                            .iter()
                            .map(|s| s.to_string()),
                    );
                } else {
                    efforts.extend(from_caps);
                }
                let current = self.connect_form.reasoning_effort.clone();
                let selected = if current.is_none() {
                    0
                } else {
                    efforts
                        .iter()
                        .position(|e| Some(e) == current.as_ref())
                        .unwrap_or(0)
                };
                self.variant_options = efforts;
                self.variant_selected = selected;
                self.mode = TuiMode::VariantPicker;
            }
            // Open the provider picker overlay.
            "/provider" => {
                let _ = self.cmd_tx.send(TuiCommand::GetProviders);
            }
            // Quit the application.
            "/quit" | "/exit" => {
                self.should_quit = true;
            }
            // Clear all chat messages and the in-progress response.
            "/clear" => {
                self.messages.clear();
                self.current_response.clear();
            }
            // Unknown command — show a help message.
            _ => {
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!("Unknown command: {}. Try /connect, /provider, /clear, /quit", cmd),
                    reasoning: None,
                    thinking: false,
                });
            }
        }
        self.input_buffer.clear();
        self.cursor_position = 0;
    }

    // Handle key presses while the connect settings form is active.
    fn handle_connect_key(&mut self, key: KeyCode) {
        match key {
            // Esc returns to normal chat mode.
            KeyCode::Esc => {
                self.mode = TuiMode::Normal;
            }
            // Tab moves to the next form field.
            KeyCode::Tab => {
                self.connect_form.next_field();
            }
            // Up: scroll model list up, or move to previous form field.
            KeyCode::Up => {
                if self.connect_form.show_model_list {
                    self.connect_form.model_list_index =
                        self.connect_form.model_list_index.saturating_sub(1);
                } else {
                    self.connect_form.previous_field();
                }
            }
            // Down: scroll model list down, or move to next form field.
            KeyCode::Down => {
                if self.connect_form.show_model_list {
                    if self.connect_form.model_list_index + 1 < self.connect_form.model_list.len() {
                        self.connect_form.model_list_index += 1;
                    }
                } else {
                    self.connect_form.next_field();
                }
            }
            // Left toggles provider type when on the provider field.
            KeyCode::Left => {
                self.connect_form.on_left();
            }
            // Right toggles provider type when on the provider field.
            KeyCode::Right => {
                self.connect_form.on_right();
            }
            // Type into the currently selected text field.
            KeyCode::Char(c) => {
                self.connect_form.on_char(c);
            }
            // Delete last character from the selected text field.
            KeyCode::Backspace => {
                self.connect_form.on_backspace();
            }
            // Enter: select from model list, fetch models, save, or cancel.
            KeyCode::Enter => {
                if self.connect_form.show_model_list && !self.connect_form.model_list.is_empty() {
                    self.connect_form.select_model();
                } else {
                    match self.connect_form.selected_field {
                        // Field 4 (model): fetch the model list from the provider.
                        4 => {
                            let provider = self.connect_form.provider_label().to_string();
                            let base_url = self.connect_form.base_url.clone();
                            let api_key = self.connect_form.api_key.clone();
                            let _ = self.cmd_tx.send(TuiCommand::FetchModels {
                                provider,
                                base_url,
                                api_key,
                            });
                            self.connect_form.fetching_models = true;
                            self.connect_form.status_message = "Fetching models...".into();
                        }
                        // Field 5 (save): apply and persist connection settings.
                        5 => {
                            self.save_connect_settings();
                        }
                        // Field 6 (cancel): return to normal mode without saving.
                        6 => {
                            self.mode = TuiMode::Normal;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Handle key presses while the provider picker is active.
    fn handle_provider_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => self.mode = TuiMode::Normal,
            KeyCode::Up => {
                self.provider_selected = self.provider_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                let max = self.providers.len(); // allow selection of [+ New Provider]
                if self.provider_selected < max {
                    self.provider_selected += 1;
                }
            }
            KeyCode::Enter => {
                if self.provider_selected < self.providers.len() {
                    let name = self.providers[self.provider_selected].clone();
                    self.provider_action_selected = 0;
                    self.mode = TuiMode::ProviderAction {
                        provider_name: name,
                    };
                } else {
                    // [+ New Provider]
                    self.mode = TuiMode::ConnectSettings;
                }
            }
            KeyCode::Char('d') => {
                if self.provider_selected < self.providers.len() {
                    let name = self.providers[self.provider_selected].clone();
                    let _ = self.cmd_tx.send(TuiCommand::DeleteProvider(name));
                    // Re-request list
                    let _ = self.cmd_tx.send(TuiCommand::GetProviders);
                }
            }
            _ => {}
        }
    }

    // Handle key presses in the provider action sub-menu (Apply / Modify).
    fn handle_provider_action_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => self.mode = TuiMode::ProviderPicker,
            KeyCode::Up => {
                self.provider_action_selected = self.provider_action_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                if self.provider_action_selected < 1 {
                    self.provider_action_selected += 1;
                }
            }
            KeyCode::Enter => {
                if let TuiMode::ProviderAction { ref provider_name } = self.mode {
                    match self.provider_action_selected {
                        0 => {
                            let _ = self.cmd_tx
                                .send(TuiCommand::SetActiveProvider(provider_name.clone()));
                            self.mode = TuiMode::Normal;
                        }
                        1 => {
                            let _ = self.cmd_tx
                                .send(TuiCommand::GetProviderConfig(provider_name.clone()));
                            // Mode switches to ConnectSettings when ProviderConfig event arrives.
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Handle key presses while the variant/reasoning-effort picker is active.
    fn handle_variant_key(&mut self, key: KeyCode) {
        match key {
            // Esc cancels and returns to normal chat mode.
            KeyCode::Esc => {
                self.mode = TuiMode::Normal;
            }
            // Move selection up through the variant list.
            KeyCode::Up => {
                self.variant_selected = self.variant_selected.saturating_sub(1);
            }
            // Move selection down through the variant list.
            KeyCode::Down => {
                if self.variant_selected + 1 < self.variant_options.len() {
                    self.variant_selected += 1;
                }
            }
            // Enter confirms the selected effort level, sends reconfigure, and returns to normal mode.
            KeyCode::Enter => {
                let level = self
                    .variant_options
                    .get(self.variant_selected)
                    .cloned();
                let is_none = level.as_deref() == Some("(none)");
                self.connect_form.reasoning_effort = if is_none { None } else { level.clone() };
                let config = self.connect_form.to_llm_config();
                self.model_name = self.connect_form.model.clone();
                let _ = self.cmd_tx.send(TuiCommand::Reconfigure(config));
                self.status_text = if is_none {
                    format!("Reasoning effort cleared for {}", self.model_name)
                } else {
                    format!(
                        "Variant set to {} for {}",
                        level.unwrap_or_default(),
                        self.model_name
                    )
                };
                self.mode = TuiMode::Normal;
            }
            _ => {}
        }
    }

    // Build the LLM config from the form and send a SaveProvider command.
    fn save_connect_settings(&mut self) {
        let name = self.connect_form.to_provider_name();
        let config = self.connect_form.to_llm_config();
        self.model_name = self.connect_form.model.clone();
        self.provider_name = name.clone();
        let _ = self.cmd_tx.send(TuiCommand::SaveProvider { name, config });
        self.status_text = format!(
            "Connected: {} @ {}",
            self.model_name,
            self.connect_form.base_url
        );
        self.mode = TuiMode::Normal;
    }

    // Process an async event from the backend channel.
    fn handle_app_event(&mut self, event: AppEvent) {
        match event {
            // Streaming response token — append to the in-progress assistant message.
            AppEvent::Token(text) => {
                let was_empty = self.current_response.is_empty();
                self.current_response.push_str(&text);
                if was_empty {
                    self.chat_scroll_offset = 0;
                    self.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: text,
                        reasoning: None,
                        thinking: false,
                    });
                } else if let Some(last) = self.messages.last_mut() {
                    if last.role == "assistant" {
                        last.content = self.current_response.clone();
                    }
                }
                self.status_text = "Generating...".into();
            }
            // Accumulate reasoning/thinking text (displayed as dim text in response).
            AppEvent::Reasoning(text) => {
                self.current_reasoning.push_str(&text);
            }
            // Update the status bar text directly.
            AppEvent::AgentStatus(status) => {
                self.status_text = status;
            }
            // Display an error — in connect mode show in form, otherwise as system message.
            AppEvent::Error(msg) => {
                if self.mode == TuiMode::ConnectSettings {
                    self.connect_form.fetching_models = false;
                    self.connect_form.status_message = format!("Error: {}", msg);
                } else {
                    self.messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!("Error: {}", msg),
                        reasoning: None,
                        thinking: false,
                    });
                    self.status_text = "Error".into();
                }
            }
            // Response streaming complete — finalize the message and reset.
            AppEvent::Done => {
                self.commit_current_response(false);
                self.status_text = "Idle".into();
            }
            // Received model list — populate the dropdown in connect mode.
            AppEvent::ModelList(models) => {
                self.connect_form.model_list = models;
                self.connect_form.show_model_list = true;
                self.connect_form.model_list_index = 0;
                self.connect_form.fetching_models = false;
                self.connect_form.status_message = format!(
                    "{} models loaded. Use Up/Down/Enter to select.",
                    self.connect_form.model_list.len()
                );
            }
            // Received provider name list — enter provider picker mode.
            AppEvent::ProviderList(names) => {
                self.providers = names.clone();
                self.provider_selected = names
                    .iter()
                    .position(|n| n == &self.provider_name)
                    .unwrap_or(0);
                self.mode = TuiMode::ProviderPicker;
            }
            // Provider was switched externally — update active name and model.
            AppEvent::ProviderSwitched { name, model } => {
                self.provider_name = name.clone();
                self.model_name = model;
                self.status_text = format!("Switched to {}", name);
            }
            // Received full config for a specific provider (from "Modify" action).
            AppEvent::ProviderConfig { name, config } => {
                self.connect_form = connect::ConnectForm::from_entry(&name, &config);
                self.mode = TuiMode::ConnectSettings;
            }
            // Tool call requested by LLM — commit in-progress text and prepare for tool execution.
            AppEvent::ToolCallStart { name, .. } => {
                self.commit_current_response(true);
                self.status_text = format!("Calling tool: {}", name);
            }
            // Tool call args streaming — not surfaced to TUI yet.
            AppEvent::ToolCallArgs { .. } => {}
            // Tool call complete — no UI action needed.
            AppEvent::ToolCallEnd { .. } => {}
            // Tool execution started.
            AppEvent::ToolExecStart { name, .. } => {
                self.status_text = format!("Running: {}", name);
            }
            // Tool execution finished — update status with result.
            AppEvent::ToolExecEnd { name, success, .. } => {
                self.status_text = if success {
                    format!("Tool {} completed", name)
                } else {
                    format!("Tool {} failed", name)
                };
            }
        }
    }

    // Save the current in-progress response as a complete message and clear for the next turn.
    fn commit_current_response(&mut self, is_thinking: bool) {
        if !self.current_response.is_empty() || !self.current_reasoning.is_empty() {
            if let Some(last) = self.messages.last_mut() {
                if last.role == "assistant" {
                    last.content = self.current_response.clone();
                    if !self.current_reasoning.is_empty() {
                        last.reasoning = Some(self.current_reasoning.clone());
                    }
                    last.thinking = is_thinking;
                }
            }
            self.current_response.clear();
            self.current_reasoning.clear();
        }
    }

    // Render the full TUI frame: content area, input line, and status bar.
    fn render(&mut self, f: &mut ratatui::Frame) {
        let size = f.area();

        // Split the screen into three rows: content, input, status bar.
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(size);

        // Render the top content area depending on the current mode.
        match self.mode {
            TuiMode::Normal => {
                self.chat_area = chunks[0];
                self.render_chat_area(chunks[0], f);
            }
            TuiMode::ConnectSettings => {
                f.render_widget(self.render_connect_settings(), chunks[0]);
            }
            TuiMode::VariantPicker => {
                f.render_widget(self.render_variant_picker(), chunks[0]);
            }
            TuiMode::ProviderPicker => {
                f.render_widget(self.render_provider_picker(), chunks[0]);
            }
            TuiMode::ProviderAction { .. } => {
                f.render_widget(self.render_provider_action(), chunks[0]);
            }
        }

        // Build the input line with cursor highlighting.
        let mut input_spans = vec![Span::raw("> ")];
        for (i, ch) in self.input_buffer.char_indices() {
            if i == self.cursor_position {
                // Highlight the character at cursor position with inverted colors.
                input_spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().bg(Color::White).fg(Color::Black),
                ));
            } else {
                input_spans.push(Span::raw(ch.to_string()));
            }
        }
        // If cursor is past end of buffer, show a highlighted space.
        if self.cursor_position >= self.input_buffer.len() {
            input_spans.push(Span::styled(
                " ",
                Style::default().bg(Color::White).fg(Color::Black),
            ));
        }

        // Render the input line widget with a top border.
        let input_line = Line::from(input_spans);
        let input_widget = Paragraph::new(input_line).block(Block::default().borders(Borders::TOP));
        f.render_widget(input_widget, chunks[1]);

        // Calculate cursor position accounting for unicode width and prompt prefix.
        let before_cursor =
            &self.input_buffer[..self.cursor_position.min(self.input_buffer.len())];
        let prompt_width = 2;
        let visual_before = unicode_width::UnicodeWidthStr::width(before_cursor);
        let visual_total = prompt_width + visual_before;
        let inner_width = chunks[1].width as usize;

        let cursor_x = (visual_total % inner_width.max(1)) as u16;
        let cursor_y = chunks[1].y + 1 + (visual_total / inner_width.max(1)) as u16;
        f.set_cursor_position(Position::new(cursor_x, cursor_y));

        // Status bar: color-coded status, provider name, model name, and keyboard shortcuts.
        let status_style = match self.status_text.as_str() {
            s if s.starts_with("Idle") => Style::default().fg(Color::Green),
            s if s.starts_with("Waiting")
                || s.starts_with("Thinking")
                || s.starts_with("Connected")
                || s.starts_with("Switched") =>
            {
                Style::default().fg(Color::Yellow)
            }
            s if s.starts_with("Generating") => Style::default().fg(Color::Cyan),
            s if s.starts_with("Error") => Style::default().fg(Color::Red),
            _ => Style::default(),
        };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(
                " DriftCLI ",
                Style::default().fg(Color::Black).bg(Color::White),
            ),
            Span::raw(" | "),
            Span::styled(self.status_text.clone(), status_style),
            Span::raw(" | "),
            Span::styled(&self.provider_name, Style::default().fg(Color::Cyan)),
            Span::raw(" | "),
            Span::styled(&self.model_name, Style::default().fg(Color::Magenta)),
            Span::styled(
                self.connect_form
                    .reasoning_effort
                    .as_ref()
                    .map(|e| format!(" [{}]", e))
                    .unwrap_or_default(),
                Style::default().fg(Color::DarkGray),
            ),

        ]));
        f.render_widget(status, chunks[2]);
    }

    // Copy text to the system clipboard. Tries arboard first, falls back to OSC 52.
    fn copy_to_clipboard(text: &str) -> anyhow::Result<()> {
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => {
                clipboard.set_text(text)?;
            }
            Err(_) => {
                // Fallback: OSC 52 terminal clipboard (works over SSH).
                use std::io::Write;
                use base64::Engine as _;
                let b64 = base64::engine::general_purpose::STANDARD.encode(text);
                let osc52 = format!("\x1b]52;c;{}\x1b\\", b64);
                let mut stdout = stdout();
                stdout.write_all(osc52.as_bytes())?;
                stdout.flush()?;
            }
        }
        Ok(())
    }

    // Render the scrollable chat message area with scroll offset and selection highlighting.
    fn render_chat_area(&mut self, area: ratatui::layout::Rect, f: &mut ratatui::Frame) {
        // Build flat lines for all messages with role-based styling.
        let mut lines: Vec<Line> = Vec::new();

        for msg in &self.messages {
            let text_style = match msg.role.as_str() {
                "system" => Style::default().fg(Color::Yellow),
                _ if msg.thinking => Style::default().fg(Color::DarkGray),
                _ => Style::default(),
            };

            if let Some(reasoning) = &msg.reasoning {
                for line in reasoning.lines() {
                    lines.push(Line::from(Span::styled(
                        line.to_string(),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }

            if msg.role == "user" {
                let width = area.width.saturating_sub(2) as usize;
                let top = format!("┌{}┐", "─".repeat(width));
                let bottom = format!("└{}┘", "─".repeat(width));
                lines.push(Line::from(Span::styled(top, Style::default().fg(Color::Cyan))));
                for content_line in msg.content.lines() {
                    let vis = unicode_width::UnicodeWidthStr::width(content_line);
                    let inner = area.width.saturating_sub(4) as usize;
                    let pad = inner.saturating_sub(vis);
                    let padded = format!("│ {} {}│", content_line, " ".repeat(pad));
                    lines.push(Line::from(Span::styled(padded, Style::default().fg(Color::Cyan))));
                }
                lines.push(Line::from(Span::styled(bottom, Style::default().fg(Color::Cyan))));
            } else {
                for line in msg.content.lines() {
                    lines.push(Line::from(Span::styled(line.to_string(), text_style)));
                }
            }
        }

        if lines.is_empty() {
            return;
        }

        // Render the paragraph with scrolling.
        let total = lines.len();
        let visible = area.height as usize;
        let max_scroll = total.saturating_sub(visible);
        let top_offset = max_scroll.saturating_sub(self.chat_scroll_offset).min(max_scroll);
        let paragraph = Paragraph::new(lines).scroll((top_offset as u16, 0));
        f.render_widget(paragraph, area);

        // Overlay the selection highlight on top of the rendered content.
        self.selection.render(f.buffer_mut());
    }

    // Render the connection settings form with provider name, type, URL, key, model fields, and buttons.
    fn render_connect_settings(&self) -> Paragraph<'_> {
        let form = &self.connect_form;

        // Highlight flags for each form field. Fields: 0=name, 1=provider, 2=url, 3=key, 4=model, 5=save, 6=cancel.
        let name_highlight = form.selected_field == 0;
        let provider_highlight = form.selected_field == 1;
        let url_highlight = form.selected_field == 2;
        let key_highlight = form.selected_field == 3;
        let model_highlight = form.selected_field == 4;
        let save_highlight = form.selected_field == 5;
        let cancel_highlight = form.selected_field == 6;

        // Obfuscate the API key for display.
        let key_display = if form.api_key.is_empty() {
            "(not set)".to_string()
        } else {
            let len = form.api_key.len();
            if len <= 8 {
                "***".to_string()
            } else {
                format!(
                    "{}...{}",
                    &form.api_key[..4],
                    &form.api_key[len - 4..]
                )
            }
        };

        // Build styled spans for each form field based on highlight state.
        let name_field = if name_highlight {
            Span::styled(
                format!("> {}", form.provider_name),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", form.provider_name),
                Style::default().fg(Color::White),
            )
        };

        let provider_options = if provider_highlight {
            Span::styled(
                format!("[ {} ]  (Left/Right to change)", form.provider_label()),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}   ", form.provider_label()),
                Style::default().fg(Color::Cyan),
            )
        };

        let url_field = if url_highlight {
            Span::styled(
                format!("> {}", form.base_url),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", form.base_url),
                Style::default().fg(Color::White),
            )
        };

        let key_field = if key_highlight {
            Span::styled(
                format!("> {}", key_display),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(format!("  {}", key_display), Style::default().fg(Color::White))
        };

        let model_field = if model_highlight {
            Span::styled(
                format!("> {}", form.model),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(format!("  {}", form.model), Style::default().fg(Color::White))
        };

        let save_btn = if save_highlight {
            Span::styled(
                "[ Save & Apply ]",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green),
            )
        } else {
            Span::styled(
                "  Save & Apply  ",
                Style::default().fg(Color::Green),
            )
        };

        let cancel_btn = if cancel_highlight {
            Span::styled(
                "[ Cancel ]",
                Style::default().fg(Color::Black).bg(Color::Red),
            )
        } else {
            Span::styled("  Cancel  ", Style::default().fg(Color::Red))
        };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(vec![
            Span::raw("  Name:       "),
            name_field,
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Provider:   "),
            provider_options,
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Base URL:   "),
            url_field,
        ]));
        lines.push(Line::from(vec![
            Span::raw("  API Key:    "),
            key_field,
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Model:      "),
            model_field,
            Span::raw(" "),
            Span::styled(
                if form.fetching_models {
                    "(fetching...)"
                } else {
                    "(Enter to fetch)"
                },
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        lines.push(Line::from(Span::raw("")));

        // Render the model list dropdown if visible.
        if form.show_model_list && !form.model_list.is_empty() {
            let start = form
                .model_list_index
                .saturating_sub(4);
            let end = (start + 9).min(form.model_list.len());
            for i in start..end {
                let entry = &form.model_list[i];
                let model_line = if i == form.model_list_index {
                    Line::from(Span::styled(
                        format!("    > {}", entry.id),
                        Style::default().fg(Color::White).bg(Color::DarkGray),
                    ))
                } else {
                    Line::from(Span::raw(format!("      {}", entry.id)))
                };
                lines.push(model_line);
            }
            if form.model_list.len() > 9 {
                lines.push(Line::from(Span::styled(
                    format!(
                        "      ... ({} total, scroll with Up/Down)",
                        form.model_list.len()
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            lines.push(Line::from(Span::raw("")));
        }

        // Display status message (e.g. fetch result) if set.
        if !form.status_message.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  {}", form.status_message),
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(Span::raw("")));
        }

        lines.push(Line::from(vec![save_btn, Span::raw("  "), cancel_btn]));

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Tab/Up/Down: navigate  Left/Right: switch  Enter: confirm  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));

        // Decorative block border and title.
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /connect — Configure Connection ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the provider picker overlay for switching between configured providers.
    fn render_provider_picker(&self) -> Paragraph<'_> {
        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Select provider:",
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        for (i, name) in self.providers.iter().enumerate() {
            let active = name == &self.provider_name;
            let arrow = if i == self.provider_selected { ">" } else { " " };
            let text = if active {
                format!("  {} * {}", arrow, name)
            } else {
                format!("    {} {}", arrow, name)
            };
            let style = if i == self.provider_selected {
                Style::default().fg(Color::Black).bg(Color::Green)
            } else if active {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(Span::styled(text, style)));
        }

        // [+ New Provider] option
        let new_idx = self.providers.len();
        let arrow = if self.provider_selected == new_idx { ">" } else { " " };
        lines.push(Line::from(Span::raw("")));
        let text = format!("    {} [+ New Provider]", arrow);
        let style = if self.provider_selected == new_idx {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        lines.push(Line::from(Span::styled(text, style)));

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: select  d: delete  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /provider — Select Provider ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the provider action sub-menu (Apply / Modify) after selecting a provider.
    fn render_provider_action(&self) -> Paragraph<'_> {
        let name = match &self.mode {
            TuiMode::ProviderAction { provider_name } => provider_name.clone(),
            _ => return Paragraph::new(vec![]),
        };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            format!("  Provider: {}", name),
            Style::default().fg(Color::Yellow),
        )));
        lines.push(Line::from(Span::raw("")));

        for (i, label) in ["  Apply  (switch to this provider)", "  Modify (edit provider config)"].iter().enumerate() {
            let style = if i == self.provider_action_selected {
                Style::default().fg(Color::Black).bg(Color::Green)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(Span::styled(*label, style)));
        }

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: confirm  Esc: back",
            Style::default().fg(Color::DarkGray),
        )));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /provider — Action ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the variant / reasoning-effort picker overlay.
    fn render_variant_picker(&self) -> Paragraph<'_> {
        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Select reasoning effort for the current model:",
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        // Check if the model has known capability info.
        let has_capabilities = self
            .connect_form
            .model_list
            .iter()
            .find(|m| m.id == self.connect_form.model)
            .map(|m| !m.effort_levels.is_empty())
            .unwrap_or(false);

        // Show a fallback warning if capabilities are unknown.
        if !has_capabilities && self.variant_options.len() > 1 {
            lines.push(Line::from(Span::styled(
                "  (model capabilities unknown — showing all levels as fallback)",
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(Span::raw("")));
        }

        if self.variant_options.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No effort levels available.",
                Style::default().fg(Color::Yellow),
            )));
        } else {
            for (i, level) in self.variant_options.iter().enumerate() {
                let label = if i == self.variant_selected {
                    Span::styled(
                        format!("    > {}", level),
                        Style::default().fg(Color::Black).bg(Color::Green),
                    )
                } else if level == "(none)" {
                    Span::styled(
                        format!("      {}", level),
                        Style::default().fg(Color::DarkGray),
                    )
                } else {
                    Span::styled(
                        format!("      {}", level),
                        Style::default().fg(Color::White),
                    )
                };
                lines.push(Line::from(label));
            }
        }

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: confirm  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::raw("")));

        // Show the currently selected effort level.
        let current = self.connect_form.reasoning_effort.as_deref().unwrap_or("none");
        lines.push(Line::from(Span::styled(
            format!("  Current: {}", current),
            Style::default().fg(Color::Yellow),
        )));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /variant — Reasoning Effort ")
            .border_style(Style::default().fg(Color::Magenta));
        Paragraph::new(lines).block(block)
    }
}

pub use input::InputAction;
