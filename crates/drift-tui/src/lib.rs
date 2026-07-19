mod components;
mod connect;
mod input;
mod markdown;
mod selection;
mod slash;

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton,
        MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use drift_config::LlmConfig;
use drift_llm::ModelInfo;
use ratatui::{
    Terminal,
    layout::{Constraint, Direction, Layout, Position},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use selection::SelectionState;
use slash::{SlashCommand, filter_commands};
use std::io::{self, stdout};
use std::time::Instant;
use tokio::sync::mpsc;

// Events sent from the backend to the TUI via async channel.
#[derive(Debug, Clone)]
pub enum AppEvent {
    Token(String),
    Reasoning(String),
    ReasoningComplete {
        duration_ms: u64,
    },
    AgentStatus(String),
    /// Status update emitted by an MCP server lifecycle task.
    McpStatus {
        server_id: String,
        status: String,
    },
    Error(String),
    Done,
    Interrupted,
    ModelList(Vec<ModelInfo>),
    ProviderList(Vec<String>),
    ProviderSwitched {
        name: String,
        model: String,
    },
    // Full config for a specific provider loaded from the backend.
    ProviderConfig {
        name: String,
        config: LlmConfig,
    },
    // Tool call started with id and name
    ToolCallStart {
        id: String,
        name: String,
    },
    // Tool call arguments streaming
    ToolCallArgs {
        id: String,
        delta: String,
    },
    // Tool call completed
    ToolCallEnd {
        id: String,
    },
    // Tool execution started
    ToolExecStart {
        id: String,
        name: String,
    },
    // Tool execution finished with result summary
    ToolExecEnd {
        id: String,
        name: String,
        success: bool,
    },
    // Carry session metadata arrays back to TUI for /sessions modal list
    SessionList(Vec<drift_storage::SessionMeta>),
    // Signal TUI that a specific session has been reconstructed from store
    SessionLoaded {
        session_id: uuid::Uuid,
        messages: Vec<ChatMessage>,
    },
    // Permission system: agent asks for user approval before executing a tool.
    PermissionRequest {
        request_id: String,
        tool_name: String,
        args_summary: String,
        reason: String,
    },
    // Permission was granted or denied.
    PermissionResolved {
        request_id: String,
        allowed: bool,
    },
}

// Commands sent from the TUI to the backend (chat, fetch models, reconfigure, provider management).
#[derive(Debug, Clone)]
pub enum TuiCommand {
    Chat(String),
    Interrupt,
    FetchModels {
        provider: String,
        base_url: String,
        api_key: String,
    },
    Reconfigure(LlmConfig),
    // User response to a permission request.
    PermissionResponse {
        request_id: String,
        allowed: bool,
        /// Whether to persist this decision for the rest of the session.
        remember: bool,
    },
    // Save or update a named provider configuration.
    SaveProvider {
        name: String,
        config: LlmConfig,
    },
    // Switch to a different configured provider.
    SetActiveProvider(String),
    // Request the list of configured provider names.
    GetProviders,
    // Delete a named provider from the configuration.
    DeleteProvider(String),
    // Get the full config for a named provider (used for "Modify" in the provider picker).
    GetProviderConfig(String),
    // Get the full list of saved historical sessions.
    GetSessions,
    // Switch to a different historical session by UUID.
    SwitchSession(uuid::Uuid),
}

/// Pending permission prompt — blocks normal input until the user responds.
#[derive(Debug, Clone)]
struct PermissionPromptState {
    request_id: String,
    tool_name: String,
    args_summary: String,
    reason: String,
    risk_level: String,
}

struct SlashCompletionState {
    filtered: Vec<SlashCommand>,
    selected: usize,
}

/// Target of a reasoning header click — either a committed message or the live streaming block.
#[derive(Debug, Clone, Copy)]
enum ReasoningTarget {
    /// A committed ChatMessage at the given index in self.messages.
    Message(usize),
    /// The live streaming reasoning block (self.current_reasoning).
    Live,
}

// Central application state for the DriftCLI TUI.
pub struct TuiApp {
    messages: Vec<ChatMessage>,
    current_response: String,
    current_reasoning: String,
    current_reasoning_duration: Option<u64>,
    reasoning_start_time: Option<Instant>,
    current_reasoning_collapsed: bool,
    current_thinking_tools: Vec<String>,
    reasoning_header_positions: Vec<(ReasoningTarget, usize)>,
    total_chat_lines: usize,
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
    provider_name: String,
    // Active session metadata and switching list
    session_id: uuid::Uuid,
    session_list: Vec<drift_storage::SessionMeta>,
    session_selected: usize,
    event_rx: mpsc::UnboundedReceiver<AppEvent>,
    cmd_tx: mpsc::UnboundedSender<TuiCommand>,
    should_quit: bool,
    history: Vec<String>,
    history_index: Option<usize>,
    chat_scroll_offset: usize,
    selection: SelectionState,
    slash_completion: Option<SlashCompletionState>,
    chat_area: ratatui::layout::Rect,
    /// Active permission prompt — when set, normal input is blocked until resolved.
    permission_prompt: Option<PermissionPromptState>,
    /// Whether Ctrl+C has been pressed once and is waiting for confirmation.
    quit_confirmation_pending: bool,
}

// Which screen/overlay the TUI is currently displaying.
#[derive(Debug, Clone, PartialEq)]
pub enum TuiMode {
    Normal,
    ConnectSettings,
    VariantPicker,
    // Provider switcher overlay (list configured providers with delete option)
    ProviderPicker,
    // Session list and load picker overlay
    SessionPicker,
}

// A single chat message with optional reasoning/thinking content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
    pub thinking: bool,
    pub reasoning_duration_ms: Option<u64>,
    pub reasoning_collapsed: bool,
    pub thinking_tools: Vec<String>,
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
            current_reasoning_duration: None,
            reasoning_start_time: None,
            current_reasoning_collapsed: true,
            current_thinking_tools: Vec::new(),
            reasoning_header_positions: Vec::new(),
            total_chat_lines: 0,
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
            provider_name: "default".to_string(),
            session_id: uuid::Uuid::nil(),
            session_list: Vec::new(),
            session_selected: 0,
            event_rx,
            cmd_tx,
            should_quit: false,
            history: Vec::new(),
            history_index: None,
            chat_scroll_offset: 0,
            selection: SelectionState::new(),
            slash_completion: None,
            chat_area: ratatui::layout::Rect::new(0, 0, 80, 24),
            permission_prompt: None,
            quit_confirmation_pending: false,
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
            // Draw the current frame.
            terminal.draw(|f| self.render(f))?;

            // Drain any pending async events from the backend.
            if let Ok(event) = self.event_rx.try_recv() {
                self.handle_app_event(event);
            }

            // Poll for input events (non-blocking with 16 ms timeout).
            if event::poll(std::time::Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        // Require two consecutive Ctrl+C presses before quitting the TUI.
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c')
                        {
                            if self.quit_confirmation_pending {
                                self.should_quit = true;
                            } else {
                                self.quit_confirmation_pending = true;
                                self.status_text = "Press Ctrl+C again to exit".into();
                            }
                            continue;
                        }
                        self.quit_confirmation_pending = false;
                        // ── Permission prompt interceptor ──
                        // Permission prompt interceptor — only y/Y/n/N keys accepted
                        if let Some(prompt) = self.permission_prompt.as_ref() {
                            if key.code == KeyCode::Esc {
                                self.permission_prompt = None;
                                let _ = self.cmd_tx.send(TuiCommand::Interrupt);
                                self.status_text = "Interrupting...".into();
                                continue;
                            }
                            let request_id = prompt.request_id.clone();
                            let (allowed, remember) = match key.code {
                                KeyCode::Char('y') => (true, false),
                                KeyCode::Char('Y') => (true, true),
                                KeyCode::Char('n') => (false, false),
                                KeyCode::Char('N') => (false, true),
                                _ => {
                                    continue; // Ignore other keys
                                }
                            };
                            self.permission_prompt = None;
                            let _ = self.cmd_tx.send(TuiCommand::PermissionResponse {
                                request_id,
                                allowed,
                                remember,
                            });
                            continue;
                        }
                        // ── End permission interceptor ──

                        match self.mode {
                            TuiMode::Normal => {
                                // Slash completion popup key intercept
                                let mut popup_handled = false;
                                if let Some(ref mut state) = self.slash_completion {
                                    if !state.filtered.is_empty() {
                                        match key.code {
                                            KeyCode::Up => {
                                                state.selected = if state.selected == 0 {
                                                    state.filtered.len() - 1
                                                } else {
                                                    state.selected - 1
                                                };
                                                popup_handled = true;
                                            }
                                            KeyCode::Down => {
                                                state.selected =
                                                    if state.selected + 1 >= state.filtered.len() {
                                                        0
                                                    } else {
                                                        state.selected + 1
                                                    };
                                                popup_handled = true;
                                            }
                                            KeyCode::Tab => {
                                                let cmd = state.filtered[state.selected].name;
                                                self.input_buffer = cmd.to_string();
                                                self.cursor_position = cmd.len();
                                                popup_handled = true;
                                            }
                                            KeyCode::Enter => {
                                                let cmd = state.filtered[state.selected].name;
                                                self.handle_command(cmd);
                                                popup_handled = true;
                                            }
                                            KeyCode::Esc => {
                                                self.slash_completion = None;
                                                popup_handled = true;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                if !popup_handled {
                                    match key.code {
                                        KeyCode::Up => {
                                            if self.history_index.is_none()
                                                && !self.input_buffer.is_empty()
                                            {
                                                self.history.push(self.input_buffer.clone());
                                            }
                                            if !self.history.is_empty() {
                                                let idx = self
                                                    .history_index
                                                    .map_or(self.history.len(), |i| i);
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
                                                    self.input_buffer =
                                                        self.history[new_idx].clone();
                                                }
                                                self.cursor_position = self.input_buffer.len();
                                            }
                                        }
                                        // PgUp / PgDn scroll the chat area by half the viewport.
                                        KeyCode::PageUp => {
                                            self.chat_scroll_offset += 10;
                                        }
                                        KeyCode::PageDown => {
                                            self.chat_scroll_offset =
                                                self.chat_scroll_offset.saturating_sub(10);
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
                                                                reasoning_duration_ms: None,
                                                                reasoning_collapsed: false,
                                                                thinking_tools: Vec::new(),
                                                            });
                                                            self.current_response.clear();
                                                            self.current_reasoning.clear();
                                                            self.current_thinking_tools.clear();
                                                            self.reasoning_start_time = None;
                                                            self.current_reasoning_collapsed = true;
                                                            self.chat_scroll_offset = 0;
                                                            self.selection.clear();
                                                            self.status_text = "Waiting...".into();
                                                            let _ = self
                                                                .cmd_tx
                                                                .send(TuiCommand::Chat(text));
                                                        }
                                                        self.input_buffer.clear();
                                                        self.cursor_position = 0;
                                                    }
                                                }
                                                input::InputAction::Interrupt => {
                                                    let _ = self.cmd_tx.send(TuiCommand::Interrupt);
                                                }
                                                input::InputAction::Quit => self.should_quit = true,
                                                input::InputAction::ToggleConnectInfo => {}
                                            }
                                        }
                                    }
                                }
                                self.update_slash_completion();
                            }
                            TuiMode::ConnectSettings => {
                                self.handle_connect_key(key.code);
                            }
                            TuiMode::VariantPicker => {
                                self.handle_variant_key(key.code);
                            }
                            TuiMode::ProviderPicker => {
                                self.handle_provider_key(key.code);
                            }
                            TuiMode::SessionPicker => {
                                self.handle_session_picker_key(key.code);
                            }
                        }
                    }
                    Event::Mouse(mouse) => {
                        match mouse.kind {
                            MouseEventKind::ScrollUp => {
                                match self.mode {
                                    TuiMode::Normal => {
                                        self.chat_scroll_offset += 3;
                                    }
                                    TuiMode::ConnectSettings
                                        if self.connect_form.show_model_list =>
                                    {
                                        self.connect_form.model_list_index =
                                            self.connect_form.model_list_index.saturating_sub(1);
                                    }
                                    TuiMode::VariantPicker => {
                                        self.variant_selected =
                                            self.variant_selected.saturating_sub(1);
                                    }
                                    TuiMode::ProviderPicker => {
                                        self.provider_selected =
                                            self.provider_selected.saturating_sub(1);
                                    }
                                    TuiMode::SessionPicker => {
                                        self.session_selected =
                                            self.session_selected.saturating_sub(1);
                                    }
                                    _ => {}
                                }
                                // Scroll clears any active selection.
                                self.selection.clear();
                            }
                            MouseEventKind::ScrollDown => {
                                match self.mode {
                                    TuiMode::Normal => {
                                        self.chat_scroll_offset =
                                            self.chat_scroll_offset.saturating_sub(3);
                                    }
                                    TuiMode::ConnectSettings
                                        if self.connect_form.show_model_list =>
                                    {
                                        if self.connect_form.model_list_index + 1
                                            < self.connect_form.model_list.len()
                                        {
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
                                    TuiMode::SessionPicker => {
                                        if !self.session_list.is_empty()
                                            && self.session_selected + 1 < self.session_list.len()
                                        {
                                            self.session_selected += 1;
                                        }
                                    }
                                    _ => {}
                                }
                                self.selection.clear();
                            }
                            MouseEventKind::Down(button) => {
                                if self.mode == TuiMode::Normal && button == MouseButton::Left {
                                    if !self.try_toggle_reasoning(&mouse) {
                                        self.selection.handle_mouse_event(&mouse, self.chat_area);
                                    }
                                }
                            }
                            MouseEventKind::Drag(_) | MouseEventKind::Up(_) => {
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
            "/sessions" | "/history" => {
                let _ = self.cmd_tx.send(TuiCommand::GetSessions);
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
                    content: format!(
                        "Unknown command: {}. Try /connect, /provider, /sessions, /clear, /quit",
                        cmd
                    ),
                    reasoning: None,
                    thinking: false,
                    reasoning_duration_ms: None,
                    reasoning_collapsed: false,
                    thinking_tools: Vec::new(),
                });
            }
        }
        self.input_buffer.clear();
        self.cursor_position = 0;
    }

    // Update slash completion popup based on current input buffer.
    fn update_slash_completion(&mut self) {
        if self.input_buffer.starts_with('/') {
            let filter = &self.input_buffer[1..];
            let filtered = filter_commands(filter);
            if filtered.is_empty() {
                self.slash_completion = None;
            } else if let Some(ref mut state) = self.slash_completion {
                if state.selected >= filtered.len() {
                    state.selected = filtered.len().saturating_sub(1);
                }
                state.filtered = filtered;
            } else {
                self.slash_completion = Some(SlashCompletionState {
                    filtered,
                    selected: 0,
                });
            }
        } else {
            self.slash_completion = None;
        }
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
                    let _ = self.cmd_tx.send(TuiCommand::SetActiveProvider(name));
                    self.mode = TuiMode::Normal;
                } else {
                    // [+ New Provider]
                    self.mode = TuiMode::ConnectSettings;
                }
            }
            KeyCode::Char('e') => {
                if self.provider_selected < self.providers.len() {
                    let name = self.providers[self.provider_selected].clone();
                    let _ = self.cmd_tx.send(TuiCommand::GetProviderConfig(name));
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

    // Handle key presses while the session picker is active.
    fn handle_session_picker_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => self.mode = TuiMode::Normal,
            KeyCode::Up => {
                self.session_selected = self.session_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                if !self.session_list.is_empty()
                    && self.session_selected + 1 < self.session_list.len()
                {
                    self.session_selected += 1;
                }
            }
            KeyCode::Enter => {
                if !self.session_list.is_empty() && self.session_selected < self.session_list.len()
                {
                    let s_meta = &self.session_list[self.session_selected];
                    if let Ok(parsed_id) = uuid::Uuid::parse_str(&s_meta.session_id) {
                        let _ = self.cmd_tx.send(TuiCommand::SwitchSession(parsed_id));
                        self.status_text = "Loading session...".to_string();
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
                let level = self.variant_options.get(self.variant_selected).cloned();
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
            self.model_name, self.connect_form.base_url
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
                        reasoning_duration_ms: None,
                        reasoning_collapsed: false,
                        thinking_tools: Vec::new(),
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
                if self.reasoning_start_time.is_none() {
                    self.reasoning_start_time = Some(Instant::now());
                }
                self.current_reasoning.push_str(&text);
            }
            // Reasoning/thinking phase completed — accumulate elapsed time
            // across multiple tool-call iterations so the total thinking
            // time reflects all bursts within the same submit.
            // Uses max(self timer, agent duration) to prevent the live
            // timer from visually backtracking when the agent's measured
            // duration arrives slightly behind the TUI's own elapsed.
            AppEvent::ReasoningComplete { duration_ms } => {
                let prev = self.current_reasoning_duration.unwrap_or(0);
                let burst = self
                    .reasoning_start_time
                    .take()
                    .map(|start| start.elapsed().as_millis() as u64)
                    .unwrap_or(duration_ms)
                    .max(duration_ms);
                self.current_reasoning_duration = Some(prev + burst);
            }
            // Update the status bar text directly.
            AppEvent::AgentStatus(status) => {
                self.status_text = status;
            }
            AppEvent::McpStatus { server_id, status } => {
                self.status_text = format!("MCP {}: {}", server_id, status);
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
                        reasoning_duration_ms: None,
                        reasoning_collapsed: false,
                        thinking_tools: Vec::new(),
                    });
                    self.status_text = "Error".into();
                }
            }
            // Response streaming complete — finalize the message and reset.
            AppEvent::Done => {
                self.commit_current_response(false);
                self.status_text = "Idle".into();
            }
            // Keep any partial response visible, but return the TUI to an idle state.
            AppEvent::Interrupted => {
                self.commit_current_response(true);
                self.status_text = "Interrupted".into();
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
            // Tool call requested by LLM — update status, but keep current_reasoning
            // accumulating so that multiple tool-call iterations (OpenAI Compatible,
            // Anthropic extended thinking) produce a single unified thinking block.
            AppEvent::ToolCallStart { name, .. } => {
                self.status_text = format!("Calling tool: {}", name);
                self.current_thinking_tools
                    .push(format!("> Calling tool: {}", name));
            }
            // Tool call args streaming — not surfaced to TUI yet.
            AppEvent::ToolCallArgs { .. } => {}
            // Tool call complete — no UI action needed.
            AppEvent::ToolCallEnd { .. } => {}
            // Tool execution started.
            AppEvent::ToolExecStart { name, .. } => {
                self.status_text = format!("Running: {}", name);
                self.current_thinking_tools
                    .push(format!("> Running: {}", name));
            }
            // Tool execution finished — update status with result.
            AppEvent::ToolExecEnd { name, success, .. } => {
                self.status_text = if success {
                    format!("Tool {} completed", name)
                } else {
                    format!("Tool {} failed", name)
                };
                self.current_thinking_tools.push(format!(
                    "> Tool {} {}",
                    name,
                    if success { "completed" } else { "failed" }
                ));
            }
            AppEvent::SessionList(meta_list) => {
                self.session_list = meta_list;
                self.session_selected = self
                    .session_list
                    .iter()
                    .position(|s| s.session_id == self.session_id.to_string())
                    .unwrap_or(0);
                self.mode = TuiMode::SessionPicker;
            }
            AppEvent::SessionLoaded {
                session_id,
                messages,
            } => {
                self.session_id = session_id;
                self.messages = messages;
                self.current_response.clear();
                self.current_reasoning.clear();
                self.current_thinking_tools.clear();
                self.chat_scroll_offset = 0;
                self.status_text = format!("Loaded session {}", &session_id.to_string()[..8]);
                self.mode = TuiMode::Normal;
            }
            AppEvent::PermissionRequest {
                request_id,
                tool_name,
                args_summary,
                reason,
                ..
            } => {
                // Set the interactive permission prompt — blocks normal input until resolved.
                self.permission_prompt = Some(PermissionPromptState {
                    request_id,
                    tool_name,
                    args_summary,
                    reason,
                    risk_level: "medium".into(),
                });
                // Also push a system message so it appears in the chat transcript.
                let msg = format!(
                    "⚠ Permission needed: `{}` with args `{}`\nReason: {}",
                    self.permission_prompt.as_ref().unwrap().tool_name,
                    self.permission_prompt.as_ref().unwrap().args_summary,
                    self.permission_prompt.as_ref().unwrap().reason,
                );
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: msg,
                    reasoning: None,
                    thinking: false,
                    reasoning_duration_ms: None,
                    reasoning_collapsed: false,
                    thinking_tools: Vec::new(),
                });
            }
            AppEvent::PermissionResolved { .. } => {
                // For now, resolved notifications are logged but not displayed distinctly
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
                    last.reasoning_duration_ms = {
                        let accumulated = self.current_reasoning_duration.take().unwrap_or(0);
                        let remaining = self
                            .reasoning_start_time
                            .take()
                            .map(|start| start.elapsed().as_millis() as u64)
                            .unwrap_or(0);
                        Some(accumulated + remaining)
                    };
                    last.reasoning_collapsed = self.current_reasoning_collapsed;
                    last.thinking = is_thinking;
                    last.thinking_tools = std::mem::take(&mut self.current_thinking_tools);
                }
            }
            self.current_response.clear();
            self.current_reasoning.clear();
            self.current_thinking_tools.clear();
            self.current_reasoning_collapsed = true;
            self.current_reasoning_duration = None;
        }
    }

    // Render the full TUI frame: content area, input line, and status bar.
    fn render(&mut self, f: &mut ratatui::Frame) {
        let size = f.area();

        // Calculate popup height (max 8 items + 2 for borders)
        let popup_height = self.slash_completion.as_ref().and_then(|s| {
            if s.filtered.is_empty() {
                None
            } else {
                Some((s.filtered.len().min(8) + 2) as u16)
            }
        });

        // Split the screen: content, optional popup, input, status bar.
        let mut constraints = vec![
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ];
        if let Some(h) = popup_height {
            constraints.insert(1, Constraint::Length(h));
        }
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(size);
        let content_area = layout[0];
        let (input_area, status_area, popup_area) = if popup_height.is_some() {
            (layout[2], layout[3], Some(layout[1]))
        } else {
            (layout[1], layout[2], None)
        };

        // Render the top content area depending on the current mode.
        match self.mode {
            TuiMode::Normal => {
                self.chat_area = content_area;
                self.render_chat_area(content_area, f);
            }
            TuiMode::ConnectSettings => {
                f.render_widget(self.render_connect_settings(), content_area);
            }
            TuiMode::VariantPicker => {
                f.render_widget(self.render_variant_picker(), content_area);
            }
            TuiMode::ProviderPicker => {
                f.render_widget(self.render_provider_picker(), content_area);
            }
            TuiMode::SessionPicker => {
                f.render_widget(self.render_session_picker(), content_area);
            }
        }

        // Render slash completion popup between content and input if active.
        if let (Some(area), Some(state)) = (popup_area, &self.slash_completion) {
            if !state.filtered.is_empty() {
                f.render_widget(self.render_slash_popup(state), area);
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
        // Permission dialog replaces the input line when a prompt is active.
        if let Some(prompt) = &self.permission_prompt {
            let dialog = self.render_permission_dialog(prompt);
            f.render_widget(dialog, input_area);
        } else {
            let input_widget =
                Paragraph::new(input_line).block(Block::default().borders(Borders::TOP));
            f.render_widget(input_widget, input_area);
        }

        // Calculate cursor position accounting for unicode width and prompt prefix.
        let before_cursor = &self.input_buffer[..self.cursor_position.min(self.input_buffer.len())];
        let prompt_width = 2;
        let visual_before = unicode_width::UnicodeWidthStr::width(before_cursor);
        let visual_total = prompt_width + visual_before;
        let inner_width = input_area.width as usize;

        let cursor_x = (visual_total % inner_width.max(1)) as u16;
        let cursor_y = input_area.y + 1 + (visual_total / inner_width.max(1)) as u16;
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
            Span::raw(" | Session: "),
            Span::styled(
                if self.session_id.is_nil() {
                    "(none)".to_string()
                } else {
                    self.session_id.to_string()
                },
                Style::default().fg(Color::Yellow),
            ),
        ]));
        f.render_widget(status, status_area);
    }

    /// Check if a mouse click hits a reasoning header line and toggle it.
    /// Returns true if a reasoning header was clicked (and handled).
    fn try_toggle_reasoning(&mut self, mouse: &crossterm::event::MouseEvent) -> bool {
        let mouse_row = mouse.row as usize;
        let chat_top = self.chat_area.y as usize;
        if mouse_row < chat_top {
            return false;
        }
        let relative_row = mouse_row - chat_top;
        let total = self.total_chat_lines;
        let visible = self.chat_area.height as usize;
        let max_scroll = total.saturating_sub(visible);
        let top_offset = max_scroll
            .saturating_sub(self.chat_scroll_offset)
            .min(max_scroll);
        let clicked_line = relative_row + top_offset;

        for (target, line_idx) in &self.reasoning_header_positions {
            if *line_idx == clicked_line {
                match target {
                    ReasoningTarget::Message(msg_idx) => {
                        if let Some(msg) = self.messages.get_mut(*msg_idx) {
                            msg.reasoning_collapsed = !msg.reasoning_collapsed;
                        }
                    }
                    ReasoningTarget::Live => {
                        self.current_reasoning_collapsed = !self.current_reasoning_collapsed;
                    }
                }
                return true;
            }
        }
        false
    }

    /// Push wrapped reasoning lines into the lines vector with indent and dim style.
    fn push_wrapped_lines(
        lines: &mut Vec<Line<'static>>,
        text: &str,
        area_width: u16,
        indent: &str,
    ) {
        let max_w = area_width.saturating_sub(indent.len() as u16) as usize;
        for paragraph in text.split('\n') {
            let mut current = String::new();
            let mut current_width = 0usize;
            for word in paragraph.split_inclusive(|c: char| c.is_whitespace()) {
                let word_width = unicode_width::UnicodeWidthStr::width(word);
                if current_width + word_width > max_w && current_width > 0 {
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", indent, current.trim_end()),
                        Style::default().fg(Color::DarkGray),
                    )));
                    current.clear();
                    current_width = 0;
                }
                current.push_str(word);
                current_width += word_width;
            }
            if !current.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("{}{}", indent, current.trim_end()),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
    }

    /// Push the live streaming reasoning block into the lines vector.
    /// Used both for pure-thinking phase (no text yet) and interleaved
    /// thinking during generation for OpenAI Compatible models.
    #[allow(clippy::too_many_arguments)]
    fn push_live_reasoning_block(
        header_positions: &mut Vec<(ReasoningTarget, usize)>,
        lines: &mut Vec<Line<'static>>,
        current_reasoning: &str,
        current_reasoning_collapsed: bool,
        reasoning_start_time: Option<Instant>,
        current_reasoning_duration: Option<u64>,
        area_width: u16,
        thinking_tools: &[String],
    ) {
        let live_duration = match reasoning_start_time {
            Some(start) => {
                let current_ms = start.elapsed().as_millis() as u64;
                let total = current_reasoning_duration.unwrap_or(0) + current_ms;
                if total >= 1000 {
                    format!(" for {:.1}s", total as f64 / 1000.0)
                } else {
                    format!(" for {}ms", total)
                }
            }
            None => match current_reasoning_duration {
                Some(ms) if ms > 0 => {
                    if ms >= 1000 {
                        format!(" for {:.1}s", ms as f64 / 1000.0)
                    } else {
                        format!(" for {}ms", ms)
                    }
                }
                _ => String::new(),
            },
        };

        let toggle = if current_reasoning_collapsed {
            "▶"
        } else {
            "▼"
        };
        let header = format!("{} Thinking{}", toggle, live_duration);

        lines.push(Line::from(Span::raw("")));

        header_positions.push((ReasoningTarget::Live, lines.len()));

        lines.push(Line::from(Span::styled(
            header,
            Style::default().fg(Color::Gray),
        )));

        if !current_reasoning_collapsed {
            Self::push_wrapped_lines(lines, current_reasoning, area_width, "  ");
            for tool_line in thinking_tools {
                lines.push(Line::from(Span::styled(
                    format!("  {}", tool_line),
                    Style::default().fg(Color::Cyan),
                )));
            }
        }

        lines.push(Line::from(Span::raw("")));
    }

    // Render the scrollable chat message area with scroll offset and selection highlighting.
    fn render_chat_area(&mut self, area: ratatui::layout::Rect, f: &mut ratatui::Frame) {
        self.reasoning_header_positions.clear();
        self.total_chat_lines = 0;

        // Build flat lines for all messages with role-based styling.
        let mut lines: Vec<Line> = Vec::new();
        let mut live_rendered = false;

        for (msg_idx, msg) in self.messages.iter().enumerate() {
            let text_style = match msg.role.as_str() {
                "system" => Style::default().fg(Color::Yellow),
                _ if msg.thinking => Style::default().fg(Color::DarkGray),
                _ => Style::default(),
            };

            if let Some(reasoning) = &msg.reasoning {
                let duration_str = msg.reasoning_duration_ms.map_or_else(
                    || String::new(),
                    |ms| {
                        if ms >= 1000 {
                            format!(" for {:.1}s", ms as f64 / 1000.0)
                        } else {
                            format!(" for {}ms", ms)
                        }
                    },
                );

                let toggle = if msg.reasoning_collapsed {
                    "▶"
                } else {
                    "▼"
                };
                let header = format!("{} Thinking{}", toggle, duration_str);

                lines.push(Line::from(Span::raw("")));

                // Record header position for mouse hit-testing
                self.reasoning_header_positions
                    .push((ReasoningTarget::Message(msg_idx), lines.len()));

                lines.push(Line::from(Span::styled(
                    header,
                    Style::default().fg(Color::Gray),
                )));

                if !msg.reasoning_collapsed {
                    Self::push_wrapped_lines(&mut lines, reasoning, area.width, "  ");
                    for tool_line in &msg.thinking_tools {
                        lines.push(Line::from(Span::styled(
                            format!("  {}", tool_line),
                            Style::default().fg(Color::Cyan),
                        )));
                    }
                }

                lines.push(Line::from(Span::raw("")));
            }

            // If this is the last message being streamed and live reasoning exists,
            // render the live block before the content (so interleaved thinking
            // appears above the generating text for OpenAI Compatible models).
            let is_last = msg_idx + 1 == self.messages.len();
            if is_last && msg.role == "assistant" && !self.current_reasoning.is_empty() {
                Self::push_live_reasoning_block(
                    &mut self.reasoning_header_positions,
                    &mut lines,
                    &self.current_reasoning,
                    self.current_reasoning_collapsed,
                    self.reasoning_start_time,
                    self.current_reasoning_duration,
                    area.width,
                    &self.current_thinking_tools,
                );
                live_rendered = true;
            }

            if msg.role == "user" {
                let width = area.width.saturating_sub(2) as usize;
                let top = format!("┌{}┐", "─".repeat(width));
                let bottom = format!("└{}┘", "─".repeat(width));
                lines.push(Line::from(Span::styled(
                    top,
                    Style::default().fg(Color::Cyan),
                )));
                for content_line in msg.content.lines() {
                    let vis = unicode_width::UnicodeWidthStr::width(content_line);
                    let inner = area.width.saturating_sub(4) as usize;
                    let pad = inner.saturating_sub(vis);
                    let padded = format!("│ {} {}│", content_line, " ".repeat(pad));
                    lines.push(Line::from(Span::styled(
                        padded,
                        Style::default().fg(Color::Cyan),
                    )));
                }
                lines.push(Line::from(Span::styled(
                    bottom,
                    Style::default().fg(Color::Cyan),
                )));
            } else if msg.role == "assistant" && !msg.thinking {
                let md_lines = markdown::render_markdown(&msg.content, area.width);
                for line in md_lines {
                    lines.push(line);
                }
            } else {
                for line in msg.content.lines() {
                    lines.push(Line::from(Span::styled(line.to_string(), text_style)));
                }
            }
        }

        // If the live block wasn't rendered inline (no assistant message yet),
        // render it at the end for the pure thinking phase.
        if !live_rendered && !self.current_reasoning.is_empty() {
            Self::push_live_reasoning_block(
                &mut self.reasoning_header_positions,
                &mut lines,
                &self.current_reasoning,
                self.current_reasoning_collapsed,
                self.reasoning_start_time,
                self.current_reasoning_duration,
                area.width,
                &self.current_thinking_tools,
            );
        }

        if lines.is_empty() {
            return;
        }

        // Render the paragraph with scrolling.
        let total = lines.len();
        self.total_chat_lines = total;
        let visible = area.height as usize;
        let max_scroll = total.saturating_sub(visible);
        let top_offset = max_scroll
            .saturating_sub(self.chat_scroll_offset)
            .min(max_scroll);
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
                format!("{}...{}", &form.api_key[..4], &form.api_key[len - 4..])
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
            Span::styled(
                format!("  {}", key_display),
                Style::default().fg(Color::White),
            )
        };

        let model_field = if model_highlight {
            Span::styled(
                format!("> {}", form.model),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", form.model),
                Style::default().fg(Color::White),
            )
        };

        let save_btn = if save_highlight {
            Span::styled(
                "[ Save & Apply ]",
                Style::default().fg(Color::Black).bg(Color::Green),
            )
        } else {
            Span::styled("  Save & Apply  ", Style::default().fg(Color::Green))
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
        lines.push(Line::from(vec![Span::raw("  Name:       "), name_field]));
        lines.push(Line::from(vec![
            Span::raw("  Provider:   "),
            provider_options,
        ]));
        lines.push(Line::from(vec![Span::raw("  Base URL:   "), url_field]));
        lines.push(Line::from(vec![Span::raw("  API Key:    "), key_field]));
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
            let start = form.model_list_index.saturating_sub(4);
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
            let arrow = if i == self.provider_selected {
                ">"
            } else {
                " "
            };
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
        let arrow = if self.provider_selected == new_idx {
            ">"
        } else {
            " "
        };
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
            "  Up/Down: choose  Enter: switch  e: edit  d: delete  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /provider — Select Provider ")
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
        let current = self
            .connect_form
            .reasoning_effort
            .as_deref()
            .unwrap_or("none");
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

    // Render the persistent session picker overlay list.
    fn render_session_picker(&self) -> Paragraph<'_> {
        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Saved Historical Sessions:",
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        if self.session_list.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No historical sessions found.",
                Style::default().fg(Color::Yellow),
            )));
        } else {
            for (i, meta) in self.session_list.iter().enumerate() {
                let is_active = meta.session_id == self.session_id.to_string();
                let active_marker = if is_active { "*" } else { " " };

                let text = format!(
                    "  {} [{}]  Created: {}  Dir: {}  Model: {}",
                    active_marker,
                    &meta.session_id[..8.min(meta.session_id.len())],
                    meta.created_at,
                    meta.working_dir,
                    meta.model
                );

                let style = if i == self.session_selected {
                    Style::default().fg(Color::Black).bg(Color::Green)
                } else if is_active {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::White)
                };

                lines.push(Line::from(Span::styled(text, style)));
            }
        }

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: switch session  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::raw("")));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /sessions — Saved Chat History ")
            .border_style(Style::default().fg(Color::Yellow));
        Paragraph::new(lines).block(block)
    }

    // Set full visual ChatMessage messages in TUI (used on loading a session)
    pub fn set_messages(&mut self, messages: Vec<ChatMessage>) {
        self.messages = messages;
    }

    // Set the currently active session ID
    pub fn set_session_id(&mut self, session_id: uuid::Uuid) {
        self.session_id = session_id;
    }

    // Set the provider name loaded from persistent configuration.
    pub fn set_provider_name(&mut self, provider_name: String) {
        self.provider_name = provider_name;
    }

    // Render the slash command completion popup with bordered list.
    fn render_slash_popup(&self, state: &SlashCompletionState) -> Paragraph<'_> {
        let max_visible = 8usize;
        let total = state.filtered.len();
        let visible = total.min(max_visible);

        // Scroll window to keep selected item visible
        let start = if total <= max_visible {
            0
        } else {
            state
                .selected
                .saturating_sub(max_visible / 2)
                .min(total - max_visible)
        };

        let mut lines: Vec<Line> = Vec::new();
        for i in start..start + visible {
            let cmd = &state.filtered[i];
            let label = if i == state.selected {
                Span::styled(
                    format!("> {}  {}", cmd.name, cmd.description),
                    Style::default().fg(Color::Black).bg(Color::White),
                )
            } else {
                Span::styled(
                    format!("  {}  {}", cmd.name, cmd.description),
                    Style::default().fg(Color::White),
                )
            };
            lines.push(Line::from(label));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Commands ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the permission prompt dialog in place of the input line.
    fn render_permission_dialog(&self, prompt: &PermissionPromptState) -> Paragraph<'_> {
        let risk_color = match prompt.risk_level.as_str() {
            "Critical" => Color::Red,
            "High" => Color::LightRed,
            "Medium" => Color::Yellow,
            _ => Color::White,
        };
        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled(
            format!("  Permission Required — {}", prompt.tool_name),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!("  {}", prompt.args_summary),
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::styled(
            format!("  Reason: {}", prompt.reason),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::styled(
            format!("  Risk:   {}", prompt.risk_level),
            Style::default().fg(risk_color),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  [y] Allow once    [Y] Always allow    [n] Deny once    [N] Always deny",
            Style::default().fg(Color::Cyan),
        )));
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Permission ")
            .border_style(Style::default().fg(Color::Yellow));
        Paragraph::new(lines).block(block)
    }
}

pub use input::InputAction;

#[cfg(test)]
mod tests {
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
}
