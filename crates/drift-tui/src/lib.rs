mod components;
mod connect;
mod input;
mod markdown;
mod runtime;
mod selection;
mod slash;
mod update;
mod view;

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
    // Tool call started; only its name is needed for the transient status bar.
    ToolCallStart {
        name: String,
    },
    // Tool execution started
    ToolExecStart {
        name: String,
    },
    // Tool execution finished; the next model pass may begin.
    ToolExecEnd,
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
        risk_level: String,
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
    reasoning_start_time: Option<Instant>,
    // Duration accumulated before a reasoning-only message resumes streaming.
    current_reasoning_duration_ms: u64,
    current_reasoning_collapsed: bool,
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
    pub reasoning_duration_ms: Option<u64>,
    pub reasoning_collapsed: bool,
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
            reasoning_start_time: None,
            current_reasoning_duration_ms: 0,
            current_reasoning_collapsed: true,
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
}

pub use input::InputAction;

#[cfg(test)]
#[path = "tui_tests.rs"]
mod tests;
