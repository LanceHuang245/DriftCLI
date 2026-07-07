use drift_config::LlmConfig;
use drift_llm::ModelInfo;
use serde::{Deserialize, Serialize};

// EventMsg: typed events emitted by the Agent core and consumed by the TUI event bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventMsg {
    // A single text token from the streaming LLM response, forwarded to TUI.
    Token(String),
    // Reasoning/thinking text chunk (for extended-thinking models), streamed before the final answer.
    Reasoning(String),
    // Signal that the reasoning/thinking phase has completed, carrying the elapsed wall-clock time (ms).
    ReasoningComplete {
        duration_ms: u64,
    },
    // Token usage stats (input + output) reported after a turn completes.
    TokenUsage {
        input: usize,
        output: usize,
    },
    // Agent state transition notification (idle → thinking → generating → error).
    AgentState(AgentState),
    // Context compaction has started — TUI may display a "compacting" indicator.
    ContextCompacting,
    // Context compaction finished with a summary and the number of tokens saved.
    ContextCompacted {
        summary: String,
        saved_tokens: usize,
    },
    // A recoverable or non-recoverable error occurred during processing.
    Error {
        message: String,
        recoverable: bool,
    },
    // Agent finished processing for the current turn; TUI can accept new input.
    Done,
    // Retrieved list of available model IDs from the provider's API.
    ModelList(Vec<ModelInfo>),
    // List of configured provider names for the /provider picker.
    ProviderList(Vec<String>),
    // Full configuration for a specific provider, by name.
    ProviderConfig {
        name: String,
        config: LlmConfig,
    },
    // Notification that the provider was switched, carrying the new provider name and model.
    ProviderSwitched {
        name: String,
        model: String,
    },
    // A tool call has been requested by the LLM — emitted with the call ID and tool name.
    ToolCallStart {
        id: String,
        name: String,
    },
    // Streaming argument JSON fragment for a tool call in progress.
    ToolCallArgs {
        id: String,
        delta: String,
    },
    // A tool call has been fully received — arguments are complete.
    ToolCallEnd {
        id: String,
    },
    // Tool execution has begun — the tool name and args summary.
    ToolExecStart {
        id: String,
        name: String,
    },
    // Tool execution has produced a result — content, success flag, and error if any.
    ToolExecEnd {
        id: String,
        name: String,
        success: bool,
        error: Option<String>,
    },
    // Permission request sent when the PermissionEngine decides to Ask the user.
    // The TUI should display a dialog and send a PermissionResponse back via the mpsc channel.
    PermissionRequest {
        request_id: String,
        tool_name: String,
        args_summary: String,
        reason: String,
        risk_level: String,
    },
    // Permission granted/denied by the user (notified after the decision, for auditing).
    PermissionResolved {
        request_id: String,
        allowed: bool,
    },
    // Sent to the TUI to broadcast all historical session metadata.
    SessionList(Vec<drift_storage::SessionMeta>),
    // Sent to the TUI when a specific session has been reconstructed.
    SessionLoaded {
        session_id: uuid::Uuid,
        events: Vec<drift_storage::SessionEvent>,
    },
}

// AgentState: lifecycle states the agent transitions through during a processing turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentState {
    // Waiting for the next user input; ready to accept commands.
    Idle,
    // LLM is thinking — the request has been sent but no tokens have been returned yet.
    Thinking,
    // Streaming a text response; the inner string accumulates generated content.
    Generating(String),
    // Agent entered an error state with a human-readable description.
    Error(String),
}
