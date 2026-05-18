use drift_llm::ModelInfo;
use serde::{Deserialize, Serialize};

/// Events emitted by the Agent core and consumed by TUI/storage/telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventMsg {
    /// A text token in the streaming response
    Token(String),
    /// Reasoning/thinking text (for thinking models)
    Reasoning(String),
    /// Token usage stats for this turn
    TokenUsage { input: usize, output: usize },
    /// Agent state change
    AgentState(AgentState),
    /// Context is being compacted
    ContextCompacting,
    /// Context compaction complete
    ContextCompacted { summary: String, saved_tokens: usize },
    /// Error occurred
    Error { message: String, recoverable: bool },
    /// Agent processing complete for this turn
    Done,
    /// Model list fetched from provider
    ModelList(Vec<ModelInfo>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentState {
    /// Waiting for user input
    Idle,
    /// LLM is thinking (before any text or tool call)
    Thinking,
    /// Streaming text response
    Generating(String),
    /// Error state
    Error(String),
}
