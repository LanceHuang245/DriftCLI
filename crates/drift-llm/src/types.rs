use pin_project_lite::pin_project;
use std::pin::Pin;
use tokio_stream::Stream;

/// Provider-agnostic content part — the unified message content format.
/// Each provider converts these to/from its native wire format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ContentPart {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    Reasoning(String),
}

#[derive(Debug, Clone)]
pub struct LlmMessage {
    pub role: String,
    pub content: Vec<ContentPart>,
}

impl LlmMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: vec![ContentPart::Text(text.into())],
        }
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: vec![ContentPart::Text(text.into())],
        }
    }
}

/// Convenience: extract all Text parts as a single string.
pub fn extract_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| {
            if let ContentPart::Text(t) = p {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convenience: extract all Reasoning parts as a single string.
pub fn extract_reasoning(parts: &[ContentPart]) -> Option<String> {
    let r: String = parts
        .iter()
        .filter_map(|p| {
            if let ContentPart::Reasoning(t) = p {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect();
    if r.is_empty() {
        None
    } else {
        Some(r)
    }
}

#[derive(Debug, Clone)]
pub enum LlmChunk {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStart { id: String, name: String },
    ToolCallArgs { id: String, delta: String },
    ToolCallEnd { id: String },
    Done,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub effort_levels: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("SSE stream error: {0}")]
    Stream(String),
    #[error("API error: {status} — {message}")]
    Api { status: u16, message: String },
    #[error("Invalid API key or authentication failed")]
    Unauthorized,
    #[error("Configuration error: {0}")]
    Config(String),
}

pin_project! {
    pub struct LlmResponseStream {
        #[pin]
        pub inner: Pin<Box<dyn Stream<Item = Result<LlmChunk, LlmError>> + Send>>,
    }
}

impl LlmResponseStream {
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<LlmChunk, LlmError>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    pub async fn next(&mut self) -> Option<Result<LlmChunk, LlmError>> {
        use futures::StreamExt;
        self.inner.as_mut().next().await
    }
}
