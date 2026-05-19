use pin_project_lite::pin_project;
use std::pin::Pin;
use tokio_stream::Stream;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct LlmMessage {
    pub role: String,
    pub content: serde_json::Value,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<ToolCallInfo>>,
}

impl LlmMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: serde_json::Value::String(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: serde_json::Value::String(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }
    pub fn tool_result(tool_call_id: String, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: serde_json::Value::String(content.into()),
            tool_call_id: Some(tool_call_id),
            tool_calls: None,
        }
    }
    pub fn assistant_with_tools(text: impl Into<String>, tool_calls: Vec<ToolCallInfo>) -> Self {
        let text = text.into();
        Self {
            role: "assistant".into(),
            content: if text.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(text)
            },
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        }
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
