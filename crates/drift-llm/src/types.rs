use pin_project_lite::pin_project;
use std::pin::Pin;
use tokio_stream::Stream;

#[derive(Debug, Clone)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

impl LlmMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LlmChunk {
    TextDelta(String),
    ReasoningDelta(String),
    Done,
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
