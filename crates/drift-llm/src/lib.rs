pub mod anthropic;
pub mod openai_compat;
pub mod types;

use async_trait::async_trait;
pub use types::*;

use bytes::Bytes;
use futures::{StreamExt, ready};
use reqwest::Response;
use serde::Deserialize;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;

struct SseLineStream {
    byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, LlmError>> + Send>>,
    buffer: Vec<u8>,
    failed: bool,
}

const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;

impl SseLineStream {
    fn new(response: Response) -> Self {
        Self::from_stream(
            response
                .bytes_stream()
                .map(|chunk| chunk.map_err(LlmError::Http)),
        )
    }

    fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<Bytes, LlmError>> + Send + 'static,
    {
        Self {
            byte_stream: Box::pin(stream),
            buffer: Vec::new(),
            failed: false,
        }
    }
}

impl Stream for SseLineStream {
    type Item = Result<String, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.failed {
            return Poll::Ready(None);
        }

        loop {
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                if pos > MAX_SSE_LINE_BYTES {
                    self.buffer.clear();
                    self.failed = true;
                    return Poll::Ready(Some(Err(LlmError::Stream(format!(
                        "SSE line exceeds {MAX_SSE_LINE_BYTES} bytes"
                    )))));
                }
                let line_bytes: Vec<u8> = self.buffer[..pos].to_vec();
                self.buffer.drain(..pos + 1);
                let text = match String::from_utf8(line_bytes) {
                    Ok(text) => text,
                    Err(error) => {
                        self.failed = true;
                        return Poll::Ready(Some(Err(LlmError::Stream(format!(
                            "invalid UTF-8 in SSE stream: {error}"
                        )))));
                    }
                };
                let trimmed = text.trim_end_matches('\r');
                if !trimmed.is_empty() {
                    return Poll::Ready(Some(Ok(trimmed.to_string())));
                }
                continue;
            }

            if self.buffer.len() > MAX_SSE_LINE_BYTES {
                self.buffer.clear();
                self.failed = true;
                return Poll::Ready(Some(Err(LlmError::Stream(format!(
                    "SSE line exceeds {MAX_SSE_LINE_BYTES} bytes"
                )))));
            }

            match ready!(self.byte_stream.as_mut().poll_next(cx)) {
                Some(Ok(bytes)) => self.buffer.extend_from_slice(&bytes),
                Some(Err(error)) => {
                    self.failed = true;
                    return Poll::Ready(Some(Err(error)));
                }
                None => {
                    if !self.buffer.is_empty() {
                        let remaining = std::mem::take(&mut self.buffer);
                        let text = match String::from_utf8(remaining) {
                            Ok(text) => text,
                            Err(error) => {
                                self.failed = true;
                                return Poll::Ready(Some(Err(LlmError::Stream(format!(
                                    "invalid UTF-8 in SSE stream: {error}"
                                )))));
                            }
                        };
                        let trimmed = text.trim_end_matches('\r');
                        return if trimmed.is_empty() {
                            Poll::Ready(None)
                        } else {
                            Poll::Ready(Some(Ok(trimmed.to_string())))
                        };
                    }
                    return Poll::Ready(None);
                }
            }
        }
    }
}

pub fn sse_text_stream(response: Response) -> impl Stream<Item = Result<String, LlmError>> {
    SseLineStream::new(response)
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn provider_id(&self) -> &str;
    fn model_name(&self) -> &str;
    fn context_window(&self) -> usize;

    async fn stream_chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_output_tokens: Option<usize>,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<LlmResponseStream, LlmError>;

    async fn chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
    ) -> Result<String, LlmError> {
        let mut stream = self
            .stream_chat(messages, system_prompt, None, None, None)
            .await?;
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(LlmChunk::TextDelta(t)) => text.push_str(&t),
                Ok(LlmChunk::ReasoningDelta(_)) => {}
                Ok(LlmChunk::ToolCallStart { .. }) => {}
                Ok(LlmChunk::ToolCallArgs { .. }) => {}
                Ok(LlmChunk::ToolCallEnd { .. }) => {}
                Ok(LlmChunk::Done) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(text)
    }
}

pub fn create_provider(config: &drift_config::LlmConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    match config {
        drift_config::LlmConfig::Anthropic {
            api_key,
            model,
            base_url,
            reasoning_effort,
        } => Ok(Box::new(anthropic::AnthropicProvider::new(
            api_key.clone(),
            model.clone(),
            base_url.clone(),
            reasoning_effort.clone(),
        ))),
        drift_config::LlmConfig::OpenAiCompatible {
            api_key,
            model,
            base_url,
            ..
        } => Ok(Box::new(openai_compat::OpenAiCompatibleProvider::new(
            api_key.clone(),
            model.clone(),
            base_url.clone(),
        ))),
    }
}

// ---------- Model fetching ----------

#[derive(Deserialize)]
struct SimpleModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct AnthropicModelEntry {
    id: String,
    #[serde(default)]
    capabilities: Option<AnthropicCapabilities>,
}

#[derive(Deserialize)]
struct AnthropicCapabilities {
    #[serde(default)]
    effort: Option<AnthropicEffort>,
}

#[derive(Deserialize)]
struct AnthropicEffort {
    #[serde(default)]
    supported: bool,
    #[serde(default)]
    low: CapabilitySupport,
    #[serde(default)]
    medium: CapabilitySupport,
    #[serde(default)]
    high: CapabilitySupport,
    #[serde(default)]
    max: CapabilitySupport,
}

#[derive(Deserialize, Default)]
struct CapabilitySupport {
    #[serde(default)]
    supported: bool,
}

/// Try fetching models from an Anthropic or Anthropic-compatible endpoint.
/// Tries multiple URLs and auth approaches to handle various proxy setups
/// (e.g. https://api.deepseek.com/anthropic where /models is only at the root).
pub async fn fetch_anthropic_models(
    api_key: &str,
    base_url: &str,
) -> Result<Vec<ModelInfo>, LlmError> {
    let base = base_url.trim_end_matches('/').to_string();

    // Build fallback URLs:
    //   1. {base}/models            (primary)
    //   2. {base}/v1/models        (some proxies nest under /v1)
    //   3. {parent}/models         (strip /anthropic suffix)
    let mut urls = vec![format!("{}/models", base), format!("{}/v1/models", base)];
    if let Some(slash) = base.rfind('/') {
        let parent = &base[..slash];
        urls.push(format!("{}/models", parent));
        urls.push(format!("{}/v1/models", parent));
    }

    let client = reqwest::Client::new();

    for url in &urls {
        let response = client
            .get(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            continue;
        }

        let body_text = match response.text().await {
            Ok(t) => t,
            Err(_) => continue,
        };

        if let Ok(models) =
            parse_anthropic_list(&body_text).or_else(|_| parse_simple_list(&body_text))
        {
            if !models.is_empty() {
                return Ok(models);
            }
        }
    }

    Err(LlmError::Config(format!(
        "No model list found. Tried: {}",
        urls.join(", ")
    )))
}

fn parse_anthropic_list(body: &str) -> Result<Vec<ModelInfo>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Wrapper {
        data: Vec<AnthropicModelEntry>,
    }
    let resp: Wrapper = serde_json::from_str(body)?;
    Ok(resp
        .data
        .into_iter()
        .map(|m| {
            let effort_levels = m
                .capabilities
                .and_then(|c| c.effort)
                .filter(|e| e.supported)
                .map(|e| {
                    let mut levels = Vec::new();
                    if e.low.supported {
                        levels.push("low".into());
                    }
                    if e.medium.supported {
                        levels.push("medium".into());
                    }
                    if e.high.supported {
                        levels.push("high".into());
                    }
                    if e.max.supported {
                        levels.push("max".into());
                    }
                    levels
                })
                .unwrap_or_default();
            ModelInfo {
                id: m.id,
                effort_levels,
            }
        })
        .collect())
}

fn parse_simple_list(body: &str) -> Result<Vec<ModelInfo>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Wrapper {
        data: Vec<SimpleModelEntry>,
    }
    let resp: Wrapper = serde_json::from_str(body)?;
    Ok(resp
        .data
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            effort_levels: vec![],
        })
        .collect())
}

pub async fn fetch_openai_compat_models(
    api_key: &str,
    base_url: &str,
) -> Result<Vec<String>, LlmError> {
    let base = base_url.trim_end_matches('/').to_string();
    let mut urls = vec![format!("{}/models", base), format!("{}/v1/models", base)];
    if let Some(slash) = base.rfind('/') {
        let parent = &base[..slash];
        urls.push(format!("{}/models", parent));
        urls.push(format!("{}/v1/models", parent));
    }

    let client = reqwest::Client::new();

    for url in &urls {
        let response = client
            .get(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            continue;
        }

        let body_text = match response.text().await {
            Ok(t) => t,
            Err(_) => continue,
        };

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<SimpleModelEntry>,
        }

        if let Ok(models) = serde_json::from_str::<ModelsResponse>(&body_text) {
            let ids: Vec<String> = models.data.into_iter().map(|m| m.id).collect();
            if !ids.is_empty() {
                return Ok(ids);
            }
        }
    }

    Err(LlmError::Config(format!(
        "No model list found. Tried: {}",
        urls.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::{LlmError, MAX_SSE_LINE_BYTES, SseLineStream};
    use bytes::Bytes;
    use futures::StreamExt;

    #[tokio::test]
    async fn sse_stream_propagates_transport_errors() {
        // Transport failures must remain distinguishable from a clean EOF.
        let source = futures::stream::iter(vec![Err(LlmError::Stream("transport".into()))]);
        let mut stream = SseLineStream::from_stream(source);

        let error = stream.next().await.unwrap().unwrap_err();

        assert!(matches!(error, LlmError::Stream(message) if message == "transport"));
    }

    #[tokio::test]
    async fn sse_stream_rejects_invalid_utf8() {
        // Invalid event bytes must be reported instead of silently discarded.
        let source = futures::stream::iter(vec![Ok(Bytes::from_static(b"\xff\n"))]);
        let mut stream = SseLineStream::from_stream(source);

        let error = stream.next().await.unwrap().unwrap_err();

        assert!(matches!(error, LlmError::Stream(message) if message.contains("invalid UTF-8")));
    }

    #[tokio::test]
    async fn sse_stream_rejects_oversized_lines() {
        // A line cap prevents an unbounded SSE buffer from exhausting memory.
        let bytes = Bytes::from(vec![b'x'; MAX_SSE_LINE_BYTES + 1]);
        let source = futures::stream::iter(vec![Ok(bytes)]);
        let mut stream = SseLineStream::from_stream(source);

        let error = stream.next().await.unwrap().unwrap_err();

        assert!(matches!(error, LlmError::Stream(message) if message.contains("exceeds")));
    }
}
