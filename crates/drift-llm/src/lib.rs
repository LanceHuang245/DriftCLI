pub mod anthropic;
pub mod openai_compat;
pub mod types;

use async_trait::async_trait;
pub use types::*;

use bytes::Bytes;
use futures::ready;
use reqwest::Response;
use serde::Deserialize;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;

struct SseLineStream {
    byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buffer: Vec<u8>,
}

impl SseLineStream {
    fn new(response: Response) -> Self {
        Self {
            byte_stream: Box::pin(response.bytes_stream()),
            buffer: Vec::new(),
        }
    }
}

impl Stream for SseLineStream {
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = self.buffer[..pos].to_vec();
                self.buffer.drain(..pos + 1);
                if let Ok(text) = String::from_utf8(line_bytes) {
                    let trimmed = text.trim_end_matches('\r');
                    if !trimmed.is_empty() {
                        return Poll::Ready(Some(trimmed.to_string()));
                    }
                    continue;
                }
                continue;
            }

            match ready!(self.byte_stream.as_mut().poll_next(cx)) {
                Some(Ok(bytes)) => self.buffer.extend_from_slice(&bytes),
                Some(Err(_)) => return Poll::Ready(None),
                None => {
                    if !self.buffer.is_empty() {
                        let remaining = std::mem::take(&mut self.buffer);
                        return if let Ok(text) = String::from_utf8(remaining) {
                            let trimmed = text.trim_end_matches('\r');
                            if !trimmed.is_empty() {
                                Poll::Ready(Some(trimmed.to_string()))
                            } else {
                                Poll::Ready(None)
                            }
                        } else {
                            Poll::Ready(None)
                        };
                    }
                    return Poll::Ready(None);
                }
            }
        }
    }
}

pub fn sse_text_stream(response: Response) -> impl Stream<Item = String> {
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
    ) -> Result<LlmResponseStream, LlmError>;

    async fn chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
    ) -> Result<String, LlmError> {
        let mut stream = self
            .stream_chat(messages, system_prompt, None, None)
            .await?;
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(LlmChunk::TextDelta(t)) => text.push_str(&t),
                Ok(LlmChunk::ReasoningDelta(_)) => {}
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
        } => Ok(Box::new(anthropic::AnthropicProvider::new(
            api_key.clone(),
            model.clone(),
            base_url.clone(),
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

#[derive(Deserialize)]
struct RemoteModelEntry {
    id: String,
}

pub async fn fetch_anthropic_models(api_key: &str, base_url: &str) -> Result<Vec<String>, LlmError> {
    let client = reqwest::Client::new();
    let url = format!("{}/models", base_url.trim_end_matches('/'));

    let response = client
        .get(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return if status.as_u16() == 401 {
            Err(LlmError::Unauthorized)
        } else {
            Err(LlmError::Api {
                status: status.as_u16(),
                message: error_text,
            })
        };
    }

    #[derive(Deserialize)]
    struct ModelsResponse {
        data: Vec<RemoteModelEntry>,
    }

    let models: ModelsResponse = response.json().await?;
    Ok(models.data.into_iter().map(|m| m.id).collect())
}

pub async fn fetch_openai_compat_models(api_key: &str, base_url: &str) -> Result<Vec<String>, LlmError> {
    let client = reqwest::Client::new();
    let url = format!("{}/models", base_url.trim_end_matches('/'));

    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return if status.as_u16() == 401 {
            Err(LlmError::Unauthorized)
        } else {
            Err(LlmError::Api {
                status: status.as_u16(),
                message: error_text,
            })
        };
    }

    #[derive(Deserialize)]
    struct ModelsResponse {
        data: Vec<RemoteModelEntry>,
    }

    let models: ModelsResponse = response.json().await?;
    Ok(models.data.into_iter().map(|m| m.id).collect())
}
