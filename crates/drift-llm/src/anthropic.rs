use crate::types::*;
use crate::LlmProvider;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    base_url: String,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn provider_id(&self) -> &str {
        "anthropic"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
    fn context_window(&self) -> usize {
        200_000
    }

    async fn stream_chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_output_tokens: Option<usize>,
    ) -> Result<LlmResponseStream, LlmError> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let anthropic_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role.as_str() {
                    "assistant" => "assistant",
                    _ => "user",
                };
                serde_json::json!({
                    "role": role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_output_tokens.unwrap_or(4096),
            "messages": anthropic_messages,
            "stream": true,
        });

        if let Some(sp) = system_prompt {
            body["system"] = serde_json::Value::String(sp);
        }
        if let Some(t) = temperature {
            body["temperature"] = serde_json::json!(t);
        }

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
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

        let byte_stream = response.bytes_stream();
        let line_stream = byte_stream.filter_map(|result| async move {
            match result {
                Ok(bytes) => Some(String::from_utf8_lossy(&bytes).to_string()),
                Err(_) => None,
            }
        });

        let event_stream = line_stream.filter_map(|line| async move {
            if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if data == "[DONE]" {
                    return Some(Ok(LlmChunk::Done));
                }
                match serde_json::from_str::<AnthropicStreamEvent>(data) {
                    Ok(event) => match event.event_type.as_str() {
                        "content_block_delta" => {
                            if let Some(delta) = &event.delta {
                                if let Some(text) = &delta.text {
                                    Some(Ok(LlmChunk::TextDelta(text.clone())))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        "message_stop" => Some(Ok(LlmChunk::Done)),
                        _ => None,
                    },
                    Err(_) => None,
                }
            } else {
                None
            }
        });

        Ok(LlmResponseStream::new(event_stream))
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicDelta>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    text: Option<String>,
}
