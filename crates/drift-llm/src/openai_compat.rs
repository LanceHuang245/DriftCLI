use crate::types::*;
use crate::LlmProvider;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

pub struct OpenAiCompatibleProvider {
    api_key: String,
    model: String,
    base_url: String,
    client: Client,
}

impl OpenAiCompatibleProvider {
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
impl LlmProvider for OpenAiCompatibleProvider {
    fn provider_id(&self) -> &str {
        "openai_compatible"
    }
    fn model_name(&self) -> &str {
        &self.model
    }
    fn context_window(&self) -> usize {
        128_000
    }

    async fn stream_chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_output_tokens: Option<usize>,
    ) -> Result<LlmResponseStream, LlmError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut api_messages: Vec<serde_json::Value> = Vec::new();

        if let Some(sp) = system_prompt {
            api_messages.push(serde_json::json!({
                "role": "system",
                "content": sp,
            }));
        }

        for m in &messages {
            let role = match m.role.as_str() {
                "assistant" => "assistant",
                _ => "user",
            };
            api_messages.push(serde_json::json!({
                "role": role,
                "content": m.content,
            }));
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": api_messages,
            "stream": true,
        });

        if let Some(t) = temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(mt) = max_output_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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

        use futures::StreamExt;

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
                if let Ok(event) = serde_json::from_str::<OpenAiStreamEvent>(data) {
                    for choice in &event.choices {
                        if let Some(delta) = &choice.delta {
                            if let Some(content) = &delta.content {
                                return Some(Ok(LlmChunk::TextDelta(content.clone())));
                            }
                        }
                        if let Some(reason) = &choice.finish_reason {
                            if reason == "stop" {
                                return Some(Ok(LlmChunk::Done));
                            }
                        }
                    }
                }
            }
            None
        });

        Ok(LlmResponseStream::new(event_stream))
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamEvent {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: Option<OpenAiDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
    #[allow(dead_code)]
    role: Option<String>,
}
