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
        tools: Option<Vec<serde_json::Value>>,
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
                "tool" => "tool",
                _ => "user",
            };
            let mut msg = serde_json::json!({
                "role": role,
                "content": m.content,
            });
            if let Some(ref tci) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::json!(tci);
            }
            api_messages.push(msg);
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
        if let Some(ref tool_list) = tools {
            let openai_tools: Vec<serde_json::Value> = tool_list
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["input_schema"],
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(openai_tools);
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

        let line_stream = crate::sse_text_stream(response);

        let event_stream = line_stream.filter_map(|line| async move {
            if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if data == "[DONE]" {
                    return Some(Ok(LlmChunk::Done));
                }
                if let Ok(event) = serde_json::from_str::<OpenAiStreamEvent>(data) {
                    for choice in &event.choices {
                        if let Some(delta) = &choice.delta {
                            if let Some(ref tool_calls) = delta.tool_calls {
                                for tc in tool_calls {
                                    if let Some(ref name) = tc.function.name {
                                        if !name.is_empty() {
                                            return Some(Ok(LlmChunk::ToolCallStart {
                                                id: tc.id.clone().unwrap_or_default(),
                                                name: name.clone(),
                                            }));
                                        }
                                    }
                                    if let Some(ref args) = tc.function.arguments {
                                        if !args.is_empty() {
                                            return Some(Ok(LlmChunk::ToolCallArgs {
                                                id: tc.id.clone().unwrap_or_default(),
                                                delta: args.clone(),
                                            }));
                                        }
                                    }
                                }
                            }
                            if let Some(content) = &delta.content {
                                return Some(Ok(LlmChunk::TextDelta(content.clone())));
                            }
                        }
                        if let Some(reason) = &choice.finish_reason {
                            if reason == "stop" {
                                return Some(Ok(LlmChunk::Done));
                            }
                            if reason == "tool_calls" {
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
    #[serde(default)]
    content: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: OpenAiToolCallFunction,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
