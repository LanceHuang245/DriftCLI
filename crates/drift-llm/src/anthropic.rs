use crate::sse_text_stream;
use crate::types::*;
use crate::LlmProvider;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    base_url: String,
    reasoning_effort: Option<String>,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String, base_url: String, reasoning_effort: Option<String>) -> Self {
        Self {
            api_key,
            model,
            base_url,
            reasoning_effort,
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
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<LlmResponseStream, LlmError> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let anthropic_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role.as_str() {
                    "assistant" => "assistant",
                    _ => "user",
                };
                let content_blocks: Vec<serde_json::Value> = m
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text(t) => {
                            Some(serde_json::json!({"type": "text", "text": t}))
                        }
                        ContentPart::ToolCall { id, name, arguments } => {
                            let input = serde_json::from_str::<serde_json::Value>(arguments)
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                            Some(serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            }))
                        }
                        ContentPart::ToolResult { tool_use_id, content, is_error } => {
                            Some(serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                                "is_error": is_error,
                            }))
                        }
                        ContentPart::Reasoning(r) => {
                            Some(serde_json::json!({"type": "thinking", "thinking": r}))
                        }
                    })
                    .collect();
                serde_json::json!({
                    "role": role,
                    "content": content_blocks,
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
        if let Some(ref effort) = self.reasoning_effort {
            body["reasoning_effort"] = serde_json::Value::String(effort.clone());
        }
        if let Some(ref tool_list) = tools {
            tracing::info!(count = tool_list.len(), "anthropic: adding tools to request");
            body["tools"] = serde_json::json!(tool_list);
        } else {
            tracing::info!("anthropic: no tools in request");
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

        let line_stream = sse_text_stream(response);

        let event_stream = line_stream.filter_map(|line| {
            // Track which block indices correspond to tool_use blocks and map index → actual tool_use id.
            let mut tool_block_indices: HashMap<usize, String> = HashMap::new();
            async move {
                if let Some(data) = line.strip_prefix("data: ") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        return Some(Ok(LlmChunk::Done));
                    }
                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                        match event.event_type.as_str() {
                            "content_block_start" => {
                                if let Some(ref block) = event.content_block {
                                    if block.block_type == "tool_use" {
                                        let actual_id = block.id.clone().unwrap_or_default();
                                        let name = block.name.clone().unwrap_or_default();
                                        if actual_id.is_empty() || name.is_empty() {
                                            return None;
                                        }
                                        if let Some(index) = event.index {
                                            tool_block_indices.insert(index, actual_id.clone());
                                        }
                                        return Some(Ok(LlmChunk::ToolCallStart {
                                            id: actual_id,
                                            name,
                                        }));
                                    }
                                }
                            }
                            "content_block_delta" => {
                                if let Some(delta) = &event.delta {
                                    if delta.delta_type == "input_json_delta" {
                                        if let Some(ref partial) = delta.partial_json {
                                            let id = event
                                                .index
                                                .and_then(|i| tool_block_indices.get(&i).cloned())
                                                .unwrap_or_default();
                                            return Some(Ok(LlmChunk::ToolCallArgs {
                                                id,
                                                delta: partial.clone(),
                                            }));
                                        }
                                    } else if let Some(thinking) = &delta.thinking {
                                        return Some(Ok(LlmChunk::ReasoningDelta(
                                            thinking.clone(),
                                        )));
                                    } else if let Some(text) = &delta.text {
                                        return Some(Ok(LlmChunk::TextDelta(text.clone())));
                                    }
                                }
                            }
                            "content_block_stop" => {
                                if let Some(index) = event.index {
                                    if let Some(actual_id) =
                                        tool_block_indices.remove(&index)
                                    {
                                        return Some(Ok(LlmChunk::ToolCallEnd {
                                            id: actual_id,
                                        }));
                                    }
                                }
                            }
                            "message_stop" => return Some(Ok(LlmChunk::Done)),
                            _ => {}
                        }
                    }
                }
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
    #[serde(default)]
    index: Option<usize>,
    delta: Option<AnthropicDelta>,
    content_block: Option<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct AnthropicDelta {
    #[serde(rename = "type", default)]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
}
