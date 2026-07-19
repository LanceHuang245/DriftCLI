use crate::LlmProvider;
use crate::types::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

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
            // Emit tool results as role:"tool" messages first (agent sends them
            // as role:"user" with ContentPart::ToolResult for Anthropic compat).
            for part in &m.content {
                if let ContentPart::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = part
                {
                    api_messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content,
                    }));
                }
            }

            // Build a user/assistant message from non-tool-result parts.
            let has_msg_parts = m
                .content
                .iter()
                .any(|p| !matches!(p, ContentPart::ToolResult { .. }));
            if !has_msg_parts {
                continue;
            }

            let role = if m.role.as_str() == "assistant" {
                "assistant"
            } else {
                "user"
            };
            let text = extract_text(&m.content);
            let reasoning = extract_reasoning(&m.content);
            let tc_list: Vec<ContentPart> = m
                .content
                .iter()
                .filter(|p| matches!(p, ContentPart::ToolCall { .. }))
                .cloned()
                .collect();

            let mut msg = serde_json::json!({
                "role": role,
                "content": text,
            });
            if let Some(r) = reasoning {
                msg["reasoning_content"] = serde_json::json!(r);
            }
            if !tc_list.is_empty() {
                let tc_json: Vec<serde_json::Value> = tc_list
                    .iter()
                    .map(|tc| {
                        if let ContentPart::ToolCall {
                            id,
                            name,
                            arguments,
                        } = tc
                        {
                            serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": { "name": name, "arguments": arguments }
                            })
                        } else {
                            unreachable!()
                        }
                    })
                    .collect();
                msg["tool_calls"] = serde_json::json!(tc_json);
                // Some models reject empty-string content when tool_calls are present.
                // Omit the field to let the API-side default handle it.
                if text.is_empty() {
                    msg.as_object_mut().unwrap().remove("content");
                }
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
        // OpenAI identifies each streamed tool call by index after omitting its ID.
        let tool_call_ids = Arc::new(Mutex::new(Vec::<Option<String>>::new()));

        let event_stream = line_stream
            .filter_map({
                let tool_call_ids = Arc::clone(&tool_call_ids);
                move |line| {
                    let tool_call_ids = Arc::clone(&tool_call_ids);
                    async move {
                        let line = match line {
                            Ok(line) => line,
                            Err(error) => return Some(vec![Err(error)]),
                        };
                        if let Some(data) = line.strip_prefix("data: ") {
                            let data = data.trim();
                            if data == "[DONE]" {
                                return Some(vec![Ok(LlmChunk::Done)]);
                            }
                            let event = match serde_json::from_str::<OpenAiStreamEvent>(data) {
                                Ok(event) => event,
                                Err(error) => {
                                    return Some(vec![Err(LlmError::Stream(format!(
                                        "invalid OpenAI SSE event: {error}"
                                    )))]);
                                }
                            };
                            let mut chunks: Vec<Result<LlmChunk, LlmError>> = Vec::new();
                            for choice in &event.choices {
                                if let Some(delta) = &choice.delta {
                                    if let Some(ref tool_calls) = delta.tool_calls {
                                        for tc in tool_calls {
                                            chunks.extend(openai_tool_call_chunks(
                                                tc,
                                                &mut tool_call_ids.lock().unwrap(),
                                            ));
                                        }
                                    }
                                    if let Some(content) = &delta.content {
                                        chunks.push(Ok(LlmChunk::TextDelta(content.clone())));
                                    }
                                    if let Some(reasoning) = &delta.reasoning_content {
                                        if !reasoning.is_empty() {
                                            chunks.push(Ok(LlmChunk::ReasoningDelta(
                                                reasoning.clone(),
                                            )));
                                        }
                                    }
                                }
                                if let Some(reason) = &choice.finish_reason {
                                    if reason == "stop" || reason == "tool_calls" {
                                        chunks.push(Ok(LlmChunk::Done));
                                    }
                                }
                            }
                            if !chunks.is_empty() {
                                return Some(chunks);
                            }
                        }
                        None
                    }
                }
            })
            .flat_map(|chunks| futures::stream::iter(chunks));

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
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    #[serde(default)]
    index: usize,
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
    arguments: Option<serde_json::Value>,
}

/// Normalize the arguments value to a plain JSON string.
/// DeepSeek (and other providers) sometimes send `arguments` as a JSON
/// object instead of the spec-mandated JSON-encoded string.
fn normalize_args(args: &serde_json::Value) -> String {
    match args {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Correlate one OpenAI tool-call delta with the ID stored for its stream index.
fn openai_tool_call_chunks(
    tool_call: &OpenAiToolCall,
    ids: &mut Vec<Option<String>>,
) -> Vec<Result<LlmChunk, LlmError>> {
    if ids.len() <= tool_call.index {
        ids.resize(tool_call.index + 1, None);
    }
    if let Some(id) = tool_call.id.as_ref().filter(|id| !id.is_empty()) {
        ids[tool_call.index] = Some(id.clone());
    }

    let id = ids[tool_call.index].clone().unwrap_or_default();
    let mut chunks = Vec::new();
    if let Some(name) = tool_call
        .function
        .name
        .as_ref()
        .filter(|name| !name.is_empty())
    {
        if id.is_empty() {
            return vec![Err(LlmError::Stream(format!(
                "tool call at index {} is missing an ID",
                tool_call.index
            )))];
        }
        chunks.push(Ok(LlmChunk::ToolCallStart {
            id: id.clone(),
            name: name.clone(),
        }));
    }

    if let Some(arguments) = &tool_call.function.arguments {
        let delta = normalize_args(arguments);
        if !delta.is_empty() {
            if id.is_empty() {
                return vec![Err(LlmError::Stream(format!(
                    "tool-call arguments at index {} have no known ID",
                    tool_call.index
                )))];
            }
            chunks.push(Ok(LlmChunk::ToolCallArgs { id, delta }));
        }
    }

    chunks
}

#[cfg(test)]
#[path = "openai_compat_tests.rs"]
mod tests;
