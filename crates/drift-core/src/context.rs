use drift_llm::{ContentPart, LlmMessage};
use drift_tools::ToolDefinition;
use serde_json::Value;

const CHARS_PER_TOKEN: usize = 4;
const MAX_TOOL_OUTPUT_TOKENS: usize = 4_000;
const BUILTIN_SYSTEM_PROMPT: &str = "You are DriftCLI, a terminal-based AI coding agent with direct access to tools.\n\nYou can read files, edit code, run shell commands, search code, fetch web pages, and manage tasks.\n\nCRITICAL RULES:\n\n1. When the user asks you to do something, CALL THE TOOL immediately. Do NOT narrate your plan.\n\n2. Do NOT say things like \"Let me check\", \"I'll try\", \"I need to\" — just invoke the tool.\n\n3. Your first response to any file/action request should be a tool call, NOT a text description.\n\n4. Only output text AFTER you have tool results to summarize.\n\n5. For simple greetings or questions requiring no file access, respond directly.\n\nBe concise. Act, don't talk.";

/// The fully assembled request context passed to an LLM provider.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    pub system_prompt: Option<String>,
    pub messages: Vec<LlmMessage>,
    pub tools: Option<Vec<Value>>,
    pub estimated_tokens: usize,
    pub compacted: bool,
    pub saved_tokens: usize,
}

/// Owns system instructions, conversation history, tool definitions, and local compaction.
pub struct ContextManager {
    messages: Vec<LlmMessage>,
    system_prompt: String,
    context_window: usize,
    compaction_threshold: f64,
    compaction_target: f64,
}

impl ContextManager {
    /// Create a context manager with the built-in DriftCLI system prompt.
    pub fn new(context_window: usize, compaction_threshold: f64, compaction_target: f64) -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: BUILTIN_SYSTEM_PROMPT.to_string(),
            context_window,
            compaction_threshold,
            compaction_target,
        }
    }

    /// Keep the context budget aligned with the active provider after a provider switch.
    pub fn set_context_window(&mut self, context_window: usize) {
        self.context_window = context_window;
    }

    /// Replace the conversation history when a stored session is loaded.
    pub fn set_messages(&mut self, messages: Vec<LlmMessage>) {
        self.messages = messages;
    }

    /// Append one message to the active conversation.
    pub fn push_message(&mut self, message: LlmMessage) {
        self.messages.push(message);
    }

    /// Check whether the current context is above the configured compaction threshold.
    pub fn needs_compaction(&self, tool_defs: &[ToolDefinition]) -> bool {
        let limit = self.context_window as f64 * self.compaction_threshold;
        self.estimate_tokens(tool_defs) as f64 >= limit
    }

    /// Build the provider request and apply deterministic local compaction when enabled.
    pub fn build_context(
        &mut self,
        tool_defs: &[ToolDefinition],
        auto_compaction: bool,
    ) -> BuiltContext {
        let before = self.estimate_tokens(tool_defs);
        let truncated = self.truncate_tool_outputs();
        let dropped =
            auto_compaction && self.needs_compaction(tool_defs) && self.drop_old_turns(tool_defs);
        let estimated_tokens = self.estimate_tokens(tool_defs);
        let tools = if tool_defs.is_empty() {
            None
        } else {
            Some(
                tool_defs
                    .iter()
                    .map(|definition| {
                        serde_json::json!({
                            "name": definition.name,
                            "description": definition.description,
                            "input_schema": definition.input_schema,
                        })
                    })
                    .collect(),
            )
        };

        BuiltContext {
            system_prompt: (!self.system_prompt.is_empty()).then_some(self.system_prompt.clone()),
            messages: self.messages.clone(),
            tools,
            estimated_tokens,
            compacted: truncated || dropped,
            saved_tokens: before.saturating_sub(estimated_tokens),
        }
    }

    fn truncate_tool_outputs(&mut self) -> bool {
        let max_chars = MAX_TOOL_OUTPUT_TOKENS * CHARS_PER_TOKEN;
        let mut changed = false;

        for message in &mut self.messages {
            for part in &mut message.content {
                if let ContentPart::ToolResult { content, .. } = part
                    && content.chars().count() > max_chars
                {
                    let truncated: String = content.chars().take(max_chars).collect();
                    *content = format!("{truncated}\n[tool output truncated]");
                    changed = true;
                }
            }
        }

        changed
    }

    fn drop_old_turns(&mut self, tool_defs: &[ToolDefinition]) -> bool {
        let target = self.context_window as f64 * self.compaction_target;
        let mut changed = false;

        while self.estimate_tokens(tool_defs) as f64 > target {
            let turn_starts: Vec<usize> = self
                .messages
                .iter()
                .enumerate()
                .filter_map(|(index, message)| {
                    (message.role == "user" && !is_tool_result_only(message)).then_some(index)
                })
                .collect();

            if turn_starts.len() < 2 {
                break;
            }

            // Remove a complete oldest turn, including its assistant tool call and results.
            self.messages.drain(..turn_starts[1]);
            changed = true;
        }

        changed
    }

    fn estimate_tokens(&self, tool_defs: &[ToolDefinition]) -> usize {
        let system_tokens = estimate_text(&self.system_prompt);
        let message_tokens = self.messages.iter().map(estimate_message).sum::<usize>();
        let tool_tokens = tool_defs
            .iter()
            .map(|definition| {
                estimate_text(&definition.name)
                    + estimate_text(&definition.description)
                    + estimate_text(&definition.input_schema.to_string())
            })
            .sum::<usize>();

        system_tokens + message_tokens + tool_tokens
    }
}

fn is_tool_result_only(message: &LlmMessage) -> bool {
    !message.content.is_empty()
        && message
            .content
            .iter()
            .all(|part| matches!(part, ContentPart::ToolResult { .. }))
}

fn estimate_message(message: &LlmMessage) -> usize {
    let content_tokens = message
        .content
        .iter()
        .map(|part| match part {
            ContentPart::Text(text) | ContentPart::Reasoning(text) => estimate_text(text),
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => estimate_text(id) + estimate_text(name) + estimate_text(arguments),
            ContentPart::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => estimate_text(tool_use_id) + estimate_text(content) + usize::from(*is_error),
        })
        .sum::<usize>();

    estimate_text(&message.role) + content_tokens
}

fn estimate_text(text: &str) -> usize {
    text.chars().count().div_ceil(CHARS_PER_TOKEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_result(content: &str) -> LlmMessage {
        LlmMessage {
            role: "user".into(),
            content: vec![ContentPart::ToolResult {
                tool_use_id: "call-1".into(),
                content: content.into(),
                is_error: false,
            }],
        }
    }

    fn tool_call() -> LlmMessage {
        LlmMessage {
            role: "assistant".into(),
            content: vec![ContentPart::ToolCall {
                id: "call-1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }],
        }
    }

    #[test]
    fn truncates_large_tool_results() {
        let mut manager = ContextManager::new(128_000, 0.75, 0.4);
        manager.set_messages(vec![tool_result(&"x".repeat(20_000))]);

        let built = manager.build_context(&[], true);
        let ContentPart::ToolResult { content, .. } = &built.messages[0].content[0] else {
            panic!("expected tool result");
        };

        assert!(content.contains("[tool output truncated]"));
        assert!(built.compacted);
    }

    #[test]
    fn drops_complete_old_turns() {
        let mut manager = ContextManager::new(200, 0.75, 0.4);
        manager.set_messages(vec![
            LlmMessage::user(&"old user input ".repeat(10)),
            tool_call(),
            tool_result("old result"),
            LlmMessage::user(&"new user input ".repeat(10)),
        ]);

        let built = manager.build_context(&[], true);

        assert!(built.compacted);
        assert_eq!(built.messages.len(), 1);
        assert!(matches!(built.messages[0].content[0], ContentPart::Text(_)));
    }

    #[test]
    fn keeps_system_prompt_and_tool_definitions_in_context() {
        let mut manager = ContextManager::new(128_000, 0.75, 0.4);
        let built = manager.build_context(
            &[ToolDefinition {
                name: "read".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            false,
        );

        assert!(built.system_prompt.unwrap().contains("DriftCLI"));
        assert_eq!(built.tools.unwrap().len(), 1);
    }

    #[test]
    fn does_not_drop_history_before_threshold() {
        let mut manager = ContextManager::new(128_000, 0.75, 0.4);
        manager.set_messages(vec![
            LlmMessage::user("keep this"),
            LlmMessage::user("and this"),
        ]);

        let built = manager.build_context(&[], true);

        assert!(!built.compacted);
        assert_eq!(built.messages.len(), 2);
    }
}
