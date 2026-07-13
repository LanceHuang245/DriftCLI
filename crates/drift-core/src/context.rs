use std::path::Path;

use drift_config::AppConfig;
use drift_llm::{ContentPart, LlmMessage, LlmProvider};
use drift_tools::ToolDefinition;
use serde_json::Value;

const CHARS_PER_TOKEN: usize = 4;
const MAX_TOOL_OUTPUT_TOKENS: usize = 4_000;
const BUILTIN_SYSTEM_PROMPT: &str = "You are DriftCLI, a terminal-based AI coding agent with direct access to tools.\n\nYou can read files, edit code, run shell commands, search code, fetch web pages, and manage tasks.\n\nCRITICAL RULES:\n\n1. When the user asks you to do something, CALL THE TOOL immediately. Do NOT narrate your plan.\n\n2. Do NOT say things like \"Let me check\", \"I'll try\", \"I need to\" — just invoke the tool.\n\n3. Your first response to any file/action request should be a tool call, NOT a text description.\n\n4. Only output text AFTER you have tool results to summarize.\n\n5. For simple greetings or questions requiring no file access, respond directly.\n\nBe concise. Act, don't talk.";
// Instruct the active provider to produce state-only conversation summaries.
const SUMMARY_SYSTEM_PROMPT: &str = "You are a conversation-state summarizer. Return only a concise structured summary. Preserve the user's goal, constraints, decisions, changed files, commands and test results, failures, and unfinished work. Do not add advice or commentary.";

/// A candidate context state that can be committed after persistence succeeds.
#[derive(Debug, Clone)]
pub struct CompactionSnapshot {
    /// Messages that will become the committed active context.
    pub messages: Vec<LlmMessage>,
    /// Summary rendered into the committed system prompt, when available.
    pub summary: Option<String>,
}

/// Errors produced while preparing an automatically summarized context.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("context summary provider is unavailable")]
    SummaryUnavailable,
    #[error("context summary failed: {0}")]
    SummaryFailed(String),
    #[error("context summary returned empty content")]
    EmptySummary,
    #[error("context exceeds budget after compaction: {estimated_tokens} tokens")]
    BudgetExceeded { estimated_tokens: usize },
}

/// The fully assembled request context passed to an LLM provider.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    pub system_prompt: Option<String>,
    pub messages: Vec<LlmMessage>,
    pub tools: Option<Vec<Value>>,
    pub estimated_tokens: usize,
    pub compacted: bool,
    pub saved_tokens: usize,
    /// Candidate state to persist before committing it to the manager.
    pub compaction: Option<CompactionSnapshot>,
}

/// Owns system instructions, conversation history, and transactional local compaction.
pub struct ContextManager {
    messages: Vec<LlmMessage>,
    /// Built-in system instructions shared by every workspace.
    builtin_system_prompt: String,
    /// Workspace and global instruction blocks appended to the system prompt.
    instruction_blocks: Vec<String>,
    /// Latest persisted conversation summary, when compaction has occurred.
    conversation_summary: Option<String>,
    context_window: usize,
    compaction_threshold: f64,
    compaction_target: f64,
}

impl ContextManager {
    /// Create a context manager with only the built-in DriftCLI system prompt.
    pub fn new(context_window: usize, compaction_threshold: f64, compaction_target: f64) -> Self {
        Self {
            messages: Vec::new(),
            builtin_system_prompt: BUILTIN_SYSTEM_PROMPT.to_string(),
            instruction_blocks: Vec::new(),
            conversation_summary: None,
            context_window,
            compaction_threshold,
            compaction_target,
        }
    }

    /// Create a context manager with workspace and global AGENTS.md instructions.
    pub fn for_workspace(
        context_window: usize,
        compaction_threshold: f64,
        compaction_target: f64,
        cwd: &Path,
    ) -> Self {
        let global_dir = AppConfig::global_config_dir();
        Self::with_instruction_blocks(
            context_window,
            compaction_threshold,
            compaction_target,
            load_instruction_blocks(cwd, global_dir.as_deref()),
        )
    }

    /// Build a manager from preloaded instruction blocks for production and tests.
    fn with_instruction_blocks(
        context_window: usize,
        compaction_threshold: f64,
        compaction_target: f64,
        instruction_blocks: Vec<String>,
    ) -> Self {
        Self {
            messages: Vec::new(),
            builtin_system_prompt: BUILTIN_SYSTEM_PROMPT.to_string(),
            instruction_blocks,
            conversation_summary: None,
            context_window,
            compaction_threshold,
            compaction_target,
        }
    }

    /// Keep the context budget aligned with the active provider after a provider switch.
    pub fn set_context_window(&mut self, context_window: usize) {
        self.context_window = context_window;
    }

    /// Replace the conversation history and discard any prior summary.
    pub fn set_messages(&mut self, messages: Vec<LlmMessage>) {
        self.messages = messages;
        self.conversation_summary = None;
    }

    /// Replace both the conversation history and its persisted summary.
    pub fn set_compacted_state(&mut self, messages: Vec<LlmMessage>, summary: Option<String>) {
        self.messages = messages;
        self.conversation_summary = summary;
    }

    /// Commit a candidate only after its persistence event has been flushed successfully.
    pub fn apply_compaction(&mut self, snapshot: &CompactionSnapshot) {
        self.messages = snapshot.messages.clone();
        self.conversation_summary = snapshot.summary.clone();
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

    /// Build a request without mutating state; summaries are committed separately.
    pub async fn build_context(
        &self,
        tool_defs: &[ToolDefinition],
        auto_compaction: bool,
        summarizer: Option<&dyn LlmProvider>,
    ) -> Result<BuiltContext, ContextError> {
        let before = self.estimate_tokens(tool_defs);
        let mut candidate_messages = self.messages.clone();
        let truncated = truncate_tool_outputs(&mut candidate_messages);
        let truncated_tokens = self.estimate_tokens_for(
            tool_defs,
            &candidate_messages,
            self.conversation_summary.as_deref(),
        );
        let threshold = self.context_window as f64 * self.compaction_threshold;
        let target = self.context_window as f64 * self.compaction_target;
        let mut candidate_summary = self.conversation_summary.clone();
        let mut summary_compacted = false;

        if truncated_tokens as f64 > threshold && auto_compaction {
            let (kept_messages, removed_messages) = select_messages(&candidate_messages, 3);
            let provider = summarizer.ok_or(ContextError::SummaryUnavailable)?;
            let response = provider
                .chat(
                    vec![LlmMessage::user(format_summary_input(
                        self.conversation_summary.as_deref(),
                        &removed_messages,
                    ))],
                    Some(SUMMARY_SYSTEM_PROMPT.to_string()),
                )
                .await
                .map_err(|error| ContextError::SummaryFailed(error.to_string()))?;
            if response.trim().is_empty() {
                return Err(ContextError::EmptySummary);
            }
            candidate_summary = Some(response.trim().to_string());
            candidate_messages = kept_messages;
            summary_compacted = true;

            let mut estimated_tokens = self.estimate_tokens_for(
                tool_defs,
                &candidate_messages,
                candidate_summary.as_deref(),
            );
            if estimated_tokens as f64 > target {
                let (reduced_messages, _) = select_messages(&candidate_messages, 1);
                candidate_messages = reduced_messages;
                estimated_tokens = self.estimate_tokens_for(
                    tool_defs,
                    &candidate_messages,
                    candidate_summary.as_deref(),
                );
            }
            if estimated_tokens as f64 > target {
                return Err(ContextError::BudgetExceeded { estimated_tokens });
            }
        }

        let estimated_tokens =
            self.estimate_tokens_for(tool_defs, &candidate_messages, candidate_summary.as_deref());
        let changed = truncated || summary_compacted;
        let compaction = changed.then(|| CompactionSnapshot {
            messages: candidate_messages.clone(),
            summary: candidate_summary.clone(),
        });
        let system_prompt = self.render_system_prompt(candidate_summary.as_deref());
        let tools = build_tools(tool_defs);

        Ok(BuiltContext {
            system_prompt: (!system_prompt.is_empty()).then_some(system_prompt),
            messages: candidate_messages,
            tools,
            estimated_tokens,
            compacted: changed,
            saved_tokens: before.saturating_sub(estimated_tokens),
            compaction,
        })
    }

    /// Render the exact system prompt used for token accounting and provider requests.
    fn render_system_prompt(&self, summary: Option<&str>) -> String {
        let mut prompt = self.builtin_system_prompt.clone();
        for block in &self.instruction_blocks {
            prompt.push_str("\n\n");
            prompt.push_str(block);
        }
        if let Some(summary) = summary {
            prompt.push_str("\n\n<conversation_summary>");
            prompt.push_str(summary);
            prompt.push_str("</conversation_summary>");
        }
        prompt
    }

    /// Estimate the current provider request, including rendered system instructions.
    fn estimate_tokens(&self, tool_defs: &[ToolDefinition]) -> usize {
        self.estimate_tokens_for(
            tool_defs,
            &self.messages,
            self.conversation_summary.as_deref(),
        )
    }

    /// Estimate a candidate message/summary state without mutating the manager.
    fn estimate_tokens_for(
        &self,
        tool_defs: &[ToolDefinition],
        messages: &[LlmMessage],
        summary: Option<&str>,
    ) -> usize {
        let system_tokens = estimate_text(&self.render_system_prompt(summary));
        let message_tokens = messages.iter().map(estimate_message).sum::<usize>();
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

/// Read instruction files in project-first order, warning but skipping unreadable files.
fn load_instruction_blocks(cwd: &Path, global_dir: Option<&Path>) -> Vec<String> {
    let candidates = [
        (
            cwd.join(".drift").join("AGENTS.md"),
            "project_instructions",
            ".drift/AGENTS.md",
        ),
        (cwd.join("AGENTS.md"), "project_instructions", "AGENTS.md"),
    ];
    let mut blocks = Vec::new();
    for (path, tag, source) in candidates {
        if let Some(block) = read_instruction_block(&path, tag, source) {
            blocks.push(block);
        }
    }
    if let Some(global_dir) = global_dir {
        let path = global_dir.join("AGENTS.md");
        if let Some(block) = read_instruction_block(&path, "user_instructions", "global/AGENTS.md")
        {
            blocks.push(block);
        }
    }
    blocks
}

/// Read one instruction file and wrap it with its provider-facing source tag.
fn read_instruction_block(path: &Path, tag: &str, source: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => Some(format!("<{tag} source=\"{source}\">\n{content}\n</{tag}>")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "failed to read instruction file");
            None
        }
    }
}

/// Convert internal tool definitions into the provider-neutral JSON shape.
fn build_tools(tool_defs: &[ToolDefinition]) -> Option<Vec<Value>> {
    (!tool_defs.is_empty()).then(|| {
        tool_defs
            .iter()
            .map(|definition| {
                serde_json::json!({
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": definition.input_schema,
                })
            })
            .collect()
    })
}

/// Enforce the per-tool-result output cap on a candidate message list.
fn truncate_tool_outputs(messages: &mut [LlmMessage]) -> bool {
    let max_chars = MAX_TOOL_OUTPUT_TOKENS * CHARS_PER_TOKEN;
    let mut changed = false;

    for message in messages {
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

/// Select complete turns while preserving the initial user request and recent turns.
fn select_messages(
    messages: &[LlmMessage],
    recent_turns: usize,
) -> (Vec<LlmMessage>, Vec<LlmMessage>) {
    let starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (message.role == "user" && !is_tool_result_only(message)).then_some(index)
        })
        .collect();
    if starts.is_empty() {
        return (messages.to_vec(), Vec::new());
    }

    let mut ranges = Vec::with_capacity(starts.len());
    for (turn, &start) in starts.iter().enumerate() {
        let range_start = if turn == 0 { 0 } else { start };
        let end = starts.get(turn + 1).copied().unwrap_or(messages.len());
        ranges.push((range_start, end));
    }

    let recent_start = ranges.len().saturating_sub(recent_turns);
    let mut keep = vec![false; ranges.len()];
    keep[0] = true;
    for keep_flag in keep.iter_mut().skip(recent_start) {
        *keep_flag = true;
    }

    let mut kept = Vec::new();
    let mut removed = Vec::new();
    for (index, &(start, end)) in ranges.iter().enumerate() {
        if keep[index] {
            kept.extend_from_slice(&messages[start..end]);
        } else {
            removed.extend_from_slice(&messages[start..end]);
        }
    }
    (kept, removed)
}

/// Serialize prior summary and removed messages into the summarizer prompt.
fn format_summary_input(previous_summary: Option<&str>, messages: &[LlmMessage]) -> String {
    let mut output = String::from("<previous_summary>");
    if let Some(summary) = previous_summary {
        output.push_str(summary);
    }
    output.push_str("</previous_summary>\n<messages>\n");
    for message in messages {
        output.push_str(&format!("[role={}]\n", message.role));
        for part in &message.content {
            match part {
                ContentPart::Text(text) => output.push_str(&format!("text: {text}\n")),
                ContentPart::Reasoning(text) => output.push_str(&format!("reasoning: {text}\n")),
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => output.push_str(&format!(
                    "tool-call(name={name} id={id} arguments={arguments})\n"
                )),
                ContentPart::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => output.push_str(&format!(
                    "tool-result(id={tool_use_id} is_error={is_error} content={content})\n"
                )),
            }
        }
    }
    output.push_str("</messages>");
    output
}

/// Identify synthetic user messages that contain only tool results.
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

    #[tokio::test]
    async fn truncates_large_tool_results() {
        let mut manager = ContextManager::new(128_000, 0.75, 0.4);
        manager.set_messages(vec![tool_result(&"x".repeat(20_000))]);

        let built = manager.build_context(&[], true, None).await.unwrap();
        let ContentPart::ToolResult { content, .. } = &built.messages[0].content[0] else {
            panic!("expected tool result");
        };

        assert!(content.contains("[tool output truncated]"));
        assert!(built.compacted);
        assert!(built.compaction.is_some());
    }

    #[tokio::test]
    async fn drops_complete_old_turns_only_with_summary_provider() {
        let mut manager = ContextManager::new(200, 0.75, 0.4);
        manager.set_messages(vec![
            LlmMessage::user("old user input ".repeat(10)),
            tool_call(),
            tool_result("old result"),
            LlmMessage::user("new user input ".repeat(10)),
        ]);

        let result = manager.build_context(&[], true, None).await;
        assert!(matches!(result, Err(ContextError::SummaryUnavailable)));
    }

    #[tokio::test]
    async fn keeps_system_prompt_and_tool_definitions_in_context() {
        let manager = ContextManager::new(128_000, 0.75, 0.4);
        let built = manager
            .build_context(
                &[ToolDefinition {
                    name: "read".into(),
                    description: "Read a file".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }],
                false,
                None,
            )
            .await
            .unwrap();

        assert!(built.system_prompt.unwrap().contains("DriftCLI"));
        assert_eq!(built.tools.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn does_not_drop_history_before_threshold() {
        let mut manager = ContextManager::new(128_000, 0.75, 0.4);
        manager.set_messages(vec![
            LlmMessage::user("keep this"),
            LlmMessage::user("and this"),
        ]);

        let built = manager.build_context(&[], true, None).await.unwrap();

        assert!(!built.compacted);
        assert!(built.compaction.is_none());
        assert_eq!(built.messages.len(), 2);
    }

    // Verify workspace instructions precede global instructions in the final prompt.
    #[tokio::test]
    async fn loads_workspace_instructions_in_project_first_order() {
        let root = std::env::temp_dir().join(format!("drift-context-{}", uuid::Uuid::new_v4()));
        let global = root.join("global");
        std::fs::create_dir_all(root.join(".drift")).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        std::fs::write(root.join(".drift/AGENTS.md"), "dot drift").unwrap();
        std::fs::write(root.join("AGENTS.md"), "root").unwrap();
        std::fs::write(global.join("AGENTS.md"), "global").unwrap();

        let blocks = load_instruction_blocks(&root, Some(&global));
        assert_eq!(blocks.len(), 3);
        assert!(blocks[0].contains("source=\".drift/AGENTS.md\""));
        assert!(blocks[1].contains("source=\"AGENTS.md\""));
        assert!(blocks[2].contains("source=\"global/AGENTS.md\""));
        assert!(blocks[0].contains("dot drift"));
        assert!(blocks[1].contains("root"));
        assert!(blocks[2].contains("global"));
        let manager = ContextManager::with_instruction_blocks(128_000, 0.75, 0.4, blocks);
        let prompt = manager
            .build_context(&[], false, None)
            .await
            .unwrap()
            .system_prompt
            .unwrap();
        assert!(prompt.find("dot drift").unwrap() < prompt.find("root").unwrap());
        assert!(prompt.find("root").unwrap() < prompt.find("global").unwrap());

        std::fs::remove_dir_all(root).ok();
    }
}

#[cfg(test)]
mod compaction_tests {
    use super::*;

    // Empty provider stream used to keep fake providers network-free.
    struct EmptyStream;

    impl tokio_stream::Stream for EmptyStream {
        type Item = Result<drift_llm::LlmChunk, drift_llm::LlmError>;

        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(None)
        }
    }

    // Configurable provider fixture that records summary requests.
    struct FakeProvider {
        summary: String,
        calls: std::sync::atomic::AtomicUsize,
        inputs: std::sync::Arc<tokio::sync::Mutex<Vec<String>>>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl LlmProvider for FakeProvider {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_name(&self) -> &str {
            "fake-model"
        }

        fn context_window(&self) -> usize {
            2400
        }

        async fn stream_chat(
            &self,
            _messages: Vec<LlmMessage>,
            _system_prompt: Option<String>,
            _temperature: Option<f64>,
            _max_output_tokens: Option<usize>,
            _tools: Option<Vec<Value>>,
        ) -> Result<drift_llm::LlmResponseStream, drift_llm::LlmError> {
            Ok(drift_llm::LlmResponseStream::new(EmptyStream))
        }

        async fn chat(
            &self,
            messages: Vec<LlmMessage>,
            _system_prompt: Option<String>,
        ) -> Result<String, drift_llm::LlmError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let input = messages
                .first()
                .and_then(|message| message.content.first())
                .and_then(|part| match part {
                    ContentPart::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            self.inputs.lock().await.push(input);
            if self.fail {
                Err(drift_llm::LlmError::Stream("fake summary failure".into()))
            } else {
                Ok(self.summary.clone())
            }
        }
    }

    // Build enough complete turns to force deterministic compaction.
    fn long_conversation() -> Vec<LlmMessage> {
        let mut messages = vec![LlmMessage::user(format!(
            "initial request {}",
            "x".repeat(400)
        ))];
        for index in 0..5 {
            messages.push(LlmMessage {
                role: "assistant".into(),
                content: vec![
                    ContentPart::Text(format!("turn {index} assistant {}", "a".repeat(200))),
                    ContentPart::ToolCall {
                        id: format!("call-{index}"),
                        name: "read".into(),
                        arguments: "{}".into(),
                    },
                ],
            });
            messages.push(LlmMessage {
                role: "user".into(),
                content: vec![ContentPart::ToolResult {
                    tool_use_id: format!("call-{index}"),
                    content: format!("tool result {index} {}", "r".repeat(200)),
                    is_error: false,
                }],
            });
            messages.push(LlmMessage::user(format!(
                "turn {index} request {}",
                "u".repeat(400)
            )));
        }
        messages
    }

    // Create a successful fake summarizer with isolated call tracking.
    fn fake_provider(summary: &str) -> FakeProvider {
        FakeProvider {
            summary: summary.into(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            inputs: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            fail: false,
        }
    }

    // Verify summary preparation preserves boundaries and leaves state transactional.
    #[tokio::test]
    async fn summarizes_transactionally_and_preserves_recent_complete_turns() {
        let mut manager = ContextManager::new(2400, 0.25, 0.5);
        manager.set_messages(long_conversation());
        let provider = fake_provider("## summary ##");

        let built = manager
            .build_context(&[], true, Some(&provider))
            .await
            .unwrap();
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            built
                .system_prompt
                .as_deref()
                .unwrap()
                .contains("## summary ##")
        );
        assert!(built.messages.iter().any(|message| {
            message.content.iter().any(
                |part| matches!(part, ContentPart::Text(text) if text.contains("initial request")),
            )
        }));
        assert!(built.messages.iter().any(|message| {
            message.content.iter().any(
                |part| matches!(part, ContentPart::Text(text) if text.contains("turn 4 request")),
            )
        }));
        assert!(!built.messages.iter().any(|message| {
            message.content.iter().any(
                |part| matches!(part, ContentPart::Text(text) if text.contains("turn 0 request")),
            )
        }));
        let call_index = built
            .messages
            .iter()
            .position(|message| {
                message
                    .content
                    .iter()
                    .any(|part| matches!(part, ContentPart::ToolCall { id, .. } if id == "call-0"))
            })
            .expect("tool call retained");
        let result_index = built
            .messages
            .iter()
            .position(|message| {
                message.content.iter().any(|part| {
                    matches!(part, ContentPart::ToolResult { tool_use_id, .. } if tool_use_id == "call-0")
                })
            })
            .expect("tool result retained");
        assert_eq!(result_index, call_index + 1);
        let untouched = manager.build_context(&[], false, None).await.unwrap();
        assert_eq!(untouched.messages.len(), 16);

        let snapshot = built.compaction.clone().unwrap();
        manager.apply_compaction(&snapshot);
        let applied = manager.build_context(&[], false, None).await.unwrap();
        assert!(
            applied
                .system_prompt
                .unwrap()
                .contains("<conversation_summary>## summary ##</conversation_summary>")
        );
    }

    // Verify provider errors do not commit candidate messages or summaries.
    #[tokio::test]
    async fn summary_failure_is_transactional() {
        let mut manager = ContextManager::new(2400, 0.25, 0.5);
        manager.set_messages(long_conversation());
        let provider = FakeProvider {
            summary: String::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            inputs: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
            fail: true,
        };

        let result = manager.build_context(&[], true, Some(&provider)).await;
        assert!(matches!(result, Err(ContextError::SummaryFailed(_))));
        let untouched = manager.build_context(&[], false, None).await.unwrap();
        assert_eq!(untouched.messages.len(), 16);
        assert!(
            !untouched
                .system_prompt
                .unwrap()
                .contains("<conversation_summary>")
        );
    }

    // Verify successive compactions feed the previous summary forward once.
    #[tokio::test]
    async fn subsequent_summary_input_contains_previous_summary() {
        let mut manager = ContextManager::new(2400, 0.25, 0.5);
        manager.set_messages(long_conversation());
        let first = fake_provider("first summary");
        let first_built = manager
            .build_context(&[], true, Some(&first))
            .await
            .unwrap();
        manager.apply_compaction(&first_built.compaction.unwrap());
        manager.push_message(LlmMessage::user(format!("new request {}", "n".repeat(500))));
        let second = fake_provider("second summary");
        let second_built = manager
            .build_context(&[], true, Some(&second))
            .await
            .unwrap();
        let inputs = second.inputs.lock().await;
        assert!(inputs[0].contains("<previous_summary>first summary</previous_summary>"));
        assert_eq!(
            second_built
                .system_prompt
                .as_deref()
                .unwrap()
                .matches("<conversation_summary>")
                .count(),
            1
        );
        assert!(
            second_built
                .system_prompt
                .unwrap()
                .contains("second summary")
        );
    }
}
