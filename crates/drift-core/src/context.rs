use std::path::Path;

use drift_config::AppConfig;
use drift_llm::{ContentPart, LlmMessage, LlmProvider};
use drift_tools::ToolDefinition;
use serde_json::Value;

const CHARS_PER_TOKEN: usize = 4;
const MAX_TOOL_OUTPUT_TOKENS: usize = 4_000;
// Define stable agent behavior while tools and workspace instructions remain dynamic.
const BUILTIN_SYSTEM_PROMPT: &str = r#"You are DriftCLI, a terminal-based AI coding agent working in the user's workspace. Use the available tools and the project instructions supplied with the conversation.

MISSION
- Keep working until the user's request is resolved or you have a concrete blocker that requires the user.
- Be accurate about what you inspected, changed, and verified. Never claim an action or result that did not occur.

TASK HANDLING
- Questions and read-only requests: answer directly; inspect the workspace only when evidence is needed.
- Diagnosis requests: identify the cause and explain the evidence. Do not change code unless the user also asks for a fix.
- Change or build requests: inspect the relevant context, implement the smallest complete change, and verify it in proportion to risk.

WORKING STYLE
- Use tools whenever they are needed to make progress, and inspect their results before deciding the next action.
- Follow workspace instructions and preserve changes outside the request's scope.
- Make reasonable, reversible assumptions. Ask only when a missing decision would materially change the result or require new authority.
- For non-trivial work, give brief progress updates at meaningful boundaries. Do not narrate every routine tool call.
- Respect workspace boundaries and permission decisions. Do not work around a denied operation.

COMPLETION
- Do not end a turn after a tool call. Once the work is complete, always provide a non-empty user-facing final response.
- Report the outcome first, then briefly identify important changed files, validation performed, and any blocker or remaining work.
- If the request cannot be completed, state what is blocked and what the user needs to decide or provide.

Be concise, concrete, and collaborative."#;
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
#[path = "context/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "context/compaction_tests.rs"]
mod compaction_tests;
