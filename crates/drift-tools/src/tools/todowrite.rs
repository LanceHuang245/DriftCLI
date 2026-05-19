use crate::{Tool, ToolContext, ToolError, ToolResult};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// In-memory storage for todo lists, keyed by session ID.
static TODO_STORE: LazyLock<Mutex<HashMap<String, Vec<TodoItem>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct TodoItem {
    content: String,
    status: String,
    priority: String,
}

pub struct TodoWriteTool;

#[async_trait::async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "todowrite"
    }

    fn description(&self) -> &str {
        "Create and manage a structured task list. Pass a JSON array of todo items, each with \
         content, status (pending/in_progress/completed/cancelled), and priority (high/medium/low). \
         Exactly one item may be in_progress at a time."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The list of todo items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Description of the task"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                                "description": "Current status of the task"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Priority level of the task"
                            }
                        },
                        "required": ["content", "status", "priority"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Parse the todos array from the JSON arguments
        let todos: Vec<TodoItem> = serde_json::from_value(
            args["todos"].clone(),
        )
        .map_err(|e| ToolError::InvalidArgs(format!("invalid todos array: {e}")))?;

        // Validate: exactly one item may be in_progress, OR all items are completed/cancelled
        let in_progress_count = todos
            .iter()
            .filter(|t| t.status == "in_progress")
            .count();

        let all_terminal = todos.iter().all(|t| {
            t.status == "completed" || t.status == "cancelled"
        });

        if !all_terminal && in_progress_count > 1 {
            return Err(ToolError::InvalidArgs(format!(
                "multiple items ({in_progress_count}) have status 'in_progress'; \
                 exactly one item may be in_progress at a time"
            )));
        }

        if !all_terminal && in_progress_count == 0 && !todos.is_empty() {
            return Err(ToolError::InvalidArgs(
                "no item has status 'in_progress'; exactly one item must be in_progress \
                 when not all items are completed/cancelled"
                    .into(),
            ));
        }

        // Count statuses for the summary
        let pending = todos.iter().filter(|t| t.status == "pending").count();
        let in_progress = todos.iter().filter(|t| t.status == "in_progress").count();
        let completed = todos.iter().filter(|t| t.status == "completed").count();
        let cancelled = todos.iter().filter(|t| t.status == "cancelled").count();

        // Store the todo list in the session-scoped map
        {
            let mut store = TODO_STORE.lock().map_err(|e| {
                ToolError::Other(format!("todo store lock poisoned: {e}"))
            })?;
            store.insert(ctx.session_id.to_string(), todos);
        }

        let summary = format!(
            "Todo list updated: {pending} pending, {in_progress} in progress, \
             {completed} completed, {cancelled} cancelled"
        );

        Ok(ToolResult {
            success: true,
            content: summary,
            error: None,
        })
    }
}
