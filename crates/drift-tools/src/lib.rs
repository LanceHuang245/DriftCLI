//! drift-tools: Tool execution framework — core types, `Tool` trait, and `ToolRegistry`.

pub mod tools;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// A tool definition sent to the LLM describing name, purpose, and JSON Schema for arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Context available to a tool at execution time (session, working dir, call ID).
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub session_id: Uuid,
    pub working_dir: PathBuf,
    pub tool_call_id: String,
    pub file_access: Arc<drift_security::FileAccessGuard>,
    pub network: Arc<drift_security::NetworkGuard>,
    pub process_sandbox: Arc<drift_security::ProcessSandbox>,
}

/// Result returned by a tool after execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub content: String,
    pub error: Option<String>,
}

/// Errors that can occur during tool lookup or execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("tool execution timeout")]
    Timeout,
    #[error("tool was cancelled")]
    Cancelled,
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("tool execution failed: {0}")]
    ExecutionFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tool error: {0}")]
    Other(String),
}

/// Every tool implements this trait: define metadata and execute with arguments + context.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> serde_json::Value;

    /// Returns a `ToolDefinition` built from the trait methods.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }

    /// Execute the tool with the given JSON arguments and session context.
    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError>;
}

/// Registry that holds built-in tools and dynamic (runtime-registered) tools.
pub struct ToolRegistry {
    builtins: HashMap<String, Arc<dyn Tool>>,
    dynamic: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            builtins: HashMap::new(),
            dynamic: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tool that lives for the lifetime of the registry (built-in).
    pub fn register_builtin(&mut self, tool: Arc<dyn Tool>) {
        self.builtins.insert(tool.name().to_string(), tool);
    }

    /// Number of built-in tools registered.
    pub fn builtin_count(&self) -> usize {
        self.builtins.len()
    }

    /// Register a tool whose lifecycle can be managed at runtime.
    pub async fn register_dynamic(&self, tool: Arc<dyn Tool>) {
        self.dynamic
            .write()
            .await
            .insert(tool.name().to_string(), tool);
    }

    /// Remove a dynamically registered tool by name.
    pub async fn unregister(&self, name: &str) {
        self.dynamic.write().await.remove(name);
    }

    /// Remove several dynamic tools while holding one write lock.
    pub async fn unregister_many<I, S>(&self, names: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut dynamic = self.dynamic.write().await;
        for name in names {
            dynamic.remove(name.as_ref());
        }
    }

    /// Remove all dynamic tools without touching built-ins.
    pub async fn clear_dynamic(&self) {
        self.dynamic.write().await.clear();
    }

    /// Look up a tool by name (synchronous, blocks for a read lock if needed).
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.builtins
            .get(name)
            .cloned()
            .or_else(|| self.dynamic.blocking_read().get(name).cloned())
    }

    /// Look up a tool by name (async, prefers non-blocking read).
    pub async fn get_async(&self, name: &str) -> Option<Arc<dyn Tool>> {
        if let Some(tool) = self.builtins.get(name) {
            return Some(tool.clone());
        }
        self.dynamic.read().await.get(name).cloned()
    }

    /// Collect all `ToolDefinition`s from both built-in and dynamic tools.
    pub async fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> =
            self.builtins.values().map(|t| t.definition()).collect();
        let dynamic = self.dynamic.read().await;
        for tool in dynamic.values() {
            defs.push(tool.definition());
        }
        defs
    }

    /// Execute a named tool with arguments and context.
    pub async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let tool = self
            .get_async(name)
            .await
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.execute(args, ctx).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool {
        name: String,
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "dummy"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                success: true,
                content: self.name.clone(),
                error: None,
            })
        }
    }

    #[tokio::test]
    async fn unregister_many_only_removes_requested_dynamic_tools() {
        let registry = ToolRegistry::new();
        registry
            .register_dynamic(Arc::new(DummyTool { name: "one".into() }))
            .await;
        registry
            .register_dynamic(Arc::new(DummyTool { name: "two".into() }))
            .await;
        registry
            .register_dynamic(Arc::new(DummyTool {
                name: "three".into(),
            }))
            .await;

        registry.unregister_many(["one", "three"]).await;

        assert!(registry.get_async("one").await.is_none());
        assert!(registry.get_async("three").await.is_none());
        assert!(registry.get_async("two").await.is_some());
    }
}
