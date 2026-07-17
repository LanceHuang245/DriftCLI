use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use drift_config::{McpConfig, McpServerConfig, McpTransport};
use drift_tools::{Tool, ToolContext, ToolError, ToolRegistry, ToolResult};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, ContentBlock};
use rmcp::service::{Peer, RoleClient, RunningService};
use rmcp::transport::{TokioChildProcess, which_command};
use thiserror::Error;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::warn;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Owns MCP child processes and their dynamically registered tools.
pub struct McpManager {
    config: McpConfig,
    tool_registry: Arc<ToolRegistry>,
    status_tx: Option<mpsc::UnboundedSender<(String, String)>>,
    shutdown: CancellationToken,
    connections: Mutex<HashMap<String, McpConnection>>,
}

struct McpConnection {
    service: RunningService<RoleClient, ()>,
    tool_names: Vec<String>,
}

#[derive(Debug, Error)]
enum McpRuntimeError {
    #[error("MCP command error: {0}")]
    Command(#[from] std::io::Error),
    #[error("MCP initialization failed: {0}")]
    Initialize(String),
    #[error("MCP tool discovery failed: {0}")]
    Discovery(String),
    #[error("MCP tool name conflict: {0}")]
    ToolConflict(String),
    #[error("MCP duplicate tool name: {0}")]
    DuplicateTool(String),
    #[error("MCP duplicate server id: {0}")]
    DuplicateServer(String),
    #[error("MCP tool name is not provider-safe: {0}")]
    InvalidToolName(String),
    #[error("MCP environment variable '{name}' is not set for server '{server}'")]
    MissingEnvironment { server: String, name: String },
    #[error("MCP environment value has an unclosed placeholder for server '{0}'")]
    InvalidEnvironmentPlaceholder(String),
    #[error("MCP server '{0}' only supports stdio")]
    UnsupportedTransport(String),
    #[error("MCP startup was cancelled")]
    Cancelled,
}

impl McpManager {
    /// Creates a manager that shares the caller's registry.
    pub fn new(config: McpConfig, tool_registry: Arc<ToolRegistry>) -> Self {
        Self {
            config,
            tool_registry,
            status_tx: None,
            shutdown: CancellationToken::new(),
            connections: Mutex::new(HashMap::new()),
        }
    }

    /// Creates a manager with a status channel for the TUI bridge.
    pub fn with_status_sender(
        config: McpConfig,
        tool_registry: Arc<ToolRegistry>,
        status_tx: mpsc::UnboundedSender<(String, String)>,
    ) -> Self {
        Self {
            config,
            tool_registry,
            status_tx: Some(status_tx),
            shutdown: CancellationToken::new(),
            connections: Mutex::new(HashMap::new()),
        }
    }

    /// Starts configured auto-start servers in declaration order.
    pub async fn start_auto_servers(self: Arc<Self>) {
        if !self.config.enabled {
            return;
        }

        let mut server_ids = HashSet::with_capacity(self.config.servers.len());
        for server in self
            .config
            .servers
            .iter()
            .filter(|server| server.auto_start)
        {
            if self.shutdown.is_cancelled() {
                break;
            }
            if !server_ids.insert(&server.id) {
                let error = McpRuntimeError::DuplicateServer(server.id.clone());
                warn!(server = %server.id, error = %error, "MCP server was skipped");
                self.send_status(&server.id, format!("failed: {error}"));
                continue;
            }
            self.send_status(&server.id, "connecting".into());
            let result = tokio::time::timeout(STARTUP_TIMEOUT, self.connect_server(server)).await;
            match result {
                Ok(Ok(connection)) => {
                    let count = connection.tool_names.len();
                    let cancelled_connection = {
                        let mut connections = self.connections.lock().await;
                        if self.shutdown.is_cancelled() {
                            Some(connection)
                        } else {
                            connections.insert(server.id.clone(), connection);
                            None
                        }
                    };
                    if let Some(mut connection) = cancelled_connection {
                        // A shutdown that raced startup must release this child and its tools.
                        if let Err(error) = connection.service.close().await {
                            warn!(server = %server.id, error = %error, "MCP server close failed");
                        }
                        self.tool_registry
                            .unregister_many(&connection.tool_names)
                            .await;
                        self.send_status(&server.id, "stopped".into());
                    } else {
                        self.send_status(&server.id, format!("connected ({count} tools)"));
                    }
                }
                Ok(Err(McpRuntimeError::Cancelled)) if self.shutdown.is_cancelled() => break,
                Ok(Err(error)) => {
                    warn!(server = %server.id, error = %error, "MCP server failed to start");
                    self.send_status(&server.id, format!("failed: {error}"));
                }
                Err(_) => {
                    let error = format!(
                        "startup timed out after {} seconds",
                        STARTUP_TIMEOUT.as_secs()
                    );
                    warn!(server = %server.id, "MCP server startup timed out");
                    self.send_status(&server.id, format!("failed: {error}"));
                }
            }
        }
    }

    /// Closes every child process and removes only its dynamic tools.
    pub async fn shutdown(&self) {
        self.shutdown.cancel();
        let connections = {
            let mut guard = self.connections.lock().await;
            std::mem::take(&mut *guard)
        };
        for (server_id, mut connection) in connections {
            if let Err(error) = connection.service.close().await {
                warn!(server = %server_id, error = %error, "MCP server close failed");
            }
            self.tool_registry
                .unregister_many(&connection.tool_names)
                .await;
            self.send_status(&server_id, "stopped".into());
        }
    }

    async fn connect_server(
        &self,
        server: &McpServerConfig,
    ) -> Result<McpConnection, McpRuntimeError> {
        if !matches!(server.transport, McpTransport::Stdio) {
            return Err(McpRuntimeError::UnsupportedTransport(server.id.clone()));
        }

        let mut command = which_command(&server.command)?;
        command.args(&server.args);
        for (name, value) in &server.env {
            command.env(name, expand_environment_value(&server.id, value)?);
        }

        let transport = TokioChildProcess::new(command)?;
        let mut service = tokio::select! {
            _ = self.shutdown.cancelled() => return Err(McpRuntimeError::Cancelled),
            result = ().serve(transport) => {
                result.map_err(|error| McpRuntimeError::Initialize(error.to_string()))?
            }
        };
        let peer = service.peer().clone();
        let tools = tokio::select! {
            _ = self.shutdown.cancelled() => {
                if let Err(error) = service.close().await {
                    warn!(server = %server.id, error = %error, "MCP server close failed");
                }
                return Err(McpRuntimeError::Cancelled);
            }
            result = peer.list_all_tools() => {
                result.map_err(|error| McpRuntimeError::Discovery(error.to_string()))?
            }
        };

        let mut public_names = HashSet::with_capacity(tools.len());
        let mut adapters = Vec::with_capacity(tools.len());
        for tool in tools {
            let raw_name = tool.name.to_string();
            if !public_names.insert(raw_name.clone()) {
                return Err(McpRuntimeError::DuplicateTool(raw_name));
            }
            let public_name = format!("mcp__{}__{}", server.id, raw_name);
            if raw_name.is_empty() || !is_provider_safe_tool_name(&public_name) {
                return Err(McpRuntimeError::InvalidToolName(public_name));
            }
            if self.tool_registry.get_async(&public_name).await.is_some() {
                return Err(McpRuntimeError::ToolConflict(public_name));
            }
            let description = tool
                .description
                .as_deref()
                .unwrap_or("MCP tool")
                .to_string();
            let input_schema = serde_json::to_value(tool.input_schema.as_ref())
                .map_err(|error| McpRuntimeError::Discovery(error.to_string()))?;
            adapters.push((
                public_name,
                Arc::new(McpTool {
                    server_id: server.id.clone(),
                    raw_name,
                    name: format!("mcp__{}__{}", server.id, tool.name),
                    description,
                    input_schema,
                    peer: peer.clone(),
                }) as Arc<dyn Tool>,
            ));
        }

        if self.shutdown.is_cancelled() {
            if let Err(error) = service.close().await {
                warn!(server = %server.id, error = %error, "MCP server close failed");
            }
            return Err(McpRuntimeError::Cancelled);
        }

        let tool_names = adapters
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        for (_, tool) in adapters {
            self.tool_registry.register_dynamic(tool).await;
        }
        Ok(McpConnection {
            service,
            tool_names,
        })
    }

    fn send_status(&self, server_id: &str, status: String) {
        if let Some(sender) = &self.status_tx {
            let _ = sender.send((server_id.to_string(), status));
        }
    }
}

// Keep generated names within the common OpenAI-compatible function-name contract.
fn is_provider_safe_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
}

struct McpTool {
    server_id: String,
    raw_name: String,
    name: String,
    description: String,
    input_schema: serde_json::Value,
    peer: Peer<RoleClient>,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        self.input_schema.clone()
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        tracing::trace!(server = %self.server_id, tool = %self.raw_name, "calling MCP tool");
        let arguments = match args {
            serde_json::Value::Object(arguments) => Some(arguments),
            serde_json::Value::Null => None,
            _ => {
                return Err(ToolError::InvalidArgs(
                    "MCP tool arguments must be a JSON object".into(),
                ));
            }
        };
        let mut params = CallToolRequestParams::new(Cow::Owned(self.raw_name.clone()));
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        let result = self
            .peer
            .call_tool(params)
            .await
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        let content = render_tool_result(&result);
        if result.is_error == Some(true) {
            Ok(ToolResult {
                success: false,
                content: content.clone(),
                error: Some(content),
            })
        } else {
            Ok(ToolResult {
                success: true,
                content,
                error: None,
            })
        }
    }
}

fn render_tool_result(result: &rmcp::model::CallToolResult) -> String {
    let text = result
        .content
        .iter()
        .filter_map(ContentBlock::as_text)
        .map(|content| content.text.as_str())
        .collect::<Vec<_>>();
    if !text.is_empty() {
        return text.join("\n");
    }
    if let Some(structured) = &result.structured_content {
        return structured.to_string();
    }
    serde_json::to_string(&result.content).unwrap_or_else(|_| "[]".into())
}

fn expand_environment_value(server: &str, value: &str) -> Result<String, McpRuntimeError> {
    let mut result = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        result.push_str(&rest[..start]);
        let variable = &rest[start + 2..];
        let Some(end) = variable.find('}') else {
            return Err(McpRuntimeError::InvalidEnvironmentPlaceholder(
                server.to_string(),
            ));
        };
        let name = &variable[..end];
        let replacement = std::env::var(name).map_err(|_| McpRuntimeError::MissingEnvironment {
            server: server.to_string(),
            name: name.to_string(),
        })?;
        result.push_str(&replacement);
        rest = &variable[end + 1..];
    }
    result.push_str(rest);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::is_provider_safe_tool_name;

    #[test]
    fn provider_safe_tool_names_reject_invalid_characters_and_length() {
        assert!(is_provider_safe_tool_name("mcp__server__echo"));
        assert!(!is_provider_safe_tool_name("mcp__server__bad.name"));
        assert!(!is_provider_safe_tool_name(&"a".repeat(65)));
    }
}
