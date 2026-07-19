use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::context::ContextManager;
use crate::event::{AgentState, EventMsg};
use drift_config::{AppConfig, LlmConfig};
use drift_llm::{
    ContentPart, LlmChunk, LlmError, LlmMessage, LlmProvider, ModelInfo, create_provider,
    fetch_anthropic_models, fetch_openai_compat_models,
};
use drift_security::{PermissionDecision, PermissionEngine, ProcessSandbox, SecurityConfig};
use drift_tools::{ToolContext, ToolRegistry};
use tokio::sync::broadcast;
use tracing::{info, warn};

mod history;
mod provider;
mod turn;

use history::{redact_session_event, replay_history};

// Agent coordinates provider, tools, permissions, context, and session persistence.
pub struct Agent {
    config: AppConfig,
    llm: Box<dyn LlmProvider>,
    tool_registry: std::sync::Arc<ToolRegistry>,
    /// Permission engine for tool call approval.
    permission_engine: PermissionEngine,
    /// Channel the bridge task writes user permission responses into.
    permission_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<(String, drift_security::PermissionResponse)>>,
    event_tx: broadcast::Sender<EventMsg>,
    context: ContextManager,
    cwd: PathBuf,
    session_id: uuid::Uuid,
    session_store: std::sync::Arc<drift_storage::SessionStore>,
    file_access: std::sync::Arc<drift_security::FileAccessGuard>,
    network: std::sync::Arc<drift_security::NetworkGuard>,
    process_sandbox: std::sync::Arc<ProcessSandbox>,
}

impl Agent {
    // Create a new agent: builds the LLM provider, tool registry, and permission engine.
    pub fn new(
        config: AppConfig,
        cwd: PathBuf,
        tool_registry: std::sync::Arc<ToolRegistry>,
        session_id: uuid::Uuid,
        session_store: std::sync::Arc<drift_storage::SessionStore>,
        security_config: &SecurityConfig,
        security_profile: &str,
    ) -> Result<Self, LlmError> {
        let llm = create_provider(config.active_llm_config().unwrap())?;
        let (event_tx, _) = broadcast::channel(256);
        let permission_engine = PermissionEngine::new(security_config, security_profile);
        let file_access = std::sync::Arc::new(
            permission_engine
                .file_access_guard(&cwd)
                .map_err(|error| LlmError::Config(format!("file access guard: {:?}", error)))?,
        );
        let network = std::sync::Arc::new(permission_engine.network_guard());
        let process_sandbox = std::sync::Arc::new(
            ProcessSandbox::new(permission_engine.sandbox_mode(), &cwd)
                .map_err(|error| LlmError::Config(format!("process sandbox: {error}")))?,
        );
        let context = ContextManager::for_workspace(
            llm.context_window(),
            config.agent.compaction_threshold,
            config.agent.compaction_target,
            &cwd,
        );

        info!(
            provider = %llm.provider_id(),
            model = %llm.model_name(),
            security_profile = permission_engine.profile_name(),
            approval_policy = ?permission_engine.approval_policy(),
            sandbox_mode = ?permission_engine.sandbox_mode(),
            "Agent created"
        );

        Ok(Self {
            config,
            llm,
            tool_registry,
            event_tx,
            context,
            cwd,
            session_id,
            session_store,
            file_access,
            network,
            process_sandbox,
            permission_engine,
            permission_rx: None,
        })
    }

    // Set up the channel through which the TUI bridge sends permission responses back to the agent loop.
    pub fn set_permission_channel(
        &mut self,
        rx: tokio::sync::mpsc::UnboundedReceiver<(String, drift_security::PermissionResponse)>,
    ) {
        self.permission_rx = Some(rx);
    }

    // Set the full conversation history from a reconstructed LlmMessage list (used when resuming a session).
    pub fn set_messages(&mut self, messages: Vec<LlmMessage>) {
        self.context.set_messages(messages);
    }

    // Retrieve active session ID.
    pub fn session_id(&self) -> uuid::Uuid {
        self.session_id
    }

    /// Share the immutable process boundary with MCP child-process management.
    pub fn process_sandbox(&self) -> std::sync::Arc<ProcessSandbox> {
        self.process_sandbox.clone()
    }

    // Switch the active session and rebuild the LLM history from its transcript.
    pub fn switch_session(
        &mut self,
        session_id: uuid::Uuid,
        events: &[drift_storage::SessionEvent],
    ) {
        self.session_id = session_id;
        self.reconstruct_history(events);
    }

    // Reconstruct the core messages history list from a vector of storage events.
    pub fn reconstruct_history(&mut self, events: &[drift_storage::SessionEvent]) {
        let (messages, summary) = replay_history(events);
        self.context.set_compacted_state(messages, summary);
    }

    // Subscribe returns a new broadcast receiver for consuming agent events in the TUI bridge.
    pub fn subscribe(&self) -> broadcast::Receiver<EventMsg> {
        self.event_tx.subscribe()
    }

    /// Redact all transcript payloads immediately before writing them to disk.
    fn append_session_event(
        &self,
        mut event: drift_storage::SessionEvent,
    ) -> Result<(), drift_storage::StorageError> {
        redact_session_event(&mut event)?;
        self.session_store.append_event(self.session_id, &event)
    }
}
#[cfg(test)]
#[path = "agent/tests.rs"]
mod tests;
