use clap::{Parser, Subcommand};
use drift_config::AppConfig;
use drift_core::{Agent, EventMsg};
use drift_mcp::McpManager;
use drift_security::types::PermissionResponse;
use drift_tools::{
    ToolRegistry,
    tools::{
        bash::BashTool, edit::EditTool, glob::GlobTool, grep::GrepTool, read::ReadTool,
        todowrite::TodoWriteTool, webfetch::WebFetchTool, websearch::WebSearchTool,
        write::WriteTool,
    },
};
use drift_tui::{AppEvent, ChatMessage, TuiApp, TuiCommand};
use std::env;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

// Cli: top-level argument struct parsed by clap — model, api_key, subcommands, and runtime options.
#[derive(Parser)]
#[command(
    name = "drift",
    version,
    about = "High-performance terminal AI coding agent"
)]
struct Cli {
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    config: Option<String>,
    /// Security profile: "default", "auto", "readonly", "danger"
    #[arg(long = "security-profile", short = 'P')]
    security_profile: Option<String>,
    #[arg(long, value_name = "MODE")]
    permission_mode: Option<String>,
    #[arg(long, default_value = "info")]
    log_level: String,
    #[arg(long)]
    session: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

// Command: subcommands for one-shot operations (init project config, show connection info).
#[derive(Subcommand)]
enum Command {
    Init,
    Config,
}

// Helper function to translate persistent storage SessionEvents into TUI ChatMessages
fn translate_events_to_chat_messages(events: &[drift_storage::SessionEvent]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    for event in events {
        match event {
            drift_storage::SessionEvent::Message {
                role,
                content,
                reasoning,
            } => {
                messages.push(ChatMessage {
                    role: role.clone(),
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    thinking: false,
                    reasoning_duration_ms: None,
                    reasoning_collapsed: true,
                    thinking_tools: Vec::new(),
                });
            }
            drift_storage::SessionEvent::ToolCall {
                call_id: _,
                name,
                args,
            } => {
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" {
                        last.thinking_tools
                            .push(format!("> Calling tool: {} with {}", name, args));
                    }
                }
            }
            drift_storage::SessionEvent::ToolResult {
                call_id: _,
                name,
                success,
                content: _,
                error,
            } => {
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" {
                        last.thinking_tools.push(format!(
                            "> Tool {} {}",
                            name,
                            if *success { "completed" } else { "failed" }
                        ));
                        if let Some(err) = error {
                            last.thinking_tools.push(format!("  Error: {}", err));
                        }
                    }
                }
            }
            // Compaction snapshots are internal context state, not chat messages.
            drift_storage::SessionEvent::ContextCompacted { .. } => {}
        }
    }

    messages
}

// Main entry point: parses CLI, loads config, and either runs a subcommand or boots the TUI with an event bridge.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up tracing subscriber — writes to stderr (visible after TUI exit).
    // Override with RUST_LOG env var (e.g. RUST_LOG=warn,drift_core=debug,drift_llm=debug).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    match &cli.command {
        // `drift init` — create a .drift/config.toml in the current directory.
        Some(Command::Init) => {
            let cwd = env::current_dir()?;
            let path = AppConfig::init_project(&cwd)?;
            println!("Created project config: {}", path.display());
            println!("Edit this file to set your LLM provider and API key.");
            return Ok(());
        }
        // `drift config` — load config and print the current connection summary.
        Some(Command::Config) => {
            let cwd = env::current_dir()?;
            let explicit_config = cli.config.as_deref().map(Path::new);
            let config = AppConfig::load_for_workspace(
                &cwd,
                explicit_config,
                cli.model.as_deref(),
                cli.api_key.as_deref(),
            )?;
            println!("{}", config.connection_summary());
            return Ok(());
        }
        // Default: interactive mode — load config, start the agent, bridge events to the TUI, and run.
        None => {
            let cwd = env::current_dir()?;
            let explicit_config = cli.config.as_deref().map(Path::new);
            let mut config = AppConfig::load_for_workspace(
                &cwd,
                explicit_config,
                cli.model.as_deref(),
                cli.api_key.as_deref(),
            )?;

            let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<TuiCommand>();

            // Build tool registry with all built-in tools
            let mut tool_registry = ToolRegistry::new();
            tool_registry.register_builtin(Arc::new(BashTool));
            tool_registry.register_builtin(Arc::new(ReadTool));
            tool_registry.register_builtin(Arc::new(WriteTool));
            tool_registry.register_builtin(Arc::new(EditTool));
            tool_registry.register_builtin(Arc::new(GrepTool));
            tool_registry.register_builtin(Arc::new(GlobTool));
            tool_registry.register_builtin(Arc::new(WebFetchTool));
            tool_registry.register_builtin(Arc::new(WebSearchTool));
            tool_registry.register_builtin(Arc::new(TodoWriteTool));
            let tool_registry = Arc::new(tool_registry);

            // Establish Persistence Paths & Handle Boot Bootstrapping
            let drift_dir = AppConfig::global_config_dir()
                .unwrap_or_else(|| std::path::PathBuf::from(".drift"));
            let session_store = std::sync::Arc::new(drift_storage::SessionStore::new(drift_dir)?);

            let (session_id, history_events) = if let Some(ref s_str) = cli.session {
                if let Ok(parsed_id) = uuid::Uuid::parse_str(s_str) {
                    if let Ok(events) = session_store.read_events(parsed_id) {
                        (parsed_id, events)
                    } else {
                        let (new_id, _) =
                            session_store.create(&cwd.to_string_lossy(), &config.agent.model)?;
                        (new_id, Vec::new())
                    }
                } else {
                    let (new_id, _) =
                        session_store.create(&cwd.to_string_lossy(), &config.agent.model)?;
                    (new_id, Vec::new())
                }
            } else {
                let (new_id, _) =
                    session_store.create(&cwd.to_string_lossy(), &config.agent.model)?;
                (new_id, Vec::new())
            };

            // Use CLI flag profile if set; otherwise default
            let profile_name = cli
                .security_profile
                .as_deref()
                .unwrap_or(&config.security.default_profile)
                .to_string();
            if let Some(mode) = cli.permission_mode.as_deref() {
                config.apply_permission_mode(&profile_name, mode)?;
            }
            let security_cfg = config.security.clone();
            let mut agent = Agent::new(
                config.clone(),
                cwd.clone(),
                tool_registry.clone(),
                session_id,
                session_store.clone(),
                &security_cfg,
                &profile_name,
            )?;

            // Set up permission response channel (TUI → Agent)
            let (perm_tx, perm_rx) = mpsc::unbounded_channel();
            agent.set_permission_channel(perm_rx);
            if !history_events.is_empty() {
                agent.reconstruct_history(&history_events);
            }

            let (mcp_status_tx, mcp_status_rx) = mpsc::unbounded_channel();
            let process_sandbox = agent.process_sandbox();
            let mcp_manager = Arc::new(McpManager::with_status_sender(
                config.mcp.clone(),
                tool_registry,
                mcp_status_tx,
                process_sandbox,
            ));
            let mcp_start_task = tokio::spawn(mcp_manager.clone().start_auto_servers());

            let event_tx = agent.event_sender();
            let llm_config = config.active_llm_config().cloned().unwrap();

            let (tui_tx, tui_rx) = mpsc::unbounded_channel();
            let mcp_tui_tx = tui_tx.clone();
            let mcp_bridge_task = tokio::spawn(async move {
                let mut status_rx = mcp_status_rx;
                while let Some((server_id, status)) = status_rx.recv().await {
                    let _ = mcp_tui_tx.send(AppEvent::McpStatus { server_id, status });
                }
            });
            let mut core_rx = agent.subscribe();
            let agent = Arc::new(tokio::sync::Mutex::new(agent));
            // Event bridge task: subscribes to core Agent events and forwards them to the TUI via an mpsc channel.
            tokio::spawn(async move {
                loop {
                    match core_rx.recv().await {
                        Ok(EventMsg::Token(text)) => {
                            let _ = tui_tx.send(AppEvent::Token(text));
                        }
                        Ok(EventMsg::Reasoning(text)) => {
                            let _ = tui_tx.send(AppEvent::Reasoning(text));
                        }
                        Ok(EventMsg::ReasoningComplete { duration_ms }) => {
                            let _ = tui_tx.send(AppEvent::ReasoningComplete { duration_ms });
                        }
                        Ok(EventMsg::AgentState(state)) => {
                            let status = match state {
                                drift_core::AgentState::Idle => "Idle",
                                drift_core::AgentState::Thinking => "Thinking...",
                                drift_core::AgentState::Generating(_) => "Generating...",
                                drift_core::AgentState::Error(e) => {
                                    let _ = tui_tx.send(AppEvent::Error(e.clone()));
                                    "Error"
                                }
                            };
                            let _ = tui_tx.send(AppEvent::AgentStatus(status.to_string()));
                        }
                        Ok(EventMsg::Error { message, .. }) => {
                            let _ = tui_tx.send(AppEvent::Error(message));
                        }
                        Ok(EventMsg::Done) => {
                            let _ = tui_tx.send(AppEvent::Done);
                        }
                        Ok(EventMsg::Interrupted) => {
                            let _ = tui_tx.send(AppEvent::Interrupted);
                        }
                        Ok(EventMsg::ModelList(models)) => {
                            let _ = tui_tx.send(AppEvent::ModelList(models));
                        }
                        Ok(EventMsg::ProviderList(names)) => {
                            let _ = tui_tx.send(AppEvent::ProviderList(names));
                        }
                        Ok(EventMsg::ProviderSwitched { name, model }) => {
                            let _ = tui_tx.send(AppEvent::ProviderSwitched { name, model });
                        }
                        Ok(EventMsg::ProviderConfig { name, config }) => {
                            let _ = tui_tx.send(AppEvent::ProviderConfig { name, config });
                        }
                        Ok(EventMsg::ContextCompacting) => {
                            let _ = tui_tx
                                .send(AppEvent::AgentStatus("Compacting context...".to_string()));
                        }
                        Ok(EventMsg::ContextCompacted { saved_tokens, .. }) => {
                            let _ = tui_tx.send(AppEvent::AgentStatus(format!(
                                "Context compacted (-{} tokens)",
                                saved_tokens
                            )));
                        }
                        Ok(EventMsg::ToolCallStart { id, name }) => {
                            let _ = tui_tx.send(AppEvent::ToolCallStart { id, name });
                        }
                        Ok(EventMsg::ToolCallArgs { id, delta }) => {
                            let _ = tui_tx.send(AppEvent::ToolCallArgs { id, delta });
                        }
                        Ok(EventMsg::ToolCallEnd { id }) => {
                            let _ = tui_tx.send(AppEvent::ToolCallEnd { id });
                        }
                        Ok(EventMsg::ToolExecStart { id, name }) => {
                            let _ = tui_tx.send(AppEvent::ToolExecStart { id, name });
                        }
                        Ok(EventMsg::ToolExecEnd {
                            id, name, success, ..
                        }) => {
                            let _ = tui_tx.send(AppEvent::ToolExecEnd { id, name, success });
                        }
                        Ok(EventMsg::SessionList(meta_list)) => {
                            let _ = tui_tx.send(AppEvent::SessionList(meta_list));
                        }
                        Ok(EventMsg::SessionLoaded { session_id, events }) => {
                            let _ = tui_tx.send(AppEvent::SessionLoaded {
                                session_id,
                                messages: translate_events_to_chat_messages(&events),
                            });
                        }
                        Ok(EventMsg::PermissionRequest {
                            request_id,
                            tool_name,
                            args_summary,
                            reason,
                            ..
                        }) => {
                            let _ = tui_tx.send(AppEvent::PermissionRequest {
                                request_id,
                                tool_name,
                                args_summary,
                                reason,
                            });
                        }
                        Ok(EventMsg::PermissionResolved {
                            request_id,
                            allowed,
                        }) => {
                            let _ = tui_tx.send(AppEvent::PermissionResolved {
                                request_id,
                                allowed,
                            });
                        }
                        _ => {}
                    }
                }
            });

            // Command handling task: receives TUI commands (chat, fetch models, reconfigure, provider management) and dispatches to the agent.
            tokio::spawn(async move {
                let mut active_task: Option<tokio::task::JoinHandle<()>> = None;

                while let Some(cmd) = cmd_rx.recv().await {
                    match cmd {
                        TuiCommand::Chat(msg) => {
                            // A new input replaces an active turn, matching terminal Agent behavior.
                            if let Some(handle) = active_task.take() {
                                if handle.is_finished() {
                                    let _ = handle.await;
                                } else {
                                    handle.abort();
                                    let _ = handle.await;
                                    let _ = event_tx.send(EventMsg::Interrupted);
                                }
                            }
                            let agent = Arc::clone(&agent);
                            active_task = Some(tokio::spawn(async move {
                                let mut agent = agent.lock().await;
                                agent.submit(msg).await;
                            }));
                        }
                        TuiCommand::Interrupt => {
                            // Aborting the submit task drops the active LLM/tool future immediately.
                            if let Some(handle) = active_task.take() {
                                if !handle.is_finished() {
                                    handle.abort();
                                    let _ = handle.await;
                                    let _ = event_tx.send(EventMsg::Interrupted);
                                } else {
                                    let _ = handle.await;
                                }
                            }
                        }
                        TuiCommand::FetchModels {
                            provider,
                            base_url,
                            api_key,
                        } => match Agent::fetch_models(&provider, &base_url, &api_key).await {
                            Ok(models) => {
                                let _ = event_tx.send(EventMsg::ModelList(models));
                            }
                            Err(e) => {
                                let _ = event_tx.send(EventMsg::Error {
                                    message: format!("Failed to fetch models: {}", e),
                                    recoverable: true,
                                });
                            }
                        },
                        TuiCommand::Reconfigure(config) => {
                            let result = {
                                let mut agent = agent.lock().await;
                                agent.reconfigure(config).await
                            };
                            match result {
                                Ok(()) => {
                                    let _ = event_tx
                                        .send(EventMsg::AgentState(drift_core::AgentState::Idle));
                                }
                                Err(e) => {
                                    let _ = event_tx.send(EventMsg::Error {
                                        message: format!("Reconfiguration failed: {}", e),
                                        recoverable: true,
                                    });
                                }
                            }
                        }
                        TuiCommand::SaveProvider { name, config } => {
                            let result = {
                                let mut agent = agent.lock().await;
                                agent.save_provider(name, config).await
                            };
                            match result {
                                Ok(()) => {
                                    let _ = event_tx
                                        .send(EventMsg::AgentState(drift_core::AgentState::Idle));
                                }
                                Err(e) => {
                                    let _ = event_tx.send(EventMsg::Error {
                                        message: format!("Save failed: {}", e),
                                        recoverable: true,
                                    });
                                }
                            }
                        }
                        TuiCommand::SetActiveProvider(name) => {
                            let result = {
                                let mut agent = agent.lock().await;
                                agent.activate_provider(&name).await
                            };
                            match result {
                                Ok(()) => {
                                    let _ = event_tx
                                        .send(EventMsg::AgentState(drift_core::AgentState::Idle));
                                }
                                Err(e) => {
                                    let _ = event_tx.send(EventMsg::Error {
                                        message: format!("Switch failed: {}", e),
                                        recoverable: true,
                                    });
                                }
                            }
                        }
                        TuiCommand::GetProviders => {
                            let names = agent.lock().await.list_providers();
                            let _ = event_tx.send(EventMsg::ProviderList(names));
                        }
                        TuiCommand::DeleteProvider(name) => {
                            let result = {
                                let mut agent = agent.lock().await;
                                agent.remove_provider(&name).await
                            };
                            match result {
                                Ok(()) => {
                                    let names = agent.lock().await.list_providers();
                                    let _ = event_tx.send(EventMsg::ProviderList(names));
                                    let _ = event_tx
                                        .send(EventMsg::AgentState(drift_core::AgentState::Idle));
                                }
                                Err(e) => {
                                    let _ = event_tx.send(EventMsg::Error {
                                        message: format!("Delete failed: {}", e),
                                        recoverable: true,
                                    });
                                }
                            }
                        }
                        TuiCommand::GetProviderConfig(name) => {
                            match agent.lock().await.get_provider_config(&name) {
                                Some(config) => {
                                    let _ =
                                        event_tx.send(EventMsg::ProviderConfig { name, config });
                                }
                                None => {
                                    let _ = event_tx.send(EventMsg::Error {
                                        message: format!("Provider '{}' not found", name),
                                        recoverable: true,
                                    });
                                }
                            }
                        }
                        TuiCommand::GetSessions => {
                            if let Ok(meta_list) = session_store.list() {
                                let _ = event_tx.send(EventMsg::SessionList(meta_list));
                            }
                        }
                        TuiCommand::SwitchSession(target_id) => {
                            if let Ok(events) = session_store.read_events(target_id) {
                                agent.lock().await.switch_session(target_id, &events);
                                let _ = event_tx.send(EventMsg::SessionLoaded {
                                    session_id: target_id,
                                    events,
                                });
                            } else {
                                let _ = event_tx.send(EventMsg::Error {
                                    message: format!("Failed to read session {}", target_id),
                                    recoverable: true,
                                });
                            }
                        }
                        TuiCommand::PermissionResponse {
                            request_id,
                            allowed,
                            remember,
                        } => {
                            // Map the TUI decision to a PermissionResponse variant.
                            let resp = match (allowed, remember) {
                                (true, false) => PermissionResponse::Allow,
                                (true, true) => PermissionResponse::AllowAlways,
                                (false, false) => PermissionResponse::Deny,
                                (false, true) => PermissionResponse::DenyAlways,
                            };
                            // Correlate the decision with the exact request shown by the TUI.
                            let _ = perm_tx.send((request_id.clone(), resp));
                            // Mark the request as resolved for TUI display.
                            let _ = event_tx.send(EventMsg::PermissionResolved {
                                request_id,
                                allowed,
                            });
                        }
                    }
                }

                if let Some(handle) = active_task {
                    handle.abort();
                    let _ = handle.await;
                }
            });

            // Start the TUI app on the main thread — blocks until the user exits.
            let mut tui = TuiApp::new(&llm_config, tui_rx, cmd_tx);
            tui.set_provider_name(config.active_provider.clone());
            if !history_events.is_empty() {
                tui.set_messages(translate_events_to_chat_messages(&history_events));
            }
            tui.set_session_id(session_id);
            let tui_result = tui.run();
            mcp_manager.shutdown().await;
            let _ = mcp_start_task.await;
            drop(mcp_manager);
            let _ = mcp_bridge_task.await;
            tui_result?;
        }
    }

    Ok(())
}
