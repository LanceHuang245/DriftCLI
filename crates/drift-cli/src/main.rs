mod bridge;
mod command;

use clap::{Parser, Subcommand};
use drift_config::AppConfig;
use drift_core::Agent;
use drift_mcp::McpManager;
use drift_tools::{
    ToolRegistry,
    tools::{
        bash::BashTool, edit::EditTool, glob::GlobTool, grep::GrepTool, read::ReadTool,
        todowrite::TodoWriteTool, webfetch::WebFetchTool, websearch::WebSearchTool,
        write::WriteTool,
    },
};
use drift_tui::{AppEvent, TuiApp, TuiCommand};
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

            let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<TuiCommand>();

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
            let core_rx = agent.subscribe();
            let agent = Arc::new(tokio::sync::Mutex::new(agent));
            // Event bridge task: subscribes to core Agent events and forwards them to the TUI via an mpsc channel.
            let _event_bridge_task = bridge::spawn_event_bridge(core_rx, tui_tx);
            let _command_task =
                command::spawn_command_loop(cmd_rx, agent, event_tx, session_store, perm_tx);

            // Start the TUI app on the main thread — blocks until the user exits.
            let mut tui = TuiApp::new(&llm_config, tui_rx, cmd_tx);
            tui.set_provider_name(config.active_provider.clone());
            if !history_events.is_empty() {
                tui.set_messages(bridge::translate_events_to_chat_messages(&history_events));
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
