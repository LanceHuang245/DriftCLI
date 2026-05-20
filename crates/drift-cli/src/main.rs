use clap::{Parser, Subcommand};
use drift_config::AppConfig;
use drift_core::{Agent, EventMsg};
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
    #[arg(long, default_value = "ask")]
    permission_mode: String,
    #[arg(long, default_value = "info")]
    log_level: String,
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
            let config = AppConfig::load(cli.model.as_deref(), cli.api_key.as_deref())?;
            println!("{}", config.connection_summary());
            return Ok(());
        }
        // Default: interactive mode — load config, start the agent, bridge events to the TUI, and run.
        None => {
            let mut config = AppConfig::load(cli.model.as_deref(), cli.api_key.as_deref())?;

            let cwd = env::current_dir()?;
            let _ = config.apply_project_override(&cwd);

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

            let mut agent = Agent::new(config.clone(), cwd.clone(), tool_registry)?;
            let event_tx = agent.event_sender();
            let llm_config = config.active_llm_config().cloned().unwrap();

            let (tui_tx, tui_rx) = mpsc::unbounded_channel();
            let mut core_rx = agent.subscribe();
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
                        _ => {}
                    }
                }
            });

            // Command handling task: receives TUI commands (chat, fetch models, reconfigure, provider management) and dispatches to the agent.
            tokio::spawn(async move {
                while let Some(cmd) = cmd_rx.recv().await {
                    match cmd {
                        TuiCommand::Chat(msg) => {
                            agent.submit(msg).await;
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
                        TuiCommand::Reconfigure(config) => match agent.reconfigure(config).await {
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
                        },
                        TuiCommand::SaveProvider { name, config } => {
                            match agent.save_provider(name, config).await {
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
                            match agent.activate_provider(&name).await {
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
                            let names = agent.list_providers();
                            let _ = event_tx.send(EventMsg::ProviderList(names));
                        }
                        TuiCommand::DeleteProvider(name) => {
                            match agent.remove_provider(&name).await {
                                Ok(()) => {
                                    let names = agent.list_providers();
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
                            match agent.get_provider_config(&name) {
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
                    }
                }
            });

            // Start the TUI app on the main thread — blocks until the user exits.
            let mut tui = TuiApp::new(&llm_config, tui_rx, cmd_tx);
            tui.run()?;
        }
    }

    Ok(())
}
