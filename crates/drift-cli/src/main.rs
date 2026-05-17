use clap::{Parser, Subcommand};
use drift_config::AppConfig;
use drift_core::Agent;
use drift_tui::TuiApp;
use std::env;
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(name = "drift", version, about = "High-performance terminal AI coding agent")]
struct Cli {
    /// Override the model
    #[arg(long)]
    model: Option<String>,
    /// Override API key
    #[arg(long)]
    api_key: Option<String>,
    /// Custom config file path
    #[arg(long)]
    config: Option<String>,
    /// Permission mode: deny | ask | allow
    #[arg(long, default_value = "ask")]
    permission_mode: String,
    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a default config file
    Init,
    /// Show current configuration
    Config,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Command::Init) => {
            let cwd = env::current_dir()?;
            let path = AppConfig::init_project(&cwd)?;
            println!("Created project config: {}", path.display());
            println!("Edit this file to set your LLM provider and API key.");
            return Ok(());
        }
        Some(Command::Config) => {
            let config = AppConfig::load(cli.model.as_deref(), cli.api_key.as_deref())?;
            println!("{}", config.connection_summary());
            return Ok(());
        }
        None => {
            // --- TUI Mode ---
            let mut config = AppConfig::load(cli.model.as_deref(), cli.api_key.as_deref())?;

            // Apply project-level config override (current directory)
            let cwd = env::current_dir()?;
            let _ = config.apply_project_override(&cwd);

            // Create channels for bidirectional communication
            let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<String>();

            // Create the Agent
            let mut agent = Agent::new(config)?;
            let connection_summary = agent.connection_summary();

            // Bridge: core EventMsg -> tui AppEvent
            let (tui_tx, tui_rx) = mpsc::unbounded_channel();
            let mut core_rx = agent.subscribe();
            tokio::spawn(async move {
                loop {
                    match core_rx.recv().await {
                        Ok(drift_core::EventMsg::Token(text)) => {
                            let _ = tui_tx.send(drift_tui::AppEvent::Token(text));
                        }
                        Ok(drift_core::EventMsg::AgentState(state)) => {
                            let status = match state {
                                drift_core::AgentState::Idle => "Idle",
                                drift_core::AgentState::Thinking => "Thinking...",
                                drift_core::AgentState::Generating(_) => "Generating...",
                                drift_core::AgentState::Error(e) => {
                                    let _ = tui_tx.send(drift_tui::AppEvent::Error(e.clone()));
                                    "Error"
                                }
                            };
                            let _ = tui_tx.send(drift_tui::AppEvent::AgentStatus(
                                status.to_string(),
                            ));
                        }
                        Ok(drift_core::EventMsg::Error { message, .. }) => {
                            let _ = tui_tx.send(drift_tui::AppEvent::Error(message));
                        }
                        Ok(drift_core::EventMsg::Done) => {
                            let _ = tui_tx.send(drift_tui::AppEvent::Done);
                        }
                        _ => {}
                    }
                }
            });

            // Spawn task to forward TUI commands to Agent
            let agent_handle = tokio::spawn(async move {
                while let Some(msg) = cmd_rx.recv().await {
                    agent.submit(msg).await;
                }
            });

            // Create and run TUI (blocking on main thread)
            let mut tui = TuiApp::new(connection_summary, tui_rx, cmd_tx);
            tui.run()?;

            // Cleanup
            agent_handle.abort();
        }
    }

    Ok(())
}
