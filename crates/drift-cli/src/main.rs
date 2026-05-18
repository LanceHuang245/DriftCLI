use clap::{Parser, Subcommand};
use drift_config::AppConfig;
use drift_core::{Agent, EventMsg};
use drift_tui::{AppEvent, TuiApp, TuiCommand};
use std::env;
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(name = "drift", version, about = "High-performance terminal AI coding agent")]
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

#[derive(Subcommand)]
enum Command {
    Init,
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
            let mut config = AppConfig::load(cli.model.as_deref(), cli.api_key.as_deref())?;

            let cwd = env::current_dir()?;
            let _ = config.apply_project_override(&cwd);

            let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<TuiCommand>();

            let mut agent = Agent::new(config.clone(), cwd.clone())?;
            let event_tx = agent.event_sender();
            let llm_config = config.llm.clone();

            let (tui_tx, tui_rx) = mpsc::unbounded_channel();
            let mut core_rx = agent.subscribe();
            tokio::spawn(async move {
                loop {
                    match core_rx.recv().await {
                        Ok(EventMsg::Token(text)) => {
                            let _ = tui_tx.send(AppEvent::Token(text));
                        }
                        Ok(EventMsg::Reasoning(text)) => {
                            let _ = tui_tx.send(AppEvent::Reasoning(text));
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
                        _ => {}
                    }
                }
            });

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
                    }
                }
            });

            let mut tui = TuiApp::new(&llm_config, tui_rx, cmd_tx);
            tui.run()?;
        }
    }

    Ok(())
}
