use drift_core::{Agent, EventMsg};
use drift_security::types::PermissionResponse;
use drift_tui::TuiCommand;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, mpsc};

// Spawns the command adapter that dispatches TUI intent to core services.
pub(crate) fn spawn_command_loop(
    mut cmd_rx: mpsc::UnboundedReceiver<TuiCommand>,
    agent: Arc<Mutex<Agent>>,
    event_tx: broadcast::Sender<EventMsg>,
    session_store: Arc<drift_storage::SessionStore>,
    perm_tx: mpsc::UnboundedSender<(String, PermissionResponse)>,
) -> tokio::task::JoinHandle<()> {
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
                            let _ =
                                event_tx.send(EventMsg::AgentState(drift_core::AgentState::Idle));
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
                            let _ =
                                event_tx.send(EventMsg::AgentState(drift_core::AgentState::Idle));
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
                            let _ =
                                event_tx.send(EventMsg::AgentState(drift_core::AgentState::Idle));
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
                            let _ =
                                event_tx.send(EventMsg::AgentState(drift_core::AgentState::Idle));
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
                            let _ = event_tx.send(EventMsg::ProviderConfig { name, config });
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
    })
}
