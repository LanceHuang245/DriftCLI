use super::*;

impl TuiApp {
    pub(super) fn handle_command(&mut self, cmd: &str) {
        let cmd = cmd.trim();
        match cmd {
            // Enter connection settings form.
            "/connect" => {
                self.connect_form = connect::ConnectForm::new();
                self.mode = TuiMode::ConnectSettings;
            }
            // Open the variant / reasoning-effort picker for the current model.
            "/variant" => {
                let model = &self.connect_form.model;
                let from_caps: Vec<String> = self
                    .connect_form
                    .model_list
                    .iter()
                    .find(|m| m.id == *model)
                    .map(|m| m.effort_levels.clone())
                    .unwrap_or_default();
                let mut efforts = vec!["(none)".to_string()];
                if from_caps.is_empty() {
                    efforts.extend(
                        ["low", "medium", "high", "max"]
                            .iter()
                            .map(|s| s.to_string()),
                    );
                } else {
                    efforts.extend(from_caps);
                }
                let current = self.connect_form.reasoning_effort.clone();
                let selected = if current.is_none() {
                    0
                } else {
                    efforts
                        .iter()
                        .position(|e| Some(e) == current.as_ref())
                        .unwrap_or(0)
                };
                self.variant_options = efforts;
                self.variant_selected = selected;
                self.mode = TuiMode::VariantPicker;
            }
            // Open the provider picker overlay.
            "/provider" => {
                let _ = self.cmd_tx.send(TuiCommand::GetProviders);
            }
            "/sessions" | "/history" => {
                let _ = self.cmd_tx.send(TuiCommand::GetSessions);
            }
            // Quit the application.
            "/quit" | "/exit" => {
                self.should_quit = true;
            }
            // Clear both committed history and every transient turn accumulator.
            "/clear" => {
                self.messages.clear();
                self.current_response.clear();
                self.current_reasoning.clear();
                self.reasoning_start_time = None;
                self.current_reasoning_collapsed = true;
            }
            // Unknown command — show a help message.
            _ => {
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!(
                        "Unknown command: {}. Try /connect, /provider, /sessions, /clear, /quit",
                        cmd
                    ),
                    reasoning: None,
                    reasoning_duration_ms: None,
                    reasoning_collapsed: false,
                });
            }
        }
        self.input_buffer.clear();
        self.cursor_position = 0;
    }

    // Update slash completion popup based on current input buffer.
    pub(super) fn update_slash_completion(&mut self) {
        if self.input_buffer.starts_with('/') {
            let filter = &self.input_buffer[1..];
            let filtered = filter_commands(filter);
            if filtered.is_empty() {
                self.slash_completion = None;
            } else if let Some(ref mut state) = self.slash_completion {
                if state.selected >= filtered.len() {
                    state.selected = filtered.len().saturating_sub(1);
                }
                state.filtered = filtered;
            } else {
                self.slash_completion = Some(SlashCompletionState {
                    filtered,
                    selected: 0,
                });
            }
        } else {
            self.slash_completion = None;
        }
    }

    // Handle key presses while the connect settings form is active.
    pub(super) fn handle_connect_key(&mut self, key: KeyCode) {
        match key {
            // Esc returns to normal chat mode.
            KeyCode::Esc => {
                self.mode = TuiMode::Normal;
            }
            // Tab moves to the next form field.
            KeyCode::Tab => {
                self.connect_form.next_field();
            }
            // Up: scroll model list up, or move to previous form field.
            KeyCode::Up => {
                if self.connect_form.show_model_list {
                    self.connect_form.model_list_index =
                        self.connect_form.model_list_index.saturating_sub(1);
                } else {
                    self.connect_form.previous_field();
                }
            }
            // Down: scroll model list down, or move to next form field.
            KeyCode::Down => {
                if self.connect_form.show_model_list {
                    if self.connect_form.model_list_index + 1 < self.connect_form.model_list.len() {
                        self.connect_form.model_list_index += 1;
                    }
                } else {
                    self.connect_form.next_field();
                }
            }
            // Left toggles provider type when on the provider field.
            KeyCode::Left => {
                self.connect_form.on_left();
            }
            // Right toggles provider type when on the provider field.
            KeyCode::Right => {
                self.connect_form.on_right();
            }
            // Type into the currently selected text field.
            KeyCode::Char(c) => {
                self.connect_form.on_char(c);
            }
            // Delete last character from the selected text field.
            KeyCode::Backspace => {
                self.connect_form.on_backspace();
            }
            // Enter: select from model list, fetch models, save, or cancel.
            KeyCode::Enter => {
                if self.connect_form.show_model_list && !self.connect_form.model_list.is_empty() {
                    self.connect_form.select_model();
                } else {
                    match self.connect_form.selected_field {
                        // Field 4 (model): fetch the model list from the provider.
                        4 => {
                            let provider = self.connect_form.provider_label().to_string();
                            let base_url = self.connect_form.base_url.clone();
                            let api_key = self.connect_form.api_key.clone();
                            let _ = self.cmd_tx.send(TuiCommand::FetchModels {
                                provider,
                                base_url,
                                api_key,
                            });
                            self.connect_form.fetching_models = true;
                            self.connect_form.status_message = "Fetching models...".into();
                        }
                        // Field 5 (save): apply and persist connection settings.
                        5 => {
                            self.save_connect_settings();
                        }
                        // Field 6 (cancel): return to normal mode without saving.
                        6 => {
                            self.mode = TuiMode::Normal;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Handle key presses while the provider picker is active.
    pub(super) fn handle_provider_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => self.mode = TuiMode::Normal,
            KeyCode::Up => {
                self.provider_selected = self.provider_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                let max = self.providers.len(); // allow selection of [+ New Provider]
                if self.provider_selected < max {
                    self.provider_selected += 1;
                }
            }
            KeyCode::Enter => {
                if self.provider_selected < self.providers.len() {
                    let name = self.providers[self.provider_selected].clone();
                    let _ = self.cmd_tx.send(TuiCommand::SetActiveProvider(name));
                    self.mode = TuiMode::Normal;
                } else {
                    // [+ New Provider]
                    self.mode = TuiMode::ConnectSettings;
                }
            }
            KeyCode::Char('e') => {
                if self.provider_selected < self.providers.len() {
                    let name = self.providers[self.provider_selected].clone();
                    let _ = self.cmd_tx.send(TuiCommand::GetProviderConfig(name));
                }
            }
            KeyCode::Char('d') => {
                if self.provider_selected < self.providers.len() {
                    let name = self.providers[self.provider_selected].clone();
                    let _ = self.cmd_tx.send(TuiCommand::DeleteProvider(name));
                    // Re-request list
                    let _ = self.cmd_tx.send(TuiCommand::GetProviders);
                }
            }
            _ => {}
        }
    }

    // Handle key presses while the session picker is active.
    pub(super) fn handle_session_picker_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => self.mode = TuiMode::Normal,
            KeyCode::Up => {
                self.session_selected = self.session_selected.saturating_sub(1);
            }
            KeyCode::Down => {
                if !self.session_list.is_empty()
                    && self.session_selected + 1 < self.session_list.len()
                {
                    self.session_selected += 1;
                }
            }
            KeyCode::Enter => {
                if !self.session_list.is_empty() && self.session_selected < self.session_list.len()
                {
                    let s_meta = &self.session_list[self.session_selected];
                    if let Ok(parsed_id) = uuid::Uuid::parse_str(&s_meta.session_id) {
                        let _ = self.cmd_tx.send(TuiCommand::SwitchSession(parsed_id));
                        self.status_text = "Loading session...".to_string();
                    }
                }
            }
            _ => {}
        }
    }

    // Handle key presses while the variant/reasoning-effort picker is active.
    pub(super) fn handle_variant_key(&mut self, key: KeyCode) {
        match key {
            // Esc cancels and returns to normal chat mode.
            KeyCode::Esc => {
                self.mode = TuiMode::Normal;
            }
            // Move selection up through the variant list.
            KeyCode::Up => {
                self.variant_selected = self.variant_selected.saturating_sub(1);
            }
            // Move selection down through the variant list.
            KeyCode::Down => {
                if self.variant_selected + 1 < self.variant_options.len() {
                    self.variant_selected += 1;
                }
            }
            // Enter confirms the selected effort level, sends reconfigure, and returns to normal mode.
            KeyCode::Enter => {
                let level = self.variant_options.get(self.variant_selected).cloned();
                let is_none = level.as_deref() == Some("(none)");
                self.connect_form.reasoning_effort = if is_none { None } else { level.clone() };
                let config = self.connect_form.to_llm_config();
                self.model_name = self.connect_form.model.clone();
                let _ = self.cmd_tx.send(TuiCommand::Reconfigure(config));
                self.status_text = if is_none {
                    format!("Reasoning effort cleared for {}", self.model_name)
                } else {
                    format!(
                        "Variant set to {} for {}",
                        level.unwrap_or_default(),
                        self.model_name
                    )
                };
                self.mode = TuiMode::Normal;
            }
            _ => {}
        }
    }

    // Build the LLM config from the form and send a SaveProvider command.
    pub(super) fn save_connect_settings(&mut self) {
        let name = self.connect_form.to_provider_name();
        let config = self.connect_form.to_llm_config();
        self.model_name = self.connect_form.model.clone();
        self.provider_name = name.clone();
        let _ = self.cmd_tx.send(TuiCommand::SaveProvider { name, config });
        self.status_text = format!(
            "Connected: {} @ {}",
            self.model_name, self.connect_form.base_url
        );
        self.mode = TuiMode::Normal;
    }

    // Process an async event from the backend channel.
    pub(super) fn handle_app_event(&mut self, event: AppEvent) {
        match event {
            // Streaming response token — append to the in-progress assistant message.
            AppEvent::Token(text) => {
                let was_empty = self.current_response.is_empty();
                self.current_response.push_str(&text);
                if was_empty {
                    self.chat_scroll_offset = 0;
                    self.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: text,
                        reasoning: None,
                        reasoning_duration_ms: None,
                        reasoning_collapsed: false,
                    });
                } else if let Some(last) = self.messages.last_mut() {
                    if last.role == "assistant" && last.reasoning.is_none() {
                        last.content = self.current_response.clone();
                    }
                }
                self.status_text = "Generating...".into();
            }
            // A reasoning delta always belongs to the current independent thinking phase.
            AppEvent::Reasoning(text) => {
                if self.current_reasoning.is_empty() {
                    self.reasoning_start_time = Some(Instant::now());
                    self.current_reasoning_collapsed = false;
                }
                self.current_reasoning.push_str(&text);
                self.status_text = "Thinking...".into();
            }
            // Complete and persist this phase before a tool or response begins.
            AppEvent::ReasoningComplete { duration_ms } => {
                self.finish_current_reasoning(Some(duration_ms));
            }
            // Update the status bar text directly.
            AppEvent::AgentStatus(status) => {
                self.status_text = status;
            }
            AppEvent::McpStatus { server_id, status } => {
                self.status_text = format!("MCP {}: {}", server_id, status);
            }
            // Display an error — in connect mode show in form, otherwise as system message.
            AppEvent::Error(msg) => {
                if self.mode == TuiMode::ConnectSettings {
                    self.connect_form.fetching_models = false;
                    self.connect_form.status_message = format!("Error: {}", msg);
                } else {
                    self.messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!("Error: {}", msg),
                        reasoning: None,
                        reasoning_duration_ms: None,
                        reasoning_collapsed: false,
                    });
                    self.status_text = "Error".into();
                }
            }
            // Response streaming complete — finalize the message and reset.
            AppEvent::Done => {
                self.finish_current_turn();
                self.status_text = "Idle".into();
            }
            // Keep any partial response visible, but return the TUI to an idle state.
            AppEvent::Interrupted => {
                self.finish_current_turn();
                self.status_text = "Interrupted".into();
            }
            // Received model list — populate the dropdown in connect mode.
            AppEvent::ModelList(models) => {
                self.connect_form.model_list = models;
                self.connect_form.show_model_list = true;
                self.connect_form.model_list_index = 0;
                self.connect_form.fetching_models = false;
                self.connect_form.status_message = format!(
                    "{} models loaded. Use Up/Down/Enter to select.",
                    self.connect_form.model_list.len()
                );
            }
            // Received provider name list — enter provider picker mode.
            AppEvent::ProviderList(names) => {
                self.providers = names.clone();
                self.provider_selected = names
                    .iter()
                    .position(|n| n == &self.provider_name)
                    .unwrap_or(0);
                self.mode = TuiMode::ProviderPicker;
            }
            // Provider was switched externally — update active name and model.
            AppEvent::ProviderSwitched { name, model } => {
                self.provider_name = name.clone();
                self.model_name = model;
                self.status_text = format!("Switched to {}", name);
            }
            // Received full config for a specific provider (from "Modify" action).
            AppEvent::ProviderConfig { name, config } => {
                self.connect_form = connect::ConnectForm::from_entry(&name, &config);
                self.mode = TuiMode::ConnectSettings;
            }
            // Tool details are transient: close the current blocks and use only the status bar.
            AppEvent::ToolCallStart { name } => {
                self.finish_current_reasoning(None);
                self.finish_current_response_segment();
                self.status_text = format!("Calling tool: {}", name);
            }
            // Tool execution started.
            AppEvent::ToolExecStart { name } => {
                self.status_text = format!("Running: {}", name);
            }
            // Once execution ends, the agent is preparing the next model pass.
            AppEvent::ToolExecEnd => {
                self.status_text = "Thinking...".into();
            }
            AppEvent::SessionList(meta_list) => {
                self.session_list = meta_list;
                self.session_selected = self
                    .session_list
                    .iter()
                    .position(|s| s.session_id == self.session_id.to_string())
                    .unwrap_or(0);
                self.mode = TuiMode::SessionPicker;
            }
            AppEvent::SessionLoaded {
                session_id,
                messages,
            } => {
                self.session_id = session_id;
                self.messages = messages;
                self.current_response.clear();
                self.current_reasoning.clear();
                self.reasoning_start_time = None;
                self.chat_scroll_offset = 0;
                self.status_text = format!("Loaded session {}", &session_id.to_string()[..8]);
                self.mode = TuiMode::Normal;
            }
            AppEvent::PermissionRequest {
                request_id,
                tool_name,
                args_summary,
                reason,
                risk_level,
            } => {
                // Set the interactive permission prompt — blocks normal input until resolved.
                self.permission_prompt = Some(PermissionPromptState {
                    request_id,
                    tool_name,
                    args_summary,
                    reason,
                    risk_level,
                });
                // Also push a system message so it appears in the chat transcript.
                let msg = format!(
                    "⚠ Permission needed: `{}` with args `{}`\nReason: {}",
                    self.permission_prompt.as_ref().unwrap().tool_name,
                    self.permission_prompt.as_ref().unwrap().args_summary,
                    self.permission_prompt.as_ref().unwrap().reason,
                );
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: msg,
                    reasoning: None,
                    reasoning_duration_ms: None,
                    reasoning_collapsed: false,
                });
            }
            AppEvent::PermissionResolved { .. } => {
                // For now, resolved notifications are logged but not displayed distinctly
            }
        }
    }
    // Start a user turn without mutating any completed thinking history.
    pub(super) fn begin_user_turn(&mut self, text: String) {
        self.finish_current_turn();
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        self.messages.push(ChatMessage {
            role: "user".into(),
            content: text.clone(),
            reasoning: None,
            reasoning_duration_ms: None,
            reasoning_collapsed: false,
        });
        self.chat_scroll_offset = 0;
        self.selection.clear();
        self.status_text = "Waiting...".into();
        let _ = self.cmd_tx.send(TuiCommand::Chat(text));
    }

    // Persist a completed thinking phase as its own chat block and reset its timer.
    fn finish_current_reasoning(&mut self, duration_hint: Option<u64>) {
        let elapsed = self
            .reasoning_start_time
            .take()
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);
        if self.current_reasoning.is_empty() {
            self.current_reasoning_collapsed = true;
            return;
        }
        let duration_ms = elapsed.max(duration_hint.unwrap_or(0));
        self.messages.push(ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            reasoning: Some(std::mem::take(&mut self.current_reasoning)),
            reasoning_duration_ms: Some(duration_ms),
            reasoning_collapsed: true,
        });
        self.current_reasoning_collapsed = true;
    }

    // The streamed text already lives in its message; only its live accumulator is transient.
    fn finish_current_response_segment(&mut self) {
        self.current_response.clear();
    }

    // Flush any live blocks when a turn completes or a new user turn begins.
    fn finish_current_turn(&mut self) {
        self.finish_current_reasoning(None);
        self.finish_current_response_segment();
    }

    // Render the full TUI frame: content area, input line, and status bar.
}
