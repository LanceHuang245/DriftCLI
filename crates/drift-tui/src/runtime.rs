use super::*;

impl TuiApp {
    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = ratatui::backend::CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.main_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        result
    }

    // Main event loop: draws the frame, processes async events, and handles key input.
    fn main_loop(
        &mut self,
        terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    ) -> anyhow::Result<()> {
        while !self.should_quit {
            // Draw the current frame.
            terminal.draw(|f| self.render(f))?;

            // Drain any pending async events from the backend.
            if let Ok(event) = self.event_rx.try_recv() {
                self.handle_app_event(event);
            }

            // Poll for input events (non-blocking with 16 ms timeout).
            if event::poll(std::time::Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        // Require two consecutive Ctrl+C presses before quitting the TUI.
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c')
                        {
                            if self.quit_confirmation_pending {
                                self.should_quit = true;
                            } else {
                                self.quit_confirmation_pending = true;
                                self.status_text = "Press Ctrl+C again to exit".into();
                            }
                            continue;
                        }
                        self.quit_confirmation_pending = false;
                        // ── Permission prompt interceptor ──
                        // Permission prompt interceptor — only y/Y/n/N keys accepted
                        if let Some(prompt) = self.permission_prompt.as_ref() {
                            if key.code == KeyCode::Esc {
                                self.permission_prompt = None;
                                let _ = self.cmd_tx.send(TuiCommand::Interrupt);
                                self.status_text = "Interrupting...".into();
                                continue;
                            }
                            let request_id = prompt.request_id.clone();
                            let (allowed, remember) = match key.code {
                                KeyCode::Char('y') => (true, false),
                                KeyCode::Char('Y') => (true, true),
                                KeyCode::Char('n') => (false, false),
                                KeyCode::Char('N') => (false, true),
                                _ => {
                                    continue; // Ignore other keys
                                }
                            };
                            self.permission_prompt = None;
                            let _ = self.cmd_tx.send(TuiCommand::PermissionResponse {
                                request_id,
                                allowed,
                                remember,
                            });
                            continue;
                        }
                        // ── End permission interceptor ──

                        match self.mode {
                            TuiMode::Normal => {
                                // Slash completion popup key intercept
                                let mut popup_handled = false;
                                if let Some(ref mut state) = self.slash_completion {
                                    if !state.filtered.is_empty() {
                                        match key.code {
                                            KeyCode::Up => {
                                                state.selected = if state.selected == 0 {
                                                    state.filtered.len() - 1
                                                } else {
                                                    state.selected - 1
                                                };
                                                popup_handled = true;
                                            }
                                            KeyCode::Down => {
                                                state.selected =
                                                    if state.selected + 1 >= state.filtered.len() {
                                                        0
                                                    } else {
                                                        state.selected + 1
                                                    };
                                                popup_handled = true;
                                            }
                                            KeyCode::Tab => {
                                                let cmd = state.filtered[state.selected].name;
                                                self.input_buffer = cmd.to_string();
                                                self.cursor_position = cmd.len();
                                                popup_handled = true;
                                            }
                                            KeyCode::Enter => {
                                                let cmd = state.filtered[state.selected].name;
                                                self.handle_command(cmd);
                                                popup_handled = true;
                                            }
                                            KeyCode::Esc => {
                                                self.slash_completion = None;
                                                popup_handled = true;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                if !popup_handled {
                                    match key.code {
                                        KeyCode::Up => {
                                            if self.history_index.is_none()
                                                && !self.input_buffer.is_empty()
                                            {
                                                self.history.push(self.input_buffer.clone());
                                            }
                                            if !self.history.is_empty() {
                                                let idx = self
                                                    .history_index
                                                    .map_or(self.history.len(), |i| i);
                                                let new_idx = idx.saturating_sub(1);
                                                self.history_index = Some(new_idx);
                                                self.input_buffer = self.history[new_idx].clone();
                                                self.cursor_position = self.input_buffer.len();
                                            }
                                        }
                                        KeyCode::Down => {
                                            if let Some(idx) = self.history_index {
                                                let new_idx = idx + 1;
                                                if new_idx >= self.history.len() {
                                                    self.history_index = None;
                                                    self.input_buffer.clear();
                                                } else {
                                                    self.history_index = Some(new_idx);
                                                    self.input_buffer =
                                                        self.history[new_idx].clone();
                                                }
                                                self.cursor_position = self.input_buffer.len();
                                            }
                                        }
                                        // PgUp / PgDn scroll the chat area by half the viewport.
                                        KeyCode::PageUp => {
                                            self.chat_scroll_offset += 10;
                                        }
                                        KeyCode::PageDown => {
                                            self.chat_scroll_offset =
                                                self.chat_scroll_offset.saturating_sub(10);
                                        }
                                        _ => {
                                            self.history_index = None;
                                            let action = input::process_key(
                                                key.code,
                                                key.modifiers,
                                                &mut self.input_buffer,
                                                &mut self.cursor_position,
                                            );
                                            match action {
                                                input::InputAction::Submit(text) => {
                                                    if !text.trim().is_empty() {
                                                        if text.starts_with('/') {
                                                            self.handle_command(&text);
                                                        } else {
                                                            if self.history.last() != Some(&text) {
                                                                self.history.push(text.clone());
                                                            }
                                                            self.messages.push(ChatMessage {
                                                                role: "user".into(),
                                                                content: text.clone(),
                                                                reasoning: None,
                                                                thinking: false,
                                                                reasoning_duration_ms: None,
                                                                reasoning_collapsed: false,
                                                                thinking_tools: Vec::new(),
                                                            });
                                                            self.current_response.clear();
                                                            self.current_reasoning.clear();
                                                            self.current_thinking_tools.clear();
                                                            self.reasoning_start_time = None;
                                                            self.current_reasoning_collapsed = true;
                                                            self.chat_scroll_offset = 0;
                                                            self.selection.clear();
                                                            self.status_text = "Waiting...".into();
                                                            let _ = self
                                                                .cmd_tx
                                                                .send(TuiCommand::Chat(text));
                                                        }
                                                        self.input_buffer.clear();
                                                        self.cursor_position = 0;
                                                    }
                                                }
                                                input::InputAction::Interrupt => {
                                                    let _ = self.cmd_tx.send(TuiCommand::Interrupt);
                                                }
                                                input::InputAction::Quit => self.should_quit = true,
                                                input::InputAction::ToggleConnectInfo => {}
                                            }
                                        }
                                    }
                                }
                                self.update_slash_completion();
                            }
                            TuiMode::ConnectSettings => {
                                self.handle_connect_key(key.code);
                            }
                            TuiMode::VariantPicker => {
                                self.handle_variant_key(key.code);
                            }
                            TuiMode::ProviderPicker => {
                                self.handle_provider_key(key.code);
                            }
                            TuiMode::SessionPicker => {
                                self.handle_session_picker_key(key.code);
                            }
                        }
                    }
                    Event::Mouse(mouse) => {
                        match mouse.kind {
                            MouseEventKind::ScrollUp => {
                                match self.mode {
                                    TuiMode::Normal => {
                                        self.chat_scroll_offset += 3;
                                    }
                                    TuiMode::ConnectSettings
                                        if self.connect_form.show_model_list =>
                                    {
                                        self.connect_form.model_list_index =
                                            self.connect_form.model_list_index.saturating_sub(1);
                                    }
                                    TuiMode::VariantPicker => {
                                        self.variant_selected =
                                            self.variant_selected.saturating_sub(1);
                                    }
                                    TuiMode::ProviderPicker => {
                                        self.provider_selected =
                                            self.provider_selected.saturating_sub(1);
                                    }
                                    TuiMode::SessionPicker => {
                                        self.session_selected =
                                            self.session_selected.saturating_sub(1);
                                    }
                                    _ => {}
                                }
                                // Scroll clears any active selection.
                                self.selection.clear();
                            }
                            MouseEventKind::ScrollDown => {
                                match self.mode {
                                    TuiMode::Normal => {
                                        self.chat_scroll_offset =
                                            self.chat_scroll_offset.saturating_sub(3);
                                    }
                                    TuiMode::ConnectSettings
                                        if self.connect_form.show_model_list =>
                                    {
                                        if self.connect_form.model_list_index + 1
                                            < self.connect_form.model_list.len()
                                        {
                                            self.connect_form.model_list_index += 1;
                                        }
                                    }
                                    TuiMode::VariantPicker => {
                                        if self.variant_selected + 1 < self.variant_options.len() {
                                            self.variant_selected += 1;
                                        }
                                    }
                                    TuiMode::ProviderPicker => {
                                        if self.provider_selected <= self.providers.len() {
                                            self.provider_selected += 1;
                                        }
                                    }
                                    TuiMode::SessionPicker => {
                                        if !self.session_list.is_empty()
                                            && self.session_selected + 1 < self.session_list.len()
                                        {
                                            self.session_selected += 1;
                                        }
                                    }
                                    _ => {}
                                }
                                self.selection.clear();
                            }
                            MouseEventKind::Down(button) => {
                                if self.mode == TuiMode::Normal && button == MouseButton::Left {
                                    if !self.try_toggle_reasoning(&mouse) {
                                        self.selection.handle_mouse_event(&mouse, self.chat_area);
                                    }
                                }
                            }
                            MouseEventKind::Drag(_) | MouseEventKind::Up(_) => {
                                if self.mode == TuiMode::Normal {
                                    self.selection.handle_mouse_event(&mouse, self.chat_area);
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    // Parse and execute slash commands like /connect, /variant, /provider, /clear, /quit.
}
