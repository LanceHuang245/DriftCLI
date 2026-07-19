use super::*;

impl TuiApp {
    pub(super) fn render(&mut self, f: &mut ratatui::Frame) {
        let size = f.area();

        // Calculate popup height (max 8 items + 2 for borders)
        let popup_height = self.slash_completion.as_ref().and_then(|s| {
            if s.filtered.is_empty() {
                None
            } else {
                Some((s.filtered.len().min(8) + 2) as u16)
            }
        });

        // Give the permission prompt enough rows to keep every response key visible.
        let input_height = if self.permission_prompt.is_some() {
            9
        } else {
            3
        };
        // Split the screen: content, optional popup, input, status bar.
        let mut constraints = vec![
            Constraint::Min(3),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ];
        if let Some(h) = popup_height {
            constraints.insert(1, Constraint::Length(h));
        }
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(size);
        let content_area = layout[0];
        let (input_area, status_area, popup_area) = if popup_height.is_some() {
            (layout[2], layout[3], Some(layout[1]))
        } else {
            (layout[1], layout[2], None)
        };

        // Render the top content area depending on the current mode.
        match self.mode {
            TuiMode::Normal => {
                self.chat_area = content_area;
                self.render_chat_area(content_area, f);
            }
            TuiMode::ConnectSettings => {
                f.render_widget(self.render_connect_settings(), content_area);
            }
            TuiMode::VariantPicker => {
                f.render_widget(self.render_variant_picker(), content_area);
            }
            TuiMode::ProviderPicker => {
                f.render_widget(self.render_provider_picker(), content_area);
            }
            TuiMode::SessionPicker => {
                f.render_widget(self.render_session_picker(), content_area);
            }
        }

        // Render slash completion popup between content and input if active.
        if let (Some(area), Some(state)) = (popup_area, &self.slash_completion) {
            if !state.filtered.is_empty() {
                f.render_widget(self.render_slash_popup(state), area);
            }
        }

        // Build the input line with cursor highlighting.
        let mut input_spans = vec![Span::raw("> ")];
        for (i, ch) in self.input_buffer.char_indices() {
            if i == self.cursor_position {
                // Highlight the character at cursor position with inverted colors.
                input_spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().bg(Color::White).fg(Color::Black),
                ));
            } else {
                input_spans.push(Span::raw(ch.to_string()));
            }
        }
        // If cursor is past end of buffer, show a highlighted space.
        if self.cursor_position >= self.input_buffer.len() {
            input_spans.push(Span::styled(
                " ",
                Style::default().bg(Color::White).fg(Color::Black),
            ));
        }

        // Render the input line widget with a top border.
        let input_line = Line::from(input_spans);
        // Permission dialog replaces the input line when a prompt is active.
        if let Some(prompt) = &self.permission_prompt {
            let dialog = self.render_permission_dialog(prompt);
            f.render_widget(dialog, input_area);
        } else {
            let input_widget =
                Paragraph::new(input_line).block(Block::default().borders(Borders::TOP));
            f.render_widget(input_widget, input_area);
        }

        // Keep the text cursor hidden while keyboard input belongs to the permission prompt.
        if self.permission_prompt.is_none() {
            let before_cursor =
                &self.input_buffer[..self.cursor_position.min(self.input_buffer.len())];
            let prompt_width = 2;
            let visual_before = unicode_width::UnicodeWidthStr::width(before_cursor);
            let visual_total = prompt_width + visual_before;
            let inner_width = input_area.width as usize;

            let cursor_x = (visual_total % inner_width.max(1)) as u16;
            let cursor_y = input_area.y + 1 + (visual_total / inner_width.max(1)) as u16;
            f.set_cursor_position(Position::new(cursor_x, cursor_y));
        }

        // Status bar: color-coded status, provider name, model name, and keyboard shortcuts.
        let status_style = match self.status_text.as_str() {
            s if s.starts_with("Idle") => Style::default().fg(Color::Green),
            s if s.starts_with("Waiting")
                || s.starts_with("Thinking")
                || s.starts_with("Connected")
                || s.starts_with("Switched") =>
            {
                Style::default().fg(Color::Yellow)
            }
            s if s.starts_with("Generating") => Style::default().fg(Color::Cyan),
            s if s.starts_with("Error") => Style::default().fg(Color::Red),
            _ => Style::default(),
        };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(
                " DriftCLI ",
                Style::default().fg(Color::Black).bg(Color::White),
            ),
            Span::raw(" | "),
            Span::styled(self.status_text.clone(), status_style),
            Span::raw(" | "),
            Span::styled(&self.provider_name, Style::default().fg(Color::Cyan)),
            Span::raw(" | "),
            Span::styled(&self.model_name, Style::default().fg(Color::Magenta)),
            Span::styled(
                self.connect_form
                    .reasoning_effort
                    .as_ref()
                    .map(|e| format!(" [{}]", e))
                    .unwrap_or_default(),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" | Session: "),
            Span::styled(
                if self.session_id.is_nil() {
                    "(none)".to_string()
                } else {
                    self.session_id.to_string()
                },
                Style::default().fg(Color::Yellow),
            ),
        ]));
        f.render_widget(status, status_area);
    }

    /// Check if a mouse click hits a reasoning header line and toggle it.
    /// Returns true if a reasoning header was clicked (and handled).
    pub(super) fn try_toggle_reasoning(&mut self, mouse: &crossterm::event::MouseEvent) -> bool {
        let mouse_row = mouse.row as usize;
        let chat_top = self.chat_area.y as usize;
        if mouse_row < chat_top {
            return false;
        }
        let relative_row = mouse_row - chat_top;
        let total = self.total_chat_lines;
        let visible = self.chat_area.height as usize;
        let max_scroll = total.saturating_sub(visible);
        let top_offset = max_scroll
            .saturating_sub(self.chat_scroll_offset)
            .min(max_scroll);
        let clicked_line = relative_row + top_offset;

        for (target, line_idx) in &self.reasoning_header_positions {
            if *line_idx == clicked_line {
                match target {
                    ReasoningTarget::Message(msg_idx) => {
                        if let Some(msg) = self.messages.get_mut(*msg_idx) {
                            msg.reasoning_collapsed = !msg.reasoning_collapsed;
                        }
                    }
                    ReasoningTarget::Live => {
                        self.current_reasoning_collapsed = !self.current_reasoning_collapsed;
                    }
                }
                return true;
            }
        }
        false
    }

    /// Push wrapped reasoning lines into the lines vector with indent and dim style.
    fn push_wrapped_lines(
        lines: &mut Vec<Line<'static>>,
        text: &str,
        area_width: u16,
        indent: &str,
    ) {
        let max_w = area_width.saturating_sub(indent.len() as u16) as usize;
        for paragraph in text.split('\n') {
            let mut current = String::new();
            let mut current_width = 0usize;
            for word in paragraph.split_inclusive(|c: char| c.is_whitespace()) {
                let word_width = unicode_width::UnicodeWidthStr::width(word);
                if current_width + word_width > max_w && current_width > 0 {
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", indent, current.trim_end()),
                        Style::default().fg(Color::DarkGray),
                    )));
                    current.clear();
                    current_width = 0;
                }
                current.push_str(word);
                current_width += word_width;
            }
            if !current.is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("{}{}", indent, current.trim_end()),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
    }

    /// Push the live streaming reasoning block into the lines vector.
    /// Used both for pure-thinking phase (no text yet) and interleaved
    /// thinking during generation for OpenAI Compatible models.
    #[allow(clippy::too_many_arguments)]
    fn push_live_reasoning_block(
        header_positions: &mut Vec<(ReasoningTarget, usize)>,
        lines: &mut Vec<Line<'static>>,
        current_reasoning: &str,
        current_reasoning_collapsed: bool,
        reasoning_start_time: Option<Instant>,
        current_reasoning_duration_ms: u64,
        area_width: u16,
    ) {
        let live_duration = match reasoning_start_time {
            Some(start) => {
                let elapsed_ms = current_reasoning_duration_ms
                    .saturating_add(start.elapsed().as_millis() as u64);
                if elapsed_ms >= 1000 {
                    format!(" for {:.1}s", elapsed_ms as f64 / 1000.0)
                } else {
                    format!(" for {}ms", elapsed_ms)
                }
            }
            None => String::new(),
        };

        let toggle = if current_reasoning_collapsed {
            "▶"
        } else {
            "▼"
        };
        let header = format!("{} Thinking{}", toggle, live_duration);

        lines.push(Line::from(Span::raw("")));

        header_positions.push((ReasoningTarget::Live, lines.len()));

        lines.push(Line::from(Span::styled(
            header,
            Style::default().fg(Color::Gray),
        )));

        if !current_reasoning_collapsed {
            Self::push_wrapped_lines(lines, current_reasoning, area_width, "  ");
        }

        lines.push(Line::from(Span::raw("")));
    }

    // Render the scrollable chat message area with scroll offset and selection highlighting.
    fn render_chat_area(&mut self, area: ratatui::layout::Rect, f: &mut ratatui::Frame) {
        self.reasoning_header_positions.clear();
        self.total_chat_lines = 0;

        // Build flat lines for all messages with role-based styling.
        let mut lines: Vec<Line> = Vec::new();
        for (msg_idx, msg) in self.messages.iter().enumerate() {
            let text_style = match msg.role.as_str() {
                "system" => Style::default().fg(Color::Yellow),
                _ => Style::default(),
            };

            if let Some(reasoning) = &msg.reasoning {
                let duration_str = msg.reasoning_duration_ms.map_or_else(String::new, |ms| {
                    if ms >= 1000 {
                        format!(" for {:.1}s", ms as f64 / 1000.0)
                    } else {
                        format!(" for {}ms", ms)
                    }
                });

                let toggle = if msg.reasoning_collapsed {
                    "▶"
                } else {
                    "▼"
                };
                let header = format!("{} Thinking{}", toggle, duration_str);

                lines.push(Line::from(Span::raw("")));

                // Record header position for mouse hit-testing
                self.reasoning_header_positions
                    .push((ReasoningTarget::Message(msg_idx), lines.len()));

                lines.push(Line::from(Span::styled(
                    header,
                    Style::default().fg(Color::Gray),
                )));

                if !msg.reasoning_collapsed {
                    Self::push_wrapped_lines(&mut lines, reasoning, area.width, "  ");
                }

                lines.push(Line::from(Span::raw("")));
            }

            if msg.role == "user" {
                let width = area.width.saturating_sub(2) as usize;
                let top = format!("┌{}┐", "─".repeat(width));
                let bottom = format!("└{}┘", "─".repeat(width));
                lines.push(Line::from(Span::styled(
                    top,
                    Style::default().fg(Color::Cyan),
                )));
                for content_line in msg.content.lines() {
                    let vis = unicode_width::UnicodeWidthStr::width(content_line);
                    let inner = area.width.saturating_sub(4) as usize;
                    let pad = inner.saturating_sub(vis);
                    let padded = format!("│ {} {}│", content_line, " ".repeat(pad));
                    lines.push(Line::from(Span::styled(
                        padded,
                        Style::default().fg(Color::Cyan),
                    )));
                }
                lines.push(Line::from(Span::styled(
                    bottom,
                    Style::default().fg(Color::Cyan),
                )));
            } else if msg.role == "assistant" {
                let md_lines = markdown::render_markdown(&msg.content, area.width);
                for line in md_lines {
                    lines.push(line);
                }
            } else {
                for line in msg.content.lines() {
                    lines.push(Line::from(Span::styled(line.to_string(), text_style)));
                }
            }
        }

        // A live thinking phase always follows committed history chronologically.
        if !self.current_reasoning.is_empty() {
            Self::push_live_reasoning_block(
                &mut self.reasoning_header_positions,
                &mut lines,
                &self.current_reasoning,
                self.current_reasoning_collapsed,
                self.reasoning_start_time,
                self.current_reasoning_duration_ms,
                area.width,
            );
        }

        if lines.is_empty() {
            return;
        }

        // Render the paragraph with scrolling.
        let total = lines.len();
        self.total_chat_lines = total;
        let visible = area.height as usize;
        let max_scroll = total.saturating_sub(visible);
        let top_offset = max_scroll
            .saturating_sub(self.chat_scroll_offset)
            .min(max_scroll);
        let paragraph = Paragraph::new(lines).scroll((top_offset as u16, 0));
        f.render_widget(paragraph, area);

        // Overlay the selection highlight on top of the rendered content.
        self.selection.render(f.buffer_mut());
    }

    // Render the connection settings form with provider name, type, URL, key, model fields, and buttons.
    fn render_connect_settings(&self) -> Paragraph<'_> {
        let form = &self.connect_form;

        // Highlight flags for each form field. Fields: 0=name, 1=provider, 2=url, 3=key, 4=model, 5=save, 6=cancel.
        let name_highlight = form.selected_field == 0;
        let provider_highlight = form.selected_field == 1;
        let url_highlight = form.selected_field == 2;
        let key_highlight = form.selected_field == 3;
        let model_highlight = form.selected_field == 4;
        let save_highlight = form.selected_field == 5;
        let cancel_highlight = form.selected_field == 6;

        // Obfuscate the API key for display.
        let key_display = if form.api_key.is_empty() {
            "(not set)".to_string()
        } else {
            let len = form.api_key.len();
            if len <= 8 {
                "***".to_string()
            } else {
                format!("{}...{}", &form.api_key[..4], &form.api_key[len - 4..])
            }
        };

        // Build styled spans for each form field based on highlight state.
        let name_field = if name_highlight {
            Span::styled(
                format!("> {}", form.provider_name),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", form.provider_name),
                Style::default().fg(Color::White),
            )
        };

        let provider_options = if provider_highlight {
            Span::styled(
                format!("[ {} ]  (Left/Right to change)", form.provider_label()),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}   ", form.provider_label()),
                Style::default().fg(Color::Cyan),
            )
        };

        let url_field = if url_highlight {
            Span::styled(
                format!("> {}", form.base_url),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", form.base_url),
                Style::default().fg(Color::White),
            )
        };

        let key_field = if key_highlight {
            Span::styled(
                format!("> {}", key_display),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", key_display),
                Style::default().fg(Color::White),
            )
        };

        let model_field = if model_highlight {
            Span::styled(
                format!("> {}", form.model),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(
                format!("  {}", form.model),
                Style::default().fg(Color::White),
            )
        };

        let save_btn = if save_highlight {
            Span::styled(
                "[ Save & Apply ]",
                Style::default().fg(Color::Black).bg(Color::Green),
            )
        } else {
            Span::styled("  Save & Apply  ", Style::default().fg(Color::Green))
        };

        let cancel_btn = if cancel_highlight {
            Span::styled(
                "[ Cancel ]",
                Style::default().fg(Color::Black).bg(Color::Red),
            )
        } else {
            Span::styled("  Cancel  ", Style::default().fg(Color::Red))
        };

        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(vec![Span::raw("  Name:       "), name_field]));
        lines.push(Line::from(vec![
            Span::raw("  Provider:   "),
            provider_options,
        ]));
        lines.push(Line::from(vec![Span::raw("  Base URL:   "), url_field]));
        lines.push(Line::from(vec![Span::raw("  API Key:    "), key_field]));
        lines.push(Line::from(vec![
            Span::raw("  Model:      "),
            model_field,
            Span::raw(" "),
            Span::styled(
                if form.fetching_models {
                    "(fetching...)"
                } else {
                    "(Enter to fetch)"
                },
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        lines.push(Line::from(Span::raw("")));

        // Render the model list dropdown if visible.
        if form.show_model_list && !form.model_list.is_empty() {
            let start = form.model_list_index.saturating_sub(4);
            let end = (start + 9).min(form.model_list.len());
            for i in start..end {
                let entry = &form.model_list[i];
                let model_line = if i == form.model_list_index {
                    Line::from(Span::styled(
                        format!("    > {}", entry.id),
                        Style::default().fg(Color::White).bg(Color::DarkGray),
                    ))
                } else {
                    Line::from(Span::raw(format!("      {}", entry.id)))
                };
                lines.push(model_line);
            }
            if form.model_list.len() > 9 {
                lines.push(Line::from(Span::styled(
                    format!(
                        "      ... ({} total, scroll with Up/Down)",
                        form.model_list.len()
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
            }
            lines.push(Line::from(Span::raw("")));
        }

        // Display status message (e.g. fetch result) if set.
        if !form.status_message.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  {}", form.status_message),
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(Span::raw("")));
        }

        lines.push(Line::from(vec![save_btn, Span::raw("  "), cancel_btn]));

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Tab/Up/Down: navigate  Left/Right: switch  Enter: confirm  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));

        // Decorative block border and title.
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /connect — Configure Connection ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the provider picker overlay for switching between configured providers.
    fn render_provider_picker(&self) -> Paragraph<'_> {
        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Select provider:",
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        for (i, name) in self.providers.iter().enumerate() {
            let active = name == &self.provider_name;
            let arrow = if i == self.provider_selected {
                ">"
            } else {
                " "
            };
            let text = if active {
                format!("  {} * {}", arrow, name)
            } else {
                format!("    {} {}", arrow, name)
            };
            let style = if i == self.provider_selected {
                Style::default().fg(Color::Black).bg(Color::Green)
            } else if active {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(Span::styled(text, style)));
        }

        // [+ New Provider] option
        let new_idx = self.providers.len();
        let arrow = if self.provider_selected == new_idx {
            ">"
        } else {
            " "
        };
        lines.push(Line::from(Span::raw("")));
        let text = format!("    {} [+ New Provider]", arrow);
        let style = if self.provider_selected == new_idx {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        lines.push(Line::from(Span::styled(text, style)));

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: switch  e: edit  d: delete  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /provider — Select Provider ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the variant / reasoning-effort picker overlay.
    fn render_variant_picker(&self) -> Paragraph<'_> {
        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Select reasoning effort for the current model:",
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        // Check if the model has known capability info.
        let has_capabilities = self
            .connect_form
            .model_list
            .iter()
            .find(|m| m.id == self.connect_form.model)
            .map(|m| !m.effort_levels.is_empty())
            .unwrap_or(false);

        // Show a fallback warning if capabilities are unknown.
        if !has_capabilities && self.variant_options.len() > 1 {
            lines.push(Line::from(Span::styled(
                "  (model capabilities unknown — showing all levels as fallback)",
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(Span::raw("")));
        }

        if self.variant_options.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No effort levels available.",
                Style::default().fg(Color::Yellow),
            )));
        } else {
            for (i, level) in self.variant_options.iter().enumerate() {
                let label = if i == self.variant_selected {
                    Span::styled(
                        format!("    > {}", level),
                        Style::default().fg(Color::Black).bg(Color::Green),
                    )
                } else if level == "(none)" {
                    Span::styled(
                        format!("      {}", level),
                        Style::default().fg(Color::DarkGray),
                    )
                } else {
                    Span::styled(
                        format!("      {}", level),
                        Style::default().fg(Color::White),
                    )
                };
                lines.push(Line::from(label));
            }
        }

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: confirm  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::raw("")));

        // Show the currently selected effort level.
        let current = self
            .connect_form
            .reasoning_effort
            .as_deref()
            .unwrap_or("none");
        lines.push(Line::from(Span::styled(
            format!("  Current: {}", current),
            Style::default().fg(Color::Yellow),
        )));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /variant — Reasoning Effort ")
            .border_style(Style::default().fg(Color::Magenta));
        Paragraph::new(lines).block(block)
    }

    // Render the persistent session picker overlay list.
    fn render_session_picker(&self) -> Paragraph<'_> {
        let mut lines = Vec::new();
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Saved Historical Sessions:",
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        if self.session_list.is_empty() {
            lines.push(Line::from(Span::styled(
                "  No historical sessions found.",
                Style::default().fg(Color::Yellow),
            )));
        } else {
            for (i, meta) in self.session_list.iter().enumerate() {
                let is_active = meta.session_id == self.session_id.to_string();
                let active_marker = if is_active { "*" } else { " " };

                let text = format!(
                    "  {} [{}]  Created: {}  Dir: {}  Model: {}",
                    active_marker,
                    &meta.session_id[..8.min(meta.session_id.len())],
                    meta.created_at,
                    meta.working_dir,
                    meta.model
                );

                let style = if i == self.session_selected {
                    Style::default().fg(Color::Black).bg(Color::Green)
                } else if is_active {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::White)
                };

                lines.push(Line::from(Span::styled(text, style)));
            }
        }

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            "  Up/Down: choose  Enter: switch session  Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::raw("")));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /sessions — Saved Chat History ")
            .border_style(Style::default().fg(Color::Yellow));
        Paragraph::new(lines).block(block)
    }

    // Set full visual ChatMessage messages in TUI (used on loading a session)
    fn render_slash_popup(&self, state: &SlashCompletionState) -> Paragraph<'_> {
        let max_visible = 8usize;
        let total = state.filtered.len();
        let visible = total.min(max_visible);

        // Scroll window to keep selected item visible
        let start = if total <= max_visible {
            0
        } else {
            state
                .selected
                .saturating_sub(max_visible / 2)
                .min(total - max_visible)
        };

        let mut lines: Vec<Line> = Vec::new();
        for i in start..start + visible {
            let cmd = &state.filtered[i];
            let label = if i == state.selected {
                Span::styled(
                    format!("> {}  {}", cmd.name, cmd.description),
                    Style::default().fg(Color::Black).bg(Color::White),
                )
            } else {
                Span::styled(
                    format!("  {}  {}", cmd.name, cmd.description),
                    Style::default().fg(Color::White),
                )
            };
            lines.push(Line::from(label));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Commands ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }

    // Render the permission prompt dialog in place of the input line.
    fn render_permission_dialog(&self, prompt: &PermissionPromptState) -> Paragraph<'_> {
        let risk_color = match prompt.risk_level.as_str() {
            "Critical" => Color::Red,
            "High" => Color::LightRed,
            "Medium" => Color::Yellow,
            _ => Color::White,
        };
        let mut lines = Vec::new();
        lines.push(Line::from(Span::styled(
            format!("  Permission Required — {}", prompt.tool_name),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!("  {}", prompt.args_summary),
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::styled(
            format!("  Reason: {}", prompt.reason),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::styled(
            format!("  Risk: {}", prompt.risk_level),
            Style::default().fg(risk_color),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  [y] Allow once       [Y] Always allow",
            Style::default().fg(Color::Cyan),
        )));
        lines.push(Line::from(Span::styled(
            "  [n] Deny once        [N] Always deny       [Esc] Cancel",
            Style::default().fg(Color::Cyan),
        )));
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Permission ")
            .border_style(Style::default().fg(Color::Yellow));
        Paragraph::new(lines).block(block)
    }
}
