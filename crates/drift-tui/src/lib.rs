mod connect;
mod components;
mod input;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use drift_config::LlmConfig;
use ratatui::{
    layout::{Constraint, Direction, Layout, Position},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::io::{self, stdout};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum AppEvent {
    Token(String),
    Reasoning(String),
    AgentStatus(String),
    Error(String),
    Done,
    ModelList(Vec<String>),
}

#[derive(Debug, Clone)]
pub enum TuiCommand {
    Chat(String),
    FetchModels {
        provider: String,
        base_url: String,
        api_key: String,
    },
    Reconfigure(LlmConfig),
}

pub struct TuiApp {
    messages: Vec<ChatMessage>,
    current_response: String,
    current_reasoning: String,
    input_buffer: String,
    status_text: String,
    model_name: String,
    mode: TuiMode,
    connect_form: connect::ConnectForm,
    event_rx: mpsc::UnboundedReceiver<AppEvent>,
    cmd_tx: mpsc::UnboundedSender<TuiCommand>,
    should_quit: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TuiMode {
    Normal,
    ConnectSettings,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
}

impl TuiApp {
    pub fn new(
        llm_config: &LlmConfig,
        event_rx: mpsc::UnboundedReceiver<AppEvent>,
        cmd_tx: mpsc::UnboundedSender<TuiCommand>,
    ) -> Self {
        let model_name = match llm_config {
            LlmConfig::Anthropic { model, .. } => model.clone(),
            LlmConfig::OpenAiCompatible { model, .. } => model.clone(),
        };
        Self {
            messages: Vec::new(),
            current_response: String::new(),
            current_reasoning: String::new(),
            input_buffer: String::new(),
            status_text: "Idle".into(),
            model_name,
            mode: TuiMode::Normal,
            connect_form: connect::ConnectForm::from_config(llm_config),
            event_rx,
            cmd_tx,
            should_quit: false,
        }
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = ratatui::backend::CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.main_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        result
    }

    fn main_loop(
        &mut self,
        terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    ) -> anyhow::Result<()> {
        while !self.should_quit {
            terminal.draw(|f| self.render(f))?;

            if let Ok(event) = self.event_rx.try_recv() {
                self.handle_app_event(event);
            }

            if event::poll(std::time::Duration::from_millis(16))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match self.mode {
                            TuiMode::Normal => {
                                let action = input::process_key(key.code, &mut self.input_buffer);
                                match action {
                                    input::InputAction::Submit(text) => {
                                        if !text.trim().is_empty() {
                                            if text.starts_with('/') {
                                                self.handle_command(&text);
                                            } else {
                                                self.messages.push(ChatMessage {
                                                    role: "user".into(),
                                                    content: text.clone(),
                                                    reasoning: None,
                                                });
                                                self.input_buffer.clear();
                                                self.current_response.clear();
                                                self.current_reasoning.clear();
                                                self.status_text = "Waiting...".into();
                                                let _ = self.cmd_tx.send(TuiCommand::Chat(text));
                                            }
                                        }
                                    }
                                    input::InputAction::Quit => self.should_quit = true,
                                    input::InputAction::ToggleConnectInfo => {}
                                }
                            }
                            TuiMode::ConnectSettings => {
                                self.handle_connect_key(key.code);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_command(&mut self, cmd: &str) {
        let cmd = cmd.trim();
        match cmd {
            "/connect" => {
                self.mode = TuiMode::ConnectSettings;
            }
            "/quit" | "/exit" => {
                self.should_quit = true;
            }
            "/clear" => {
                self.messages.clear();
                self.current_response.clear();
            }
            _ => {
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!("Unknown command: {}. Try /connect, /clear, /quit", cmd),
                    reasoning: None,
                });
            }
        }
        self.input_buffer.clear();
    }

    fn handle_connect_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => {
                self.mode = TuiMode::Normal;
            }
            KeyCode::Tab => {
                self.connect_form.next_field();
            }
            KeyCode::Up => {
                if self.connect_form.show_model_list {
                    self.connect_form.model_list_index =
                        self.connect_form.model_list_index.saturating_sub(1);
                } else {
                    self.connect_form.previous_field();
                }
            }
            KeyCode::Down => {
                if self.connect_form.show_model_list {
                    if self.connect_form.model_list_index + 1 < self.connect_form.model_list.len() {
                        self.connect_form.model_list_index += 1;
                    }
                } else {
                    self.connect_form.next_field();
                }
            }
            KeyCode::Left => {
                self.connect_form.on_left();
            }
            KeyCode::Right => {
                self.connect_form.on_right();
            }
            KeyCode::Char(c) => {
                self.connect_form.on_char(c);
            }
            KeyCode::Backspace => {
                self.connect_form.on_backspace();
            }
            KeyCode::Enter => {
                if self.connect_form.show_model_list && !self.connect_form.model_list.is_empty() {
                    self.connect_form.select_model();
                } else {
                    match self.connect_form.selected_field {
                        3 => {
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
                        4 => {
                            self.save_connect_settings();
                        }
                        5 => {
                            self.mode = TuiMode::Normal;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn save_connect_settings(&mut self) {
        let config = self.connect_form.to_llm_config();
        self.model_name = self.connect_form.model.clone();
        let summary = format!(
            "{} @ {}",
            self.connect_form.model,
            self.connect_form.base_url
        );
        let _ = self.cmd_tx.send(TuiCommand::Reconfigure(config));
        self.status_text = format!("Connected: {}", summary);
        self.mode = TuiMode::Normal;
    }

    fn handle_app_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Token(text) => {
                self.current_response.push_str(&text);
                if let Some(last) = self.messages.last_mut() {
                    if last.role == "assistant" {
                        last.content = self.current_response.clone();
                    } else {
                        self.messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: self.current_response.clone(),
                            reasoning: None,
                        });
                    }
                } else {
                    self.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: self.current_response.clone(),
                        reasoning: None,
                    });
                }
                self.status_text = "Generating...".into();
            }
            AppEvent::Reasoning(text) => {
                self.current_reasoning.push_str(&text);
            }
            AppEvent::AgentStatus(status) => {
                self.status_text = status;
            }
            AppEvent::Error(msg) => {
                if self.mode == TuiMode::ConnectSettings {
                    self.connect_form.fetching_models = false;
                    self.connect_form.status_message = format!("Error: {}", msg);
                } else {
                    self.messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!("Error: {}", msg),
                        reasoning: None,
                    });
                    self.status_text = "Error".into();
                }
            }
            AppEvent::Done => {
                self.status_text = "Idle".into();
                if !self.current_reasoning.is_empty() {
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == "assistant" {
                            last.reasoning = Some(self.current_reasoning.clone());
                        }
                    }
                }
                self.current_response.clear();
                self.current_reasoning.clear();
            }
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
        }
    }

    fn render(&self, f: &mut ratatui::Frame) {
        let size = f.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(size);

        match self.mode {
            TuiMode::Normal => {
                self.render_chat_area(chunks[0], f);
            }
            TuiMode::ConnectSettings => {
                f.render_widget(self.render_connect_settings(), chunks[0]);
            }
        }

        let prompt_text = format!("> {}", self.input_buffer);
        let cursor_abs = self.input_buffer.len() + 2;
        let cursor_x = (cursor_abs % size.width as usize) as u16;
        let cursor_y = chunks[1].y;
        let prompt = Paragraph::new(prompt_text).block(Block::default().borders(Borders::TOP));
        f.render_widget(prompt, chunks[1]);
        f.set_cursor_position(Position::new(cursor_x, cursor_y));

        let status_style = match self.status_text.as_str() {
            s if s.starts_with("Idle") => Style::default().fg(Color::Green),
            s if s.starts_with("Waiting")
                || s.starts_with("Thinking")
                || s.starts_with("Connected") =>
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
            Span::raw(" | model: "),
            Span::styled(&self.model_name, Style::default().fg(Color::Magenta)),
            Span::raw(" | Ctrl+C: Interrupt | Ctrl+D: Quit | /connect: Configure"),
        ]));
        f.render_widget(status, chunks[2]);
    }

    fn render_chat_area(&self, area: ratatui::layout::Rect, f: &mut ratatui::Frame) {
        if self.messages.is_empty() {
            return;
        }

        let mut heights: Vec<usize> = Vec::new();
        for msg in &self.messages {
            let content_lines = msg.content.lines().count().max(1);
            let reasoning_lines = msg.reasoning.as_ref().map(|r| r.lines().count()).unwrap_or(0);
            let frame_padding = if msg.role == "user" { 2 } else { 0 };
            heights.push(content_lines + reasoning_lines + frame_padding);
        }

        let available = area.height as usize;
        let mut total = 0usize;
        let mut start_idx = heights.len();
        for i in (0..heights.len()).rev() {
            if total + heights[i] > available {
                break;
            }
            total += heights[i];
            start_idx = i;
        }
        if start_idx >= self.messages.len() {
            return;
        }

        let visible_heights: Vec<usize> = heights[start_idx..].to_vec();
        if visible_heights.is_empty() {
            return;
        }

        let constraints: Vec<Constraint> = visible_heights
            .iter()
            .map(|&h| Constraint::Length(h as u16))
            .collect();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        for (j, msg_index) in (start_idx..self.messages.len()).enumerate() {
            if j >= chunks.len() { break; }
            let msg = &self.messages[msg_index];
            let rect = chunks[j];

            f.render_widget(self.render_message(msg), rect);
        }
    }

    fn render_message(&self, msg: &ChatMessage) -> Paragraph<'_> {
        let mut lines: Vec<Line> = Vec::new();

        if let Some(reasoning) = &msg.reasoning {
            for line in reasoning.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }

        for line in msg.content.lines() {
            lines.push(Line::from(Span::raw(line.to_string())));
        }

        let block = match msg.role.as_str() {
            "user" => Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
            "system" => Block::default()
                .borders(Borders::NONE)
                .style(Style::default().fg(Color::Yellow)),
            _ => Block::default().borders(Borders::NONE),
        };

        let style = match msg.role.as_str() {
            "user" => Style::default().fg(Color::White),
            "system" => Style::default().fg(Color::Yellow),
            _ => Style::default(),
        };

        Paragraph::new(lines).block(block).style(style)
    }

    fn render_connect_settings(&self) -> Paragraph<'_> {
        let form = &self.connect_form;

        let provider_highlight = form.selected_field == 0;
        let url_highlight = form.selected_field == 1;
        let key_highlight = form.selected_field == 2;
        let model_highlight = form.selected_field == 3;
        let save_highlight = form.selected_field == 4;
        let cancel_highlight = form.selected_field == 5;

        let key_display = if form.api_key.is_empty() {
            "(not set)".to_string()
        } else {
            let len = form.api_key.len();
            if len <= 8 {
                "***".to_string()
            } else {
                format!(
                    "{}...{}",
                    &form.api_key[..4],
                    &form.api_key[len - 4..]
                )
            }
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
            Span::styled(format!("  {}", key_display), Style::default().fg(Color::White))
        };

        let model_field = if model_highlight {
            Span::styled(
                format!("> {}", form.model),
                Style::default().fg(Color::White).bg(Color::DarkGray),
            )
        } else {
            Span::styled(format!("  {}", form.model), Style::default().fg(Color::White))
        };

        let save_btn = if save_highlight {
            Span::styled(
                "[ Save & Apply ]",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green),
            )
        } else {
            Span::styled(
                "  Save & Apply  ",
                Style::default().fg(Color::Green),
            )
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
        lines.push(Line::from(vec![
            Span::raw("  Provider:   "),
            provider_options,
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Base URL:   "),
            url_field,
        ]));
        lines.push(Line::from(vec![
            Span::raw("  API Key:    "),
            key_field,
        ]));
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

        if form.show_model_list && !form.model_list.is_empty() {
            let start = form
                .model_list_index
                .saturating_sub(4);
            let end = (start + 9).min(form.model_list.len());
            for i in start..end {
                let entry = &form.model_list[i];
                let model_line = if i == form.model_list_index {
                    Line::from(Span::styled(
                        format!("    > {}", entry),
                        Style::default().fg(Color::White).bg(Color::DarkGray),
                    ))
                } else {
                    Line::from(Span::raw(format!("      {}", entry)))
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

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" /connect — Configure Connection ")
            .border_style(Style::default().fg(Color::Cyan));
        Paragraph::new(lines).block(block)
    }
}

pub use input::InputAction;
