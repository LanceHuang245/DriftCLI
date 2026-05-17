mod components;
mod input;

use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
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
    AgentStatus(String),
    Error(String),
    Done,
}

pub struct TuiApp {
    messages: Vec<ChatMessage>,
    current_response: String,
    input_buffer: String,
    status_text: String,
    show_connect_info: bool,
    connection_summary: String,
    event_rx: mpsc::UnboundedReceiver<AppEvent>,
    cmd_tx: mpsc::UnboundedSender<String>,
    should_quit: bool,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl TuiApp {
    pub fn new(
        connection_summary: String,
        event_rx: mpsc::UnboundedReceiver<AppEvent>,
        cmd_tx: mpsc::UnboundedSender<String>,
    ) -> Self {
        Self {
            messages: Vec::new(),
            current_response: String::new(),
            input_buffer: String::new(),
            status_text: "Idle".into(),
            show_connect_info: false,
            connection_summary,
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
                                        });
                                        self.input_buffer.clear();
                                        self.current_response.clear();
                                        self.status_text = "Waiting...".into();
                                        let _ = self.cmd_tx.send(text);
                                    }
                                }
                            }
                            input::InputAction::Quit => self.should_quit = true,
                            input::InputAction::ToggleConnectInfo => {
                                self.show_connect_info = !self.show_connect_info;
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
                self.show_connect_info = !self.show_connect_info;
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
                });
            }
        }
        self.input_buffer.clear();
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
                        });
                    }
                } else {
                    self.messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: self.current_response.clone(),
                    });
                }
                self.status_text = "Generating...".into();
            }
            AppEvent::AgentStatus(status) => {
                self.status_text = status;
            }
            AppEvent::Error(msg) => {
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!("Error: {}", msg),
                });
                self.status_text = "Error".into();
            }
            AppEvent::Done => {
                self.status_text = "Idle".into();
                self.current_response.clear();
            }
        }
    }

    fn render(&self, f: &mut ratatui::Frame) {
        let size = f.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(3),
            ])
            .split(size);

        let status_style = match self.status_text.as_str() {
            "Idle" => Style::default().fg(Color::Green),
            "Waiting..." | "Thinking..." => Style::default().fg(Color::Yellow),
            "Generating..." => Style::default().fg(Color::Cyan),
            "Error" => Style::default().fg(Color::Red),
            _ => Style::default(),
        };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(" DriftCLI ", Style::default().fg(Color::Black).bg(Color::White)),
            Span::raw(" | "),
            Span::styled(self.status_text.clone(), status_style),
            Span::raw(" | Ctrl+C: Interrupt | Ctrl+D: Quit | /connect: Show config"),
        ]));
        f.render_widget(status, chunks[0]);

        if self.show_connect_info {
            let chat_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

            f.render_widget(self.render_chat(), chat_chunks[0]);

            let connect_block = Block::default()
                .borders(Borders::ALL)
                .title(" /connect — Connection Info ");
            let connect_text = Paragraph::new(self.connection_summary.as_str())
                .block(connect_block)
                .style(Style::default().fg(Color::Cyan));
            f.render_widget(connect_text, chat_chunks[1]);
        } else {
            f.render_widget(self.render_chat(), chunks[1]);
        }

        let prompt_text = format!("> {}", self.input_buffer);
        let cursor_pos = self.input_buffer.len() + 2;
        let prompt = Paragraph::new(prompt_text).block(Block::default().borders(Borders::TOP));
        f.render_widget(prompt, chunks[2]);

        f.set_cursor_position(ratatui::layout::Position::new(
            (cursor_pos % size.width as usize) as u16,
            chunks[2].y,
        ));
    }

    fn render_chat(&self) -> Paragraph<'_> {
        let mut lines: Vec<Line> = Vec::new();

        for msg in &self.messages {
            let prefix = match msg.role.as_str() {
                "user" => Span::styled("You: ", Style::default().fg(Color::Green)),
                "assistant" => Span::styled("Drift: ", Style::default().fg(Color::Cyan)),
                "system" => Span::styled("System: ", Style::default().fg(Color::Yellow)),
                _ => Span::raw(""),
            };

            for line in msg.content.lines() {
                lines.push(Line::from(vec![prefix.clone(), Span::raw(line.to_string())]));
            }
        }

        let block = Block::default().borders(Borders::NONE);
        Paragraph::new(lines).block(block)
    }
}

pub use input::InputAction;
