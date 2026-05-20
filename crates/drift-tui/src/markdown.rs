use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

pub fn render_markdown(input: &str, area_width: u16) -> Vec<Line<'static>> {
    let parser = Parser::new(input);
    let mut writer = Writer::new(area_width);
    for event in parser {
        writer.handle(event);
    }
    writer.commit_line();
    writer.lines
}

struct Writer {
    width: u16,
    lines: Vec<Line<'static>>,
    spans: Vec<Span<'static>>,
    line_w: usize,
    bq_depth: usize,
    list_stack: Vec<ListKind>,
    ordered_counters: Vec<u64>,
    item_first_line: bool,
    in_code_block: bool,
    heading: Option<HeadingLevel>,
    style_stack: Vec<Style>,
    pending_space: bool,
}

enum ListKind {
    Unordered,
    Ordered,
}

impl Writer {
    fn new(width: u16) -> Self {
        Self {
            width,
            lines: Vec::new(),
            spans: Vec::new(),
            line_w: 0,
            bq_depth: 0,
            list_stack: Vec::new(),
            ordered_counters: Vec::new(),
            item_first_line: false,
            in_code_block: false,
            heading: None,
            style_stack: Vec::new(),
            pending_space: false,
        }
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) => self.text(&text),
            Event::Code(text) => self.code(&text),
            Event::Html(_)
            | Event::InlineHtml(_)
            | Event::FootnoteReference(_)
            | Event::TaskListMarker(_)
            | Event::InlineMath(_)
            | Event::DisplayMath(_) => {}
            Event::SoftBreak => self.soft_break(),
            Event::HardBreak => self.hard_break(),
            Event::Rule => self.rule(),
        }
    }

    fn start_tag(&mut self, tag: Tag) {
        match tag {
            Tag::Paragraph => {
                if self.line_w > 0 {
                    self.commit_line();
                }
            }
            Tag::Heading { level, .. } => {
                self.commit_line();
                self.heading = Some(level);
            }
            Tag::BlockQuote(_) => {
                self.commit_line();
                self.bq_depth += 1;
            }
            Tag::List(start) => {
                self.commit_line();
                self.list_stack.push(match start {
                    Some(_) => ListKind::Ordered,
                    None => ListKind::Unordered,
                });
                self.ordered_counters.push(start.unwrap_or(0));
            }
            Tag::Item => {
                self.commit_line();
                if let Some(kind) = self.list_stack.last() {
                    if matches!(kind, ListKind::Ordered) {
                        if let Some(counter) = self.ordered_counters.last_mut() {
                            *counter += 1;
                        }
                    }
                }
                self.item_first_line = true;
            }
            Tag::CodeBlock(_) => {
                self.commit_line();
                self.in_code_block = true;
            }
            Tag::Emphasis => {
                let base = self.style_stack.last().copied().unwrap_or_default();
                self.style_stack.push(base.add_modifier(Modifier::ITALIC));
            }
            Tag::Strong => {
                let base = self.style_stack.last().copied().unwrap_or_default();
                self.style_stack.push(base.add_modifier(Modifier::BOLD));
            }
            Tag::Strikethrough => {
                let base = self.style_stack.last().copied().unwrap_or_default();
                self.style_stack
                    .push(base.add_modifier(Modifier::CROSSED_OUT));
            }
            Tag::Link { .. } => {
                let base = self.style_stack.last().copied().unwrap_or_default();
                self.style_stack
                    .push(base.fg(Color::Blue).add_modifier(Modifier::UNDERLINED));
            }
            Tag::Image { .. } => {}
            Tag::FootnoteDefinition(_)
            | Tag::Table(_)
            | Tag::TableHead
            | Tag::TableRow
            | Tag::TableCell
            | Tag::HtmlBlock
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition
            | Tag::MetadataBlock(_) => {
                self.commit_line();
            }
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.commit_line(),
            TagEnd::Heading(_) => {
                self.commit_line();
                self.heading = None;
            }
            TagEnd::BlockQuote(_) => {
                self.commit_line();
                self.bq_depth = self.bq_depth.saturating_sub(1);
            }
            TagEnd::List(_) => {
                self.commit_line();
                self.list_stack.pop();
                self.ordered_counters.pop();
                self.item_first_line = false;
            }
            TagEnd::Item => {
                self.commit_line();
                self.item_first_line = false;
            }
            TagEnd::CodeBlock => {
                self.commit_line();
                self.in_code_block = false;
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.style_stack.pop();
            }
            TagEnd::Link | TagEnd::Image => {
                self.style_stack.pop();
            }
            _ => {}
        }
    }

    fn text(&mut self, text: &str) {
        if self.in_code_block {
            self.push_code_block_lines(text);
            return;
        }

        let style = self.current_text_style();

        if std::mem::take(&mut self.pending_space) {
            let combined = format!(" {}", text);
            self.push_words(&combined, style);
        } else {
            self.push_words(text, style);
        }
    }

    fn code(&mut self, text: &str) {
        let style = Style::default().fg(Color::Cyan);
        self.push_words(text, style);
    }

    fn soft_break(&mut self) {
        self.pending_space = true;
    }

    fn hard_break(&mut self) {
        self.pending_space = false;
        self.commit_line();
    }

    fn rule(&mut self) {
        self.commit_line();
        let count = (self.width as usize / 3).max(1);
        let rule = "───".repeat(count);
        let w = (self.width as usize).min(rule.len());
        self.lines.push(Line::from(Span::styled(
            rule[..w].to_string(),
            Style::default().fg(Color::DarkGray),
        )));
    }

    fn push_code_block_lines(&mut self, text: &str) {
        for line_str in text.lines() {
            if self.line_w > 0 {
                self.commit_line();
            }
            let indent = self.bq_indent() + "  ";
            self.spans.push(Span::styled(
                format!("{}{}", indent, line_str),
                Style::default().fg(Color::Cyan),
            ));
            self.lines
                .push(Line::from(self.spans.drain(..).collect::<Vec<_>>()));
            self.line_w = 0;
        }
    }

    fn push_words(&mut self, text: &str, style: Style) {
        for word in split_words(text) {
            let w = UnicodeWidthStr::width(word);
            if w == 0 {
                continue;
            }

            self.ensure_prefix();

            let available = self.width.saturating_sub(self.line_w as u16) as usize;
            if w > available && self.line_w > self.prefix_width() {
                self.commit_line();
                self.ensure_prefix();
            }

            let trimmed = if self.line_w == self.prefix_width() {
                word.trim_start()
            } else {
                word
            };

            if trimmed.is_empty() {
                continue;
            }

            let tw = UnicodeWidthStr::width(trimmed);
            self.spans.push(Span::styled(trimmed.to_string(), style));
            self.line_w += tw;
        }
    }

    fn ensure_prefix(&mut self) {
        if self.line_w > 0 {
            return;
        }

        let prefix = self.compute_prefix();
        let prefix_style = self.prefix_style();

        if !prefix.is_empty() {
            self.spans.push(Span::styled(prefix, prefix_style));
            self.line_w = 0;
            for span in &self.spans {
                self.line_w += UnicodeWidthStr::width(span.content.as_ref());
            }
        }
    }

    fn commit_line(&mut self) {
        if !self.spans.is_empty() {
            self.lines
                .push(Line::from(self.spans.drain(..).collect::<Vec<_>>()));
        }
        self.line_w = 0;
        if self.item_first_line {
            self.item_first_line = false;
        }
    }

    fn compute_prefix(&self) -> String {
        let mut p = String::new();
        for _ in 0..self.bq_depth {
            p.push_str("│ ");
        }
        for (i, kind) in self.list_stack.iter().enumerate() {
            if i == self.list_stack.len() - 1 && self.item_first_line {
                match kind {
                    ListKind::Unordered => p.push_str("- "),
                    ListKind::Ordered => {
                        let n = self.ordered_counters.get(i).copied().unwrap_or(1);
                        p.push_str(&format!("{}. ", n));
                    }
                }
            } else {
                p.push_str("  ");
            }
        }
        p
    }

    fn prefix_width(&self) -> usize {
        UnicodeWidthStr::width(self.compute_prefix().as_str())
    }

    fn bq_indent(&self) -> String {
        let mut s = String::new();
        for _ in 0..self.bq_depth {
            s.push_str("│ ");
        }
        s
    }

    fn prefix_style(&self) -> Style {
        if self.bq_depth > 0 {
            Style::default().fg(Color::DarkGray)
        } else if !self.list_stack.is_empty() {
            Style::default().fg(Color::LightBlue)
        } else {
            Style::default()
        }
    }

    fn current_text_style(&self) -> Style {
        if let Some(level) = self.heading {
            match level {
                HeadingLevel::H1 => {
                    Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                }
                HeadingLevel::H2 => Style::default().add_modifier(Modifier::BOLD),
                HeadingLevel::H3 => {
                    Style::default().add_modifier(Modifier::BOLD | Modifier::ITALIC)
                }
                _ => Style::default().add_modifier(Modifier::BOLD),
            }
        } else {
            self.style_stack.last().copied().unwrap_or_default()
        }
    }
}

fn split_words(s: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            if start < i {
                words.push(&s[start..i]);
            }
            let end = i + c.len_utf8();
            words.push(&s[i..end]);
            start = end;
        }
    }
    if start < s.len() {
        words.push(&s[start..]);
    }
    words
}
