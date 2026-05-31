use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

pub fn render_markdown(input: &str, area_width: u16) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(input, opts);
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
    in_table: bool,
    in_table_head: bool,
    table_alignments: Vec<Alignment>,
    table_header: Vec<TableCell>,
    table_body: Vec<Vec<TableCell>>,
    current_table_row: Vec<TableCell>,
    current_cell_spans: Vec<Span<'static>>,
}

enum ListKind {
    Unordered,
    Ordered,
}

#[derive(Clone)]
struct TableCell {
    spans: Vec<Span<'static>>,
}

impl TableCell {
    fn width(&self) -> usize {
        self.spans
            .iter()
            .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
            .sum()
    }
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
            in_table: false,
            in_table_head: false,
            table_alignments: Vec::new(),
            table_header: Vec::new(),
            table_body: Vec::new(),
            current_table_row: Vec::new(),
            current_cell_spans: Vec::new(),
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
                self.ordered_counters.push(start.map_or(0, |s| s.saturating_sub(1)));
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
            Tag::Table(alignments) => {
                self.commit_line();
                self.in_table = true;
                self.table_alignments = alignments;
                self.in_table_head = false;
            }
            Tag::TableHead => {
                self.in_table_head = true;
                self.current_table_row.clear();
            }
            Tag::TableRow => {
                self.current_table_row.clear();
            }
            Tag::TableCell => {
                self.current_cell_spans.clear();
            }
            Tag::FootnoteDefinition(_)
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
                if !self.spans.is_empty() {
                    self.commit_line();
                }
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
            TagEnd::TableCell => {
                let cell = TableCell {
                    spans: std::mem::take(&mut self.current_cell_spans),
                };
                self.current_table_row.push(cell);
            }
            TagEnd::TableRow => {
                let row = std::mem::take(&mut self.current_table_row);
                if !row.is_empty() {
                    self.table_body.push(row);
                }
            }
            TagEnd::TableHead => {
                let row = std::mem::take(&mut self.current_table_row);
                if !row.is_empty() {
                    self.table_header = row;
                }
                self.in_table_head = false;
            }
            TagEnd::Table => {
                self.render_table();
                self.in_table = false;
                self.table_alignments.clear();
                self.table_header.clear();
                self.table_body.clear();
                self.current_table_row.clear();
                self.current_cell_spans.clear();
                self.in_table_head = false;
            }
            _ => {}
        }
    }

    fn text(&mut self, text: &str) {
        if self.in_table {
            let style = self.current_text_style();
            if std::mem::take(&mut self.pending_space) {
                self.current_cell_spans
                    .push(Span::styled(" ".to_string(), style));
            }
            self.current_cell_spans
                .push(Span::styled(text.to_string(), style));
            return;
        }
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
        if self.in_table {
            let style = Style::default().fg(Color::Cyan);
            self.current_cell_spans
                .push(Span::styled(text.to_string(), style));
            return;
        }
        let style = Style::default().fg(Color::Cyan);
        self.push_words(text, style);
    }

    fn soft_break(&mut self) {
        if self.in_table {
            let style = self.current_text_style();
            self.current_cell_spans
                .push(Span::styled(" ".to_string(), style));
            return;
        }
        self.pending_space = true;
    }

    fn hard_break(&mut self) {
        if self.in_table {
            let style = self.current_text_style();
            self.current_cell_spans
                .push(Span::styled(" ".to_string(), style));
            return;
        }
        self.pending_space = false;
        self.commit_line();
    }

    fn rule(&mut self) {
        self.commit_line();
        let rule = "─".repeat(self.width as usize);
        self.lines.push(Line::from(Span::styled(
            rule,
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
        let had_content = !self.spans.is_empty();
        if had_content {
            self.lines
                .push(Line::from(self.spans.drain(..).collect::<Vec<_>>()));
        }
        self.line_w = 0;
        if had_content && self.item_first_line {
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

    fn render_table(&mut self) {
        let alignments = std::mem::take(&mut self.table_alignments);
        let header = std::mem::take(&mut self.table_header);
        let body = std::mem::take(&mut self.table_body);

        let num_cols = alignments.len();
        if num_cols == 0 || (header.is_empty() && body.is_empty()) {
            return;
        }

        let mut col_widths = vec![3usize; num_cols];
        for (i, cell) in header.iter().enumerate() {
            col_widths[i] = col_widths[i].max(cell.width());
        }
        for row in &body {
            for (i, cell) in row.iter().enumerate().take(num_cols) {
                col_widths[i] = col_widths[i].max(cell.width());
            }
        }

        let available = self.width as usize;
        loop {
            let total: usize = col_widths.iter().sum::<usize>() + 3 * num_cols + 1;
            if total <= available {
                break;
            }
            let mut max_idx = 0;
            let mut max_w = 0;
            for (i, &w) in col_widths.iter().enumerate() {
                if w > 3 && w > max_w {
                    max_w = w;
                    max_idx = i;
                }
            }
            if max_w == 0 {
                break;
            }
            col_widths[max_idx] -= 1;
        }

        let border_style = Style::default().fg(Color::DarkGray);

        self.lines.push(Line::from(Span::styled(
            Self::build_border(&col_widths, '┌', '┬', '┐'),
            border_style,
        )));

        if !header.is_empty() {
            self.lines.push(Self::build_row(
                &header,
                &col_widths,
                &alignments,
                border_style,
            ));
        }

        for row in &body {
            self.lines.push(Line::from(Span::styled(
                Self::build_border(&col_widths, '├', '┼', '┤'),
                border_style,
            )));
            self.lines.push(Self::build_row(
                row,
                &col_widths,
                &alignments,
                border_style,
            ));
        }

        self.lines.push(Line::from(Span::styled(
            Self::build_border(&col_widths, '└', '┴', '┘'),
            border_style,
        )));
    }

    fn build_border(widths: &[usize], left: char, mid: char, right: char) -> String {
        let mut s = String::new();
        s.push(left);
        for (i, &w) in widths.iter().enumerate() {
            if i > 0 {
                s.push(mid);
            }
            s.push_str(&"─".repeat(w + 2));
        }
        s.push(right);
        s
    }

    fn build_row(
        cells: &[TableCell],
        widths: &[usize],
        alignments: &[Alignment],
        border_style: Style,
    ) -> Line<'static> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        spans.push(Span::styled("│ ", border_style));
        for (i, cell) in cells.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" │ ", border_style));
            }
            let cell_w = widths.get(i).copied().unwrap_or(3);
            let content_w = cell.width();
            let pad = cell_w.saturating_sub(content_w);
            let (left_pad, right_pad) = match alignments.get(i).unwrap_or(&Alignment::None) {
                Alignment::Left | Alignment::None => (0, pad),
                Alignment::Right => (pad, 0),
                Alignment::Center => (pad / 2, pad - pad / 2),
            };
            if left_pad > 0 {
                spans.push(Span::raw(" ".repeat(left_pad)));
            }
            spans.extend(cell.spans.iter().cloned());
            if right_pad > 0 {
                spans.push(Span::raw(" ".repeat(right_pad)));
            }
        }
        spans.push(Span::styled(" │", border_style));
        Line::from(spans)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn lines_to_string(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn unordered_list() {
        let lines = render_markdown("- Item 1\n- Item 2\n", 80);
        let text = lines_to_string(&lines);
        assert!(text.contains("- Item 1"), "text was: {text}");
        assert!(text.contains("- Item 2"), "text was: {text}");
    }

    #[test]
    fn ordered_list() {
        let lines = render_markdown("1. First\n2. Second\n", 80);
        let text = lines_to_string(&lines);
        assert!(text.contains("1. First"), "text was: {text}");
        assert!(text.contains("2. Second"), "text was: {text}");
    }

    #[test]
    fn tight_unordered_list() {
        let lines = render_markdown("- A\n- B\n- C\n", 80);
        let text = lines_to_string(&lines);
        assert!(text.contains("- A"), "text was: {text}");
        assert!(text.contains("- B"), "text was: {text}");
        assert!(text.contains("- C"), "text was: {text}");
    }

    #[test]
    fn tight_ordered_list() {
        let lines = render_markdown("1. Alpha\n2. Beta\n3. Gamma\n", 80);
        let text = lines_to_string(&lines);
        assert!(text.contains("1. Alpha"), "text was: {text}");
        assert!(text.contains("2. Beta"), "text was: {text}");
        assert!(text.contains("3. Gamma"), "text was: {text}");
    }

    #[test]
    fn nested_list() {
        let lines = render_markdown("- Outer\n  - Inner\n", 80);
        let text = lines_to_string(&lines);
        assert!(text.contains("- Outer"), "text was: {text}");
        assert!(text.contains("- Inner"), "text was: {text}");
    }

    #[test]
    fn basic_table() {
        let md = "| A | B |\n| - | - |\n| 1 | 2 |\n";
        let lines = render_markdown(md, 80);
        // Should produce bordered table rows (at least top border, header, separator)
        assert!(lines.len() >= 4, "expected at least 4 lines for bordered table");
        let text = lines_to_string(&lines);
        assert!(text.contains("┌"), "table top border missing: {text}");
        assert!(text.contains("├"), "table header sep missing: {text}");
        assert!(text.contains("└"), "table bottom border missing: {text}");
        assert!(text.contains("A"), "header A missing: {text}");
        assert!(text.contains("B"), "header B missing: {text}");
        assert!(text.contains("1"), "cell 1 missing: {text}");
        assert!(text.contains("2"), "cell 2 missing: {text}");
    }

    #[test]
    fn table_with_styled_cell() {
        let md = "| **Bold** | Normal |\n| - | - |\n| data | more |\n";
        let lines = render_markdown(md, 80);
        let text = lines_to_string(&lines);
        assert!(text.contains("Bold"), "bold cell missing: {text}");
        assert!(text.contains("Normal"), "normal cell missing: {text}");
    }

    #[test]
    fn table_not_separate_lines() {
        let md = "| X | Y |\n| - | - |\n| a | b |\n";
        let lines = render_markdown(md, 80);
        let text = lines_to_string(&lines);
        // Should NOT have standalone "X" or "Y" lines — they should be in the table row
        let line_list: Vec<&str> = text.lines().collect();
        for line in &line_list {
            assert!(!line.trim().eq("X"), "found orphan X line: {text}");
            assert!(!line.trim().eq("Y"), "found orphan Y line: {text}");
        }
    }

    #[test]
    fn table_multi_body_rows_have_separators() {
        let md = "| A | B |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |\n";
        let lines = render_markdown(md, 80);
        let text = lines_to_string(&lines);
        // Count occurrences of the separator border ├
        let sep_count = text.chars().filter(|&c| c == '├').count();
        assert_eq!(sep_count, 2, "expected 2 separators (header-body + body-body), got {sep_count}: {text}");
    }
}
