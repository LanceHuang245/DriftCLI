use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

/// A point in terminal screen coordinates (column, row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionPoint {
    pub column: u16,
    pub row: u16,
}

/// A normalized (canonically ordered) selection range.
/// Guarantees `start <= end` in (row, column) tuple order.
#[derive(Debug, Clone, Copy)]
pub struct SelectionRange {
    pub start: SelectionPoint,
    pub end: SelectionPoint,
}

impl SelectionRange {
    /// Create a normalized range regardless of drag direction.
    pub fn new(anchor: SelectionPoint, current: SelectionPoint) -> Self {
        if (anchor.row, anchor.column) <= (current.row, current.column) {
            Self {
                start: anchor,
                end: current,
            }
        } else {
            Self {
                start: current,
                end: anchor,
            }
        }
    }

    /// Returns (start_col, end_col) exclusive for the given row within the area.
    /// Returns None if the row is outside the selected range.
    fn columns_for_row(&self, row: u16, area: Rect) -> Option<(u16, u16)> {
        if row < self.start.row || row > self.end.row {
            return None;
        }
        let start_col = if row == self.start.row {
            self.start.column
        } else {
            area.left()
        };
        let end_col = if row == self.end.row {
            self.end.column.saturating_add(1)
        } else {
            area.right()
        };
        Some((start_col.max(area.left()), end_col.min(area.right())))
    }
}

/// Active selection being dragged.
#[derive(Debug, Clone, Copy)]
pub struct ActiveSelection {
    area: Rect,
    anchor: SelectionPoint,
    current: SelectionPoint,
    dragged: bool,
}

impl ActiveSelection {
    pub fn range(&self) -> SelectionRange {
        SelectionRange::new(self.anchor, self.current)
    }
}

/// The result of processing a mouse event through SelectionState.
#[derive(Debug, Clone)]
pub enum SelectionResult {
    /// No change needed.
    None,
    /// Redraw needed (selection state changed, or highlight cleared).
    Redraw,
}

/// Manages the lifecycle of a mouse drag text selection.
/// Uses raw buffer coordinates so selection highlight is independent of content scroll.
///
/// On mouse-up after a drag, the selection highlight persists until cleared
/// by a new click, scroll, or message submit.
#[derive(Debug, Default)]
pub struct SelectionState {
    active: Option<ActiveSelection>,
}

impl SelectionState {
    pub fn new() -> Self {
        Self { active: None }
    }

    /// Clear the current selection highlight.
    pub fn clear(&mut self) {
        self.active = None;
    }

    /// Handle a mouse event within the given area.
    /// Returns Redraw when visual state changes.
    pub fn handle_mouse_event(&mut self, event: &MouseEvent, area: Rect) -> SelectionResult {
        match event.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(point) = point_in_area(event, area) {
                    self.active = Some(ActiveSelection {
                        area,
                        anchor: point,
                        current: point,
                        dragged: false,
                    });
                    SelectionResult::Redraw
                } else {
                    self.active = None;
                    SelectionResult::Redraw
                }
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some(ref mut active) = self.active {
                    let clamped = clamp_to_area(event, area);
                    active.current = clamped;
                    active.dragged = true;
                    SelectionResult::Redraw
                } else {
                    SelectionResult::None
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if let Some(ref active) = self.active {
                    if !active.dragged {
                        // Click without drag: clear any previous selection.
                        self.active = None;
                    }
                    // Drag completed: keep highlight visible.
                    SelectionResult::Redraw
                } else {
                    SelectionResult::None
                }
            }
            _ => {
                // Scroll, right-click, or any other mouse event clears selection.
                if self.active.take().is_some() {
                    SelectionResult::Redraw
                } else {
                    SelectionResult::None
                }
            }
        }
    }

    /// Render selection highlighting directly onto the buffer.
    /// Call after the main content (Paragraph) has been rendered to the same area.
    pub fn render(&self, buf: &mut Buffer) {
        let Some(active) = self.active.as_ref() else {
            return;
        };
        let range = active.range();
        let style = Style::default().bg(Color::White).fg(Color::Black);
        let start_y = range.start.row.max(active.area.top());
        let end_y = range.end.row.min(active.area.bottom().saturating_sub(1));
        for y in start_y..=end_y {
            if let Some((start_col, end_col)) = range.columns_for_row(y, active.area) {
                let width = end_col.saturating_sub(start_col);
                if width > 0 {
                    buf.set_style(Rect::new(start_col, y, width, 1), style);
                }
            }
        }
    }

    /// Extract selected text from a buffer (for debugging or future use).
    #[allow(dead_code)]
    pub fn extract_text(range: &SelectionRange, area: Rect, buf: &Buffer) -> Option<String> {
        let start_y = range.start.row;
        let end_y = range.end.row;
        let mut lines: Vec<String> =
            Vec::with_capacity((end_y.saturating_sub(start_y) + 1) as usize);
        for y in start_y..=end_y {
            let (start_col, end_col) = range.columns_for_row(y, area)?;
            let mut line = String::new();
            for x in start_col..end_col.min(area.right()) {
                if let Some(cell) = buf.cell((x, y)) {
                    line.push_str(cell.symbol());
                }
            }
            if end_col >= area.right() {
                line = line.trim_end().to_string();
            }
            lines.push(line);
        }
        let text = lines.join("\n");
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

/// Clamp a mouse event to within the given area.
fn clamp_to_area(event: &MouseEvent, area: Rect) -> SelectionPoint {
    SelectionPoint {
        column: event.column.clamp(area.left(), area.right().saturating_sub(1)),
        row: event.row.clamp(area.top(), area.bottom().saturating_sub(1)),
    }
}

/// Return a SelectionPoint if the mouse event falls within the given area.
fn point_in_area(event: &MouseEvent, area: Rect) -> Option<SelectionPoint> {
    if event.row < area.top()
        || event.row >= area.bottom()
        || event.column < area.left()
        || event.column >= area.right()
    {
        return None;
    }
    Some(SelectionPoint {
        column: event.column,
        row: event.row,
    })
}
