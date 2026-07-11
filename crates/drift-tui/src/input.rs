use crossterm::event::{KeyCode, KeyModifiers};

// Actions produced by input processing: submit text, interrupt work, quit the TUI, or toggle connect info overlay.
#[derive(Debug, Clone)]
pub enum InputAction {
    Submit(String),
    Interrupt,
    Quit,
    ToggleConnectInfo,
}

// Find the previous UTF-8 character boundary before the given byte position.
fn prev_char_boundary(s: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut prev = 0;
    for (i, _) in s.char_indices() {
        if i >= pos {
            return prev;
        }
        prev = i;
    }
    prev
}

// Find the next UTF-8 character boundary after the given byte position.
fn next_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    for (i, _) in s.char_indices() {
        if i > pos {
            return i;
        }
    }
    s.len()
}

// Process a single key event and return the resulting action.
// Handles editing (insert, delete, navigation) and submit/quit triggers.
pub fn process_key(
    key: KeyCode,
    modifiers: KeyModifiers,
    input_buffer: &mut String,
    cursor: &mut usize,
) -> InputAction {
    match key {
        // Submit the current input buffer content.
        KeyCode::Enter => {
            let text = input_buffer.clone();
            InputAction::Submit(text)
        }
        // Delete one character to the left of the cursor.
        KeyCode::Backspace => {
            if *cursor > 0 {
                let new_pos = prev_char_boundary(input_buffer, *cursor);
                input_buffer.remove(new_pos);
                *cursor = new_pos;
            }
            InputAction::Submit(String::new())
        }
        // Delete one character at the cursor position.
        KeyCode::Delete => {
            if *cursor < input_buffer.len() {
                input_buffer.remove(*cursor);
            }
            InputAction::Submit(String::new())
        }
        // Move cursor left by one character.
        KeyCode::Left => {
            *cursor = prev_char_boundary(input_buffer, *cursor);
            InputAction::Submit(String::new())
        }
        // Move cursor right by one character.
        KeyCode::Right => {
            *cursor = next_char_boundary(input_buffer, *cursor);
            InputAction::Submit(String::new())
        }
        // Jump to start of input line.
        KeyCode::Home => {
            *cursor = 0;
            InputAction::Submit(String::new())
        }
        // Jump to end of input line.
        KeyCode::End => {
            *cursor = input_buffer.len();
            InputAction::Submit(String::new())
        }
        // Ctrl+C is handled by the TUI loop so it can require a second press before quitting.
        KeyCode::Char(ch) if modifiers == KeyModifiers::CONTROL && ch == 'c' => InputAction::Quit,
        // Ctrl+D quits if the buffer is empty, otherwise deletes at cursor.
        KeyCode::Char(ch) if modifiers == KeyModifiers::CONTROL && ch == 'd' => {
            if input_buffer.is_empty() {
                InputAction::Quit
            } else {
                if *cursor < input_buffer.len() {
                    input_buffer.remove(*cursor);
                }
                InputAction::Submit(String::new())
            }
        }
        // Insert a typed character at the cursor position.
        KeyCode::Char(ch) => {
            input_buffer.insert_str(*cursor, &ch.to_string());
            *cursor += ch.len_utf8();
            InputAction::Submit(String::new())
        }
        // Escape interrupts the active Agent turn.
        KeyCode::Esc => InputAction::Interrupt,
        // Ignore unrecognized keys.
        _ => InputAction::Submit(String::new()),
    }
}
