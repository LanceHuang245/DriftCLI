use crossterm::event::KeyCode;

#[derive(Debug, Clone)]
pub enum InputAction {
    Submit(String),
    Quit,
    ToggleConnectInfo,
}

pub fn process_key(key: KeyCode, input_buffer: &mut String) -> InputAction {
    match key {
        KeyCode::Enter => {
            let text = input_buffer.clone();
            InputAction::Submit(text)
        }
        KeyCode::Char('c') => {
            input_buffer.push('c');
            InputAction::Submit(String::new())
        }
        KeyCode::Char('d') => {
            if input_buffer.is_empty() {
                InputAction::Quit
            } else {
                input_buffer.push('d');
                InputAction::Submit(String::new())
            }
        }
        KeyCode::Char(ch) => {
            input_buffer.push(ch);
            InputAction::Submit(String::new())
        }
        KeyCode::Backspace => {
            input_buffer.pop();
            InputAction::Submit(String::new())
        }
        KeyCode::Esc => InputAction::Quit,
        _ => InputAction::Submit(String::new()),
    }
}
