/// A registered slash command with name (including leading `/`) and description.
#[derive(Clone)]
pub struct SlashCommand {
    pub name: &'static str,
    pub description: &'static str,
}

/// All available slash commands in display order.
pub const ALL_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/connect",
        description: "Configure connection settings (API key, model, etc.)",
    },
    SlashCommand {
        name: "/provider",
        description: "Switch or manage configured AI providers",
    },
    SlashCommand {
        name: "/variant",
        description: "Change reasoning effort level for the current model",
    },
    SlashCommand {
        name: "/sessions",
        description: "Manage and switch between persistent historical chat sessions",
    },
    SlashCommand {
        name: "/clear",
        description: "Clear all chat messages",
    },
    SlashCommand {
        name: "/quit",
        description: "Exit the application",
    },
];

/// Filter commands by the given text (case-insensitive).
///
/// The filter is compared against the command name *without* the leading `/`.
/// An empty filter returns all commands.
/// Exact matches are returned before prefix matches; within each group the
/// original `ALL_COMMANDS` ordering is preserved.
pub fn filter_commands(filter: &str) -> Vec<SlashCommand> {
    if filter.is_empty() {
        return ALL_COMMANDS.to_vec();
    }

    let filter = filter.to_lowercase();

    let mut exact: Vec<SlashCommand> = Vec::new();
    let mut prefix: Vec<SlashCommand> = Vec::new();

    for cmd in ALL_COMMANDS {
        // Strip the leading '/' before comparing.
        let suffix = &cmd.name[1..];
        let lower = suffix.to_lowercase();

        if lower == filter {
            exact.push(cmd.clone());
        } else if lower.starts_with(&filter) {
            prefix.push(cmd.clone());
        }
    }

    exact.extend(prefix);
    exact
}
