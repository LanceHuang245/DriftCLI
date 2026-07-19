use super::{ConfigError, McpConfig};
use std::collections::HashSet;

impl McpConfig {
    /// Validates identifiers and process fields before MCP startup.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut server_ids = HashSet::with_capacity(self.servers.len());
        for server in &self.servers {
            if server.id.is_empty()
                || server.id.len() > 56
                || !server.id.chars().enumerate().all(|(index, ch)| {
                    (index == 0 && ch.is_ascii_alphanumeric())
                        || (index > 0 && (ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')))
                })
            {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has an invalid or overly long id",
                    server.id
                )));
            }
            if !server_ids.insert(&server.id) {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has a duplicate id",
                    server.id
                )));
            }
            if server.command.trim().is_empty() {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has an empty command",
                    server.id
                )));
            }
            if server.env.keys().any(String::is_empty) {
                return Err(ConfigError::InvalidMcp(format!(
                    "server '{}' has an empty environment key",
                    server.id
                )));
            }
        }
        Ok(())
    }
}
