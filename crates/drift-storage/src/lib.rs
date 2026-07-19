//! drift-storage: Session persistence with JSONL transcripts.
//!
//! Each session is stored as a JSONL file under `~/.drift/sessions/`.
//! Every line is a JSON object representing one event (message, tool call, etc.).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// A serializable message snapshot used to restore compacted context without depending on drift-llm.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedMessage {
    /// Provider role retained in the compacted snapshot.
    pub role: String,
    /// Ordered content parts retained without a drift-llm dependency.
    pub content: Vec<PersistedContentPart>,
}

/// A serializable content part used by [`PersistedMessage`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PersistedContentPart {
    /// Plain provider-visible text.
    Text(String),
    /// An assistant tool call and its serialized arguments.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// A tool result correlated with its originating call.
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    /// Model reasoning retained for providers that expose it.
    Reasoning(String),
}
// A single event recorded in the session transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEvent {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    #[serde(rename = "tool_call")]
    ToolCall {
        call_id: String,
        name: String,
        args: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        call_id: String,
        name: String,
        success: bool,
        content: String,
        error: Option<String>,
    },
    #[serde(rename = "context_compacted")]
    ContextCompacted {
        /// LLM summary, or None when only local tool output truncation occurred.
        summary: Option<String>,
        /// Full active message snapshot used as the replay boundary.
        messages: Vec<PersistedMessage>,
        /// Estimated tokens removed by the compaction candidate.
        saved_tokens: usize,
    },
}

// Metadata about a session (stored as the first line of each JSONL file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub created_at: String,
    pub working_dir: String,
    pub model: String,
}

// Error types for storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("session not found: {0}")]
    NotFound(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

// Manages session creation, listing, and event appending.
pub struct SessionStore {
    sessions_dir: PathBuf,
}

impl SessionStore {
    // Create a new store rooted at the given data directory.
    // Creates the sessions subdirectory if it doesn't exist.
    pub fn new(data_dir: PathBuf) -> Result<Self, StorageError> {
        let sessions_dir = data_dir.join("sessions");
        std::fs::create_dir_all(&sessions_dir)?;
        Ok(Self { sessions_dir })
    }

    // Create a new session with metadata as the first JSONL line.
    pub fn create(&self, working_dir: &str, model: &str) -> Result<(Uuid, PathBuf), StorageError> {
        let session_id = Uuid::new_v4();
        let filename = format!("{}.jsonl", session_id);
        let path = self.sessions_dir.join(&filename);

        let meta = SessionMeta {
            session_id: session_id.to_string(),
            created_at: chrono_now(),
            working_dir: working_dir.to_string(),
            model: model.to_string(),
        };

        let line = serde_json::to_string(&meta)? + "\n";
        std::fs::write(&path, line)?;

        Ok((session_id, path))
    }

    // Append an event to the session's JSONL file.
    pub fn append_event(&self, session_id: Uuid, event: &SessionEvent) -> Result<(), StorageError> {
        let filename = format!("{}.jsonl", session_id);
        let path = self.sessions_dir.join(&filename);

        if !path.exists() {
            return Err(StorageError::NotFound(session_id.to_string()));
        }

        let line = serde_json::to_string(event)? + "\n";
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
        file.write_all(line.as_bytes())?;
        file.flush()?;

        Ok(())
    }

    // List all session IDs with their metadata.
    pub fn list(&self) -> Result<Vec<SessionMeta>, StorageError> {
        let mut sessions = Vec::new();

        if !self.sessions_dir.exists() {
            return Ok(sessions);
        }

        for entry in std::fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "jsonl") {
                continue;
            }
            // Read first line for metadata
            if let Ok(content) = std::fs::read_to_string(&path)
                && let Some(line) = content.lines().next()
                && let Ok(meta) = serde_json::from_str::<SessionMeta>(line)
            {
                sessions.push(meta);
            }
        }

        // Sort newest first by created_at
        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(sessions)
    }

    // Read all events from a session JSONL file.
    pub fn read_events(&self, session_id: Uuid) -> Result<Vec<SessionEvent>, StorageError> {
        let filename = format!("{}.jsonl", session_id);
        let path = self.sessions_dir.join(&filename);

        if !path.exists() {
            return Err(StorageError::NotFound(session_id.to_string()));
        }

        let content = std::fs::read_to_string(&path)?;
        let mut events = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            // Skip metadata lines (they have type "session_meta")
            if let Ok(event) = serde_json::from_str::<SessionEvent>(line) {
                events.push(event);
            }
            // If it's a SessionMeta line, silently skip
        }

        Ok(events)
    }

    // Get the path to a session file.
    pub fn session_path(&self, session_id: Uuid) -> PathBuf {
        self.sessions_dir.join(format!("{}.jsonl", session_id))
    }
}

// Returns current UTC time as an ISO 8601 string.
fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Simple ISO 8601: YYYY-MM-DDThh:mm:ssZ
    let days_since_epoch = secs / 86400;
    let mut year = 1970i64;
    let mut remaining_days = days_since_epoch as i64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }
    let month_days = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in &month_days {
        if remaining_days < md {
            break;
        }
        remaining_days -= md;
        month += 1;
    }
    let day = remaining_days + 1;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
#[path = "storage_tests.rs"]
mod tests;
