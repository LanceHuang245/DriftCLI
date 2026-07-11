use crate::types::CircuitBreakerConfig;
use crate::pattern::PatternMatcher;
use std::collections::HashMap;
use std::time::Instant;

/// Tracks repeated tool calls to detect doom loops.
///
/// A doom loop occurs when the LLM calls the same tool with the same arguments
/// repeatedly, usually because it failed to understand the error and keeps retrying.
pub struct DoomLoopTracker {
    /// (tool_name, args_fingerprint) → (repeat_count, first_seen)
    history: HashMap<(String, String), (usize, Instant)>,
    max_repeats: usize,
    window_secs: u64,
}

impl DoomLoopTracker {
    /// Create a tracker configured with the given circuit breaker settings.
    pub fn new(config: &CircuitBreakerConfig) -> Self {
        Self {
            history: HashMap::new(),
            max_repeats: config.doom_loop_max_repeats,
            window_secs: config.doom_loop_window_secs,
        }
    }

    /// Record a tool call and return whether it triggered a doom loop.
    ///
    /// # Returns
    /// - `true` if the same (tool, args_fingerprint) has been called `max_repeats` times
    ///   within the time window — the caller should escalate to `Ask`.
    /// - `false` otherwise.
    pub fn record(&mut self, tool_name: &str, args_fingerprint: &str) -> bool {
        let key = (tool_name.to_string(), args_fingerprint.to_string());
        let now = Instant::now();

        // Clean expired entries
        self.history.retain(|_, (_, first)| {
            now.duration_since(*first).as_secs() < self.window_secs
        });

        let entry = self.history.entry(key).or_insert((0, now));
        entry.0 += 1;

        if entry.0 >= self.max_repeats {
            // Reset to avoid triggering on every subsequent call
            entry.0 = 0;
            entry.1 = now;
            true
        } else {
            false
        }
    }

    /// Check if the command matches an always-ask critical pattern.
    /// Returns true if the command should ALWAYS require user confirmation,
    /// regardless of approval policy mode.
    pub fn is_critical_command(always_ask: &[String], command: &str) -> bool {
        // Treat each critical pattern as a candidate; if any matches, it's critical.
        for pattern in always_ask {
            if PatternMatcher::matches(pattern, command) {
                return true;
            }
        }
        false
    }

    /// Build a short fingerprint string from tool arguments.
    /// For bash: the command string. For file tools: the target path.
    pub fn fingerprint(tool_name: &str, args: &serde_json::Value) -> String {
        match tool_name {
            "bash" | "Bash" => {
                args.get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "read" | "Read" | "edit" | "Edit" | "write" | "Write" => {
                args.get("path")
                    .or_else(|| args.get("file_path"))
                    .or_else(|| args.get("filePath"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "glob" | "Glob" => {
                args.get("pattern")
                    .or_else(|| args.get("path"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "grep" | "Grep" => {
                args.get("pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "web_fetch" | "webfetch" | "WebFetch" => {
                args.get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "web_search" | "websearch" | "WebSearch" => args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            _ => {
                // Use compact JSON as fingerprint
                serde_json::to_string(args).unwrap_or_default()
            }
        }
    }
}
