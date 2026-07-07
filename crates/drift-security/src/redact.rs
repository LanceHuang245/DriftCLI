/// Strip sensitive data from tool output before persisting to transcript.
///
/// Scans for API keys, private keys, tokens, and passwords using substring heuristics.
/// Matched content is replaced with safe labels.
pub struct SensitiveDataFilter;

impl SensitiveDataFilter {
    /// Substrings that indicate the entire surrounding line should be redacted.
    const BLOCK_PATTERNS: &'static [&'static str] = &[
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN DSA PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
        "Authorization: Bearer",
        "authorization: bearer",
    ];

    /// Filter the given text, replacing sensitive content with redaction markers.
    pub fn filter(text: &str) -> String {
        let mut result = String::with_capacity(text.len());

        for line in text.lines() {
            let lower = line.to_lowercase();

            // Check block patterns (whole line redacted)
            let mut block_redacted = false;
            for pattern in Self::BLOCK_PATTERNS {
                if lower.contains(&pattern.to_lowercase()) {
                    result.push_str("[REDACTED: sensitive data]\n");
                    block_redacted = true;
                    break;
                }
            }
            if block_redacted {
                continue;
            }

            // Check for API key / secret patterns and redact inline
            let filtered = Self::redact_inline(line);
            result.push_str(&filtered);
            result.push('\n');
        }

        // Trim trailing newline if the original didn't have one
        if !text.ends_with('\n') && result.ends_with('\n') {
            result.pop();
        }

        result
    }

    /// Redact known sensitive patterns within a single line.
    fn redact_inline(line: &str) -> String {
        let lower = line.to_lowercase();

        // API key prefixes: redact the prefix + following identifier characters
        let prefixes = &[
            "sk-ant-api",
            "sk-proj-",
            "nvapi-",
            "ghp_",
            "github_pat_",
            "AKIA",
        ];

        // Sensitive key=value patterns
        let sensitive_keys = &[
            "password=", "passwd=", "pwd=",
            "secret=", "api_key=", "apikey=", "token=",
        ];

        // Collect all redaction spans (start, end) before modifying the string
        let mut spans: Vec<(usize, usize)> = Vec::new();

        for prefix in prefixes {
            let prefix_lower = prefix.to_lowercase();
            if let Some(pos) = lower.find(&prefix_lower) {
                // Find end of the key: whitespace, quote, newline, or end of line
                let end = line[pos..]
                    .find(|c: char| c.is_whitespace() || c == '\'' || c == '"')
                    .map(|p| pos + p)
                    .unwrap_or(line.len());
                // Avoid overlapping spans
                if !spans.iter().any(|(s, e)| *s <= pos && pos < *e) {
                    spans.push((pos, end));
                }
            }
        }

        for key in sensitive_keys {
            let key_lower = key.to_lowercase();
            if let Some(pos) = lower.find(&key_lower) {
                let val_start = pos + key.len();
                let val_end = line[val_start..]
                    .find(|c: char| c.is_whitespace() || c == ';' || c == '&')
                    .map(|p| val_start + p)
                    .unwrap_or(line.len());
                if !spans.iter().any(|(s, e)| *s <= val_start && val_start < *e) {
                    spans.push((val_start, val_end));
                }
            }
        }

        if spans.is_empty() {
            return line.to_string();
        }

        // Sort spans and build output
        spans.sort_by_key(|(s, _)| *s);
        let mut result = String::with_capacity(line.len());
        let mut cursor = 0;
        for (start, end) in &spans {
            if *start > cursor {
                result.push_str(&line[cursor..*start]);
            }
            result.push_str("[REDACTED]");
            cursor = *end;
        }
        if cursor < line.len() {
            result.push_str(&line[cursor..]);
        }
        result
    }

    /// Quick check: does the text contain anything sensitive?
    pub fn contains_sensitive(text: &str) -> bool {
        let lower = text.to_lowercase();
        for pattern in Self::BLOCK_PATTERNS {
            if lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }
        // Check for API key prefix patterns
        let prefixes = ["sk-ant-api", "sk-proj-", "nvapi-", "ghp_", "github_pat_"];
        for p in &prefixes {
            if lower.contains(p) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_anthropic_key() {
        let input = "key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789";
        let result = SensitiveDataFilter::filter(input);
        assert!(result.contains("[REDACTED]"), "got: {}", result);
        assert!(!result.contains("sk-ant"));
    }

    #[test]
    fn test_redact_openai_key() {
        let input = "export KEY=sk-proj-mylongkeyhere12345678";
        let result = SensitiveDataFilter::filter(input);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("sk-proj"));
    }

    #[test]
    fn test_redact_private_key() {
        let input = "File contents:\n-----BEGIN PRIVATE KEY-----\nMIIEvQ...\n-----END PRIVATE KEY-----";
        let result = SensitiveDataFilter::filter(input);
        assert!(result.contains("[REDACTED"));
        assert!(!result.contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn test_clean_text_passes_through() {
        let input = "Build completed successfully.\nTests: 42 passed.";
        let result = SensitiveDataFilter::filter(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_password_redaction() {
        let input = "db: password=supersecret123 host=localhost";
        let result = SensitiveDataFilter::filter(input);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("supersecret123"));
    }
}
