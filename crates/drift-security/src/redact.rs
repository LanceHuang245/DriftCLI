/// Strip sensitive data from session payloads before persisting a transcript.
///
/// Scans text and structured arguments for API keys, private keys, tokens, and passwords.
/// Matched content is replaced with safe labels while preserving non-sensitive structure.
pub struct SensitiveDataFilter;

impl SensitiveDataFilter {
    /// Substrings that make the entire surrounding line sensitive.
    const BLOCK_PATTERNS: &'static [&'static str] = &["authorization: bearer"];

    /// Prefixes that identify common provider and service credentials.
    const SECRET_PREFIXES: &'static [&'static str] = &[
        "sk-ant-api",
        "sk-proj-",
        "nvapi-",
        "ghp_",
        "github_pat_",
        "akia",
    ];

    /// Assignments whose values should be removed from free-form text.
    const SENSITIVE_ASSIGNMENTS: &'static [&'static str] = &[
        "password=",
        "passwd=",
        "pwd=",
        "secret=",
        "api_key=",
        "apikey=",
        "token=",
    ];

    /// Filter the given text, replacing sensitive content with redaction markers.
    pub fn filter(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut in_private_key = false;

        for line in text.lines() {
            let lower = line.to_ascii_lowercase();

            // Drop every line from a private-key header through its matching footer.
            if in_private_key {
                if lower.contains("-----end ") && lower.contains("private key-----") {
                    in_private_key = false;
                }
                continue;
            }
            if lower.contains("-----begin ") && lower.contains("private key-----") {
                result.push_str("[REDACTED: private key]\n");
                in_private_key = true;
                continue;
            }

            // Replace lines whose structure cannot be preserved without exposing a secret.
            if Self::BLOCK_PATTERNS
                .iter()
                .any(|pattern| lower.contains(pattern))
            {
                result.push_str("[REDACTED: sensitive data]\n");
                continue;
            }

            // Preserve non-sensitive text around every inline secret.
            result.push_str(&Self::redact_inline(line));
            result.push('\n');
        }

        // Match the original trailing-newline behavior.
        if !text.ends_with('\n') && result.ends_with('\n') {
            result.pop();
        }

        result
    }

    /// Recursively redact strings and values under sensitive JSON keys.
    pub fn filter_json(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    Self::filter_json(value);
                }
            }
            serde_json::Value::Object(values) => {
                for (key, value) in values {
                    if Self::is_sensitive_json_key(key) {
                        *value = serde_json::Value::String("[REDACTED]".into());
                        continue;
                    }

                    // Parse serialized tool arguments so their field names remain visible.
                    if key == "arguments"
                        && let serde_json::Value::String(arguments) = value
                        && let Ok(mut parsed) = serde_json::from_str(arguments)
                    {
                        Self::filter_json(&mut parsed);
                        if let Ok(redacted) = serde_json::to_string(&parsed) {
                            *arguments = redacted;
                            continue;
                        }
                    }

                    Self::filter_json(value);
                }
            }
            serde_json::Value::String(text) => {
                *text = Self::filter(text);
            }
            _ => {}
        }
    }

    /// Redact every known sensitive pattern within a single line.
    fn redact_inline(line: &str) -> String {
        let lower = line.to_ascii_lowercase();
        let mut spans: Vec<(usize, usize)> = Vec::new();

        // Collect all prefixed credentials instead of stopping after the first match.
        for prefix in Self::SECRET_PREFIXES {
            let mut offset = 0;
            while offset < line.len() {
                let Some(relative_pos) = lower[offset..].find(prefix) else {
                    break;
                };
                let start = offset + relative_pos;
                let end = line[start..]
                    .find(|c: char| c.is_whitespace() || c == '\'' || c == '"')
                    .map(|pos| start + pos)
                    .unwrap_or(line.len());
                spans.push((start, end));
                offset = end.max(start + prefix.len());
            }
        }

        // Collect all values from sensitive key=value assignments.
        for assignment in Self::SENSITIVE_ASSIGNMENTS {
            let mut offset = 0;
            while offset < line.len() {
                let Some(relative_pos) = lower[offset..].find(assignment) else {
                    break;
                };
                let value_start = offset + relative_pos + assignment.len();
                let value_end = line[value_start..]
                    .find(|c: char| c.is_whitespace() || matches!(c, ';' | '&' | ',' | '\'' | '"'))
                    .map(|pos| value_start + pos)
                    .unwrap_or(line.len());
                spans.push((value_start, value_end));
                offset = value_end.max(value_start);
            }
        }

        if spans.is_empty() {
            return line.to_string();
        }

        // Merge overlapping matches so each secret is replaced exactly once.
        spans.sort_by_key(|(start, _)| *start);
        let mut merged_spans: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            if let Some((_, previous_end)) = merged_spans.last_mut()
                && start <= *previous_end
            {
                *previous_end = (*previous_end).max(end);
            } else {
                merged_spans.push((start, end));
            }
        }

        // Rebuild the line while retaining all non-sensitive segments.
        let mut result = String::with_capacity(line.len());
        let mut cursor = 0;
        for (start, end) in merged_spans {
            result.push_str(&line[cursor..start]);
            result.push_str("[REDACTED]");
            cursor = end;
        }
        result.push_str(&line[cursor..]);
        result
    }

    /// Identify JSON object keys that conventionally hold credentials.
    fn is_sensitive_json_key(key: &str) -> bool {
        let key = key.to_ascii_lowercase();
        matches!(
            key.as_str(),
            "password"
                | "passwd"
                | "pwd"
                | "secret"
                | "api_key"
                | "apikey"
                | "token"
                | "access_token"
                | "refresh_token"
                | "client_secret"
                | "authorization"
        ) || key.ends_with("_password")
            || key.ends_with("_secret")
            || key.ends_with("_token")
            || key.ends_with("_api_key")
    }

    /// Quick check: does the text contain anything sensitive?
    pub fn contains_sensitive(text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        (lower.contains("-----begin ") && lower.contains("private key-----"))
            || Self::BLOCK_PATTERNS
                .iter()
                .any(|pattern| lower.contains(pattern))
            || Self::SECRET_PREFIXES
                .iter()
                .any(|prefix| lower.contains(prefix))
            || Self::SENSITIVE_ASSIGNMENTS
                .iter()
                .any(|assignment| lower.contains(assignment))
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

    // Verify the complete private-key block is removed from persisted text.
    #[test]
    fn test_redact_private_key() {
        let input =
            "File contents:\n-----BEGIN PRIVATE KEY-----\nMIIEvQ...\n-----END PRIVATE KEY-----";
        let result = SensitiveDataFilter::filter(input);
        assert!(result.contains("[REDACTED"));
        assert!(!result.contains("BEGIN PRIVATE KEY"));
        assert!(!result.contains("MIIEvQ"));
        assert!(!result.contains("END PRIVATE KEY"));
    }

    // Verify every sensitive value on a line is removed, not only the first match.
    #[test]
    fn test_redact_multiple_values_on_one_line() {
        let input = "first=sk-proj-one second=sk-proj-two password=alpha token=beta";
        let result = SensitiveDataFilter::filter(input);
        assert_eq!(result.matches("[REDACTED]").count(), 4);
        assert!(!result.contains("sk-proj"));
        assert!(!result.contains("alpha"));
        assert!(!result.contains("beta"));
    }

    // Verify structured and serialized arguments redact secrets without damaging other values.
    #[test]
    fn test_redact_json_secret_fields() {
        let mut input = serde_json::json!({
            "token": "plain-secret",
            "arguments": r#"{"access_token":"nested-secret","path":"src/lib.rs"}"#,
            "nested": { "client_secret": "other-secret", "path": "src/main.rs" }
        });
        SensitiveDataFilter::filter_json(&mut input);
        assert_eq!(input["token"], "[REDACTED]");
        assert_eq!(
            input["arguments"],
            r#"{"access_token":"[REDACTED]","path":"src/lib.rs"}"#
        );
        assert_eq!(input["nested"]["client_secret"], "[REDACTED]");
        assert_eq!(input["nested"]["path"], "src/main.rs");
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
