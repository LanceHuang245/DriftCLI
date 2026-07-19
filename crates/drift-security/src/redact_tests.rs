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
    let input = "File contents:\n-----BEGIN PRIVATE KEY-----\nMIIEvQ...\n-----END PRIVATE KEY-----";
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
