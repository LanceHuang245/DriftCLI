use super::is_provider_safe_tool_name;

#[test]
fn provider_safe_tool_names_reject_invalid_characters_and_length() {
    assert!(is_provider_safe_tool_name("mcp__server__echo"));
    assert!(!is_provider_safe_tool_name("mcp__server__bad.name"));
    assert!(!is_provider_safe_tool_name(&"a".repeat(65)));
}
