use super::*;

fn is_ask(decision: PermissionDecision) -> bool {
    matches!(decision, PermissionDecision::AskUser { .. })
}

#[test]
fn mcp_tools_ask_unless_read_only_denies_execution() {
    let config = SecurityConfig::default();
    for profile in ["default", "auto", "danger"] {
        let mut engine = PermissionEngine::new(&config, profile);
        assert!(is_ask(engine.check_tool_permission(
            "mcp__server__tool",
            &serde_json::json!({})
        )));
    }
    let mut read_only = PermissionEngine::new(&config, "readonly");
    assert!(matches!(
        read_only.check_tool_permission("mcp__server__tool", &serde_json::json!({})),
        PermissionDecision::Denied { .. }
    ));
}

#[test]
fn mcp_explicit_and_session_rules_keep_priority() {
    let mut config = SecurityConfig::default();
    config
        .profiles
        .get_mut("default")
        .unwrap()
        .tool_rules
        .insert(
            "mcp__server__allow".into(),
            vec![PatternRule {
                pattern: "*".into(),
                action: PermissionAction::Allow,
            }],
        );
    config
        .profiles
        .get_mut("default")
        .unwrap()
        .tool_rules
        .insert(
            "mcp__server__deny".into(),
            vec![PatternRule {
                pattern: "*".into(),
                action: PermissionAction::Deny,
            }],
        );
    let mut engine = PermissionEngine::new(&config, "default");
    assert!(matches!(
        engine.check_tool_permission("mcp__server__allow", &serde_json::json!({})),
        PermissionDecision::Allowed { .. }
    ));
    assert!(matches!(
        engine.check_tool_permission("mcp__server__deny", &serde_json::json!({})),
        PermissionDecision::Denied { .. }
    ));

    engine.add_session_rule("mcp__server__tool", "*", PermissionAction::Allow);
    assert!(matches!(
        engine.check_tool_permission("mcp__server__tool", &serde_json::json!({})),
        PermissionDecision::Allowed { .. }
    ));
}

#[test]
fn disabled_security_allows_mcp_tools() {
    let config = SecurityConfig::default();
    let profile = config.profiles["default"].clone();
    let mut engine = PermissionEngine::with_profile(profile, false);
    assert!(matches!(
        engine.check_tool_permission("mcp__server__tool", &serde_json::json!({})),
        PermissionDecision::Allowed { .. }
    ));
}

#[test]
fn read_only_boundary_survives_disabled_approval_checks() {
    let config = SecurityConfig::default();
    let profile = config.profiles["readonly"].clone();
    let mut engine = PermissionEngine::with_profile(profile, false);

    for tool in ["bash", "write", "edit", "mcp__server__tool"] {
        assert!(matches!(
            engine.check_tool_permission(tool, &serde_json::json!({})),
            PermissionDecision::Denied { .. }
        ));
    }
}

#[test]
fn default_profile_allows_common_workspace_tools() {
    let config = SecurityConfig::default();
    let mut engine = PermissionEngine::new(&config, "default");

    for (tool, args) in [
        ("write", serde_json::json!({ "filePath": "src/new.rs" })),
        ("edit", serde_json::json!({ "filePath": "src/lib.rs" })),
        (
            "bash",
            serde_json::json!({ "command": "mkdir -p target/tmp" }),
        ),
    ] {
        assert!(
            matches!(
                engine.check_tool_permission(tool, &args),
                PermissionDecision::Allowed { .. }
            ),
            "{tool} should run without a permission prompt inside the workspace"
        );
    }
}

#[test]
fn default_profile_keeps_sensitive_operations_gated() {
    let config = SecurityConfig::default();
    let mut engine = PermissionEngine::new(&config, "default");

    assert!(matches!(
        engine.check_tool_permission("write", &serde_json::json!({ "filePath": ".env" })),
        PermissionDecision::Denied { .. }
    ));
    assert!(is_ask(engine.check_tool_permission(
        "bash",
        &serde_json::json!({ "command": "git push origin main" })
    )));
}
