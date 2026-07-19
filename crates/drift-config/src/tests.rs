use super::*;
#[test]
fn test_default_config() {
    let tmpl = AppConfig::default_template();
    assert!(tmpl.contains("[agent]"));
    assert!(tmpl.contains("[[providers]]"));
    assert!(tmpl.contains("active_provider"));
    assert!(tmpl.contains("auto_compaction = true"));
    assert!(tmpl.contains("compaction_threshold = 0.75"));
    assert!(tmpl.contains("compaction_target = 0.4"));
    toml::from_str::<toml::Value>(&tmpl).expect("default template must be valid TOML");
}

#[test]
fn test_compaction_config_merge() {
    let mut config = AppConfig::load_defaults_with_files().unwrap();
    let overlay: toml::Value = toml::from_str(
        "[agent]\nauto_compaction = false\ncompaction_threshold = 0.8\ncompaction_target = 0.3\n",
    )
    .unwrap();

    AppConfig::merge_toml_value(&mut config, &overlay).unwrap();

    assert!(!config.agent.auto_compaction);
    assert_eq!(config.agent.compaction_threshold, 0.8);
    assert_eq!(config.agent.compaction_target, 0.3);
}

#[test]
fn test_connection_summary() {
    let mut providers = HashMap::new();
    providers.insert(
        "test".into(),
        ProviderEntry {
            name: "test".into(),
            config: LlmConfig::Anthropic {
                api_key: "sk-ant-test1234".into(),
                model: "test-model".into(),
                base_url: "https://api.anthropic.com/v1".into(),
                reasoning_effort: None,
            },
        },
    );
    let config = AppConfig {
        agent: AgentConfig {
            model: "test-model".into(),
            max_iterations: 50,
            temperature: None,
            thinking_budget: None,
            reasoning_effort: None,
            auto_compaction: true,
            compaction_threshold: 0.75,
            compaction_target: 0.4,
        },
        active_provider: "test".into(),
        providers,
        security: SecurityConfig::default(),
        mcp: McpConfig::default(),
        migrated: false,
    };
    let summary = config.connection_summary();
    assert!(summary.contains("Anthropic"));
    assert!(summary.contains("sk-a...1234"));
}

#[test]
fn test_security_config_merge_and_save() {
    let mut config = AppConfig {
        agent: AgentConfig {
            model: default_model(),
            max_iterations: 50,
            temperature: None,
            thinking_budget: None,
            reasoning_effort: None,
            auto_compaction: true,
            compaction_threshold: 0.75,
            compaction_target: 0.4,
        },
        active_provider: String::new(),
        providers: HashMap::new(),
        security: SecurityConfig::default(),
        mcp: McpConfig::default(),
        migrated: false,
    };
    let overlay: toml::Value = toml::from_str(
        r#"
[security]
enabled = false
default_profile = "audit"
allowed_domains = ["example.com"]
blocked_domains = ["blocked.example.com"]

[security.profiles.audit]
approval_policy = "deny"
sandbox_mode = "read-only"
"#,
    )
    .unwrap();

    AppConfig::merge_toml_value(&mut config, &overlay).unwrap();
    assert!(!config.security.enabled);
    assert_eq!(config.security.default_profile, "audit");
    assert_eq!(config.security.allowed_domains, vec!["example.com"]);
    assert_eq!(config.security.blocked_domains, vec!["blocked.example.com"]);
    assert_eq!(
        config.security.profiles["audit"].approval_policy,
        drift_security::ApprovalPolicy::Deny
    );

    config.apply_permission_mode("audit", "allow").unwrap();
    assert_eq!(
        config.security.profiles["audit"].approval_policy,
        drift_security::ApprovalPolicy::Never
    );

    let serialized = config.to_toml_string().unwrap();
    let value: toml::Value = toml::from_str(&serialized).unwrap();
    assert_eq!(value["security"]["enabled"].as_bool(), Some(false));
    assert_eq!(
        value["security"]["profiles"]["audit"]["approval_policy"].as_str(),
        Some("never")
    );
}

#[test]
fn test_mcp_defaults_and_template() {
    let config = McpConfig::default();
    assert!(config.enabled);
    assert!(config.servers.is_empty());
    let template = AppConfig::default_template();
    assert!(template.contains("[mcp]"));
    assert!(template.contains("# [[mcp.servers]]"));
}

#[test]
fn test_mcp_parse_validation_and_layered_merge() {
    let server = |id: &str| McpServerConfig {
        id: id.into(),
        command: "server".into(),
        args: Vec::new(),
        env: HashMap::new(),
        transport: McpTransport::Stdio,
        auto_start: true,
    };
    let mut config = AppConfig::load_defaults_with_files().unwrap();
    let first: toml::Value = toml::from_str(
        r#"
[mcp]
enabled = true
[[mcp.servers]]
id = "alpha"
command = "first"
transport = "stdio"
[[mcp.servers]]
id = "keep"
command = "keep"
transport = "stdio"
"#,
    )
    .unwrap();
    let second: toml::Value = toml::from_str(
        r#"
[mcp]
[[mcp.servers]]
id = "alpha"
command = "second"
args = ["--flag"]
transport = "stdio"
"#,
    )
    .unwrap();
    AppConfig::merge_toml_value(&mut config, &first).unwrap();
    AppConfig::merge_toml_value(&mut config, &second).unwrap();
    assert_eq!(config.mcp.servers.len(), 2);
    assert_eq!(config.mcp.servers[0].command, "second");
    assert_eq!(config.mcp.servers[0].args, vec!["--flag"]);
    assert_eq!(config.mcp.servers[1].id, "keep");
    config.mcp.validate().unwrap();

    for id in ["bad id", "bad.id", &"a".repeat(57)] {
        let invalid = McpConfig {
            enabled: true,
            servers: vec![server(id)],
        };
        assert!(matches!(
            invalid.validate(),
            Err(ConfigError::InvalidMcp(_))
        ));
    }
    let duplicate = server("duplicate");
    let duplicates = McpConfig {
        enabled: true,
        servers: vec![duplicate.clone(), duplicate],
    };
    assert!(matches!(
        duplicates.validate(),
        Err(ConfigError::InvalidMcp(_))
    ));
    let invalid_transport = toml::from_str::<McpConfig>(
        r#"[[servers]]
id = "bad-transport"
command = "server"
transport = "sse"
"#,
    );
    assert!(invalid_transport.is_err());
    let empty_command = McpConfig {
        enabled: true,
        servers: vec![McpServerConfig {
            id: "empty-command".into(),
            command: "  ".into(),
            args: Vec::new(),
            env: HashMap::new(),
            transport: McpTransport::Stdio,
            auto_start: true,
        }],
    };
    assert!(matches!(
        empty_command.validate(),
        Err(ConfigError::InvalidMcp(_))
    ));
    let empty_env_key = McpConfig {
        enabled: true,
        servers: vec![McpServerConfig {
            id: "empty-env".into(),
            command: "server".into(),
            args: Vec::new(),
            env: [(String::new(), "value".into())].into_iter().collect(),
            transport: McpTransport::Stdio,
            auto_start: true,
        }],
    };
    assert!(matches!(
        empty_env_key.validate(),
        Err(ConfigError::InvalidMcp(_))
    ));
}

#[test]
fn test_mcp_serialization_round_trip() {
    let mut config = AppConfig::load_defaults_with_files().unwrap();
    let overlay: toml::Value = toml::from_str(
        r#"
[mcp]
enabled = false
[[mcp.servers]]
id = "roundtrip"
command = "server"
env = { TOKEN = "${TOKEN}" }
transport = "stdio"
auto_start = false
"#,
    )
    .unwrap();
    AppConfig::merge_toml_value(&mut config, &overlay).unwrap();
    let serialized = config.to_toml_string().unwrap();
    let parsed: toml::Value = toml::from_str(&serialized).unwrap();
    assert_eq!(parsed["mcp"]["enabled"].as_bool(), Some(false));
    assert_eq!(
        parsed["mcp"]["servers"][0]["id"].as_str(),
        Some("roundtrip")
    );
    assert_eq!(
        parsed["mcp"]["servers"][0]["auto_start"].as_bool(),
        Some(false)
    );
}
