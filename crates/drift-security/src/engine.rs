use crate::circuit::DoomLoopTracker;
use crate::network::NetworkGuard;
use crate::pattern::PatternMatcher;
use crate::types::*;
use std::collections::HashMap;

/// Core permission engine: checks tool calls against the active security profile.
///
/// Decision flow:
/// 1. If security is disabled → Allow
/// 2. If tool is in safe_tools → Allow
/// 3. If sandbox is ReadOnly and tool mutates → Deny
/// 4. Check do_loop circuit breaker → escalate to Ask
/// 5. Check critical command patterns → escalate to Ask
/// 6. Match tool_rules patterns (last-match-wins) → Allow/Ask/Deny
/// 7. Fall back to approval_policy:
///    - Never → Allow
///    - OnRequest → Ask for mutate tools, Allow for read tools
///    - Untrusted → Ask for everything not in safe_tools
pub struct PermissionEngine {
    /// Active security profile
    profile: SecurityProfile,
    /// Whether permission checks are enabled at all
    enabled: bool,
    /// Doom loop tracker (lives for the session)
    doom_loop: DoomLoopTracker,
    /// Counter for generating unique request IDs
    request_counter: u64,
    /// Session-persistent rules added by AllowAlways/DenyAlways user decisions.
    /// Each entry: (tool_name, args_pattern, action). Checked before profile rules.
    session_rules: Vec<(String, String, PermissionAction)>,
    /// Shared network policy used by network-capable tools.
    network_guard: NetworkGuard,
}

impl PermissionEngine {
    /// Create a PermissionEngine from a SecurityConfig, picking the named profile.
    /// Falls back to the "default" profile if the named one is not found.
    pub fn new(config: &SecurityConfig, profile_name: &str) -> Self {
        let profile = config
            .profiles
            .get(profile_name)
            .cloned()
            .unwrap_or_else(|| {
                tracing::warn!(
                    profile = profile_name,
                    "Security profile not found, falling back to default"
                );
                config
                    .profiles
                    .get("default")
                    .cloned()
                    .unwrap_or_else(|| SecurityProfile {
                        name: "default".into(),
                        approval_policy: ApprovalPolicy::OnRequest,
                        sandbox_mode: SandboxMode::WorkspaceWrite,
                        tool_rules: HashMap::new(),
                        circuit_breakers: CircuitBreakerConfig::default(),
                        protected_paths: vec![],
                        safe_tools: vec![
                            "read".into(), "grep".into(), "glob".into(), "todowrite".into(),
                        ],
                    })
            });

        let doom_loop = DoomLoopTracker::new(&profile.circuit_breakers);

        Self {
            profile,
            enabled: config.enabled,
            doom_loop,
            request_counter: 0,
            session_rules: Vec::new(),
            network_guard: NetworkGuard::new(
                &config.allowed_domains,
                &config.blocked_domains,
            ),
        }
    }

    /// Create an engine with an explicit profile (for testing or programmatic use).
    pub fn with_profile(profile: SecurityProfile, enabled: bool) -> Self {
        let doom_loop = DoomLoopTracker::new(&profile.circuit_breakers);
        Self {
            profile,
            enabled,
            doom_loop,
            request_counter: 0,
            session_rules: Vec::new(),
            network_guard: NetworkGuard::new(&["*".into()], &[]),
        }
    }

    /// The main entry point: check whether a tool call should be allowed.
    ///
    /// Returns a `PermissionDecision`:
    /// - `Allowed` → agent proceeds immediately
    /// - `Denied` → agent returns error to LLM
    /// - `AskUser` → agent pauses and waits for user response via channel
    pub fn check_tool_permission(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> PermissionDecision {
        // 0. Security disabled → allow everything
        if !self.enabled {
            return PermissionDecision::Allowed {
                rule: "security_disabled".into(),
            };
        }

        // 1. Safe tools always auto-approve
        if self.is_safe_tool(tool_name) {
            return PermissionDecision::Allowed {
                rule: "safe_tool".into(),
            };
        }
        // 1. Session-persistent rules (AllowAlways/DenyAlways decisions)
        let arg_text = self.extract_arg_text(tool_name, args);
        for (rule_tool, rule_pattern, action) in &self.session_rules {
            if rule_tool == tool_name && PatternMatcher::matches(rule_pattern, &arg_text) {
                return match action {
                    PermissionAction::Allow => PermissionDecision::Allowed {
                        rule: "session_rule".into(),
                    },
                    PermissionAction::Deny => PermissionDecision::Denied {
                        reason: format!("Session rule denies '{}'", arg_text),
                    },
                    PermissionAction::Ask => {
                        return self.make_ask(tool_name, args, "Session rule requires confirmation");
                    }
                };
            }
        }
        // 2. ReadOnly sandbox — deny all mutating tools
        if self.profile.sandbox_mode == SandboxMode::ReadOnly && self.is_mutating_tool(tool_name)
        {
            return PermissionDecision::Denied {
                reason: format!(
                    "Tool '{}' is not allowed in ReadOnly sandbox mode",
                    tool_name
                ),
            };
        }

        // 3. Doom loop circuit breaker
        if self.profile.circuit_breakers.doom_loop {
            let fp = DoomLoopTracker::fingerprint(tool_name, args);
            if self.doom_loop.record(tool_name, &fp) {
                tracing::warn!(
                    tool = tool_name,
                    fingerprint = %fp,
                    "Doom loop detected — forcing Ask"
                );
                return self.make_ask(tool_name, args, "Doom loop detected: repeated identical call");
            }
        }

        // 4. Critical command always-ask (even in Never mode)
        if tool_name == "bash" || tool_name == "Bash" {
            let cmd = args
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if DoomLoopTracker::is_critical_command(
                &self.profile.circuit_breakers.always_ask_commands,
                cmd,
            ) {
                return self.make_ask(tool_name, args, "Critical/destructive command");
            }
        }

        // 5. Tool-specific pattern rules (last-match-wins)
        let arg_text = self.extract_arg_text(tool_name, args);
        if let Some(action) = self.match_tool_rules(tool_name, &arg_text) {
            return match action {
                PermissionAction::Allow => PermissionDecision::Allowed {
                    rule: format!("tool_rule:{}", tool_name),
                },
                PermissionAction::Deny => PermissionDecision::Denied {
                    reason: format!(
                        "Tool '{}' denied by pattern rule for '{}'",
                        tool_name, arg_text
                    ),
                },
                PermissionAction::Ask => {
                    self.make_ask(tool_name, args, "Matched tool rule requiring confirmation")
                }
            };
        }

        // 6. Fall back to approval policy
        match self.profile.approval_policy {
            ApprovalPolicy::Never => PermissionDecision::Allowed {
                rule: "approval_policy:never".into(),
            },
            ApprovalPolicy::OnRequest => {
                if self.is_mutating_tool(tool_name) {
                    self.make_ask(tool_name, args, "Mutating tool requires confirmation")
                } else {
                    PermissionDecision::Allowed {
                        rule: "approval_policy:on-request".into(),
                    }
                }
            }
            ApprovalPolicy::Untrusted => {
                // Untrusted: only safe tools auto-pass (already handled above)
                self.make_ask(tool_name, args, "Untrusted mode requires confirmation")
            }
            ApprovalPolicy::Deny => PermissionDecision::Denied {
                reason: "Permission mode is set to 'deny'".into(),
            },
        }
    }

    /// Build an AskUser decision with a unique request ID.
    fn make_ask(&mut self, tool_name: &str, args: &serde_json::Value, reason: &str) -> PermissionDecision {
        self.request_counter += 1;
        let request_id = format!("perm-{}", self.request_counter);

        let args_summary = self.summarize_args(tool_name, args);

        PermissionDecision::AskUser {
            request: PermissionRequest {
                request_id,
                tool_name: tool_name.to_string(),
                args_summary,
                reason: reason.to_string(),
                risk_level: self.assess_risk(tool_name, args),
            },
        }
    }

    /// Check if a tool is in the safe_tools list.
    fn is_safe_tool(&self, tool_name: &str) -> bool {
        self.profile.safe_tools.iter().any(|t| t == tool_name)
    }

    /// Check if a tool mutates files or executes commands.
    fn is_mutating_tool(&self, tool_name: &str) -> bool {
        matches!(
            tool_name,
            "bash" | "Bash"
                | "write" | "Write"
                | "edit" | "Edit"
                | "web_fetch" | "webfetch" | "WebFetch"
                | "web_search" | "websearch" | "WebSearch"
        )
    }

    /// Extract a human-readable arg string for pattern matching.
    fn extract_arg_text(&self, tool_name: &str, args: &serde_json::Value) -> String {
        match tool_name {
            "bash" | "Bash" => args
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            "read" | "Read" | "edit" | "Edit" | "write" | "Write" => args
                .get("path")
                .or_else(|| args.get("file_path"))
                .or_else(|| args.get("filePath"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            "web_fetch" | "webfetch" | "WebFetch" => args
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            "web_search" | "websearch" | "WebSearch" => args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            "glob" | "Glob" => args
                .get("pattern")
                .or_else(|| args.get("path"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            "grep" | "Grep" => args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            _ => serde_json::to_string(args).unwrap_or_default(),
        }
    }

    /// Match tool-specific pattern rules. Returns the action from the last matching rule.
    fn match_tool_rules(&self, tool_name: &str, arg_text: &str) -> Option<PermissionAction> {
        // Check exact tool name first, then "*" fallback
        if let Some(rules) = self.profile.tool_rules.get(tool_name) {
            let result = PatternMatcher::evaluate_rules(rules, arg_text);
            if result.is_some() {
                return result;
            }
        }
        // "*" wildcard rules
        if let Some(rules) = self.profile.tool_rules.get("*") {
            return PatternMatcher::evaluate_rules(rules, arg_text);
        }
        None
    }

    /// Summarize tool arguments for display in the TUI permission dialog.
    fn summarize_args(&self, tool_name: &str, args: &serde_json::Value) -> String {
        match tool_name {
            "bash" | "Bash" => {
                let cmd = args
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no command)");
                // Truncate long commands
                if cmd.len() > 120 {
                    format!("{}…", &cmd[..117])
                } else {
                    cmd.to_string()
                }
            }
            "read" | "Read" | "edit" | "Edit" | "write" | "Write" => {
                let path = args
                    .get("path")
                    .or_else(|| args.get("file_path"))
                    .or_else(|| args.get("filePath"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no path)");
                path.to_string()
            }
            "web_fetch" | "webfetch" | "WebFetch" => {
                let url = args
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no url)");
                if url.len() > 80 {
                    format!("{}…", &url[..77])
                } else {
                    url.to_string()
                }
            }
            "web_search" | "websearch" | "WebSearch" => args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("(no query)")
                .to_string(),
            _ => {
                let compact = serde_json::to_string(args).unwrap_or_default();
                if compact.len() > 80 {
                    format!("{}…", &compact[..77])
                } else {
                    compact
                }
            }
        }
    }

    /// Assess risk level for a tool call.
    fn assess_risk(&self, tool_name: &str, args: &serde_json::Value) -> RiskLevel {
        match tool_name {
            "bash" | "Bash" => {
                let cmd = args
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();

                if cmd.contains("rm -rf /") || cmd.contains("dd if=") || cmd.contains("mkfs.") {
                    RiskLevel::Critical
                } else if cmd.contains("rm ")
                    || cmd.contains("sudo ")
                    || cmd.contains("chmod ")
                    || cmd.contains("chown ")
                    || cmd.contains("git push --force")
                    || cmd.contains("curl ") && cmd.contains("| sh")
                {
                    RiskLevel::High
                } else if cmd.contains("git push")
                    || cmd.contains("pip install")
                    || cmd.contains("npm install")
                    || cmd.contains("mv ")
                    || cmd.contains("cp ")
                {
                    RiskLevel::Medium
                } else {
                    RiskLevel::Low
                }
            }
            "edit" | "Edit" | "write" | "Write" => {
                let path = args
                    .get("path")
                    .or_else(|| args.get("file_path"))
                    .or_else(|| args.get("filePath"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if path.contains(".git/")
                    || path.ends_with(".env")
                    || path.contains("secrets/")
                    || path.contains("credentials")
                {
                    RiskLevel::High
                } else {
                    RiskLevel::Medium
                }
            }
            "web_fetch" | "webfetch" | "WebFetch" => RiskLevel::Medium,
            "web_search" | "websearch" | "WebSearch" => RiskLevel::Low,
            _ => RiskLevel::Low,
        }
    }


    /// Record a session-persistent rule from an AllowAlways/DenyAlways decision.
    /// The pattern string should be the tool's arg text (e.g., "cargo build" for bash).
    pub fn add_session_rule(&mut self, tool_name: &str, args_pattern: &str, action: PermissionAction) {
        self.session_rules.push((
            tool_name.to_string(),
            args_pattern.to_string(),
            action,
        ));
    }

    /// Check whether a file path is accessible given the current sandbox and protected paths.
    /// `working_dir` should be the canonical working directory from the tool context.
    /// `write` indicates whether the operation is a write/edit (enforces protected paths).
    pub fn check_file_access(&self, working_dir: &std::path::Path, file_path: &std::path::Path, write: bool) -> Result<(), crate::guard::AccessDenied> {
        let guard = self.file_access_guard(working_dir)?;
        if write {
            guard.check_write(file_path)
        } else {
            guard.check_read(file_path)
        }
    }

    /// Build the file guard used by all filesystem-capable tools.
    pub fn file_access_guard(
        &self,
        working_dir: &std::path::Path,
    ) -> Result<crate::FileAccessGuard, crate::guard::AccessDenied> {
        crate::FileAccessGuard::new(working_dir, &self.profile.protected_paths)
            .map_err(|e| crate::guard::AccessDenied::OutsideWorkspace(e.to_string()))
    }

    /// Return the network guard configured for this security profile.
    pub fn network_guard(&self) -> NetworkGuard {
        self.network_guard.clone()
    }


    // ── Accessors ──

    pub fn profile_name(&self) -> &str {
        &self.profile.name
    }

    pub fn approval_policy(&self) -> ApprovalPolicy {
        self.profile.approval_policy
    }

    pub fn sandbox_mode(&self) -> SandboxMode {
        self.profile.sandbox_mode
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}
