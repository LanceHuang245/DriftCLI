use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Two-Axis Model
// ============================================================
//    ApprovalPolicy (when to ask)  ×  SandboxMode (what OS allows)
// ============================================================

/// Approval axis: controls when the agent must ask the user for permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovalPolicy {
    /// Only safe reads auto-run; everything else requires approval.
    Untrusted,
    /// Run within current permission level; ask only to escalate (default).
    #[default]
    OnRequest,
    /// No prompts at all. OS sandbox still enforces limits.
    Never,
    /// Deny every non-safe tool without prompting.
    Deny,
}

/// Sandbox axis: controls what the operating system physically permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    /// No writes, no command execution.
    ReadOnly,
    /// Read anywhere, write only within the working directory (default).
    #[default]
    WorkspaceWrite,
    /// No OS-level restrictions. Use only in disposable environments.
    DangerFullAccess,
}

// ============================================================
// Permission Decisions
// ============================================================

/// The outcome of a single permission check.
#[derive(Debug, Clone)]
pub enum PermissionDecision {
    /// Tool call is allowed — proceed immediately.
    Allowed { rule: String },
    /// Tool call is denied — return error to LLM.
    Denied { reason: String },
    /// User confirmation required — agent pauses and waits for TUI input.
    AskUser {
        request: PermissionRequest,
    },
}

/// A permission request sent from the agent to the TUI.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub request_id: String,
    pub tool_name: String,
    pub args_summary: String,
    pub reason: String,
    pub risk_level: RiskLevel,
}

/// User's response to a permission prompt in the TUI.
#[derive(Debug, Clone)]
pub enum PermissionResponse {
    /// Allow this single invocation.
    Allow,
    /// Deny this single invocation.
    Deny,
    /// Allow and create a session-persistent rule.
    AllowAlways,
    /// Deny and create a session-persistent rule.
    DenyAlways,
}

// ============================================================
// Fine-Grained Tool Rules (OpenCode-style pattern matching)
// ============================================================

/// A single pattern rule: `pattern` → `action`. Last match wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRule {
    pub pattern: String,
    pub action: PermissionAction,
}

/// Rules for a specific tool. `tool_name: None` means "applies to all tools".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPermissionRules {
    pub rules: Vec<PatternRule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// Per-tool map: tool name → ordered list of pattern rules.
pub type ToolPermissionSet = HashMap<String, Vec<PatternRule>>;

/// Action taken when a tool pattern matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionAction {
    Allow,
    Ask,
    Deny,
}

// ============================================================
// Security Profile
// ============================================================

/// A named security profile: approval policy + sandbox mode + tool rules + circuit breakers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityProfile {
    /// Display name (e.g. "dev", "audit", "ci").
    pub name: String,

    #[serde(default)]
    pub approval_policy: ApprovalPolicy,

    #[serde(default)]
    pub sandbox_mode: SandboxMode,

    /// Per-tool fine-grained pattern rules.
    #[serde(default)]
    pub tool_rules: ToolPermissionSet,

    /// Circuit breaker configuration for this profile.
    #[serde(default)]
    pub circuit_breakers: CircuitBreakerConfig,

    /// Glob patterns for paths that are always read-only, even in WorkspaceWrite mode.
    #[serde(default)]
    pub protected_paths: Vec<String>,

    /// Tools that never require permission approval, regardless of mode.
    #[serde(default = "default_safe_tools")]
    pub safe_tools: Vec<String>,
}

/// Tools that are safe to auto-approve in every mode.
fn default_safe_tools() -> Vec<String> {
    vec![
        "read".into(),
        "grep".into(),
        "glob".into(),
        "todowrite".into(),
    ]
}

// ============================================================
// Circuit Breakers
// ============================================================

/// Circuit breaker configuration: doom-loop detection and critical-command protection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable doom-loop detection (repeated identical tool calls).
    #[serde(default = "default_true")]
    pub doom_loop: bool,
    /// Number of repeats within the window that trigger escalation to Ask.
    #[serde(default = "default_doom_loop_max")]
    pub doom_loop_max_repeats: usize,
    /// Time window in seconds for counting repeats.
    #[serde(default = "default_doom_loop_window")]
    pub doom_loop_window_secs: u64,

    /// High-risk command patterns that always require user confirmation,
    /// even when approval policy is Never.
    #[serde(default = "default_critical_commands")]
    pub always_ask_commands: Vec<String>,
}

fn default_true() -> bool {
    true
}

fn default_doom_loop_max() -> usize {
    3
}

fn default_doom_loop_window() -> u64 {
    60
}

/// Commands that are always treated as critical and force a user prompt.
fn default_critical_commands() -> Vec<String> {
    vec![
        "rm -rf /*".into(),
        "rm -rf ~/*".into(),
        "rm -rf /".into(),
        "dd if=*".into(),
        "mkfs.*".into(),
        ":(){ :|:& };:".into(),
        "> /dev/*".into(),
        "chmod 777 *".into(),
        "git push --force *".into(),
    ]
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            doom_loop: true,
            doom_loop_max_repeats: 3,
            doom_loop_window_secs: 60,
            always_ask_commands: default_critical_commands(),
        }
    }
}

// ============================================================
// Security Config (wired into AppConfig)
// ============================================================

/// Top-level security configuration, written as the `[security]` section in config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Name of the default profile to use when none is specified via CLI.
    #[serde(default = "default_profile_name")]
    pub default_profile: String,

    /// Named security profiles keyed by name.
    #[serde(default)]
    pub profiles: HashMap<String, SecurityProfile>,

    /// Master switch: when `false`, all permission checks are skipped.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Domains allowed by network-capable tools. `*` allows every domain.
    #[serde(default = "default_allowed_domains")]
    pub allowed_domains: Vec<String>,

    /// Domains denied by network-capable tools. Deny rules take precedence.
    #[serde(default)]
    pub blocked_domains: Vec<String>,
}

fn default_profile_name() -> String {
    "default".into()
}

fn default_allowed_domains() -> Vec<String> {
    vec!["*".into()]
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let mut default_profile = SecurityProfile {
            name: "default".into(),
            approval_policy: ApprovalPolicy::OnRequest,
            sandbox_mode: SandboxMode::WorkspaceWrite,
            tool_rules: HashMap::new(),
            circuit_breakers: CircuitBreakerConfig::default(),
            protected_paths: vec![],
            safe_tools: default_safe_tools(),
        };
        default_profile.tool_rules = default_tool_rules();

        let mut profiles = HashMap::new();
        profiles.insert("default".into(), default_profile.clone());

        // "readonly" profile: untrusted approval + read-only sandbox
        profiles.insert("readonly".into(), {
            let mut p = default_profile.clone();
            p.name = "readonly".into();
            p.approval_policy = ApprovalPolicy::Untrusted;
            p.sandbox_mode = SandboxMode::ReadOnly;
            p
        });

        // "auto" profile: on-request + workspace-write (same as default)
        profiles.insert("auto".into(), {
            let mut p = default_profile;
            p.name = "auto".into();
            p
        });

        // "danger" profile: never ask + no sandbox. Critical commands still gate.
        profiles.insert("danger".into(), {
            SecurityProfile {
                name: "danger".into(),
                approval_policy: ApprovalPolicy::Never,
                sandbox_mode: SandboxMode::DangerFullAccess,
                tool_rules: HashMap::new(),
                circuit_breakers: CircuitBreakerConfig {
                    doom_loop: true,
                    doom_loop_max_repeats: 3,
                    doom_loop_window_secs: 60,
                    always_ask_commands: default_critical_commands(),
                },
                protected_paths: vec![],
                safe_tools: default_safe_tools(),
            }
        });

        Self {
            default_profile: "default".into(),
            profiles,
            enabled: true,
            allowed_domains: default_allowed_domains(),
            blocked_domains: Vec::new(),
        }
    }
}

/// Default per-tool pattern rules: safe dev commands auto-allow, destructive ones deny.
///
/// Rules are evaluated **last-match-wins**: broad patterns like `"*"` go first,
/// more specific patterns like `"git push *"` go later.
fn default_tool_rules() -> ToolPermissionSet {
    let mut set = HashMap::new();

    // bash: dev commands auto-allow, mutating commands ask, destructive commands deny
    set.insert(
        "bash".into(),
        vec![
            PatternRule {
                pattern: "rm -rf /*".into(),
                action: PermissionAction::Deny,
            },
            PatternRule {
                pattern: "git push --force *".into(),
                action: PermissionAction::Ask,
            },
            PatternRule {
                pattern: "git push *".into(),
                action: PermissionAction::Ask,
            },
            PatternRule {
                pattern: "git *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "cargo *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "npm *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "pip *".into(),
                action: PermissionAction::Ask,
            },
            PatternRule {
                pattern: "grep *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "rg *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "ls *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "dir *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "pwd".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "echo *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "cat *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "head *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "tail *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "wc *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "sort *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "uniq *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "find *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "which *".into(),
                action: PermissionAction::Allow,
            },
            PatternRule {
                pattern: "*".into(),
                action: PermissionAction::Ask,
            },
        ],
    );

    // edit/write: deny sensitive paths, ask for everything else
    let file_mutate_rules = vec![
        PatternRule {
            pattern: "*.env".into(),
            action: PermissionAction::Deny,
        },
        PatternRule {
            pattern: "*.env.*".into(),
            action: PermissionAction::Deny,
        },
        PatternRule {
            pattern: ".git/*".into(),
            action: PermissionAction::Deny,
        },
        PatternRule {
            pattern: ".drift/*".into(),
            action: PermissionAction::Deny,
        },
        PatternRule {
            pattern: "*".into(),
            action: PermissionAction::Ask,
        },
    ];
    set.insert("edit".into(), file_mutate_rules.clone());
    set.insert("write".into(), file_mutate_rules);

    // web_fetch/web_search: ask by default
    set.insert(
        "web_fetch".into(),
        vec![PatternRule {
            pattern: "*".into(),
            action: PermissionAction::Ask,
        }],
    );
    set.insert(
        "web_search".into(),
        vec![PatternRule {
            pattern: "*".into(),
            action: PermissionAction::Ask,
        }],
    );

    set
}

// ============================================================
// Risk Level
// ============================================================

/// Severity of a tool call, used by the TUI to colour-code permission prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================
// Errors
// ============================================================

/// Errors produced by the permission engine.
#[derive(Debug, thiserror::Error)]
pub enum PermissionError {
    #[error("Denied: {0}")]
    Denied(String),
    #[error("Tool '{0}' not allowed in ReadOnly sandbox mode")]
    ReadOnlyViolation(String),
}
