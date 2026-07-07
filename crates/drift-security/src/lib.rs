//! drift-security: Permission engine and security model for DriftCLI.
//!
//! Two-axis security: ApprovalPolicy (when to ask) × SandboxMode (what OS allows).
//! Tool-level pattern matching for fine-grained control.
//! Circuit breakers for doom-loop detection and critical command protection.
//! Sensitive data redaction and file access guards.

pub mod circuit;
pub mod engine;
pub mod guard;
pub mod pattern;
pub mod redact;
pub mod types;

pub use circuit::DoomLoopTracker;
pub use engine::PermissionEngine;
pub use guard::FileAccessGuard;
pub use pattern::PatternMatcher;
pub use redact::SensitiveDataFilter;
pub use types::*;
