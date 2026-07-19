use super::*;

#[test]
fn test_basic_match() {
    assert!(PatternMatcher::matches("hello", "hello"));
    assert!(!PatternMatcher::matches("hello", "world"));
}

#[test]
fn test_wildcard_star() {
    assert!(PatternMatcher::matches("*", "anything"));
    assert!(PatternMatcher::matches("git *", "git status"));
    assert!(PatternMatcher::matches("git *", "git commit -m 'msg'"));
    assert!(!PatternMatcher::matches("git *", "cargo build"));
    assert!(PatternMatcher::matches("*.env", ".env"));
    assert!(PatternMatcher::matches("*.env", "prod.env"));
    assert!(!PatternMatcher::matches("*.env", ".env.example"));
}

#[test]
fn test_wildcard_question() {
    assert!(PatternMatcher::matches("???.env", "abc.env"));
    assert!(!PatternMatcher::matches("???.env", "ab.env"));
}

#[test]
fn test_case_insensitive() {
    assert!(PatternMatcher::matches("Git *", "GIT status"));
    assert!(PatternMatcher::matches("*.ENV", ".Env"));
}

#[test]
fn test_last_match_wins() {
    let rules = vec![
        super::super::types::PatternRule {
            pattern: "*".into(),
            action: super::super::types::PermissionAction::Ask,
        },
        super::super::types::PatternRule {
            pattern: "git *".into(),
            action: super::super::types::PermissionAction::Allow,
        },
        super::super::types::PatternRule {
            pattern: "git push *".into(),
            action: super::super::types::PermissionAction::Ask,
        },
    ];

    assert_eq!(
        PatternMatcher::evaluate_rules(&rules, "git status"),
        Some(super::super::types::PermissionAction::Allow)
    );
    assert_eq!(
        PatternMatcher::evaluate_rules(&rules, "git push origin main"),
        Some(super::super::types::PermissionAction::Ask)
    );
    assert_eq!(
        PatternMatcher::evaluate_rules(&rules, "cargo build"),
        Some(super::super::types::PermissionAction::Ask)
    );
    assert_eq!(
        PatternMatcher::evaluate_rules(&rules, "unknown"),
        Some(super::super::types::PermissionAction::Ask),
    );
}

#[test]
fn test_complex_patterns() {
    assert!(PatternMatcher::matches("rm -rf /*", "rm -rf /"));
    assert!(PatternMatcher::matches("rm -rf /*", "rm -rf /home/user"));
    assert!(PatternMatcher::matches(
        "dd if=*",
        "dd if=/dev/zero of=image"
    ));
    assert!(PatternMatcher::matches("> /dev/*", "> /dev/null"));
    assert!(PatternMatcher::matches(":(){ :|:& };:", ":(){ :|:& };:"));
}
