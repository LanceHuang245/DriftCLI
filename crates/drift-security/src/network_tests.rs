use super::*;

#[test]
fn allows_matching_subdomains() {
    let guard = NetworkGuard::new(&["example.com".into()], &[]);
    assert!(guard.check_url("https://api.example.com/v1").is_ok());
}

#[test]
fn blocked_domain_wins() {
    let guard = NetworkGuard::new(&["*".into()], &["example.com".into()]);
    assert!(matches!(
        guard.check_url("https://example.com"),
        Err(NetworkAccessDenied::BlockedDomain(_))
    ));
}

#[test]
fn rejects_non_http_schemes() {
    let guard = NetworkGuard::new(&["*".into()], &[]);
    assert!(matches!(
        guard.check_url("file:///tmp/secrets"),
        Err(NetworkAccessDenied::InvalidScheme(_))
    ));
}
