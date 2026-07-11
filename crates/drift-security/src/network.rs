/// Checks URLs against the configured scheme and domain allow/deny lists.
#[derive(Debug, Clone)]
pub struct NetworkGuard {
    allowed_domains: Vec<String>,
    blocked_domains: Vec<String>,
}

/// Reasons a network request was rejected before leaving the process.
#[derive(Debug, thiserror::Error)]
pub enum NetworkAccessDenied {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    #[error("URL scheme is not allowed: {0}")]
    InvalidScheme(String),
    #[error("domain is blocked: {0}")]
    BlockedDomain(String),
    #[error("domain is not in the allow list: {0}")]
    NotAllowedDomain(String),
}

impl NetworkGuard {
    /// Create a guard from the top-level security network settings.
    pub fn new(allowed_domains: &[String], blocked_domains: &[String]) -> Self {
        Self {
            allowed_domains: allowed_domains.to_vec(),
            blocked_domains: blocked_domains.to_vec(),
        }
    }

    /// Validate an HTTP(S) URL, with blocked domains taking precedence.
    pub fn check_url(&self, raw_url: &str) -> Result<(), NetworkAccessDenied> {
        let url = url::Url::parse(raw_url)
            .map_err(|error| NetworkAccessDenied::InvalidUrl(error.to_string()))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(NetworkAccessDenied::InvalidScheme(url.scheme().to_string()));
        }

        let host = url
            .host_str()
            .ok_or_else(|| NetworkAccessDenied::InvalidUrl("URL has no host".into()))?;
        if Self::matches_any(host, &self.blocked_domains) {
            return Err(NetworkAccessDenied::BlockedDomain(host.to_string()));
        }

        // Keep the default allow-all behavior while honoring explicit allow lists.
        if self.allowed_domains.iter().any(|pattern| pattern == "*")
            || Self::matches_any(host, &self.allowed_domains)
        {
            return Ok(());
        }

        Err(NetworkAccessDenied::NotAllowedDomain(host.to_string()))
    }

    fn matches_any(host: &str, patterns: &[String]) -> bool {
        patterns.iter().any(|pattern| {
            let pattern = pattern.trim().trim_start_matches("*.");
            host == pattern || host.ends_with(&format!(".{pattern}"))
        })
    }
}

#[cfg(test)]
mod tests {
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
}
