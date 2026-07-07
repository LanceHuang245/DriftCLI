use std::path::{Path, PathBuf};

/// Guards file access: ensures operations stay within the working directory
/// and don't touch protected paths.
pub struct FileAccessGuard {
    /// Absolute, canonical working directory.
    working_dir: PathBuf,
    /// Glob-like patterns for paths that are always read-only.
    protected_patterns: Vec<String>,
}

/// Reason a file access was denied.
#[derive(Debug)]
pub enum AccessDenied {
    /// The path escapes the working directory.
    OutsideWorkspace(String),
    /// The path matches a protected pattern and cannot be written.
    ProtectedPath(String),
}

impl FileAccessGuard {
    /// Create a new guard with the given canonical working directory and protected patterns.
    pub fn new(working_dir: &Path, protected_patterns: &[String]) -> Result<Self, std::io::Error> {
        let canonical = working_dir.canonicalize()?;
        Ok(Self {
            working_dir: canonical,
            protected_patterns: protected_patterns.to_vec(),
        })
    }

    /// Check that a read operation on the given path is permitted.
    /// Returns Ok(()) or an access denial reason.
    pub fn check_read(&self, path: &Path) -> Result<(), AccessDenied> {
        let resolved = self.resolve(path);
        if !resolved.starts_with(&self.working_dir) {
            return Err(AccessDenied::OutsideWorkspace(
                resolved.display().to_string(),
            ));
        }
        Ok(())
    }

    /// Check that a write/edit operation on the given path is permitted.
    /// Same as read check, but also enforces protected paths.
    pub fn check_write(&self, path: &Path) -> Result<(), AccessDenied> {
        let resolved = self.resolve(path);
        if !resolved.starts_with(&self.working_dir) {
            return Err(AccessDenied::OutsideWorkspace(
                resolved.display().to_string(),
            ));
        }
        // Check protected patterns
        let path_str = resolved.to_string_lossy();
        for pattern in &self.protected_patterns {
            if Self::simple_glob_match(pattern, &path_str) {
                return Err(AccessDenied::ProtectedPath(path_str.to_string()));
            }
        }
        Ok(())
    }

    /// Resolve a path: if absolute, use as-is; if relative, join with working_dir.
    fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            // Try to canonicalize; if it doesn't exist yet, normalize manually
            path.canonicalize().unwrap_or_else(|_| {
                // Normalize ".." and "." manually
                let mut components = Vec::new();
                for c in path.components() {
                    match c {
                        std::path::Component::ParentDir => {
                            components.pop();
                        }
                        std::path::Component::CurDir => {}
                        other => components.push(other),
                    }
                }
                let mut result = PathBuf::new();
                for c in components {
                    result.push(c);
                }
                result
            })
        } else {
            let joined = self.working_dir.join(path);
            joined.canonicalize().unwrap_or_else(|_| joined)
        }
    }

    /// Simple glob matching: `*` matches any sequence, `?` matches one char.
    /// `**` matches across directory separators.
    /// Returns true if pattern matches the full path string.
    fn simple_glob_match(pattern: &str, path: &str) -> bool {
        if pattern == "*" || pattern == "**" {
            return true;
        }

        let path_unix = path.replace('\\', "/");
        let pat_unix = pattern.replace('\\', "/");

        // ** matches across path separators
        if pat_unix.contains("**") {
            let parts: Vec<&str> = pat_unix.split("**").collect();
            let mut remaining = path_unix.as_str();
            for (i, part) in parts.iter().enumerate() {
                let part = part.trim_start_matches('/');
                if i == 0 {
                    // First part must match from start
                    if !Self::star_match(part, &remaining[..remaining.len().min(part.len() + 100)]).is_some() {
                        return false;
                    }
                    if let Some(pos) = Self::star_match(part, remaining) {
                        remaining = &remaining[pos..];
                    } else {
                        return false;
                    }
                } else {
                    // Later parts can match anywhere in remaining
                    let found = remaining.find(part);
                    match found {
                        Some(pos) => remaining = &remaining[pos + part.len()..],
                        None => return false,
                    }
                }
            }
            true
        } else {
            Self::star_match(&pat_unix, &path_unix).is_some()
        }
    }

    /// Match a pattern with `*` and `?` wildcards against input.
    /// Returns the end position of the match, or None.
    fn star_match(pattern: &str, input: &str) -> Option<usize> {
        let p: Vec<char> = pattern.chars().collect();
        let s: Vec<char> = input.chars().collect();
        let pn = p.len();
        let sn = s.len();

        let mut dp = vec![false; sn + 1];
        dp[0] = true;

        for pi in 0..pn {
            let mut next = vec![false; sn + 1];
            if p[pi] == '*' {
                for si in 0..=sn {
                    next[si] = dp[si] || (si > 0 && next[si - 1]);
                }
            } else {
                for si in 1..=sn {
                    if (p[pi] == '?' || p[pi] == s[si - 1]) && dp[si - 1] {
                        next[si] = true;
                    }
                }
            }
            dp = next;
        }

        if dp[sn] { Some(sn) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_within_workspace() {
        let dir = std::env::current_dir().unwrap();
        let guard = FileAccessGuard::new(&dir, &[]).unwrap();
        let test_path = dir.join("Cargo.toml");
        assert!(guard.check_read(&test_path).is_ok());
    }

    #[test]
    fn test_outside_workspace_denied() {
        let dir = std::env::current_dir().unwrap();
        let guard = FileAccessGuard::new(&dir, &[]).unwrap();
        // Try to access a path clearly outside workspace
        let outside = if cfg!(windows) {
            Path::new("C:\\Windows\\System32")
        } else {
            Path::new("/etc/passwd")
        };
        assert!(guard.check_read(outside).is_err());
    }
    #[test]
    fn test_simple_glob_match() {
        assert!(FileAccessGuard::simple_glob_match("*.env", ".env"));
        assert!(FileAccessGuard::simple_glob_match("*.env", "prod.env"));
        assert!(!FileAccessGuard::simple_glob_match("*.env", ".env.example"));
        // Path matching with wildcards
        assert!(FileAccessGuard::simple_glob_match(".git/*", ".git/config"));
        assert!(!FileAccessGuard::simple_glob_match(".git/*", "src/main.rs"));
    }
}
