use std::path::{Component, Path, PathBuf};

/// Guards file access: ensures operations stay within the working directory
/// and don't touch protected paths.
#[derive(Debug)]
pub struct FileAccessGuard {
    /// Absolute, canonical working directory.
    working_dir: PathBuf,
    /// Workspace-relative glob-like patterns for paths that are always read-only.
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
        // Match configured patterns against the workspace-relative path they describe.
        let path_str = resolved
            .strip_prefix(&self.working_dir)
            .expect("workspace boundary was checked above")
            .to_string_lossy();
        for pattern in &self.protected_patterns {
            if Self::simple_glob_match(pattern, &path_str) {
                return Err(AccessDenied::ProtectedPath(path_str.to_string()));
            }
        }
        Ok(())
    }

    /// Resolve existing components through symlinks while preserving a missing target suffix.
    pub fn resolve(&self, path: &Path) -> PathBuf {
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_dir.join(path)
        };

        let mut resolved = PathBuf::new();
        for component in candidate.components() {
            match component {
                Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
                Component::RootDir => resolved.push(component.as_os_str()),
                Component::CurDir => {}
                Component::ParentDir => {
                    resolved.pop();
                }
                Component::Normal(part) => {
                    resolved.push(part);
                    // Canonicalize each existing component so a symlink cannot hide an escape.
                    if let Ok(canonical) = resolved.canonicalize() {
                        resolved = canonical;
                    }
                }
            }
        }
        resolved
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
                    if Self::star_match(part, &remaining[..remaining.len().min(part.len() + 100)])
                        .is_none()
                    {
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

        for pc in p.iter().take(pn) {
            let mut next = vec![false; sn + 1];
            if *pc == '*' {
                for si in 0..=sn {
                    next[si] = dp[si] || (si > 0 && next[si - 1]);
                }
            } else {
                for si in 1..=sn {
                    if (*pc == '?' || *pc == s[si - 1]) && dp[si - 1] {
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
#[path = "guard_tests.rs"]
mod tests;
