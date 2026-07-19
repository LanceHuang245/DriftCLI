/// Simple wildcard pattern matcher: `*` matches zero or more chars, `?` matches one char.
///
/// Rules are evaluated in order; the **last matching** rule wins (OpenCode convention).
/// This allows putting a broad `"*": ask` first and more specific `"git *": allow` later.
pub struct PatternMatcher;

impl PatternMatcher {
    /// Match a tool argument string against a pattern. Returns true if the pattern matches.
    ///
    /// For bash tool: the argument string is the shell command (e.g., "git status --porcelain").
    /// For file tools: the argument string is the file path.
    /// For webfetch: the argument string is the URL.
    pub fn matches(pattern: &str, input: &str) -> bool {
        let pattern_lower = pattern.to_lowercase();
        let input_lower = input.to_lowercase();
        Self::matches_impl(&pattern_lower, &input_lower)
    }

    fn matches_impl(pattern: &str, input: &str) -> bool {
        let p_chars: Vec<char> = pattern.chars().collect();
        let i_chars: Vec<char> = input.chars().collect();
        let p_len = p_chars.len();
        let i_len = i_chars.len();

        // DP table: dp[p_idx][i_idx] = can pattern[..p_idx] match input[..i_idx]
        let mut dp = vec![false; i_len + 1];
        dp[0] = true;

        // Handle leading '*' characters
        let mut p_start = 0;
        while p_start < p_len && p_chars[p_start] == '*' {
            p_start += 1;
            for value in dp.iter_mut().take(i_len + 1) {
                *value = true;
            }
        }

        for pc in p_chars.iter().skip(p_start) {
            let mut next_dp = vec![false; i_len + 1];

            if *pc == '*' {
                // '*' matches zero or more: propagate from left or above
                for j in 0..=i_len {
                    next_dp[j] = dp[j] || (j > 0 && next_dp[j - 1]);
                }
            } else {
                for j in 1..=i_len {
                    if (*pc == '?' || *pc == i_chars[j - 1]) && dp[j - 1] {
                        next_dp[j] = true;
                    }
                }
            }

            dp = next_dp;
        }

        dp[i_len]
    }

    /// Find the last matching rule's action in an ordered list of rules.
    /// Returns None if no rule matches (caller should fall back to default behavior).
    pub fn evaluate_rules(
        rules: &[super::types::PatternRule],
        input: &str,
    ) -> Option<super::types::PermissionAction> {
        let mut last_match: Option<super::types::PermissionAction> = None;

        for rule in rules {
            if Self::matches(&rule.pattern, input) {
                last_match = Some(rule.action);
            }
        }

        last_match
    }
}

#[cfg(test)]
#[path = "pattern_tests.rs"]
mod tests;
