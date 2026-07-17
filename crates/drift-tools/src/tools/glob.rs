//! GlobTool — find files matching a glob pattern, sorted by modification time.

use crate::{Tool, ToolContext, ToolError, ToolResult};
use std::path::PathBuf;

pub struct GlobTool;

#[async_trait::async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against (supports ** for recursive matching)"
                },
                "path": {
                    "type": "string",
                    "description": "Optional subdirectory to search in (relative to working directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("pattern is required".into()))?;

        // Resolve the search directory through the shared workspace guard.
        let requested = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(std::path::Path::new)
            .unwrap_or_else(|| std::path::Path::new("."));
        ctx.file_access
            .check_read(requested)
            .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
        let search_dir = ctx.file_access.resolve(requested);

        // Validate the full pattern before glob expansion can traverse outside the workspace.
        let full_pattern = search_dir.join(pattern);
        ctx.file_access
            .check_read(&full_pattern)
            .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
        let full_pattern = full_pattern.to_string_lossy();

        // Collect matching entries with their modification times
        let mut entries: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        const MAX_RESULTS: usize = 100;

        for entry in glob::glob(&full_pattern).map_err(|e| ToolError::InvalidArgs(e.to_string()))? {
            let path = match entry {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Recheck expanded paths because wildcards may resolve through symlinks.
            ctx.file_access
                .check_read(&path)
                .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;

            // Get modification time, skip on error
            let mtime = match std::fs::metadata(&path) {
                Ok(meta) => meta.modified().unwrap_or(std::time::UNIX_EPOCH),
                Err(_) => continue,
            };

            entries.push((path, mtime));
        }

        // Sort by modification time, newest first
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.1));

        // Limit to max results
        entries.truncate(MAX_RESULTS);

        // Format output paths relative to the search directory
        let content = entries
            .iter()
            .map(|(p, _)| {
                p.strip_prefix(&search_dir)
                    .unwrap_or(p)
                    .display()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolResult {
            success: true,
            content,
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use drift_security::{FileAccessGuard, NetworkGuard, ProcessSandbox, SandboxMode};
    use std::sync::Arc;

    struct TestWorkspace {
        root: PathBuf,
        workspace: PathBuf,
    }

    impl TestWorkspace {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("drift-glob-{}", uuid::Uuid::new_v4()));
            let workspace = root.join("workspace");
            std::fs::create_dir_all(&workspace).unwrap();
            Self { root, workspace }
        }

        // Build the smallest real tool context needed to exercise workspace checks.
        fn context(&self) -> ToolContext {
            ToolContext {
                session_id: uuid::Uuid::nil(),
                working_dir: self.workspace.clone(),
                tool_call_id: "glob-test".into(),
                file_access: Arc::new(FileAccessGuard::new(&self.workspace, &[]).unwrap()),
                network: Arc::new(NetworkGuard::new(&["*".into()], &[])),
                process_sandbox: Arc::new(
                    ProcessSandbox::new(SandboxMode::DangerFullAccess, &self.workspace).unwrap(),
                ),
            }
        }
    }

    impl Drop for TestWorkspace {
        fn drop(&mut self) {
            // Remove temporary fixtures after every test run.
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[tokio::test]
    async fn rejects_pattern_that_escapes_workspace() {
        let fixture = TestWorkspace::new();
        std::fs::write(fixture.root.join("outside.txt"), "secret").unwrap();

        let result = GlobTool
            .execute(serde_json::json!({ "pattern": "../*" }), &fixture.context())
            .await;

        assert!(matches!(result, Err(ToolError::PermissionDenied(_))));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_wildcard_that_expands_through_outside_symlink() {
        let fixture = TestWorkspace::new();
        let outside = fixture.root.join("outside");
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(&outside, fixture.workspace.join("linked")).unwrap();

        let result = GlobTool
            .execute(serde_json::json!({ "pattern": "*/*" }), &fixture.context())
            .await;

        assert!(matches!(result, Err(ToolError::PermissionDenied(_))));
    }

    #[tokio::test]
    async fn matches_files_inside_workspace() {
        let fixture = TestWorkspace::new();
        std::fs::write(fixture.workspace.join("inside.rs"), "fn main() {}").unwrap();

        let result = GlobTool
            .execute(serde_json::json!({ "pattern": "*.rs" }), &fixture.context())
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.content, "inside.rs");
    }
}
