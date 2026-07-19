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
