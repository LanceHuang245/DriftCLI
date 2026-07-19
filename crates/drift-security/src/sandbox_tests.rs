use super::*;

#[test]
fn read_only_mode_rejects_process_execution() {
    let cwd = std::env::current_dir().expect("cwd must exist");
    let sandbox = ProcessSandbox::new(SandboxMode::ReadOnly, &cwd).unwrap();

    let result = sandbox.command("ignored", std::iter::empty::<&str>(), &cwd);

    assert!(matches!(result, Err(SandboxError::ReadOnly)));
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn workspace_write_only_changes_the_workspace() {
    if find_bubblewrap().is_none() {
        return;
    }

    let suffix = uuid::Uuid::new_v4();
    let workspace = std::env::temp_dir().join(format!("drift-sandbox-workspace-{suffix}"));
    let outside = std::env::temp_dir().join(format!("drift-sandbox-outside-{suffix}"));
    std::fs::create_dir(&workspace).expect("test workspace must be created");
    let inside = workspace.join("inside");
    let sandbox = ProcessSandbox::new(SandboxMode::WorkspaceWrite, &workspace).unwrap();
    let script = format!(
        "printf allowed > '{}'; printf blocked > '{}'",
        inside.display(),
        outside.display()
    );
    let mut command = sandbox
        .command("/bin/sh", ["-c", &script], &workspace)
        .unwrap();

    let output = command.output().await.expect("sandbox process must start");

    assert!(!output.status.success());
    assert_eq!(std::fs::read_to_string(&inside).unwrap(), "allowed");
    assert!(!outside.exists());
    std::fs::remove_dir_all(workspace).expect("test workspace must be removed");
}
