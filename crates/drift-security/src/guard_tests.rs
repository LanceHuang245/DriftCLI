use super::*;

struct TestWorkspace {
    root: PathBuf,
    workspace: PathBuf,
    outside: PathBuf,
}

impl TestWorkspace {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("drift-guard-{}", uuid::Uuid::new_v4()));
        let workspace = root.join("workspace");
        let outside = root.join("outside");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        Self {
            root,
            workspace,
            outside,
        }
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        // Keep temporary test data from leaking into later test runs.
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

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
fn test_missing_target_cannot_traverse_outside_workspace() {
    let fixture = TestWorkspace::new();
    let guard = FileAccessGuard::new(&fixture.workspace, &[]).unwrap();

    let error = guard
        .check_write(Path::new("../outside/new-file.txt"))
        .unwrap_err();

    assert!(matches!(error, AccessDenied::OutsideWorkspace(_)));
}

#[cfg(unix)]
#[test]
fn test_missing_target_cannot_escape_through_symlink_parent() {
    let fixture = TestWorkspace::new();
    std::os::unix::fs::symlink(&fixture.outside, fixture.workspace.join("linked")).unwrap();
    let guard = FileAccessGuard::new(&fixture.workspace, &[]).unwrap();

    let error = guard
        .check_write(Path::new("linked/new-file.txt"))
        .unwrap_err();

    assert!(matches!(error, AccessDenied::OutsideWorkspace(_)));
}

#[test]
fn test_protected_pattern_matches_workspace_relative_path() {
    let fixture = TestWorkspace::new();
    std::fs::create_dir(fixture.workspace.join(".git")).unwrap();
    let guard = FileAccessGuard::new(&fixture.workspace, &[String::from(".git/*")]).unwrap();

    let error = guard.check_write(Path::new(".git/config")).unwrap_err();

    assert!(matches!(error, AccessDenied::ProtectedPath(_)));
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
