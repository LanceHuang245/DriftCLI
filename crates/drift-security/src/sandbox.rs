use crate::SandboxMode;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Builds child-process commands that enforce the selected OS sandbox boundary.
#[derive(Debug, Clone)]
pub struct ProcessSandbox {
    mode: SandboxMode,
    workspace: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("process execution is disabled in read-only sandbox mode")]
    ReadOnly,
    #[error("working directory is outside the sandbox workspace: {0}")]
    OutsideWorkspace(PathBuf),
    #[error("bubblewrap is required for workspace-write sandbox mode")]
    BubblewrapUnavailable,
    #[error("workspace-write process sandboxing is not supported on this platform")]
    UnsupportedPlatform,
    #[error("failed to resolve sandbox path '{path}': {source}")]
    Path {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl ProcessSandbox {
    /// Canonicalize the workspace once so every child uses the same trusted boundary.
    pub fn new(mode: SandboxMode, workspace: &Path) -> Result<Self, SandboxError> {
        let workspace = workspace
            .canonicalize()
            .map_err(|source| SandboxError::Path {
                path: workspace.to_path_buf(),
                source,
            })?;
        Ok(Self { mode, workspace })
    }

    pub fn mode(&self) -> SandboxMode {
        self.mode
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Wrap a child in bubblewrap on Linux; unrestricted mode keeps the native command.
    pub fn command<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        working_dir: &Path,
    ) -> Result<Command, SandboxError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let working_dir = working_dir
            .canonicalize()
            .map_err(|source| SandboxError::Path {
                path: working_dir.to_path_buf(),
                source,
            })?;

        match self.mode {
            SandboxMode::ReadOnly => Err(SandboxError::ReadOnly),
            SandboxMode::DangerFullAccess => {
                let mut command = Command::new(program);
                command.args(args);
                command.current_dir(working_dir);
                Ok(command)
            }
            SandboxMode::WorkspaceWrite => {
                if !working_dir.starts_with(&self.workspace) {
                    return Err(SandboxError::OutsideWorkspace(working_dir));
                }
                self.workspace_write_command(program, args, &working_dir)
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn workspace_write_command<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        working_dir: &Path,
    ) -> Result<Command, SandboxError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let bubblewrap = find_bubblewrap().ok_or(SandboxError::BubblewrapUnavailable)?;
        let mut command = Command::new(bubblewrap);

        // The host root is read-only; only the canonical workspace is rebound writable.
        command.args([
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--unshare-cgroup-try",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
        ]);
        command.arg(&self.workspace).arg(&self.workspace);
        command.arg("--chdir").arg(working_dir).arg("--");
        command.arg(program).args(args);
        command.current_dir(&self.workspace);
        Ok(command)
    }

    #[cfg(not(target_os = "linux"))]
    fn workspace_write_command<I, S>(
        &self,
        _program: impl AsRef<OsStr>,
        _args: I,
        _working_dir: &Path,
    ) -> Result<Command, SandboxError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        Err(SandboxError::UnsupportedPlatform)
    }
}

#[cfg(target_os = "linux")]
fn find_bubblewrap() -> Option<PathBuf> {
    // Prefer system locations so a workspace-controlled PATH entry cannot replace the sandbox.
    ["/usr/bin/bwrap", "/bin/bwrap"]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
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
}
