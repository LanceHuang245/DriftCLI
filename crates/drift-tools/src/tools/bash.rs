//! BashTool — execute a shell command with timeout and capture output.

use crate::{Tool, ToolContext, ToolError, ToolResult};
use std::path::Path;
use std::process::Stdio;
use tokio::time::timeout;

pub struct BashTool;

/// Shell configuration for the current platform
struct ShellConfig {
    program: String,
    arg_prefix: Vec<String>,
}

impl ShellConfig {
    /// Detect the best available shell for the current platform.
    fn detect() -> Self {
        if cfg!(windows) {
            // On Windows: prefer pwsh (PowerShell 7+), then powershell, then cmd
            if which_exists("pwsh") {
                return ShellConfig {
                    program: "pwsh".into(),
                    arg_prefix: vec!["-NoProfile".into(), "-Command".into()],
                };
            }
            if which_exists("powershell") {
                return ShellConfig {
                    program: "powershell".into(),
                    arg_prefix: vec!["-NoProfile".into(), "-Command".into()],
                };
            }
            ShellConfig {
                program: "cmd".into(),
                arg_prefix: vec!["/c".into()],
            }
        } else {
            // On Unix: use $SHELL or fall back to /bin/sh
            let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into());
            ShellConfig {
                program: shell,
                arg_prefix: vec!["-c".into()],
            }
        }
    }
}

/// Check if a command exists in PATH by trying to spawn it.
fn which_exists(cmd: &str) -> bool {
    let mut c = std::process::Command::new(cmd);
    if cmd == "pwsh" || cmd == "powershell" {
        c.arg("-NoProfile").arg("-Command").arg("exit 0");
    } else {
        c.arg("--version");
    }
    c.stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|mut child| child.wait().is_ok())
        .unwrap_or(false)
}

#[async_trait::async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        if cfg!(windows) {
            "Execute a shell command via PowerShell and capture output. Available commands: ls/dir (list files), cat/type (read file), echo, mkdir, rm, cp/copy, mv/move. Use 'dir' not 'ls' for cmd compatibility; PowerShell supports both."
        } else {
            "Execute a shell command and capture output"
        }
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional working directory for the command (overrides the session working directory)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds (defaults to 120000ms / 2 minutes)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let command_text = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("command is required".into()))?;

        // Resolve the working directory through the shared workspace guard.
        let workdir = if let Some(wd) = args.get("workdir").and_then(|v| v.as_str()) {
            if wd.is_empty() {
                ctx.working_dir.clone()
            } else {
                let requested = Path::new(wd);
                ctx.file_access
                    .check_read(requested)
                    .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
                ctx.file_access.resolve(requested)
            }
        } else {
            ctx.working_dir.clone()
        };

        let canonical_workdir = workdir;

        // Parse timeout: default 120 seconds
        let timeout_ms = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(120_000);
        let timeout_duration = std::time::Duration::from_millis(timeout_ms);

        // Detect the best available shell
        let shell = ShellConfig::detect();

        // Build the shell command inside the active OS sandbox boundary.
        let mut shell_args = shell.arg_prefix;
        shell_args.push(command_text.to_string());
        let mut cmd = ctx
            .process_sandbox
            .command(&shell.program, &shell_args, &canonical_workdir)
            .map_err(|error| ToolError::PermissionDenied(error.to_string()))?;
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::null());
        cmd.kill_on_drop(true);

        // Run the command with a timeout
        let output = match timeout(timeout_duration, cmd.output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Ok(ToolResult {
                    success: false,
                    content: String::new(),
                    error: Some(format!("failed to spawn command: {}", e)),
                });
            }
            Err(_elapsed) => {
                return Err(ToolError::Timeout);
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        // Trim output to max 100000 characters
        const MAX_OUTPUT: usize = 100_000;
        let stdout_truncated = truncate_str(&stdout, MAX_OUTPUT);
        let stderr_truncated = truncate_str(&stderr, MAX_OUTPUT);

        // Build the result as a JSON string
        let result_content = serde_json::json!({
            "stdout": stdout_truncated,
            "stderr": stderr_truncated,
            "exit_code": exit_code,
        })
        .to_string();

        Ok(ToolResult {
            success: exit_code == 0,
            content: result_content,
            error: if exit_code != 0 {
                Some(format!("command exited with code {}", exit_code))
            } else {
                None
            },
        })
    }
}

/// Truncate a string to max_len characters, appending a truncation notice if needed.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!(
            "{}...\n[truncated: output exceeded {} characters]",
            &s[..max_len],
            max_len
        )
    }
}
