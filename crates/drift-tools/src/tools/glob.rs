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

        // Resolve the search directory, ensuring it is within the working directory
        let search_dir = resolve_subdir(&ctx.working_dir, args.get("path").and_then(|v| v.as_str()))?;

        // Build the full glob pattern: search_dir / pattern
        let full_pattern = format!("{}/{}", search_dir.display(), pattern);

        // Collect matching entries with their modification times
        let mut entries: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
        const MAX_RESULTS: usize = 100;

        for entry in glob::glob(&full_pattern).map_err(|e| ToolError::InvalidArgs(e.to_string()))? {
            let path = match entry {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Get modification time, skip on error
            let mtime = match std::fs::metadata(&path) {
                Ok(meta) => meta.modified().unwrap_or(std::time::UNIX_EPOCH),
                Err(_) => continue,
            };

            entries.push((path, mtime));
        }

        // Sort by modification time, newest first
        entries.sort_by(|a, b| b.1.cmp(&a.1));

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

/// Resolve a subdirectory relative to the base working directory.
/// Canonicalizes and ensures the result stays within the base directory.
fn resolve_subdir(base: &std::path::Path, subdir: Option<&str>) -> Result<std::path::PathBuf, ToolError> {
    let target = match subdir {
        Some(sub) if !sub.is_empty() => base.join(sub),
        _ => base.to_path_buf(),
    };

    let canonical = target.canonicalize().map_err(|e| ToolError::Io(e))?;
    let canonical_base = base.canonicalize().map_err(|e| ToolError::Io(e))?;

    if !canonical.starts_with(&canonical_base) {
        return Err(ToolError::PermissionDenied(
            "path traversal outside working directory is not allowed".into(),
        ));
    }

    Ok(canonical)
}
