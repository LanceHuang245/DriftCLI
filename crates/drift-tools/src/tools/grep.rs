//! GrepTool — search file contents using regex patterns.

use crate::{Tool, ToolContext, ToolError, ToolResult};
use regex::Regex;
use std::path::Path;
use walkdir::WalkDir;

pub struct GrepTool;

#[async_trait::async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex pattern"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Optional subdirectory to search in (relative to working directory)"
                },
                "include": {
                    "type": "string",
                    "description": "Optional glob file filter (e.g. \"*.rs\")"
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
        // Parse and compile the regex pattern
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("pattern is required".into()))?;
        let re = Regex::new(pattern)
            .map_err(|e| ToolError::InvalidArgs(format!("invalid regex: {}", e)))?;

        // Resolve the search directory, ensuring it is within the working directory
        let search_dir = resolve_subdir(&ctx.working_dir, args.get("path").and_then(|v| v.as_str()))?;

        // Optional glob filter for matching file paths relative to search_dir
        let include_filter = args
            .get("include")
            .and_then(|v| v.as_str())
            .and_then(|s| glob::Pattern::new(s).ok());

        let mut results = Vec::new();
        const MAX_RESULTS: usize = 50;

        for entry in WalkDir::new(&search_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }

            // Apply optional glob filter against the relative path
            if let Some(ref filter) = include_filter {
                if let Ok(rel) = entry.path().strip_prefix(&search_dir) {
                    if !filter.matches(rel.to_string_lossy().as_ref()) {
                        continue;
                    }
                }
            }

            // Read the file content, skipping binary or unreadable files
            let content = match std::fs::read_to_string(entry.path()) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Compute the relative path for output display
            let rel_path = entry
                .path()
                .strip_prefix(&search_dir)
                .unwrap_or(entry.path());

            for (line_num, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(format!(
                        "{}:{}: {}",
                        rel_path.display(),
                        line_num + 1,
                        line
                    ));
                    if results.len() >= MAX_RESULTS {
                        break;
                    }
                }
            }
            if results.len() >= MAX_RESULTS {
                break;
            }
        }

        Ok(ToolResult {
            success: true,
            content: results.join("\n"),
            error: None,
        })
    }
}

/// Resolve a subdirectory relative to the base working directory.
/// Canonicalizes and ensures the result stays within the base directory.
fn resolve_subdir(base: &Path, subdir: Option<&str>) -> Result<std::path::PathBuf, ToolError> {
    let target = match subdir {
        Some(sub) if !sub.is_empty() => base.join(sub),
        _ => base.to_path_buf(),
    };

    // Canonicalize to resolve any `..` components and get the real path
    let canonical = target
        .canonicalize()
        .map_err(|e| ToolError::Io(e))?;

    // Canonicalize the base for comparison
    let canonical_base = base
        .canonicalize()
        .map_err(|e| ToolError::Io(e))?;

    // Ensure the target is within the base directory
    if !canonical.starts_with(&canonical_base) {
        return Err(ToolError::PermissionDenied(
            "path traversal outside working directory is not allowed".into(),
        ));
    }

    Ok(canonical)
}
