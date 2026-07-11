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

        // Resolve the search directory through the shared workspace guard.
        let requested = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(Path::new)
            .unwrap_or_else(|| Path::new("."));
        ctx.file_access
            .check_read(requested)
            .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
        let search_dir = ctx.file_access.resolve(requested);

        // Optional glob filter for matching file paths relative to search_dir
        let include_filter = args
            .get("include")
            .and_then(|v| v.as_str())
            .and_then(|s| glob::Pattern::new(s).ok());

        let mut results = Vec::new();
        const MAX_RESULTS: usize = 50;

        for entry in WalkDir::new(&search_dir).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            // Apply optional glob filter against the relative path
            if let Some(ref filter) = include_filter
                && let Ok(rel) = entry.path().strip_prefix(&search_dir)
                && !filter.matches(rel.to_string_lossy().as_ref())
            {
                continue;
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
                    results.push(format!("{}:{}: {}", rel_path.display(), line_num + 1, line));
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
