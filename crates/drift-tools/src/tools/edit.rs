use crate::{Tool, ToolContext, ToolError, ToolResult};
use std::path::Path;

pub struct EditTool;

#[async_trait::async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Perform an exact string find-and-replace in a file. Fails if old_string is found zero or more than once."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": { "type": "string", "description": "Path to the file to edit, resolved relative to the working directory." },
                "oldString": { "type": "string", "description": "Exact string to find and replace." },
                "newString": { "type": "string", "description": "Replacement string." }
            },
            "required": ["filePath", "oldString", "newString"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Extract required arguments
        let file_path_str = args["filePath"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("filePath must be a string".into()))?;
        let old_string = args["oldString"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("oldString must be a string".into()))?;
        let new_string = args["newString"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("newString must be a string".into()))?;

        // Resolve and validate the file through the shared workspace guard.
        let requested = Path::new(file_path_str);
        ctx.file_access
            .check_write(requested)
            .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
        let resolved = ctx.file_access.resolve(requested);

        // Read the original file content
        let content = std::fs::read_to_string(&resolved)
            .map_err(|e| ToolError::InvalidArgs(format!("cannot read file: {e}")))?;

        // Count occurrences of old_string (non-overlapping)
        let count = content.matches(old_string).count();

        if count == 0 {
            return Ok(ToolResult {
                success: false,
                content: String::new(),
                error: Some(format!("oldString not found in file: {}", file_path_str)),
            });
        }
        if count > 1 {
            return Ok(ToolResult {
                success: false,
                content: String::new(),
                error: Some(format!(
                    "oldString found {} times in file: {}. Provide more surrounding context to make the match unique.",
                    count, file_path_str
                )),
            });
        }

        // Perform the single replacement
        let new_content = content.replacen(old_string, new_string, 1);

        // Write atomically: write to temp file then rename
        let tmp_path = resolved.with_extension("tmp");
        ctx.file_access
            .check_write(&tmp_path)
            .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
        std::fs::write(&tmp_path, new_content).map_err(ToolError::Io)?;
        std::fs::rename(&tmp_path, &resolved).map_err(ToolError::Io)?;

        Ok(ToolResult {
            success: true,
            content: format!("replaced 1 occurrence in {}", file_path_str),
            error: None,
        })
    }
}
