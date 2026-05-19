use crate::{Tool, ToolContext, ToolError, ToolResult};

pub struct WriteTool;

#[async_trait::async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file, overwriting if it already exists. Parent directories are auto-created."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": { "type": "string", "description": "Path to the file to write, resolved relative to the working directory." },
                "content": { "type": "string", "description": "Content to write to the file." }
            },
            "required": ["filePath", "content"]
        })
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        // Extract required arguments
        let file_path_str = args["filePath"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("filePath must be a string".into()))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("content must be a string".into()))?;

        // Resolve path relative to working_dir and canonically verify no escape
        let resolved = ctx.working_dir.join(file_path_str);
        let canonical = std::path::absolute(&ctx.working_dir).map_err(ToolError::Io)?;
        let file_canonical = std::path::absolute(&resolved)
            .map_err(|e| ToolError::InvalidArgs(format!("invalid file path: {e}")))?;

        if !file_canonical.starts_with(&canonical) {
            return Err(ToolError::PermissionDenied(
                "file path escapes the working directory".into(),
            ));
        }

        // Auto-create parent directories
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(ToolError::Io)?;
        }

        // Write content to file (overwrite)
        std::fs::write(&resolved, content).map_err(ToolError::Io)?;

        Ok(ToolResult {
            success: true,
            content: format!("wrote {} bytes to {}", content.len(), file_path_str),
            error: None,
        })
    }
}
