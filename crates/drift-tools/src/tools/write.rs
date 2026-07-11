use crate::{Tool, ToolContext, ToolError, ToolResult};
use std::path::Path;

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

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Extract required arguments
        let file_path_str = args["filePath"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("filePath must be a string".into()))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("content must be a string".into()))?;

        // Resolve and validate the file through the shared workspace guard.
        let requested = Path::new(file_path_str);
        ctx.file_access
            .check_write(requested)
            .map_err(|error| ToolError::PermissionDenied(format!("{error:?}")))?;
        let resolved = ctx.file_access.resolve(requested);

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
