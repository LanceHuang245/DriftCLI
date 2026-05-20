use crate::{Tool, ToolContext, ToolError, ToolResult};
use std::io::Read;

pub struct ReadTool;

#[async_trait::async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file from the local filesystem. Returns file content with line numbers prefixed like \"  123│ content\"."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": { "type": "string", "description": "Path to the file to read, resolved relative to the working directory." },
                "offset": { "type": "integer", "description": "1-indexed line number to start reading from (default: 1)." },
                "limit": { "type": "integer", "description": "Maximum number of lines to return (default: 2000)." }
            },
            "required": ["filePath"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Extract required filePath
        let file_path_str = args["filePath"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("filePath must be a string".into()))?;

        // Resolve path relative to working_dir, reject attempts to escape the working directory
        let resolved = ctx.working_dir.join(file_path_str);
        let canonical = std::path::absolute(&ctx.working_dir).map_err(ToolError::Io)?;
        let file_canonical = std::path::absolute(&resolved)
            .map_err(|e| ToolError::InvalidArgs(format!("invalid file path: {e}")))?;

        // Ensure the resolved file is within the working directory
        if !file_canonical.starts_with(&canonical) {
            return Err(ToolError::PermissionDenied(
                "file path escapes the working directory".into(),
            ));
        }

        // Check if the path exists and is a file
        if !resolved.exists() {
            return Ok(ToolResult {
                success: false,
                content: String::new(),
                error: Some(format!("file not found: {}", file_path_str)),
            });
        }
        if !resolved.is_file() {
            return Ok(ToolResult {
                success: false,
                content: String::new(),
                error: Some(format!("not a file: {}", file_path_str)),
            });
        }

        // Binary file detection: read first 8KB and check for null bytes
        let mut file = std::fs::File::open(&resolved).map_err(ToolError::Io)?;
        let mut head_buf = vec![0u8; 8192];
        let head_len = file.read(&mut head_buf).map_err(ToolError::Io)?;
        if head_buf[..head_len].contains(&0u8) {
            return Ok(ToolResult {
                success: false,
                content: String::new(),
                error: Some(format!(
                    "cannot display content of binary file: {}",
                    file_path_str
                )),
            });
        }

        // Read the full file content as a string
        let full_content = std::fs::read_to_string(&resolved).map_err(ToolError::Io)?;

        // Apply offset and limit
        let offset = args["offset"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(1)
            .max(1);
        let limit = args["limit"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(2000)
            .min(2000);

        let lines: Vec<&str> = full_content.lines().collect();
        let total_lines = lines.len();

        let start_idx = (offset - 1).min(total_lines);
        let end_idx = (start_idx + limit).min(total_lines);

        // Format output with line numbers: "  123│ content"
        let output: String = lines[start_idx..end_idx]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>6}│ {}", start_idx + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolResult {
            success: true,
            content: output,
            error: None,
        })
    }
}
