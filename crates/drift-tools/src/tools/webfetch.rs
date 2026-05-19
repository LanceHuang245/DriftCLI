use crate::{Tool, ToolContext, ToolError, ToolResult};

pub struct WebFetchTool;

impl WebFetchTool {
    /// Simple HTML tag stripper: remove everything between < and >.
    fn strip_html(input: &str) -> String {
        let mut result = String::with_capacity(input.len());
        let mut in_tag = false;
        for ch in input.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        result
    }

    /// Collapse repeated whitespace (newlines, tabs, multiple spaces) into single spaces.
    fn collapse_whitespace(input: &str) -> String {
        let mut result = String::with_capacity(input.len());
        let mut last_was_space = false;
        for ch in input.chars() {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
            }
        }
        result.trim().to_string()
    }
}

#[async_trait::async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "webfetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL and return as plain text or markdown. Handles HTML pages by \
         stripping tags and returning readable text."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from"
                },
                "maxLength": {
                    "type": "integer",
                    "description": "Maximum number of characters to return (default: 5000)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        // Extract and validate the URL argument
        let url = args["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'url' field".into()))?;

        let max_length = args["maxLength"]
            .as_u64()
            .unwrap_or(5000) as usize;

        // Build HTTP client with a 30-second timeout
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (compatible; DriftCLI/1.0)")
            .build()
            .map_err(|e| ToolError::Other(format!("failed to build HTTP client: {e}")))?;

        // Send GET request
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::Other(format!("request failed: {e}")))?;

        // Check for HTTP error status codes
        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult {
                success: true,
                content: String::new(),
                error: Some(format!(
                    "HTTP {status}: {}",
                    response.text().await.unwrap_or_default()
                )),
            });
        }

        // Determine content-type to decide if we should strip HTML
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        let body = response
            .text()
            .await
            .map_err(|e| ToolError::Other(format!("failed to read response body: {e}")))?;

        // Strip HTML tags if the content looks like HTML
        let content = if content_type.contains("text/html") || content_type.contains("text/xhtml") {
            let stripped = Self::strip_html(&body);
            Self::collapse_whitespace(&stripped)
        } else {
            body
        };

        // Truncate to max_length characters
        let truncated: String = content.chars().take(max_length).collect();

        Ok(ToolResult {
            success: true,
            content: truncated,
            error: None,
        })
    }
}
