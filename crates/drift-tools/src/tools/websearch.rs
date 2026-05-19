use crate::{Tool, ToolContext, ToolError, ToolResult};

pub struct WebSearchTool;

#[derive(Debug, Clone)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

impl WebSearchTool {
    /// Parse DuckDuckGo HTML results page into structured search results.
    /// Looks for elements with class `result__a` (title + URL) and `result__snippet`.
    fn parse_duckduckgo_results(html: &str, limit: usize) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Split by result blocks — each result is wrapped in a web-result div.
        // Strategy: find <a class="result__a" to get URL/title, then find nearest snippet.
        let mut remaining = html;

        while results.len() < limit {
            // Find next result__a element
            let link_start = match remaining.find("class=\"result__a\"") {
                Some(pos) => pos,
                None => break,
            };

            // Backtrack to find the opening <a tag with href
            let tag_start = remaining[..link_start]
                .rfind("<a ")
                .unwrap_or(link_start.saturating_sub(200));

            // Extract href attribute
            let href_pos = remaining[tag_start..].find("href=\"");
            let (mut url, title) = if let Some(pos) = href_pos {
                let abs_href = tag_start + pos + 6; // skip 'href="'
                let href_end = remaining[abs_href..].find('"').map(|e| abs_href + e).unwrap_or(abs_href + 200);
                let url_str = &remaining[abs_href..href_end];

                // Find the closing > of the <a> tag, then text until </a>
                let tag_close = remaining[href_end..]
                    .find('>')
                    .map(|i| href_end + i + 1)
                    .unwrap_or(href_end + 1);
                let title_end = remaining[tag_close..]
                    .find("</a>")
                    .map(|i| tag_close + i)
                    .unwrap_or(tag_close + 200);
                let title_str = &remaining[tag_close..title_end];

                (url_str.trim().to_string(), html_entity_decode(title_str.trim()))
            } else {
                // Advance to avoid infinite loop
                remaining = &remaining[link_start + 16..];
                continue;
            };

            // Fix DuckDuckGo redirect URLs
            if url.starts_with("//") {
                url = format!("https:{url}");
            }

            // Find the snippet after this result__a
            let snippet_start_pos = link_start + 16;
            let after_link = &remaining[snippet_start_pos..];

            let snippet = after_link
                .find("class=\"result__snippet\"")
                .and_then(|snippet_class_pos| {
                    let snippet_tag_close = after_link[snippet_class_pos..]
                        .find('>')
                        .map(|i| snippet_class_pos + i + 1)?;
                    let snippet_end = after_link[snippet_tag_close..]
                        .find("</a>")
                        .map(|i| snippet_tag_close + i)?;
                    Some(html_entity_decode(
                        &after_link[snippet_tag_close..snippet_end].trim(),
                    ))
                })
                .unwrap_or_default();

            results.push(SearchResult {
                title,
                url,
                snippet,
            });

            // Advance past the processed area
            remaining = &remaining[link_start + 16..];
        }

        results
    }
}

/// Decode common HTML entities like &amp; &lt; &gt; &quot; &#39;
fn html_entity_decode(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
}

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "websearch"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo (HTML) and return title, URL, and snippet for each result. \
         No API key required."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArgs("missing 'query' field".into()))?;

        let limit = args["limit"].as_u64().unwrap_or(10) as usize;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .map_err(|e| ToolError::Other(format!("failed to build HTTP client: {e}")))?;

        let search_url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding(&query)
        );

        let response = client
            .get(&search_url)
            .send()
            .await
            .map_err(|e| ToolError::Other(format!("search request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult {
                success: true,
                content: String::new(),
                error: Some(format!("search returned HTTP {status}")),
            });
        }

        let body = response
            .text()
            .await
            .map_err(|e| ToolError::Other(format!("failed to read response: {e}")))?;

        let results = Self::parse_duckduckgo_results(&body, limit);

        if results.is_empty() {
            return Ok(ToolResult {
                success: true,
                content: "No results found.".to_string(),
                error: None,
            });
        }

        // Format results as plain text
        let mut output = String::new();
        for (i, r) in results.iter().enumerate() {
            output.push_str(&format!("{}. {}\n", i + 1, r.title));
            output.push_str(&format!("   URL: {}\n", r.url));
            if !r.snippet.is_empty() {
                output.push_str(&format!("   {}\n", r.snippet));
            }
            output.push('\n');
        }

        Ok(ToolResult {
            success: true,
            content: output.trim_end().to_string(),
            error: None,
        })
    }
}

/// Simple URL encoding for search queries (spaces and special chars).
fn urlencoding(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b' ' => result.push('+'),
            _ => {
                result.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    result
}
