use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let hang_initialize = std::env::var_os("DRIFT_MCP_FIXTURE_HANG_INITIALIZE").is_some();
    let tool_name = std::env::var("DRIFT_MCP_FIXTURE_TOOL_NAME").unwrap_or_else(|_| "echo".into());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let method = request.get("method").and_then(serde_json::Value::as_str);
        let id = request.get("id").cloned();
        let response = match method {
            Some("initialize") if hang_initialize => None,
            Some("initialize") => id.map(|id| {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "drift-mcp-fixture", "version": "1.0" }
                    }
                })
            }),
            Some("notifications/initialized") => None,
            Some("tools/list") => id.map(|id| {
                let final_page = request
                    .pointer("/params/cursor")
                    .and_then(serde_json::Value::as_str)
                    .is_some();
                let result = if final_page {
                    serde_json::json!({ "tools": [] })
                } else {
                    serde_json::json!({
                        "tools": [{
                            "name": tool_name.as_str(),
                            "description": "Echo text",
                            "inputSchema": {
                                "type": "object",
                                "properties": { "text": { "type": "string" } },
                                "required": ["text"]
                            }
                        }],
                        "nextCursor": "final-page"
                    })
                };
                serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })
            }),
            Some("tools/call") => id.map(|id| {
                let text = request
                    .get("params")
                    .and_then(|params| params.get("arguments"))
                    .and_then(|arguments| arguments.get("text"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    }
                })
            }),
            _ => id.map(|id| {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": "method not found" }
                })
            }),
        };
        if let Some(response) = response {
            writeln!(stdout, "{}", response)?;
            stdout.flush()?;
        }
    }

    if let Ok(path) = std::env::var("DRIFT_MCP_FIXTURE_EXIT_FILE") {
        std::fs::write(path, b"stopped")?;
    }
    Ok(())
}
