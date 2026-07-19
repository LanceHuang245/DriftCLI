use super::{OpenAiToolCall, OpenAiToolCallFunction, openai_tool_call_chunks};
use crate::LlmChunk;

#[test]
fn correlates_interleaved_tool_arguments_by_index() {
    // Each continuation must reuse the ID assigned to its own stream index.
    let mut ids = Vec::new();
    let starts = [
        tool_call(0, Some("call-0"), Some("first"), Some("{")),
        tool_call(1, Some("call-1"), Some("second"), Some("{")),
    ];
    for tool_call in &starts {
        let chunks = openai_tool_call_chunks(tool_call, &mut ids);
        assert!(chunks.iter().all(Result::is_ok));
    }

    let first = openai_tool_call_chunks(&tool_call(0, None, None, Some("\"value\":0}")), &mut ids);
    let second = openai_tool_call_chunks(&tool_call(1, None, None, Some("\"value\":1}")), &mut ids);

    assert!(matches!(
        &first[0],
        Ok(LlmChunk::ToolCallArgs { id, .. }) if id == "call-0"
    ));
    assert!(matches!(
        &second[0],
        Ok(LlmChunk::ToolCallArgs { id, .. }) if id == "call-1"
    ));
}

/// Build one wire-format delta for focused index-correlation tests.
fn tool_call(
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> OpenAiToolCall {
    OpenAiToolCall {
        index,
        id: id.map(str::to_string),
        function: OpenAiToolCallFunction {
            name: name.map(str::to_string),
            arguments: arguments.map(|value| serde_json::Value::String(value.to_string())),
        },
    }
}
