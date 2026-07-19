use super::{LlmError, MAX_SSE_LINE_BYTES, SseLineStream};
use bytes::Bytes;
use futures::StreamExt;

#[tokio::test]
async fn sse_stream_propagates_transport_errors() {
    // Transport failures must remain distinguishable from a clean EOF.
    let source = futures::stream::iter(vec![Err(LlmError::Stream("transport".into()))]);
    let mut stream = SseLineStream::from_stream(source);

    let error = stream.next().await.unwrap().unwrap_err();

    assert!(matches!(error, LlmError::Stream(message) if message == "transport"));
}

#[tokio::test]
async fn sse_stream_rejects_invalid_utf8() {
    // Invalid event bytes must be reported instead of silently discarded.
    let source = futures::stream::iter(vec![Ok(Bytes::from_static(b"\xff\n"))]);
    let mut stream = SseLineStream::from_stream(source);

    let error = stream.next().await.unwrap().unwrap_err();

    assert!(matches!(error, LlmError::Stream(message) if message.contains("invalid UTF-8")));
}

#[tokio::test]
async fn sse_stream_rejects_oversized_lines() {
    // A line cap prevents an unbounded SSE buffer from exhausting memory.
    let bytes = Bytes::from(vec![b'x'; MAX_SSE_LINE_BYTES + 1]);
    let source = futures::stream::iter(vec![Ok(bytes)]);
    let mut stream = SseLineStream::from_stream(source);

    let error = stream.next().await.unwrap().unwrap_err();

    assert!(matches!(error, LlmError::Stream(message) if message.contains("exceeds")));
}
