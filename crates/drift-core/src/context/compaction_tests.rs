use super::*;

// Empty provider stream used to keep fake providers network-free.
struct EmptyStream;

impl tokio_stream::Stream for EmptyStream {
    type Item = Result<drift_llm::LlmChunk, drift_llm::LlmError>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready(None)
    }
}

// Configurable provider fixture that records summary requests.
struct FakeProvider {
    summary: String,
    calls: std::sync::atomic::AtomicUsize,
    inputs: std::sync::Arc<tokio::sync::Mutex<Vec<String>>>,
    fail: bool,
}

#[async_trait::async_trait]
impl LlmProvider for FakeProvider {
    fn provider_id(&self) -> &str {
        "fake"
    }

    fn model_name(&self) -> &str {
        "fake-model"
    }

    fn context_window(&self) -> usize {
        2400
    }

    async fn stream_chat(
        &self,
        _messages: Vec<LlmMessage>,
        _system_prompt: Option<String>,
        _temperature: Option<f64>,
        _max_output_tokens: Option<usize>,
        _tools: Option<Vec<Value>>,
    ) -> Result<drift_llm::LlmResponseStream, drift_llm::LlmError> {
        Ok(drift_llm::LlmResponseStream::new(EmptyStream))
    }

    async fn chat(
        &self,
        messages: Vec<LlmMessage>,
        _system_prompt: Option<String>,
    ) -> Result<String, drift_llm::LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let input = messages
            .first()
            .and_then(|message| message.content.first())
            .and_then(|part| match part {
                ContentPart::Text(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        self.inputs.lock().await.push(input);
        if self.fail {
            Err(drift_llm::LlmError::Stream("fake summary failure".into()))
        } else {
            Ok(self.summary.clone())
        }
    }
}

// Build enough complete turns to force deterministic compaction.
fn long_conversation() -> Vec<LlmMessage> {
    let mut messages = vec![LlmMessage::user(format!(
        "initial request {}",
        "x".repeat(400)
    ))];
    for index in 0..5 {
        messages.push(LlmMessage {
            role: "assistant".into(),
            content: vec![
                ContentPart::Text(format!("turn {index} assistant {}", "a".repeat(200))),
                ContentPart::ToolCall {
                    id: format!("call-{index}"),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ],
        });
        messages.push(LlmMessage {
            role: "user".into(),
            content: vec![ContentPart::ToolResult {
                tool_use_id: format!("call-{index}"),
                content: format!("tool result {index} {}", "r".repeat(200)),
                is_error: false,
            }],
        });
        messages.push(LlmMessage::user(format!(
            "turn {index} request {}",
            "u".repeat(400)
        )));
    }
    messages
}

// Create a successful fake summarizer with isolated call tracking.
fn fake_provider(summary: &str) -> FakeProvider {
    FakeProvider {
        summary: summary.into(),
        calls: std::sync::atomic::AtomicUsize::new(0),
        inputs: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        fail: false,
    }
}

// Verify summary preparation preserves boundaries and leaves state transactional.
#[tokio::test]
async fn summarizes_transactionally_and_preserves_recent_complete_turns() {
    let mut manager = ContextManager::new(2400, 0.25, 0.5);
    manager.set_messages(long_conversation());
    let provider = fake_provider("## summary ##");

    let built = manager
        .build_context(&[], true, Some(&provider))
        .await
        .unwrap();
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(
        built
            .system_prompt
            .as_deref()
            .unwrap()
            .contains("## summary ##")
    );
    assert!(built.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Text(text) if text.contains("initial request")))
    }));
    assert!(built.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Text(text) if text.contains("turn 4 request")))
    }));
    assert!(!built.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::Text(text) if text.contains("turn 0 request")))
    }));
    let call_index = built
        .messages
        .iter()
        .position(|message| {
            message
                .content
                .iter()
                .any(|part| matches!(part, ContentPart::ToolCall { id, .. } if id == "call-0"))
        })
        .expect("tool call retained");
    let result_index = built
        .messages
        .iter()
        .position(|message| {
            message.content.iter().any(|part| {
                matches!(part, ContentPart::ToolResult { tool_use_id, .. } if tool_use_id == "call-0")
            })
        })
        .expect("tool result retained");
    assert_eq!(result_index, call_index + 1);
    let untouched = manager.build_context(&[], false, None).await.unwrap();
    assert_eq!(untouched.messages.len(), 16);

    let snapshot = built.compaction.clone().unwrap();
    manager.apply_compaction(&snapshot);
    let applied = manager.build_context(&[], false, None).await.unwrap();
    assert!(
        applied
            .system_prompt
            .unwrap()
            .contains("<conversation_summary>## summary ##</conversation_summary>")
    );
}

// Verify provider errors do not commit candidate messages or summaries.
#[tokio::test]
async fn summary_failure_is_transactional() {
    let mut manager = ContextManager::new(2400, 0.25, 0.5);
    manager.set_messages(long_conversation());
    let provider = FakeProvider {
        summary: String::new(),
        calls: std::sync::atomic::AtomicUsize::new(0),
        inputs: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        fail: true,
    };

    let result = manager.build_context(&[], true, Some(&provider)).await;
    assert!(matches!(result, Err(ContextError::SummaryFailed(_))));
    let untouched = manager.build_context(&[], false, None).await.unwrap();
    assert_eq!(untouched.messages.len(), 16);
    assert!(
        !untouched
            .system_prompt
            .unwrap()
            .contains("<conversation_summary>")
    );
}

// Verify successive compactions feed the previous summary forward once.
#[tokio::test]
async fn subsequent_summary_input_contains_previous_summary() {
    let mut manager = ContextManager::new(2400, 0.25, 0.5);
    manager.set_messages(long_conversation());
    let first = fake_provider("first summary");
    let first_built = manager
        .build_context(&[], true, Some(&first))
        .await
        .unwrap();
    manager.apply_compaction(&first_built.compaction.unwrap());
    manager.push_message(LlmMessage::user(format!("new request {}", "n".repeat(500))));
    let second = fake_provider("second summary");
    let second_built = manager
        .build_context(&[], true, Some(&second))
        .await
        .unwrap();
    let inputs = second.inputs.lock().await;
    assert!(inputs[0].contains("<previous_summary>first summary</previous_summary>"));
    assert_eq!(
        second_built
            .system_prompt
            .as_deref()
            .unwrap()
            .matches("<conversation_summary>")
            .count(),
        1
    );
    assert!(
        second_built
            .system_prompt
            .unwrap()
            .contains("second summary")
    );
}
