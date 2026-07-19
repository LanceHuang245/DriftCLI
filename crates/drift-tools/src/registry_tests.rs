use super::*;

struct DummyTool {
    name: String,
}

#[async_trait]
impl Tool for DummyTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "dummy"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            success: true,
            content: self.name.clone(),
            error: None,
        })
    }
}

#[tokio::test]
async fn unregister_many_only_removes_requested_dynamic_tools() {
    let registry = ToolRegistry::new();
    registry
        .register_dynamic(Arc::new(DummyTool { name: "one".into() }))
        .await;
    registry
        .register_dynamic(Arc::new(DummyTool { name: "two".into() }))
        .await;
    registry
        .register_dynamic(Arc::new(DummyTool {
            name: "three".into(),
        }))
        .await;

    registry.unregister_many(["one", "three"]).await;

    assert!(registry.get_async("one").await.is_none());
    assert!(registry.get_async("three").await.is_none());
    assert!(registry.get_async("two").await.is_some());
}
