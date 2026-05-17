pub mod anthropic;
pub mod openai_compat;
pub mod types;

use async_trait::async_trait;
pub use types::*;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn provider_id(&self) -> &str;
    fn model_name(&self) -> &str;
    fn context_window(&self) -> usize;

    async fn stream_chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
        temperature: Option<f64>,
        max_output_tokens: Option<usize>,
    ) -> Result<LlmResponseStream, LlmError>;

    async fn chat(
        &self,
        messages: Vec<LlmMessage>,
        system_prompt: Option<String>,
    ) -> Result<String, LlmError> {
        let mut stream = self
            .stream_chat(messages, system_prompt, None, None)
            .await?;
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(LlmChunk::TextDelta(t)) => text.push_str(&t),
                Ok(LlmChunk::Done) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(text)
    }
}

pub fn create_provider(config: &drift_config::LlmConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    match config {
        drift_config::LlmConfig::Anthropic {
            api_key,
            model,
            base_url,
        } => Ok(Box::new(anthropic::AnthropicProvider::new(
            api_key.clone(),
            model.clone(),
            base_url.clone(),
        ))),
        drift_config::LlmConfig::OpenAiCompatible {
            api_key,
            model,
            base_url,
            ..
        } => Ok(Box::new(openai_compat::OpenAiCompatibleProvider::new(
            api_key.clone(),
            model.clone(),
            base_url.clone(),
        ))),
    }
}
