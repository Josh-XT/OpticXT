use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, debug, warn};
use image::DynamicImage;
use base64::{Engine as _, engine::general_purpose};
use crate::config::RemoteModelConfig;
use crate::models::GenerationResult;

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: ChatMessageContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ChatMessageContent {
    Text(String),
    Mixed(Vec<ContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u64>,
}

pub struct RemoteModel {
    config: RemoteModelConfig,
    client: reqwest::Client,
}

impl RemoteModel {
    pub fn new(config: RemoteModelConfig) -> Result<Self> {
        let timeout = Duration::from_secs(config.timeout_seconds.unwrap_or(60));
        
        let mut client_builder = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("OpticXT/1.0");

        // Add default headers
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        
        // Add authorization header
        let auth_header = format!("Bearer {}", config.api_key);
        default_headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&auth_header)
                .map_err(|e| anyhow!("Invalid API key format: {}", e))?,
        );

        // Add any additional headers
        if let Some(additional_headers) = &config.additional_headers {
            for (key, value) in additional_headers {
                default_headers.insert(
                    reqwest::header::HeaderName::from_bytes(key.as_bytes())
                        .map_err(|e| anyhow!("Invalid header name '{}': {}", key, e))?,
                    reqwest::header::HeaderValue::from_str(value)
                        .map_err(|e| anyhow!("Invalid header value for '{}': {}", key, e))?,
                );
            }
        }

        client_builder = client_builder.default_headers(default_headers);

        let client = client_builder.build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        info!("🌐 Initialized remote model client for: {}", config.model_name);
        info!("🔗 Base URL: {}", config.base_url);
        info!("🖼️ Vision support: {}", config.supports_vision);

        Ok(Self { config, client })
    }

    #[allow(dead_code)]
    pub async fn generate(&self, prompt: &str) -> Result<GenerationResult> {
        self.generate_multimodal(prompt, None, None).await
    }

    #[allow(dead_code)]
    pub async fn generate_with_image(&self, prompt: &str, image: DynamicImage) -> Result<GenerationResult> {
        if !self.config.supports_vision {
            warn!("⚠️ Image provided but remote model doesn't support vision, falling back to text-only");
            return self.generate(prompt).await;
        }
        self.generate_multimodal(prompt, Some(image), None).await
    }

    #[allow(dead_code)]
    pub async fn generate_with_audio(&self, prompt: &str, _audio: Vec<u8>) -> Result<GenerationResult> {
        // Most OpenAI-compatible APIs don't support audio in chat completions yet
        warn!("⚠️ Audio input not supported for remote models, falling back to text-only");
        self.generate(prompt).await
    }

    pub async fn generate_multimodal(&self, prompt: &str, image: Option<DynamicImage>, _audio: Option<Vec<u8>>) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        debug!("🌐 Generating response using remote model: {}", self.config.model_name);

        // Prepare the chat completion request
        let messages = if let Some(image) = image {
            if self.config.supports_vision {
                debug!("🖼️ Including image in multimodal request");
                let image_data = self.encode_image(image)?;
                vec![ChatMessage {
                    role: "user".to_string(),
                    content: ChatMessageContent::Mixed(vec![
                        ContentPart::Text { text: prompt.to_string() },
                        ContentPart::ImageUrl {
                            image_url: ImageUrl {
                                url: format!("data:image/jpeg;base64,{}", image_data),
                                detail: Some("high".to_string()),
                            },
                        },
                    ]),
                }]
            } else {
                warn!("⚠️ Image provided but model doesn't support vision, using text-only");
                vec![ChatMessage {
                    role: "user".to_string(),
                    content: ChatMessageContent::Text(prompt.to_string()),
                }]
            }
        } else {
            vec![ChatMessage {
                role: "user".to_string(),
                content: ChatMessageContent::Text(prompt.to_string()),
            }]
        };

        let request = ChatCompletionRequest {
            model: self.config.model_name.clone(),
            messages,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            max_tokens: self.config.max_tokens,
            stream: Some(false), // We don't support streaming yet
        };

        debug!("📤 Sending request to: {}/chat/completions", self.config.base_url);

        // Make the API call
        let response = self.client
            .post(&format!("{}/chat/completions", self.config.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send request to remote model: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("Remote model API error ({}): {}", status, error_text));
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse response from remote model: {}", e))?;

        let response_text = completion
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| anyhow!("No response content from remote model"))?
            .clone();

        let tokens_generated = completion
            .usage
            .and_then(|u| u.completion_tokens)
            .unwrap_or(0) as usize;

        let processing_time = start_time.elapsed();

        debug!("✅ Remote model response received in {:.2}ms", processing_time.as_millis());
        debug!("📊 Tokens generated: {}", tokens_generated);

        Ok(GenerationResult {
            text: response_text,
            tokens_generated,
            processing_time_ms: processing_time.as_millis(),
        })
    }

    fn encode_image(&self, image: DynamicImage) -> Result<String> {
        use std::io::Cursor;
        
        // Convert image to JPEG format for better compression
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        
        image
            .resize(1024, 1024, image::imageops::FilterType::Lanczos3) // Resize for efficiency
            .write_to(&mut cursor, image::ImageFormat::Jpeg)
            .map_err(|e| anyhow!("Failed to encode image as JPEG: {}", e))?;

        Ok(general_purpose::STANDARD.encode(&buffer))
    }
}
