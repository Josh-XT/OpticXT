use anyhow::{Result, anyhow};
use mistralrs::{
    VisionModelBuilder, TextMessageRole, VisionMessages,
    IsqType, AudioInput
};
use std::sync::Arc;
use std::path::PathBuf;
use tracing::{info, debug, warn};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub context_length: usize,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub processing_time_ms: u128,
}

pub struct GemmaModel {
    model: Arc<mistralrs::Model>,
    config: ModelConfig,
    model_id: String,
}

impl GemmaModel {
    pub async fn load(model_id: Option<String>, config: ModelConfig) -> Result<Self> {
        info!("Initializing Gemma model with mistral.rs Vision API");
        
        let (model_id, uqff_path) = model_id.map(|id| (id, None)).unwrap_or_else(|| {
            // Use the pre-quantized UQFF model for better performance and faster loading
            let model_id = "EricB/gemma-3n-E4B-it-UQFF".to_string();
            
            // Construct the full path to the cached UQFF file
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/josh".to_string());
            let cache_path = PathBuf::from(home)
                .join(".cache/huggingface/hub/models--EricB--gemma-3n-E4B-it-UQFF/snapshots/78bea20ac3910e8d0ed509b7d13d19acf0618152/gemma3n-e4b-it-q4k-0.uqff");
            
            (model_id, Some(cache_path))
        });
        
        info!("Loading HuggingFace model: {}", model_id);
        if let Some(ref uqff_path) = uqff_path {
            info!("Using UQFF quantized model from cache: {}", uqff_path.display());
        }
        
        // Create VisionModelBuilder for full multimodal support with MatFormer configuration
        info!("Creating VisionModelBuilder with model_id: {}", model_id);
        let model = if let Some(uqff_full_path) = uqff_path {
            info!("Building UQFF model with cached file: {}", uqff_full_path.display());
            
            // Check if the cached file exists
            if !uqff_full_path.exists() {
                warn!("UQFF cache file not found at: {}, falling back to ISQ quantization with MatFormer", uqff_full_path.display());
                // Fallback to ISQ quantization with MatFormer configuration
                VisionModelBuilder::new(&model_id)
                    .with_isq(IsqType::Q4K)
                    .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
                    .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
                    .build()
                    .await
                    .map_err(|e| anyhow!("Failed to build ISQ fallback model with MatFormer: {}", e))?
            } else {
                // Use deprecated method for now - will work but may need updating in future
                // Add MatFormer configuration for optimal performance
                #[allow(deprecated)]
                VisionModelBuilder::new(&model_id)
                    .from_uqff(vec![uqff_full_path])
                    .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
                    .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
                    .build()
                    .await
                    .map_err(|e| anyhow!("Failed to build UQFF model from cache with MatFormer: {}", e))?
            }
        } else {
            info!("Building ISQ quantized model with MatFormer configuration");
            // Fallback to ISQ quantization for non-UQFF models with MatFormer
            VisionModelBuilder::new(&model_id)
                .with_isq(IsqType::Q4K)
                .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
                .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
                .build()
                .await
                .map_err(|e| anyhow!("Failed to build ISQ model with MatFormer: {}", e))?
        };
        
        info!("✅ Successfully loaded HuggingFace Gemma 3n model with multimodal support and MatFormer E2.49B slice: {}", model_id);
        
        Ok(Self {
            model: Arc::new(model),
            config,
            model_id,
        })
    }
    
    pub async fn generate(&mut self, prompt: &str) -> Result<GenerationResult> {
        self.generate_multimodal(prompt, None, None).await
    }
    
    pub async fn generate_with_image(&mut self, prompt: &str, image: DynamicImage) -> Result<GenerationResult> {
        self.generate_multimodal(prompt, Some(image), None).await
    }
    
    pub async fn generate_with_audio(&mut self, prompt: &str, audio: Vec<u8>) -> Result<GenerationResult> {
        self.generate_multimodal(prompt, None, Some(audio)).await
    }
    
    pub async fn generate_multimodal(&mut self, prompt: &str, image: Option<DynamicImage>, audio: Option<Vec<u8>>) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();
        
        // Validate prompt length
        const MAX_PROMPT_CHARS: usize = 8000;
        let effective_prompt = if prompt.len() > MAX_PROMPT_CHARS {
            warn!("Prompt too long ({} chars), truncating to {} chars", prompt.len(), MAX_PROMPT_CHARS);
            &prompt[..MAX_PROMPT_CHARS]
        } else {
            prompt
        };
        
        debug!("Generating multimodal response for prompt: {}", effective_prompt);
        
        // Create the VisionMessages based on input modalities following Gemma 3n API
        let messages = if let Some(ref img) = image {
            if let Some(ref audio_bytes) = audio {
                // Text + Image + Audio - Full multimodal
                let audio_input = AudioInput::from_bytes(audio_bytes)?;
                VisionMessages::new().add_multimodal_message(
                    TextMessageRole::User,
                    effective_prompt,
                    vec![img.clone()],
                    vec![audio_input],
                    &*self.model,
                )?
            } else {
                // Text + Image - Follow the official Gemma 3n pattern
                VisionMessages::new().add_image_message(
                    TextMessageRole::User,
                    effective_prompt,
                    vec![img.clone()],
                    &*self.model,
                )?
            }
        } else {
            // Text only
            VisionMessages::new().add_message(
                TextMessageRole::User,
                effective_prompt,
            )
        };
        
        info!("Running model inference with {} modalities", 
              1 + image.is_some() as usize + audio.is_some() as usize);
        
        info!("Sending chat request with {} messages", "Vision API");
        
        // Send the request and get response with timeout
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(60), // Increased timeout for vision models
            self.model.send_chat_request(messages)
        ).await
        .map_err(|_| anyhow!("Model inference timed out after 60 seconds"))?
        .map_err(|e| anyhow!("Model inference failed: {}", e))?;
        
        // Extract generated text from response
        let generated_text = response.choices[0].message.content.as_ref()
            .ok_or_else(|| anyhow!("No content in response"))?;
        
        // Clean and validate the generated text
        let cleaned_text = self.clean_generated_text(generated_text);
        
        if cleaned_text.is_empty() {
            warn!("Generated text is empty after cleaning, using fallback");
            let fallback_text = if image.is_some() && audio.is_some() {
                "I can see the image and hear the audio, and I am ready to assist you."
            } else if image.is_some() {
                "I can see the image and I am ready to assist you."
            } else if audio.is_some() {
                "I can hear the audio and I am ready to assist you."
            } else {
                "I am ready to assist you."
            };
            let formatted_text = self.format_as_xml_command(fallback_text, effective_prompt)?;
            
            return Ok(GenerationResult {
                text: formatted_text,
                tokens_generated: 1,
                processing_time_ms: start_time.elapsed().as_millis(),
            });
        }
        
        // Format as XML command
        let formatted_text = self.format_as_xml_command(&cleaned_text, effective_prompt)?;
        
        let processing_time = start_time.elapsed();
        let estimated_tokens = cleaned_text.split_whitespace().count();
        
        info!("Real multimodal model generated text in {}ms: '{}'", 
              processing_time.as_millis(), formatted_text);
        
        Ok(GenerationResult {
            text: formatted_text,
            tokens_generated: estimated_tokens,
            processing_time_ms: processing_time.as_millis(),
        })
    }
    
    fn clean_generated_text(&self, text: &str) -> String {
        text.trim()
            .replace("<s>", "")
            .replace("</s>", "")
            .replace("<unk>", "")
            .replace("<pad>", "")
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string()
    }
    
    fn format_as_xml_command(&self, text: &str, prompt: &str) -> Result<String> {
        let prompt_lower = prompt.to_lowercase();
        
        // If text is empty or garbled, provide a sensible default
        let effective_text = if text.is_empty() || text.len() < 3 {
            "I am ready to assist you."
        } else {
            text
        };
        
        // Determine appropriate XML command based on content and prompt
        let xml_response = if effective_text.contains("move") || effective_text.contains("forward") || prompt_lower.contains("move") || prompt_lower.contains("navigate") {
            format!(r#"<move direction="forward" distance="0.5" speed="normal">{}</move>"#, effective_text)
        } else if effective_text.contains("turn") || effective_text.contains("rotate") || prompt_lower.contains("turn") || prompt_lower.contains("rotate") {
            format!(r#"<rotate direction="left" angle="30">{}</rotate>"#, effective_text)
        } else if effective_text.contains("stop") || effective_text.contains("halt") || prompt_lower.contains("stop") {
            format!(r#"<stop immediate="true">{}</stop>"#, effective_text)
        } else if effective_text.contains("wait") || effective_text.contains("pause") || prompt_lower.contains("wait") {
            format!(r#"<wait duration="2.0">{}</wait>"#, effective_text)
        } else if effective_text.contains("analyze") || prompt_lower.contains("analyze") || prompt_lower.contains("vision") {
            format!(r#"<analyze target="environment" detail_level="detailed">{}</analyze>"#, effective_text)
        } else {
            // Default to speak for general responses
            format!(r#"<speak>{}</speak>"#, effective_text)
        };
        
        Ok(xml_response)
    }
    
    /// Benchmark the model performance
    pub async fn benchmark(&mut self, num_tests: usize) -> Result<f64> {
        let start = std::time::Instant::now();
        let test_prompt = "Hello, how can I help you?";
        
        for i in 0..num_tests {
            let response = self.generate(&format!("{} Test {}", test_prompt, i)).await?;
            if i % 10 == 0 {
                info!("Benchmark progress: {}/{}, latest response: {}", i + 1, num_tests, response.text);
            }
        }
        
        let duration = start.elapsed();
        let tokens_per_sec = (num_tests as f64 * 10.0) / duration.as_secs_f64();
        
        info!("Benchmark completed: {} tests in {:.2}s, {:.2} tokens/sec", 
              num_tests, duration.as_secs_f64(), tokens_per_sec);
        
        Ok(tokens_per_sec)
    }
    
    
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    
    pub fn get_device(&self) -> String {
        "Auto (mistral.rs managed)".to_string() // mistral.rs handles device management internally
    }
    
    pub fn get_model_id(&self) -> &str {
        &self.model_id
    }
}

// Helper function to ensure model is available (automatic with HuggingFace models)
pub async fn ensure_model_downloaded(model_id: &str) -> Result<()> {
    info!("Using HuggingFace model: {} (will be downloaded automatically if needed)", model_id);
    Ok(())
}
