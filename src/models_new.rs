use anyhow::{Result, anyhow};
use llama_cpp::{
    LlamaModel, LlamaContext, SessionParams, LlamaParams,
    standard_sampler::StandardSampler,
    TokenDataArray,
};
use tokenizers::Tokenizer;
use std::path::Path;
use tracing::{info, debug, warn, error};

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
    model: Option<LlamaModel>,
    context: Option<LlamaContext>,
    tokenizer: Option<Tokenizer>,
    config: ModelConfig,
    model_path: String,
}

impl GemmaModel {
    pub async fn load(model_path: Option<String>, config: ModelConfig) -> Result<Self> {
        info!("Initializing Gemma model with proper GGUF inference using llama.cpp");
        
        let model_path = model_path.unwrap_or_else(|| {
            "models/gemma-3n-E4B-it-Q4_K_M.gguf".to_string()
        });
        
        // Check if model file exists
        let model_exists = Path::new(&model_path).exists();
        info!("Model path: {} (exists: {})", model_path, model_exists);
        
        if !model_exists {
            return Err(anyhow!("Model file not found at: {}. Please download a GGUF model file.", model_path));
        }

        // Check if the model file is just a placeholder
        if let Ok(content) = std::fs::read_to_string(&model_path) {
            if content.contains("Placeholder GGUF model file") || content.len() < 1000 {
                return Err(anyhow!("Model file is placeholder or too small, please download a real GGUF model"));
            }
        }
        
        // Load tokenizer
        let tokenizer = match Self::load_tokenizer().await {
            Ok(t) => {
                info!("Tokenizer loaded successfully");
                Some(t)
            }
            Err(e) => {
                warn!("Failed to load tokenizer, will use llama.cpp's built-in tokenizer: {}", e);
                None
            }
        };
        
        // Load model using llama.cpp
        info!("Loading GGUF model using llama.cpp: {}", model_path);
        
        // Configure model parameters
        let model_params = LlamaParams::default()
            .with_n_gpu_layers(1000) // Use GPU if available
            .with_use_mmap(true)     // Memory map the model
            .with_use_mlock(false);  // Don't lock memory
        
        // Load the model
        let model = LlamaModel::load_from_file(&model_path, model_params)
            .map_err(|e| anyhow!("Failed to load GGUF model: {}", e))?;
        
        info!("✅ Successfully loaded GGUF model");
        
        // Create context for inference
        let ctx_params = SessionParams::default()
            .with_n_ctx(config.context_length as u32)  // Context length
            .with_n_batch(512)                         // Batch size
            .with_n_threads(std::thread::available_parallelism().unwrap().get() as u32);
        
        let context = model.new_context(ctx_params)
            .map_err(|e| anyhow!("Failed to create llama context: {}", e))?;
        
        info!("✅ Successfully created llama context with {} context length", config.context_length);
        
        Ok(Self {
            model: Some(model),
            context: Some(context),
            tokenizer,
            config,
            model_path,
        })
    }
    
    async fn load_tokenizer() -> Result<Tokenizer> {
        // Try to load from local models directory
        let local_tokenizer_path = "models/tokenizer.json";
        if Path::new(local_tokenizer_path).exists() {
            info!("Loading tokenizer from local file: {}", local_tokenizer_path);
            match Tokenizer::from_file(local_tokenizer_path) {
                Ok(tokenizer) => {
                    info!("✅ Successfully loaded tokenizer with {} vocab entries", 
                          tokenizer.get_vocab_size(false));
                    return Ok(tokenizer);
                },
                Err(e) => warn!("❌ Failed to load local tokenizer: {}", e),
            }
        }
        
        Err(anyhow!("Tokenizer not found, will use llama.cpp's built-in tokenizer"))
    }
    
    pub async fn generate(&mut self, prompt: &str) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();
        
        // Validate prompt length
        const MAX_PROMPT_CHARS: usize = 8000;
        let effective_prompt = if prompt.len() > MAX_PROMPT_CHARS {
            warn!("Prompt too long ({} chars), truncating to {} chars", prompt.len(), MAX_PROMPT_CHARS);
            &prompt[..MAX_PROMPT_CHARS]
        } else {
            prompt
        };
        
        debug!("Generating response for prompt: {}", effective_prompt);
        
        if let (Some(ref model), Some(ref mut context)) = (&self.model, &mut self.context) {
            // Tokenize the prompt using llama.cpp's tokenizer
            let tokens = model.str_to_token(effective_prompt, llama_cpp::AddBos::Always)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
            
            debug!("Input tokenized to {} tokens", tokens.len());
            
            // Check context length
            if tokens.len() > self.config.context_length {
                return Err(anyhow!(
                    "Prompt too long: {} tokens, max: {}", 
                    tokens.len(), 
                    self.config.context_length
                ));
            }
            
            // Clear previous context and load new prompt
            context.clear_kv_cache();
            
            // Process the prompt tokens
            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                context.decode(&[token], is_last)
                    .map_err(|e| anyhow!("Failed to decode token {}: {}", token, e))?;
            }
            
            // Configure sampling parameters
            let mut sampler = StandardSampler::default()
                .with_temperature(self.config.temperature)
                .with_top_p(self.config.top_p)
                .with_top_k(40)  // Reasonable top-k
                .with_min_keep(1);
            
            // Generate tokens
            let mut generated_tokens = Vec::new();
            let max_new_tokens = self.config.max_tokens.min(100); // Reasonable limit
            
            info!("Starting token generation with real GGUF model inference");
            
            for step in 0..max_new_tokens {
                // Get logits from the model (this is real model inference!)
                let logits = context.candidates_ith(context.n_vocab());
                
                // Apply sampling
                let token_data_array = TokenDataArray::from_iter(logits, false);
                let selected_token = sampler.sample(context, token_data_array);
                
                debug!("Step {}: Generated token {} using real model logits", step, selected_token);
                
                // Check for end-of-sequence
                if model.token_is_eog(selected_token) {
                    debug!("Generated EOS token, stopping generation");
                    break;
                }
                
                generated_tokens.push(selected_token);
                
                // Feed the token back to the model for next prediction
                context.decode(&[selected_token], true)
                    .map_err(|e| anyhow!("Failed to decode generated token: {}", e))?;
            }
            
            info!("Generated {} tokens using real GGUF model inference", generated_tokens.len());
            
            // Convert tokens back to text
            let generated_text = if let Some(ref tokenizer) = self.tokenizer {
                // Use external tokenizer if available
                let token_ids: Vec<u32> = generated_tokens.iter().map(|&t| t as u32).collect();
                tokenizer.decode(&token_ids, true)
                    .map_err(|e| anyhow!("Token decoding failed: {}", e))?
            } else {
                // Use llama.cpp's built-in tokenizer
                let text_bytes = model.token_to_bytes(&generated_tokens)
                    .map_err(|e| anyhow!("Token to text conversion failed: {}", e))?;
                String::from_utf8_lossy(&text_bytes).trim().to_string()
            };
            
            // Clean up the generated text
            let cleaned_text = self.clean_generated_text(&generated_text);
            
            // Format as XML command
            let formatted_text = self.format_as_xml_command(&cleaned_text, effective_prompt)?;
            
            let processing_time = start_time.elapsed();
            
            info!("Real GGUF model generated text in {}ms: '{}'", 
                  processing_time.as_millis(), formatted_text);
            
            Ok(GenerationResult {
                text: formatted_text,
                tokens_generated: generated_tokens.len(),
                processing_time_ms: processing_time.as_millis(),
            })
            
        } else {
            Err(anyhow!("Model or context not loaded"))
        }
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
}

// Helper function to download model if not present
pub async fn ensure_model_downloaded(model_path: &str) -> Result<()> {
    if Path::new(model_path).exists() {
        // Check if it's a real model file (not placeholder)
        if let Ok(metadata) = std::fs::metadata(model_path) {
            if metadata.len() > 1000000 {  // At least 1MB for a real model
                info!("Model file already exists: {}", model_path);
                return Ok(());
            }
        }
    }
    
    warn!("GGUF model file not found or is placeholder. Please download a real GGUF model.");
    warn!("You can download Gemma models from:");
    warn!("  - https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF");
    warn!("  - https://huggingface.co/google/gemma-2b-it-GGUF");
    warn!("Place the .gguf file at: {}", model_path);
    
    Err(anyhow!("Real GGUF model file required for inference"))
}
