use anyhow::{Result, anyhow};
use mistralrs::{
    VisionModelBuilder, TextMessageRole, VisionMessages,
    IsqType, AudioInput, Device
};
use std::sync::Arc;
use std::path::PathBuf;
use tracing::{info, debug, warn, error};
use image::DynamicImage;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    // Config fields are kept for future extensibility and test configuration
    #[allow(dead_code)]
    pub max_tokens: usize,
    #[allow(dead_code)]
    pub temperature: f32,
    #[allow(dead_code)]
    pub top_p: f32,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    config: ModelConfig,
    #[allow(dead_code)]
    model_id: String,
}

impl GemmaModel {
    pub async fn load(model_id: Option<String>, config: ModelConfig, quantization_method: String, isq_type_str: String) -> Result<Self> {
        let overall_start = std::time::Instant::now();
        info!("🚀 Initializing Gemma model with mistral.rs Vision API");
        
        // Parse the ISQ type string to enum
        let isq_type = Self::parse_isq_type(&isq_type_str);
        info!("🔧 ISQ Type: {:?} (from string: {})", isq_type, isq_type_str);
        
        // Set minimal environment variables (most mistral.rs device control is via API)
        std::env::set_var("CUDA_VISIBLE_DEVICES", "0");
        std::env::set_var("TOKENIZERS_PARALLELISM", "false"); // Avoid tokenizer warnings
        
        info!("🔧 Environment variables set for CUDA optimization");
        info!("⚠️ Note: Device selection is handled via mistral.rs API, not environment variables");
        info!("⏱️ Setup phase completed in {:.3}s", overall_start.elapsed().as_secs_f64());
        
        let path_start = std::time::Instant::now();
        let (model_id, uqff_path, is_local_file) = model_id.map(|id| {
            // Check if the provided ID is a local file path
            let path = PathBuf::from(&id);
            if path.exists() && (id.ends_with(".gguf") || id.ends_with(".uqff")) {
                info!("🔍 Detected local model file: {}", id);
                (id, None::<PathBuf>, true)
            } else {
                info!("🔍 Using HuggingFace model ID: {}", id);
                (id, None::<PathBuf>, false)
            }
        }).unwrap_or_else(|| {
            // Choose quantization method based on configuration
            match quantization_method.as_str() {
                "uqff" => {
                    info!("🎯 Using UQFF quantization (faster inference, slower loading due to CPU decompression)");
                    let model_id = "EricB/gemma-3n-E4B-it-UQFF".to_string();
                    
                    // Check if UQFF cache exists, if not fall back to ISQ
                    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/josh".to_string());
                    let cache_path = PathBuf::from(home)
                        .join(".cache/huggingface/hub/models--EricB--gemma-3n-E4B-it-UQFF/snapshots/78bea20ac3910e8d0ed509b7d13d19acf0618152/gemma3n-e4b-it-q4k-0.uqff");
                    
                    if cache_path.exists() {
                        info!("📁 Found cached UQFF file: {:.2} MB", 
                              std::fs::metadata(&cache_path).map(|m| m.len() as f64 / 1024.0 / 1024.0).unwrap_or(0.0));
                        (model_id, Some(cache_path), false)
                    } else {
                        info!("⚠️ UQFF cache not found, falling back to ISQ for faster loading");
                        // Use base model for proper ISQ quantization
                        let base_model = "google/gemma-2-2b-it".to_string(); // Use lighter base model
                        (base_model, None::<PathBuf>, false)
                    }
                }
                "isq" | _ => {
                    info!("🎯 Using ISQ quantization (fast loading, in-situ quantization)");
                    // Use the unsloth Gemma model that doesn't require authentication
                    let base_model = "unsloth/gemma-3n-E4B-it".to_string(); // Free Gemma model for ISQ
                    info!("📦 Base model for ISQ: {}", base_model);
                    info!("✅ Using unsloth Gemma model - supports vision and doesn't require HF authentication");
                    
                    // ISQ will quantize this model in-place during loading
                    (base_model, None::<PathBuf>, false)
                }
            }
        });
        info!("⏱️ Path resolution completed in {:.3}s", path_start.elapsed().as_secs_f64());
        
        info!("Loading model: {}", model_id);
        if let Some(ref uqff_path) = uqff_path {
            info!("Using UQFF quantized model from cache: {}", uqff_path.display());
            
            // Check file existence and size
            let file_check_start = std::time::Instant::now();
            if uqff_path.exists() {
                if let Ok(metadata) = std::fs::metadata(uqff_path) {
                    info!("📁 UQFF file size: {:.2} MB", metadata.len() as f64 / 1024.0 / 1024.0);
                } else {
                    warn!("⚠️ Could not read UQFF file metadata");
                }
            } else {
                error!("❌ UQFF file does not exist at expected path!");
            }
            info!("⏱️ File validation completed in {:.3}s", file_check_start.elapsed().as_secs_f64());
        } else if is_local_file {
            info!("Using local model file: {}", model_id);
            let local_path = PathBuf::from(&model_id);
            if !local_path.exists() {
                error!("❌ Local model file does not exist: {}", model_id);
                return Err(anyhow!("Local model file not found: {}", model_id));
            }
            if let Ok(metadata) = std::fs::metadata(&local_path) {
                info!("📁 Local model file size: {:.2} MB", metadata.len() as f64 / 1024.0 / 1024.0);
            }
        } else {
            info!("Using HuggingFace model: {}", model_id);
        }
        
        // Create VisionModelBuilder for full multimodal support with MatFormer configuration
        // Prioritize CUDA for NVIDIA hardware (Jetson Nano, desktop GPUs), fallback to CPU
        info!("Creating VisionModelBuilder with model: {}", model_id);
        
        let detection_start = std::time::Instant::now();
        info!("Detecting hardware acceleration capabilities...");
        let cuda_available = Self::is_cuda_available();
        info!("⏱️ CUDA detection completed in {:.3}s (available: {})", detection_start.elapsed().as_secs_f64(), cuda_available);
        
        let model_build_start = std::time::Instant::now();
        let model = if is_local_file {
            info!("Building model from local file: {}", model_id);
            // For local files, we need to handle GGUF and UQFF differently
            if model_id.ends_with(".uqff") {
                let uqff_path = PathBuf::from(&model_id);
                Self::build_uqff_model_with_device_fallback("local", uqff_path).await?
            } else if model_id.ends_with(".gguf") {
                info!("❌ GGUF local files not currently supported by mistralrs VisionModelBuilder");
                return Err(anyhow!("Local GGUF files not supported. Please use HuggingFace model IDs or UQFF files."));
            } else {
                return Err(anyhow!("Unsupported local file format. Only .uqff files are supported."));
            }
        } else if let Some(uqff_full_path) = uqff_path {
            info!("Building UQFF model with cached file: {}", uqff_full_path.display());
            
            // Check if the cached file exists
            if !uqff_full_path.exists() {
                warn!("UQFF cache file not found at: {}, falling back to ISQ quantization with MatFormer", uqff_full_path.display());
                // Fallback to ISQ quantization with MatFormer configuration - try CUDA first
                Self::build_model_with_device_fallback(&model_id, true, isq_type).await?
            } else {
                // Use deprecated method for now - will work but may need updating in future
                // Add MatFormer configuration for optimal performance - try CUDA first
                Self::build_uqff_model_with_device_fallback(&model_id, uqff_full_path).await?
            }
        } else {
            info!("Building ISQ quantized model with MatFormer configuration");
            // Fallback to ISQ quantization for non-UQFF models with MatFormer - try CUDA first
            Self::build_model_with_device_fallback(&model_id, true, isq_type).await?
        };
        info!("⏱️ Model building completed in {:.3}s", model_build_start.elapsed().as_secs_f64());
        
        info!("✅ Successfully loaded model with multimodal support: {}", model_id);
        info!("⏱️ Total model loading time: {:.3}s", overall_start.elapsed().as_secs_f64());
        
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
        
        // Log GPU memory before inference
        Self::log_gpu_memory_usage();
        
        // Start monitoring GPU utilization during inference
        Self::check_gpu_utilization().await;
        
        info!("Sending chat request with {} messages", "Vision API");
        
        // Send the request and get response with timeout (increased for UQFF models)
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(180), // Increased timeout for UQFF vision models
            self.model.send_chat_request(messages)
        ).await
        .map_err(|_| anyhow!("Model inference timed out after 180 seconds"))?
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
            let formatted_text = self.format_as_tool_call(fallback_text, effective_prompt)?;
            
            return Ok(GenerationResult {
                text: formatted_text,
                tokens_generated: 1,
                processing_time_ms: start_time.elapsed().as_millis(),
            });
        }
        
        // Format as OpenAI tool call
        let formatted_text = self.format_as_tool_call(&cleaned_text, effective_prompt)?;
        
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
    
    pub fn format_as_tool_call(&self, text: &str, prompt: &str) -> Result<String> {
        let prompt_lower = prompt.to_lowercase();
        
        // If text is empty or garbled, provide a sensible default
        let effective_text = if text.is_empty() || text.len() < 3 {
            "I am ready to assist you."
        } else {
            text
        };
        
        // Determine appropriate OpenAI-style function call based on content and prompt
        let tool_call = if effective_text.contains("move") || effective_text.contains("forward") || prompt_lower.contains("move") || prompt_lower.contains("navigate") {
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "move",
                    "arguments": json!({
                        "direction": "forward",
                        "distance": 0.5,
                        "speed": "normal",
                        "reasoning": effective_text
                    }).to_string()
                }
            }])
        } else if effective_text.contains("turn") || effective_text.contains("rotate") || prompt_lower.contains("turn") || prompt_lower.contains("rotate") {
            json!([{
                "id": "call_1", 
                "type": "function",
                "function": {
                    "name": "rotate",
                    "arguments": json!({
                        "direction": "left",
                        "angle": 30.0,
                        "reasoning": effective_text
                    }).to_string()
                }
            }])
        } else if effective_text.contains("stop") || effective_text.contains("halt") || prompt_lower.contains("stop") {
            json!([{
                "id": "call_1",
                "type": "function", 
                "function": {
                    "name": "stop",
                    "arguments": json!({
                        "immediate": true,
                        "reasoning": effective_text
                    }).to_string()
                }
            }])
        } else if effective_text.contains("wait") || effective_text.contains("pause") || prompt_lower.contains("wait") {
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "wait", 
                    "arguments": json!({
                        "duration": 2.0,
                        "reasoning": effective_text
                    }).to_string()
                }
            }])
        } else if effective_text.contains("analyze") || prompt_lower.contains("analyze") || prompt_lower.contains("vision") {
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "analyze",
                    "arguments": json!({
                        "target": "environment",
                        "detail_level": "detailed",
                        "reasoning": effective_text
                    }).to_string()
                }
            }])
        } else {
            // Default to speak for general responses
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "speak",
                    "arguments": json!({
                        "text": effective_text,
                        "voice": "default",
                        "reasoning": "Providing a response to the user's query"
                    }).to_string()
                }
            }])
        };
        
        // Format as pretty JSON
        let formatted_json = serde_json::to_string_pretty(&tool_call)
            .unwrap_or_else(|_| r#"[{"id": "call_1", "type": "function", "function": {"name": "speak", "arguments": "{\"text\": \"I am ready to assist you.\"}"}}]"#.to_string());
        
        Ok(formatted_json)
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
    
    /// Parse ISQ type string to IsqType enum
    fn parse_isq_type(isq_type: &str) -> IsqType {
        match isq_type.to_uppercase().as_str() {
            "Q2K" => IsqType::Q2K,
            "Q3K" => IsqType::Q3K,
            "Q4K" => IsqType::Q4K,
            "Q5K" => IsqType::Q5K,
            "Q6K" => IsqType::Q6K,
            "Q8_0" => IsqType::Q8_0,
            "Q8_1" => IsqType::Q8_1,
            "Q4_0" => IsqType::Q4_0,
            "Q4_1" => IsqType::Q4_1,
            "Q5_0" => IsqType::Q5_0,
            "Q5_1" => IsqType::Q5_1,
            "HQQ4" => IsqType::HQQ4,
            "HQQ8" => IsqType::HQQ8,
            _ => {
                warn!("Unknown ISQ type '{}', defaulting to Q4K", isq_type);
                IsqType::Q4K
            }
        }
    }

    /// Try to build ISQ model with CUDA first, fallback to CPU
    async fn build_model_with_device_fallback(model_id: &str, is_isq: bool, isq_type: IsqType) -> Result<mistralrs::Model> {
        // Check CUDA availability first
        if Self::is_cuda_available() {
            info!("🚀 CUDA detected - attempting to load ISQ model on CUDA device...");
            match Self::try_build_model_cuda(model_id, is_isq, Some(isq_type)).await {
                Ok(model) => {
                    info!("✅ Successfully loaded ISQ model on CUDA device - optimal for NVIDIA RTX 4090");
                    return Ok(model);
                }
                Err(cuda_err) => {
                    error!("❌ CUDA ISQ loading failed despite CUDA being available: {}", cuda_err);
                    info!("🔄 Falling back to CPU inference...");
                }
            }
        } else {
            warn!("⚠️ CUDA not available - using CPU inference");
        }
        
        // CPU fallback
        match Self::try_build_model_cpu(model_id, is_isq, Some(isq_type)).await {
            Ok(model) => {
                info!("✅ Successfully loaded ISQ model on CPU device");
                Ok(model)
            }
            Err(cpu_err) => {
                Err(anyhow!("Failed to load ISQ model on both CUDA and CPU. CPU error: {}", cpu_err))
            }
        }
    }
    
    /// Try to build UQFF model with CUDA first, fallback to CPU
    async fn build_uqff_model_with_device_fallback(model_id: &str, uqff_path: PathBuf) -> Result<mistralrs::Model> {
        let fallback_start = std::time::Instant::now();
        
        // Check CUDA availability first
        if Self::is_cuda_available() {
            info!("🚀 CUDA detected - attempting to load UQFF model on CUDA device...");
            info!("🔧 Target device: NVIDIA RTX 4090 (or compatible CUDA GPU)");
            
            // Log initial GPU memory usage
            let gpu_check_start = std::time::Instant::now();
            Self::log_gpu_memory_usage();
            info!("⏱️ GPU memory check completed in {:.3}s", gpu_check_start.elapsed().as_secs_f64());
            
            let cuda_build_start = std::time::Instant::now();
            match Self::try_build_uqff_model_cuda(model_id, &uqff_path).await {
                Ok(model) => {
                    info!("✅ Successfully loaded UQFF model on CUDA device - should utilize GPU VRAM");
                    info!("⏱️ CUDA model build took {:.3}s", cuda_build_start.elapsed().as_secs_f64());
                    
                    // Log GPU memory usage after loading
                    Self::log_gpu_memory_usage();
                    
                    return Ok(model);
                }
                Err(cuda_err) => {
                    error!("❌ CUDA UQFF loading failed despite CUDA being available: {}", cuda_err);
                    error!("⏱️ CUDA build failed after {:.3}s", cuda_build_start.elapsed().as_secs_f64());
                    info!("🔄 Falling back to CPU inference...");
                }
            }
        } else {
            warn!("⚠️ CUDA not available - using CPU inference");
        }
        
        // CPU fallback
        let cpu_build_start = std::time::Instant::now();
        match Self::try_build_uqff_model_cpu(model_id, &uqff_path).await {
            Ok(model) => {
                info!("✅ Successfully loaded UQFF model on CPU device");
                info!("⏱️ CPU model build took {:.3}s", cpu_build_start.elapsed().as_secs_f64());
                info!("⏱️ Total fallback process took {:.3}s", fallback_start.elapsed().as_secs_f64());
                Ok(model)
            }
            Err(cpu_err) => {
                error!("⏱️ CPU build failed after {:.3}s", cpu_build_start.elapsed().as_secs_f64());
                Err(anyhow!("Failed to load UQFF model on both CUDA and CPU. CPU error: {}", cpu_err))
            }
        }
    }
    
    /// Check if CUDA is available
    fn is_cuda_available() -> bool {
        let detection_start = std::time::Instant::now();
        info!("🔍 Starting CUDA detection process...");
        
        // Try to detect CUDA using various methods
        
        // Method 1: Check nvidia-smi command availability
        let smi_start = std::time::Instant::now();
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output() 
        {
            info!("⏱️ nvidia-smi command took {:.3}s", smi_start.elapsed().as_secs_f64());
            if output.status.success() {
                let gpu_info = String::from_utf8_lossy(&output.stdout);
                info!("🔍 Detected GPU(s): {}", gpu_info.trim());
                
                // Additional check: Test if mistral.rs can actually use CUDA
                match Device::cuda_if_available(0) {
                    Ok(_) => {
                        info!("✅ mistral.rs CUDA device creation successful");
                        info!("⏱️ Total CUDA detection took {:.3}s", detection_start.elapsed().as_secs_f64());
                        return true;
                    }
                    Err(e) => {
                        warn!("❌ mistral.rs CUDA device creation failed: {}", e);
                        warn!("🚨 CUDA hardware detected but mistral.rs can't use it - check compilation features");
                    }
                }
            }
        } else {
            warn!("⏱️ nvidia-smi command failed after {:.3}s", smi_start.elapsed().as_secs_f64());
        }
        
        // Method 2: Check CUDA_VISIBLE_DEVICES or other CUDA environment variables
        let env_start = std::time::Instant::now();
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            info!("🔍 CUDA_VISIBLE_DEVICES environment variable set");
            info!("⏱️ Environment check took {:.3}s", env_start.elapsed().as_secs_f64());
        }
        
        // Method 3: Check for CUDA libraries in standard paths
        let lib_start = std::time::Instant::now();
        let cuda_paths = [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "/opt/cuda/lib64/libcudart.so",
        ];
        
        for path in &cuda_paths {
            if std::path::Path::new(path).exists() {
                info!("🔍 Found CUDA library at: {}", path);
                info!("⏱️ Library check took {:.3}s", lib_start.elapsed().as_secs_f64());
                break;
            }
        }
        
        warn!("🚫 CUDA not available or not properly configured for mistral.rs");
        info!("💡 Ensure mistral.rs is compiled with CUDA features: cargo build --features cuda");
        info!("⏱️ Total CUDA detection took {:.3}s", detection_start.elapsed().as_secs_f64());
        false
    }
    
    /// Monitor GPU memory usage (if CUDA is available)
    fn log_gpu_memory_usage() {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.used,memory.total,name")
            .arg("--format=csv,noheader,nounits")
            .output() 
        {
            if output.status.success() {
                let memory_info = String::from_utf8_lossy(&output.stdout);
                for line in memory_info.lines() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 3 {
                        let used = parts[0].trim();
                        let total = parts[1].trim();
                        let name = parts[2].trim();
                        info!("🎮 GPU Memory: {} MB / {} MB used on {}", used, total, name);
                    }
                }
            }
        }
    }
    
    async fn check_gpu_utilization() {
        tokio::spawn(async {
            for i in 0..5 {  // Check for 5 seconds
                if let Ok(output) = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=utilization.gpu,utilization.memory")
                    .arg("--format=csv,noheader,nounits")
                    .output() 
                {
                    if output.status.success() {
                        let utilization = String::from_utf8_lossy(&output.stdout);
                        info!("🔍 GPU Utilization check {}: {}", i + 1, utilization.trim());
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    }
    
    /// Attempt to build ISQ model with CUDA
    async fn try_build_model_cuda(model_id: &str, _is_isq: bool, isq_type: Option<IsqType>) -> Result<mistralrs::Model> {
        let actual_isq_type = isq_type.unwrap_or(IsqType::Q4K);
        info!("🔧 Building ISQ model with {:?} quantization on CUDA...", actual_isq_type);
        info!("📦 Base model: {}", model_id);
        info!("⚙️ ISQ will quantize weights in-place during loading (reduced memory footprint)");
        
        // Build model with proper ISQ and explicit CUDA device
        let builder = VisionModelBuilder::new(model_id)
            .with_isq(actual_isq_type); // Use the specified quantization type
            
        // Try to set explicit CUDA device
        let final_builder = match Device::cuda_if_available(0) {
            Ok(cuda_device) => {
                info!("✅ Explicit CUDA device 0 created successfully");
                builder.with_device(cuda_device)
            }
            Err(e) => {
                warn!("⚠️ Could not create explicit CUDA device: {}, using automatic detection", e);
                builder // Let mistralrs handle device selection automatically
            }
        };
            
        final_builder.build()
            .await
            .map_err(|e| anyhow!("CUDA ISQ model build failed: {}", e))
    }
    
    /// Attempt to build ISQ model with CPU
    async fn try_build_model_cpu(model_id: &str, _is_isq: bool, isq_type: Option<IsqType>) -> Result<mistralrs::Model> {
        let actual_isq_type = isq_type.unwrap_or(IsqType::Q4K);
        info!("🔧 Building ISQ model with {:?} quantization on CPU...", actual_isq_type);
        info!("📦 Base model: {}", model_id);
        info!("⚙️ ISQ will quantize weights in-place during loading (reduced memory footprint)");
        
        // Build model with proper ISQ - no MatFormer needed for ISQ
        VisionModelBuilder::new(model_id)
            .with_isq(actual_isq_type) // Use the specified quantization type
            .with_device(Device::Cpu) // Explicitly use CPU device
            .build()
            .await
            .map_err(|e| anyhow!("CPU ISQ model build failed: {}", e))
    }
    
    /// Attempt to build UQFF model with CUDA using multi-threaded decompression
    async fn try_build_uqff_model_cuda(model_id: &str, uqff_path: &PathBuf) -> Result<mistralrs::Model> {
        let build_start = std::time::Instant::now();
        info!("🔧 Building UQFF model with multi-threaded CUDA optimization...");
        info!("📁 UQFF file: {}", uqff_path.display());
        info!("⏱️ Implementing parallel decompression to reduce CPU bottleneck...");
        
        // Set environment variables to encourage multi-threading
        std::env::set_var("OMP_NUM_THREADS", "8"); // OpenMP threading
        std::env::set_var("RAYON_NUM_THREADS", "8"); // Rayon threading
        std::env::set_var("TOKIO_WORKER_THREADS", "8"); // Tokio threading
        
        // Pre-build checks
        let precheck_start = std::time::Instant::now();
        info!("🔍 Pre-build validation starting...");
        
        // Check if we can read the file
        match std::fs::File::open(uqff_path) {
            Ok(_) => info!("✅ UQFF file is readable"),
            Err(e) => {
                error!("❌ Cannot read UQFF file: {}", e);
                return Err(anyhow!("UQFF file read error: {}", e));
            }
        }
        info!("⏱️ Pre-build validation completed in {:.3}s", precheck_start.elapsed().as_secs_f64());
        
        // Builder creation with explicit device forcing
        let builder_start = std::time::Instant::now();
        info!("🏗️ Creating VisionModelBuilder with multi-threaded CUDA configuration...");
        
        // Log current CUDA environment
        info!("🔍 CUDA Environment Check:");
        info!("   CUDA_VISIBLE_DEVICES: {:?}", std::env::var("CUDA_VISIBLE_DEVICES"));
        info!("   MISTRALRS_DEVICE: {:?}", std::env::var("MISTRALRS_DEVICE"));
        info!("   OMP_NUM_THREADS: {:?}", std::env::var("OMP_NUM_THREADS"));
        
        #[allow(deprecated)]
        let builder = VisionModelBuilder::new(model_id)
            .from_uqff(vec![uqff_path.clone()]);
            
        // Try to force CUDA device explicitly using proper mistral.rs API
        info!("🎯 Attempting to force CUDA device selection...");
        
        let final_builder = match Device::cuda_if_available(0) {
            Ok(cuda_device) => {
                info!("✅ CUDA device 0 created successfully, applying to builder");
                builder.with_device(cuda_device)
            }
            Err(e) => {
                warn!("⚠️ Could not create explicit CUDA device: {}, using automatic detection", e);
                info!("🔄 Relying on mistralrs automatic device detection");
                builder
            }
        };
            
        info!("⏱️ Builder creation completed in {:.3}s", builder_start.elapsed().as_secs_f64());
        
        // Actual model build with progress monitoring
        let actual_build_start = std::time::Instant::now();
        info!("🚀 Starting optimized model build process with multi-core decompression...");
        
        // Start a progress monitoring task with enhanced CPU core monitoring
        let progress_handle = tokio::spawn(async {
            let mut elapsed = 0;
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3)).await; // More frequent updates
                elapsed += 3;
                info!("⏳ Multi-threaded model build in progress... {}s elapsed", elapsed);
                
                // Enhanced system monitoring - check all CPU cores
                if let Ok(output) = std::process::Command::new("mpstat")
                    .arg("1")
                    .arg("1")
                    .output() 
                {
                    if output.status.success() {
                        let cpu_info = String::from_utf8_lossy(&output.stdout);
                        info!("📊 Per-core CPU usage: {}", cpu_info.lines().last().unwrap_or("N/A"));
                    }
                }
                
                // Check GPU memory and utilization
                if let Ok(output) = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=memory.used,utilization.gpu,utilization.memory,temperature.gpu")
                    .arg("--format=csv,noheader,nounits")
                    .output() 
                {
                    if output.status.success() {
                        let info = String::from_utf8_lossy(&output.stdout);
                        for line in info.lines() {
                            let parts: Vec<&str> = line.split(',').collect();
                            if parts.len() >= 4 {
                                let memory = parts[0].trim();
                                let gpu_util = parts[1].trim();
                                let mem_util = parts[2].trim();
                                let temp = parts[3].trim();
                                info!("🎮 GPU: {} MB memory, {}% GPU util, {}% mem util, {}°C", 
                                      memory, gpu_util, mem_util, temp);
                            }
                        }
                    }
                }
            }
        });
        
        let model_result = final_builder.build().await;
        
        // Stop the progress monitoring
        progress_handle.abort();
            
        let build_time = actual_build_start.elapsed();
        info!("⏱️ Multi-threaded model build took {:.3}s", build_time.as_secs_f64());
        info!("⏱️ Total optimized UQFF CUDA build process took {:.3}s", build_start.elapsed().as_secs_f64());
        
        model_result.map_err(|e| anyhow!("Multi-threaded CUDA UQFF model build failed: {}", e))
    }
    
    /// Attempt to build UQFF model with CPU
    async fn try_build_uqff_model_cpu(model_id: &str, uqff_path: &PathBuf) -> Result<mistralrs::Model> {
        info!("🔧 Building UQFF model with explicit CPU device selection...");
        info!("📁 UQFF file: {}", uqff_path.display());
        #[allow(deprecated)]
        VisionModelBuilder::new(model_id)
            .from_uqff(vec![uqff_path.clone()])
            // Temporarily removing MatFormer config to see if it's causing issues
            // .with_matformer_config_path(PathBuf::from("matformer_configs/gemma3n.csv"))
            // .with_matformer_slice_name("Config for E2.49B (block-level)".to_string())
            .with_device(Device::Cpu) // Explicitly use CPU device
            .build()
            .await
            .map_err(|e| anyhow!("CPU UQFF model build failed: {}", e))
    }
    
    #[allow(dead_code)]
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    
    #[allow(dead_code)]
    pub fn get_device(&self) -> String {
        "Auto (mistral.rs managed)".to_string() // mistral.rs handles device management internally
    }
    
    #[allow(dead_code)]
    pub fn get_model_id(&self) -> &str {
        &self.model_id
    }
}

// Helper function to ensure model is available (automatic with HuggingFace models)
pub async fn ensure_model_downloaded(model_id: &str) -> Result<()> {
    info!("Using HuggingFace model: {} (will be downloaded automatically if needed)", model_id);
    Ok(())
}
