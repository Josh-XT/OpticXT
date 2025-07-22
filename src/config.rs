use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticXTConfig {
    pub vision: VisionConfig,
    pub model: ModelConfig,
    pub context: ContextConfig,
    pub commands: CommandConfig,
    pub performance: PerformanceConfig,
    pub audio: AudioConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Camera resolution width
    pub width: u32,
    /// Camera resolution height
    pub height: u32,
    /// Frames per second for video capture
    pub fps: u32,
    /// Object detection confidence threshold
    pub confidence_threshold: f32,
    /// Vision model for labeling (can be local or API-based)
    pub vision_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: String,
    /// Model context length
    pub context_length: usize,
    /// Temperature for generation
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Mandatory context system prompt
    pub system_prompt: String,
    /// Maximum context history to maintain
    pub max_context_history: usize,
    /// Whether to include timestamp in context
    pub include_timestamp: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandConfig {
    /// Available command types
    pub enabled_commands: Vec<String>,
    /// Command execution timeout in seconds
    pub timeout_seconds: u64,
    /// Whether to validate commands before execution
    pub validate_before_execution: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads for processing
    pub worker_threads: usize,
    /// Buffer size for video frames
    pub frame_buffer_size: usize,
    /// Processing interval in milliseconds
    pub processing_interval_ms: u64,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Enable audio input/output
    pub enabled: bool,
    /// Sample rate for audio
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Enable text-to-speech output
    pub enable_tts: bool,
    /// Enable speech recognition input
    pub enable_speech_recognition: bool,
}

impl Default for OpticXTConfig {
    fn default() -> Self {
        Self {
            vision: VisionConfig {
                width: 640,
                height: 480,
                fps: 30,
                confidence_threshold: 0.5,
                vision_model: "yolo".to_string(),
            },
            model: ModelConfig {
                model_path: "models/gemma-3n-E4B-it-Q4_K_M.gguf".to_string(),
                context_length: 4096,
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 512,
            },
            context: ContextConfig {
                system_prompt: include_str!("../prompts/system_prompt.txt").to_string(),
                max_context_history: 10,
                include_timestamp: true,
            },
            commands: CommandConfig {
                enabled_commands: vec![
                    "move".to_string(),
                    "rotate".to_string(),
                    "speak".to_string(),
                    "analyze".to_string(),
                    "offload".to_string(),
                ],
                timeout_seconds: 30,
                validate_before_execution: true,
            },
            performance: PerformanceConfig {
                worker_threads: 4,
                frame_buffer_size: 10,
                processing_interval_ms: 100,
                use_gpu: true,
            },
            audio: AudioConfig {
                enabled: true,
                sample_rate: 44100,
                channels: 1,
                enable_tts: true,
                enable_speech_recognition: false, // Optional dependency
            },
        }
    }
}

impl OpticXTConfig {
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        
        if !path.exists() {
            // Create default config file
            let default_config = Self::default();
            let toml_content = toml::to_string_pretty(&default_config)?;
            fs::write(path, toml_content).await?;
            return Ok(default_config);
        }
        
        let content = fs::read_to_string(path).await?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub async fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content).await?;
        Ok(())
    }
}
