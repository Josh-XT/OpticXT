use anyhow::Result;
use clap::Parser;
use tracing::{info, error};
use tracing_subscriber;

mod vision_basic;
mod context;
mod pipeline;
mod commands;
mod models;
mod config;
mod camera;
mod audio;
mod api;
mod remote_model;
// mod video_chat;  // Disabled due to winit compatibility issues

use vision_basic as vision;

use crate::config::OpticXTConfig;
use crate::pipeline::VisionActionPipeline;
// use crate::video_chat::{VideoChatInterface, VideoChatConfig, ChatMode};

#[derive(Parser)]
#[command(name = "opticxt")]
#[command(about = "Vision-Driven Autonomous Robot Control System")]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,
    
    /// Camera device index
    #[arg(short = 'd', long, default_value = "0")]
    camera_device: usize,
    
    /// Model path for GGUF file
    #[arg(short, long)]
    model_path: Option<String>,
    
    /// Run in video chat mode instead of robot control mode
    #[arg(long)]
    video_chat: bool,
    
    /// Chat mode: video-chat, assistant, or monitoring
    #[arg(long, default_value = "assistant")]
    chat_mode: String,
    
    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Run model benchmark
    #[arg(long)]
    benchmark: bool,
    
    /// Number of benchmark iterations
    #[arg(long, default_value = "50")]
    benchmark_iterations: usize,
    
    /// Start API server mode
    #[arg(long)]
    api_server: bool,
    
    /// API server port
    #[arg(long, default_value = "8080")]
    api_port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(format!("opticxt={}", log_level))
        .try_init(); // Use try_init to avoid panic if already initialized
    
    info!("Starting OpticXT - Vision-Driven Autonomous Robot Control System");
    
    // Load configuration
    let config = OpticXTConfig::load(&args.config).await?;
    info!("Configuration loaded successfully");
    
    // Check if benchmark mode is requested
    if args.benchmark {
        info!("Starting benchmark mode");
        run_benchmark(&args, &config).await?;
        return Ok(());
    }
    
    // Check if API server mode is requested
    if args.api_server {
        info!("Starting API server mode on port {}", args.api_port);
        run_api_server_mode(&args, config).await?;
        return Ok(());
    }
    
    // Check if we should run in video chat mode (for now, just run enhanced pipeline with audio)
    if args.video_chat {
        info!("Starting in video chat/assistant mode (using enhanced pipeline with audio)");
        run_enhanced_mode(&args, config).await?;
    } else {
        info!("Starting in robot control mode");
        run_robot_control_mode(&args, config).await?;
    }
    
    Ok(())
}

async fn run_enhanced_mode(args: &Args, mut config: OpticXTConfig) -> Result<()> {
    info!("Running in enhanced video chat/assistant mode with real camera and audio");
    
    // Enable audio for interactive mode
    config.audio.enabled = true;
    config.audio.enable_tts = true;
    
    // Initialize the vision-action pipeline with audio enabled
    let mut pipeline = VisionActionPipeline::new(
        config,
        args.camera_device,
        args.model_path.clone(),
    ).await?;
    
    info!("Enhanced pipeline initialized with real camera, audio, and model inference");
    
    // Start the main processing loop
    match pipeline.run().await {
        Ok(_) => info!("Enhanced mode completed successfully"),
        Err(e) => {
            error!("Enhanced mode error: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

async fn run_robot_control_mode(args: &Args, config: OpticXTConfig) -> Result<()> {
    // Initialize the vision-action pipeline
    let mut pipeline = VisionActionPipeline::new(
        config,
        args.camera_device,
        args.model_path.clone(),
    ).await?;
    
    info!("Pipeline initialized, starting main processing loop");
    
    // Start the main processing loop
    match pipeline.run().await {
        Ok(_) => info!("Pipeline completed successfully"),
        Err(e) => {
            error!("Pipeline error: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

async fn run_benchmark(args: &Args, _config: &OpticXTConfig) -> Result<()> {
    use crate::models::GemmaModel;
    use crate::config::ModelConfig;
    
    info!("Running benchmark with {} iterations", args.benchmark_iterations);
    
    // Test model performance
    let model_path = args.model_path.clone();
    let model_config = ModelConfig {
        model_path: args.model_path.clone().unwrap_or_default(),
        quantization_method: "isq".to_string(),
        isq_type: "Q4K".to_string(),
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 100,
        context_length: 2048,
        remote: None,
    };
    
    info!("Initializing model for benchmark...");
    let mut model = match GemmaModel::load(model_path, model_config.clone(), model_config.quantization_method.clone(), model_config.isq_type.clone()).await {
        Ok(m) => {
            info!("Model loaded successfully for benchmark");
            m
        },
        Err(e) => {
            error!("Failed to load model for benchmark: {}", e);
            info!("Running benchmark with simulated responses");
            return run_simulated_benchmark(args.benchmark_iterations).await;
        }
    };
    
    info!("Starting model inference benchmark...");
    match model.benchmark(args.benchmark_iterations).await {
        Ok(tokens_per_sec) => {
            info!("✅ Benchmark completed successfully!");
            info!("📊 Performance: {:.2} tokens/second", tokens_per_sec);
            info!("⚡ Total iterations: {}", args.benchmark_iterations);
            
            // Test different prompt types
            let test_prompts = vec![
                "Hello, how are you today?",
                "What can you see in the camera feed?", 
                "Move the robot forward",
                "Describe what you observe",
                "Execute navigation command",
            ];
            
            info!("Testing various prompt types...");
            for (i, prompt) in test_prompts.iter().enumerate() {
                info!("Test {}: {}", i + 1, prompt);
                match model.generate(prompt).await {
                    Ok(response) => info!("Response: {}", response.text),
                    Err(e) => error!("Error generating response: {}", e),
                }
            }
        },
        Err(e) => {
            error!("Benchmark failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

async fn run_simulated_benchmark(iterations: usize) -> Result<()> {
    use std::time::Instant;
    use rand::Rng;
    
    let start = Instant::now();
    let mut rng = rand::thread_rng();
    
    for i in 0..iterations {
        // Simulate processing delay
        tokio::time::sleep(tokio::time::Duration::from_millis(rng.gen_range(10..50))).await;
        
        if i % 10 == 0 {
            info!("Simulation benchmark progress: {}/{}", i + 1, iterations);
        }
    }
    
    let duration = start.elapsed();
    let ops_per_sec = iterations as f64 / duration.as_secs_f64();
    
    info!("✅ Simulated benchmark completed!");
    info!("📊 Performance: {:.2} operations/second", ops_per_sec);
    info!("⚡ Total iterations: {}", iterations);
    
    Ok(())
}

async fn run_api_server_mode(args: &Args, mut config: OpticXTConfig) -> Result<()> {
    info!("Initializing API server mode with vision and model capabilities");
    
    // Enable multimodal features for API usage
    config.vision.enable_multimodal_inference = true;
    config.audio.enabled = true; // Enable for potential TTS responses
    
    // Initialize model for API usage
    let model_path = args.model_path.clone().or_else(|| {
        if config.model.model_path.is_empty() {
            None
        } else {
            Some(config.model.model_path.clone())
        }
    });
    
    // Use the model config directly from the loaded configuration
    
    let model = crate::models::GemmaModel::load(
        model_path,
        config.model.clone(),
        config.model.quantization_method.clone(),
        config.model.isq_type.clone(),
    ).await?;
    
    info!("Model initialized for API server");
    
    // Start the API server
    match crate::api::start_api_server(model, args.api_port).await {
        Ok(_) => info!("API server stopped"),
        Err(e) => {
            error!("API server error: {}", e);
            return Err(anyhow::anyhow!("API server error: {}", e));
        }
    }
    
    Ok(())
}

