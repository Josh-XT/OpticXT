use anyhow::Result;
use clap::Parser;
use tracing::{info, error, warn};
use tracing_subscriber;

mod vision_basic;
mod context;
mod pipeline;
mod commands;
mod models;
mod config;
mod go2_basic;
mod camera;
mod audio;
// mod video_chat;  // Disabled due to winit compatibility issues

use vision_basic as vision;
use go2_basic as go2;

use crate::config::OpticXTConfig;
use crate::pipeline::VisionActionPipeline;
// use crate::video_chat::{VideoChatInterface, VideoChatConfig, ChatMode};
use crate::models::{GemmaModel, ModelConfig};

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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("opticxt={}", log_level))
        .init();
    
    info!("Starting OpticXT - Vision-Driven Autonomous Robot Control System");
    
    // Load configuration
    let config = OpticXTConfig::load(&args.config).await?;
    info!("Configuration loaded successfully");
    
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
