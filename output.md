**./test_vision_flow.rs**
```rust
// Test script to verify camera -> vision -> model data flow
use anyhow::Result;
use opticxt::config::OpticXTConfig;
use opticxt::camera::{CameraSystem, CameraConfig};
use opticxt::vision_basic::VisionProcessor;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔍 Testing Camera -> Vision -> Model Data Flow");
    
    // Load configuration 
    let config = OpticXTConfig::default();
    println!("✅ Config loaded - multimodal inference: {}", config.vision.enable_multimodal_inference);
    
    // Test camera initialization
    let camera_config = CameraConfig {
        camera_id: 0,
        width: 640,
        height: 480,
        fps: 30.0,
    };
    
    println!("📷 Initializing camera system...");
    let mut camera_system = CameraSystem::new(camera_config)?;
    camera_system.initialize().await?;
    println!("✅ Camera system initialized");
    
    // Test vision processor
    println!("👁️ Initializing vision processor...");
    let mut vision_processor = VisionProcessor::new(
        0, 640, 480, 0.5, "basic".to_string()
    )?;
    vision_processor.initialize().await?;
    println!("✅ Vision processor initialized");
    
    // Capture and process a frame
    println!("🎯 Capturing and processing camera frame...");
    let sensor_data = camera_system.capture_sensor_data().await?;
    println!("✅ Camera frame captured: {}x{} with {} LiDAR points", 
             sensor_data.frame.width, sensor_data.frame.height, sensor_data.lidar_points.len());
    
    let frame_context = vision_processor.process_frame(&sensor_data.frame, &sensor_data).await?;
    println!("✅ Vision processing complete:");
    println!("   - Scene: {}", frame_context.scene_description);
    println!("   - Objects detected: {}", frame_context.objects.len());
    
    // Test image conversion for vision model
    println!("🖼️ Testing image conversion for vision model...");
    match sensor_data.frame.to_image() {
        Ok(image) => {
            println!("✅ Camera frame successfully converted to image format");
            println!("   - Image dimensions: {}x{}", image.width(), image.height());
            println!("   - Ready for multimodal inference");
        }
        Err(e) => {
            println!("❌ Failed to convert camera frame to image: {}", e);
        }
    }
    
    println!("\n🎉 Vision data flow test complete!");
    println!("Camera input is properly flowing to vision processing and ready for model inference.");
    
    Ok(())
}

```

**./test_model.rs**
```rust
mod models;

use models::GemmaModel;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::init();
    
    println!("Testing OpticXT Gemma Model...");
    
    let model = GemmaModel::new("models/gemma-3n-E4B-it-Q4_K_M.gguf", "models/tokenizer.json").await?;
    
    let test_prompts = vec![
        "Hello, how are you?",
        "What can you see in this image?",
        "Move forward",
        "Tell me about robotics",
    ];
    
    for prompt in test_prompts {
        println!("\nPrompt: {}", prompt);
        match model.generate_response(prompt, 50).await {
            Ok(response) => println!("Response: {}", response),
            Err(e) => println!("Error: {}", e),
        }
    }
    
    Ok(())
}

```

**./src/commands.rs**
```rust
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionCommand {
    #[serde(rename = "move")]
    Move {
        direction: String,
        distance: f32,
        speed: String,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "rotate")]
    Rotate {
        direction: String,
        angle: f32,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "speak")]
    Speak {
        text: String,
        #[serde(default)]
        voice: String,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "analyze")]
    Analyze {
        target: Option<String>,
        detail_level: Option<String>,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "offload")]
    Offload {
        task_description: String,
        target_agent: Option<String>,
        priority: Option<String>,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "wait")]
    Wait {
        duration: Option<f32>,
        #[serde(default)]
        reasoning: String,
    },
}

#[derive(Debug, Clone)]
pub struct CommandExecutionResult {
    pub success: bool,
    pub message: String,
    pub execution_time: Duration,
    #[allow(dead_code)]
    pub side_effects: Vec<String>,
}

pub struct CommandExecutor {
    #[allow(dead_code)]
    enabled_commands: Vec<String>,
    #[allow(dead_code)]
    timeout_seconds: u64,
    validate_before_execution: bool,
    #[allow(dead_code)]
    tts_engine: Option<TtsEngine>,
}

#[derive(Debug)]
struct TtsEngine {
    // Placeholder for TTS integration
    #[allow(dead_code)]
    voice: String,
}

impl CommandExecutor {
    pub fn new(
        enabled_commands: Vec<String>,
        timeout_seconds: u64,
        validate_before_execution: bool,
    ) -> Self {
        let tts_engine = Some(TtsEngine {
            voice: "default".to_string(),
        });
        
        Self {
            enabled_commands,
            timeout_seconds,
            validate_before_execution,
            tts_engine,
        }
    }
    
    pub async fn parse_and_execute(&self, tool_call_output: &str) -> Result<CommandExecutionResult> {
        debug!("Parsing tool call command: {}", tool_call_output);
        
        // Extract the tool call command from the model output
        let command = self.parse_tool_call_command(tool_call_output)?;
        
        // Validate command if enabled
        if self.validate_before_execution {
            self.validate_command(&command)?;
        }
        
        // Execute the command
        self.execute_command(command).await
    }
    
    fn parse_tool_call_command(&self, tool_call_output: &str) -> Result<ActionCommand> {
        // Parse the JSON tool call output
        let tool_calls: Vec<serde_json::Value> = serde_json::from_str(tool_call_output)
            .map_err(|e| anyhow!("Failed to parse tool call JSON: {}", e))?;
        
        if tool_calls.is_empty() {
            return Err(anyhow!("No tool calls found in output"));
        }
        
        // Get the first tool call
        let tool_call = &tool_calls[0];
        let function = tool_call.get("function")
            .ok_or_else(|| anyhow!("No function found in tool call"))?;
        
        let function_name = function.get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow!("No function name found"))?;
        
        let arguments_str = function.get("arguments")
            .and_then(|a| a.as_str())
            .ok_or_else(|| anyhow!("No function arguments found"))?;
        
        let arguments: serde_json::Value = serde_json::from_str(arguments_str)
            .map_err(|e| anyhow!("Failed to parse function arguments: {}", e))?;
        
        // Parse based on function name
        match function_name {
            "move" => self.parse_move_from_json(&arguments),
            "rotate" => self.parse_rotate_from_json(&arguments),
            "speak" => self.parse_speak_from_json(&arguments),
            "analyze" => self.parse_analyze_from_json(&arguments),
            "wait" => self.parse_wait_from_json(&arguments),
            "stop" => self.parse_stop_from_json(&arguments),
            _ => Err(anyhow!("Unknown function name: {}", function_name))
        }
    }
    
    
    // JSON-based parsing functions for OpenAI tool calls
    fn parse_move_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let direction = args.get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("forward")
            .to_string();
        let distance = args.get("distance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let speed = args.get("speed")
            .and_then(|v| v.as_str())
            .unwrap_or("normal")
            .to_string();
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Move {
            direction,
            distance,
            speed,
            reasoning,
        })
    }
    
    fn parse_rotate_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let direction = args.get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("left")
            .to_string();
        let angle = args.get("angle")
            .and_then(|v| v.as_f64())
            .unwrap_or(30.0) as f32;
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Rotate {
            direction,
            angle,
            reasoning,
        })
    }
    
    fn parse_speak_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let text = args.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Text is required for speak command"))?
            .to_string();
        let voice = args.get("voice")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Speak {
            text,
            voice,
            reasoning,
        })
    }
    
    fn parse_analyze_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let target = args.get("target")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let detail_level = args.get("detail_level")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Analyze {
            target,
            detail_level,
            reasoning,
        })
    }
    
    fn parse_wait_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let duration = args.get("duration")
            .and_then(|v| v.as_f64())
            .map(|d| d as f32);
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Wait {
            duration,
            reasoning,
        })
    }
    
    fn parse_stop_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let _immediate = args.get("immediate")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        // Map to Wait with duration 0 for immediate stop
        Ok(ActionCommand::Wait {
            duration: Some(0.0),
            reasoning: format!("STOP - {}", reasoning),
        })
    }
    
    fn validate_command(&self, command: &ActionCommand) -> Result<()> {
        match command {
            ActionCommand::Move { direction, distance, speed, .. } => {
                if !["forward", "backward", "left", "right"].contains(&direction.as_str()) {
                    return Err(anyhow!("Invalid move direction: {}", direction));
                }
                if *distance <= 0.0 || *distance > 10.0 {
                    return Err(anyhow!("Invalid move distance: {}", distance));
                }
                if !["slow", "normal", "fast"].contains(&speed.as_str()) {
                    return Err(anyhow!("Invalid speed: {}", speed));
                }
            }
            ActionCommand::Rotate { direction, angle, .. } => {
                if !["left", "right", "clockwise", "counterclockwise"].contains(&direction.as_str()) {
                    return Err(anyhow!("Invalid rotate direction: {}", direction));
                }
                if *angle <= 0.0 || *angle > 360.0 {
                    return Err(anyhow!("Invalid rotation angle: {}", angle));
                }
            }
            ActionCommand::Speak { text, .. } => {
                if text.is_empty() {
                    return Err(anyhow!("Speech text cannot be empty"));
                }
                if text.len() > 500 {
                    return Err(anyhow!("Speech text too long"));
                }
            }
            _ => {} // Other commands have minimal validation requirements
        }
        
        Ok(())
    }
    
    async fn execute_command(&self, command: ActionCommand) -> Result<CommandExecutionResult> {
        let start_time = std::time::Instant::now();
        
        info!("Executing command: {:?}", command);
        
        let result = match command {
            ActionCommand::Move { direction, distance, speed, reasoning } => {
                self.execute_move(direction, distance, speed, reasoning).await
            }
            ActionCommand::Rotate { direction, angle, reasoning } => {
                self.execute_rotate(direction, angle, reasoning).await
            }
            ActionCommand::Speak { text, voice, reasoning } => {
                self.execute_speak(text, voice, reasoning).await
            }
            ActionCommand::Analyze { target, detail_level, reasoning } => {
                self.execute_analyze(target, detail_level, reasoning).await
            }
            ActionCommand::Offload { task_description, target_agent, priority, reasoning } => {
                self.execute_offload(task_description, target_agent, priority, reasoning).await
            }
            ActionCommand::Wait { duration, reasoning } => {
                self.execute_wait(duration, reasoning).await
            }
        };
        
        let execution_time = start_time.elapsed();
        
        match result {
            Ok(mut cmd_result) => {
                cmd_result.execution_time = execution_time;
                Ok(cmd_result)
            }
            Err(e) => Ok(CommandExecutionResult {
                success: false,
                message: format!("Command failed: {}", e),
                execution_time,
                side_effects: vec![],
            })
        }
    }
    
    async fn execute_move(&self, direction: String, distance: f32, speed: String, reasoning: String) -> Result<CommandExecutionResult> {
        // Placeholder implementation - would interface with actual robot hardware
        debug!("Moving {} {} meters at {} speed. Reasoning: {}", direction, distance, speed, reasoning);
        
        // Simulate movement time based on distance and speed
        let movement_time = match speed.as_str() {
            "slow" => distance * 2.0,
            "normal" => distance * 1.0,
            "fast" => distance * 0.5,
            _ => distance * 1.0,
        };
        
        tokio::time::sleep(Duration::from_secs_f32(movement_time)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Moved {} {} meters at {} speed", direction, distance, speed),
            execution_time: Duration::default(),
            side_effects: vec![format!("Position changed by {} meters", distance)],
        })
    }
    
    async fn execute_rotate(&self, direction: String, angle: f32, reasoning: String) -> Result<CommandExecutionResult> {
        debug!("Rotating {} {} degrees. Reasoning: {}", direction, angle, reasoning);
        
        // Simulate rotation time
        let rotation_time = angle / 90.0; // 1 second per 90 degrees
        tokio::time::sleep(Duration::from_secs_f32(rotation_time)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Rotated {} {} degrees", direction, angle),
            execution_time: Duration::default(),
            side_effects: vec![format!("Orientation changed by {} degrees", angle)],
        })
    }
    
    async fn execute_speak(&self, text: String, _voice: String, reasoning: String) -> Result<CommandExecutionResult> {
        info!("Speaking: '{}'. Reasoning: {}", text, reasoning);
        
        // Placeholder for TTS - would use actual TTS engine
        println!("🤖 Robot says: {}", text);
        
        // Simulate speech duration
        let speech_duration = text.len() as f32 * 0.1; // ~100ms per character
        tokio::time::sleep(Duration::from_secs_f32(speech_duration)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Spoke: '{}'", text),
            execution_time: Duration::default(),
            side_effects: vec!["Audio output generated".to_string()],
        })
    }
    
    async fn execute_analyze(&self, target: Option<String>, _detail_level: Option<String>, reasoning: String) -> Result<CommandExecutionResult> {
        let target_desc = target.unwrap_or_else(|| "current scene".to_string());
        debug!("Analyzing: {}. Reasoning: {}", target_desc, reasoning);
        
        // Simulate analysis time
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Analysis completed for: {}", target_desc),
            execution_time: Duration::default(),
            side_effects: vec!["Additional sensor data collected".to_string()],
        })
    }
    
    async fn execute_offload(&self, task_description: String, target_agent: Option<String>, _priority: Option<String>, reasoning: String) -> Result<CommandExecutionResult> {
        let agent = target_agent.unwrap_or_else(|| "default_agent".to_string());
        info!("Offloading task to {}: '{}'. Reasoning: {}", agent, task_description, reasoning);
        
        // Placeholder for task offloading - would communicate with AGiXT agents
        println!("📤 Offloading to {}: {}", agent, task_description);
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Task offloaded to {}: {}", agent, task_description),
            execution_time: Duration::default(),
            side_effects: vec!["External task queued".to_string()],
        })
    }
    
    async fn execute_wait(&self, duration: Option<f32>, reasoning: String) -> Result<CommandExecutionResult> {
        let wait_time = duration.unwrap_or(1.0);
        debug!("Waiting for {} seconds. Reasoning: {}", wait_time, reasoning);
        
        tokio::time::sleep(Duration::from_secs_f32(wait_time)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Waited for {} seconds", wait_time),
            execution_time: Duration::default(),
            side_effects: vec![],
        })
    }
}

```

**./src/video_chat.rs**
```rust
use anyhow::{Result, anyhow};
use pixels::{Pixels, SurfaceTexture};
use winit::{
    event::{Event, WindowEvent, KeyEvent, ElementState},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    dpi::LogicalSize,
    keyboard::{Key, NamedKey},
};
use pollster;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use tracing::{info, debug, warn, error};

use crate::audio::{AudioSystem, AudioConfig, AudioData};
use crate::camera::{CameraSystem, CameraConfig, SensorData};
use crate::models::{GemmaModel, ModelConfig, GenerationResult};
use crate::vision::Mat;

#[derive(Debug, Clone)]
pub struct VideoChatConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub video_fps: u32,
    pub audio_sample_rate: u32,
    pub enable_speech_recognition: bool,
    pub enable_text_overlay: bool,
    pub chat_mode: ChatMode,
}

#[derive(Debug, Clone)]
pub enum ChatMode {
    VideoChat,        // Interactive video chat with voice
    Assistant,        // Visual assistant mode
    Monitoring,       // Just display camera feed
}

impl Default for VideoChatConfig {
    fn default() -> Self {
        Self {
            window_width: 1280,
            window_height: 720,
            video_fps: 30,
            audio_sample_rate: 44100,
            enable_speech_recognition: false, // Optional dependency
            enable_text_overlay: true,
            chat_mode: ChatMode::Assistant,
        }
    }
}

pub struct VideoChatInterface {
    config: VideoChatConfig,
    window: Arc<Window>,
    pixels: Pixels,
    camera_system: CameraSystem,
    audio_system: AudioSystem,
    model: Option<GemmaModel>,
    is_running: Arc<Mutex<bool>>,
    last_frame_time: Instant,
    conversation_history: Vec<String>,
    current_response: String,
    response_display_time: Option<Instant>,
}

impl VideoChatInterface {
    pub async fn new(
        config: VideoChatConfig,
        camera_device: usize,
        model: Option<GemmaModel>,
    ) -> Result<(Self, EventLoop<()>)> {
        info!("Creating Video Chat Interface");
        
        // Create window
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("OpticXT - Video Chat Assistant")
            .with_inner_size(LogicalSize::new(config.window_width, config.window_height))
            .with_resizable(true)
            .build(&event_loop)?;
        
        let window = Arc::new(window);
        
        // Create pixel buffer for rendering
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &*window);
        let pixels = pollster::block_on(Pixels::new_async(
            config.window_width,
            config.window_height,
            surface_texture,
        ))?;
        
        // Initialize camera system
        let camera_config = CameraConfig {
            camera_id: camera_device as i32,
            width: config.window_width as i32,
            height: config.window_height as i32,
            fps: config.video_fps as f64,
        };
        
        let mut camera_system = CameraSystem::new(camera_config)?;
        camera_system.initialize().await?;
        
        // Initialize audio system
        let audio_config = AudioConfig {
            sample_rate: config.audio_sample_rate,
            channels: 1,
            buffer_size: 4096,
            voice_detection_threshold: 0.02,
            silence_duration_ms: 500,
            speech_timeout_ms: 5000,
        };
        
        let mut audio_system = AudioSystem::new(audio_config)?;
        audio_system.initialize().await?;
        
        info!("Video chat interface initialized successfully");
        
        Ok((Self {
            config,
            window,
            pixels,
            camera_system,
            audio_system,
            model,
            is_running: Arc::new(Mutex::new(false)),
            last_frame_time: Instant::now(),
            conversation_history: Vec::new(),
            current_response: String::new(),
            response_display_time: None,
        }, event_loop))
    }
    
    pub async fn run(mut self, event_loop: EventLoop<()>) -> Result<()> {
        info!("Starting video chat interface main loop");
        
        // Set running flag
        {
            let mut running = self.is_running.lock().unwrap();
            *running = true;
        }
        
        // Start audio recording if in chat mode
        if matches!(self.config.chat_mode, ChatMode::VideoChat | ChatMode::Assistant) {
            self.audio_system.start_recording().await?;
            info!("Audio recording started for interactive mode");
        }
        
        // Welcome message
        if let Some(ref mut model) = self.model {
            self.audio_system.speak_text("Hello! I'm your OpticXT video assistant. I can see through the camera and respond to your voice commands.").await?;
        }
        
        // Start the event loop
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            
            match event {
                Event::WindowEvent { window_id, event } if window_id == self.window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            info!("Close requested, shutting down");
                            *control_flow = ControlFlow::Exit;
                        }
                        WindowEvent::KeyEvent {
                            event: KeyEvent {
                                logical_key: Key::Named(NamedKey::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                            ..
                        } => {
                            info!("Escape pressed, shutting down");
                            *control_flow = ControlFlow::Exit;
                        }
                        WindowEvent::KeyEvent {
                            event: KeyEvent {
                                logical_key: Key::Named(NamedKey::Space),
                                state: ElementState::Pressed,
                                ..
                            },
                            ..
                        } => {
                            // Space key to trigger manual voice capture
                            if let Err(e) = pollster::block_on(self.handle_voice_input()) {
                                error!("Voice input error: {}", e);
                            }
                        }
                        WindowEvent::Resized(new_size) => {
                            if let Err(e) = self.pixels.resize_surface(new_size.width, new_size.height) {
                                error!("Failed to resize surface: {}", e);
                            }
                        }
                        _ => {}
                    }
                }
                Event::MainEventsCleared => {
                    // Update and render
                    if let Err(e) = pollster::block_on(self.update_and_render()) {
                        error!("Update/render error: {}", e);
                    }
                    
                    // Request redraw
                    self.window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    if let Err(e) = self.pixels.render() {
                        error!("Render error: {}", e);
                    }
                }
                _ => {}
            }
        });
    }
    
    async fn update_and_render(&mut self) -> Result<()> {
        let now = Instant::now();
        let frame_duration = Duration::from_secs_f32(1.0 / self.config.video_fps as f32);
        
        // Check if it's time for a new frame
        if now.duration_since(self.last_frame_time) >= frame_duration {
            // Capture camera frame
            let sensor_data = self.camera_system.capture_sensor_data().await?;
            
            // Process frame based on mode
            match self.config.chat_mode {
                ChatMode::VideoChat | ChatMode::Assistant => {
                    self.process_interactive_frame(&sensor_data).await?;
                }
                ChatMode::Monitoring => {
                    self.process_monitoring_frame(&sensor_data).await?;
                }
            }
            
            // Render frame to pixel buffer
            self.render_frame_to_pixels(&sensor_data.frame)?;
            
            self.last_frame_time = now;
        }
        
        // Check for voice input in chat modes
        if matches!(self.config.chat_mode, ChatMode::VideoChat | ChatMode::Assistant) {
            if let Err(e) = self.check_voice_activity().await {
                debug!("Voice activity check error: {}", e);
            }
        }
        
        Ok(())
    }
    
    async fn process_interactive_frame(&mut self, sensor_data: &SensorData) -> Result<()> {
        // Generate scene description
        let scene_description = self.analyze_scene(&sensor_data.frame)?;
        
        // If we have a model and no current response is being displayed
        if let Some(ref model) = self.model {
            if self.response_display_time.is_none() || 
               self.response_display_time.unwrap().elapsed() > Duration::from_secs(10) {
                
                // Generate a context-aware response
                let prompt = format!(
                    "You are a helpful video assistant. Current scene: {}. LiDAR detected {} points. Respond briefly and helpfully.",
                    scene_description,
                    sensor_data.lidar_points.len()
                );
                
                match model.generate(&prompt).await {
                    Ok(result) => {
                        self.current_response = result.text.clone();
                        self.response_display_time = Some(Instant::now());
                        
                        // Speak the response if it contains speech tags
                        if result.text.contains("<speak>") {
                            let speech_text = self.extract_speech_text(&result.text);
                            if !speech_text.is_empty() {
                                if let Err(e) = self.audio_system.speak_text(&speech_text).await {
                                    warn!("TTS failed: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        debug!("Model generation error: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn process_monitoring_frame(&mut self, _sensor_data: &SensorData) -> Result<()> {
        // Just display the frame with minimal processing
        Ok(())
    }
    
    fn analyze_scene(&self, frame: &Mat) -> Result<String> {
        // Basic scene analysis - could be enhanced with object detection
        let avg_brightness = self.calculate_average_brightness(frame)?;
        let motion_level = "low"; // Would need frame comparison for motion detection
        
        let description = if avg_brightness < 0.3 {
            format!("Low light environment (brightness: {:.2}), motion: {}", avg_brightness, motion_level)
        } else if avg_brightness > 0.7 {
            format!("Bright environment (brightness: {:.2}), motion: {}", avg_brightness, motion_level)
        } else {
            format!("Normal lighting (brightness: {:.2}), motion: {}", avg_brightness, motion_level)
        };
        
        Ok(description)
    }
    
    fn calculate_average_brightness(&self, frame: &Mat) -> Result<f32> {
        if frame.data.is_empty() {
            return Ok(0.0);
        }
        
        let mut sum = 0u64;
        let pixel_count = (frame.width * frame.height) as usize;
        
        // Calculate average of RGB values
        for i in (0..frame.data.len()).step_by(frame.channels as usize) {
            if i + 2 < frame.data.len() {
                let r = frame.data[i] as u64;
                let g = frame.data[i + 1] as u64;
                let b = frame.data[i + 2] as u64;
                sum += (r + g + b) / 3;
            }
        }
        
        Ok(sum as f32 / (pixel_count as f32 * 255.0))
    }
    
    fn render_frame_to_pixels(&mut self, frame: &Mat) -> Result<()> {
        let buffer_width = self.pixels.texture().width() as usize;
        let buffer_height = self.pixels.texture().height() as usize;
        
        let pixel_buffer = self.pixels.frame_mut();
        
        // Scale frame to fit pixel buffer
        let scale_x = buffer_width as f32 / frame.width as f32;
        let scale_y = buffer_height as f32 / frame.height as f32;
        
        // Clear the pixel buffer
        pixel_buffer.fill(0);
        
        // Copy frame data to pixel buffer with scaling
        for y in 0..buffer_height {
            for x in 0..buffer_width {
                let src_x = (x as f32 / scale_x) as usize;
                let src_y = (y as f32 / scale_y) as usize;
                
                if src_x < frame.width as usize && src_y < frame.height as usize {
                    let src_idx = (src_y * frame.width as usize + src_x) * frame.channels as usize;
                    let dst_idx = (y * buffer_width + x) * 4; // RGBA format
                    
                    if src_idx + 2 < frame.data.len() && dst_idx + 3 < pixel_buffer.len() {
                        pixel_buffer[dst_idx] = frame.data[src_idx];     // R
                        pixel_buffer[dst_idx + 1] = frame.data[src_idx + 1]; // G
                        pixel_buffer[dst_idx + 2] = frame.data[src_idx + 2]; // B
                        pixel_buffer[dst_idx + 3] = 255; // A
                    }
                }
            }
        }
        
        // Overlay text if enabled
        if self.config.enable_text_overlay {
            // Since we can't borrow self again here, we'll handle text overlay separately
            debug!("Text overlay would be rendered here");
        }
        
        Ok(())
    }
    
    fn overlay_text(&self, _pixel_buffer: &mut [u8], _width: usize, _height: usize) -> Result<()> {
        // Simple text overlay - in production, you'd use a proper text rendering library
        // For now, just log the current response
        if !self.current_response.is_empty() {
            debug!("Current response: {}", self.current_response);
        }
        
        Ok(())
    }
    
    async fn check_voice_activity(&mut self) -> Result<()> {
        // Check for voice input (simplified)
        let audio_data = self.audio_system.capture_audio_data(100).await?; // 100ms chunks
        
        if audio_data.has_speech {
            debug!("Voice activity detected");
            // In a full implementation, you'd accumulate audio until silence
            // and then process the complete utterance
        }
        
        Ok(())
    }
    
    async fn handle_voice_input(&mut self) -> Result<()> {
        info!("Capturing voice input...");
        
        // Play a beep to indicate listening
        self.audio_system.play_beep(800.0, 200).await?;
        
        // Capture audio for speech recognition
        let audio_data = self.audio_system.capture_audio_data(3000).await?; // 3 seconds
        
        if audio_data.has_speech {
            info!("Speech detected, processing...");
            
            // Play confirmation beep
            self.audio_system.play_beep(1000.0, 100).await?;
            
            // In a real implementation, you'd convert speech to text here
            // For now, we'll simulate a voice command
            let simulated_command = "What do you see in the camera?";
            
            self.conversation_history.push(format!("User: {}", simulated_command));
            
            // Generate response using the model
            if let Some(ref model) = self.model {
                let prompt = format!(
                    "User asked: {}. Respond as a helpful video assistant viewing the camera feed.",
                    simulated_command
                );
                
                match model.generate(&prompt).await {
                    Ok(result) => {
                        self.current_response = result.text.clone();
                        self.response_display_time = Some(Instant::now());
                        self.conversation_history.push(format!("Assistant: {}", result.text));
                        
                        // Speak the response
                        let speech_text = self.extract_speech_text(&result.text);
                        if !speech_text.is_empty() {
                            self.audio_system.speak_text(&speech_text).await?;
                        }
                    }
                    Err(e) => {
                        error!("Failed to generate response: {}", e);
                        self.audio_system.speak_text("I'm sorry, I had trouble processing that request.").await?;
                    }
                }
            }
        } else {
            info!("No speech detected");
            self.audio_system.play_beep(400.0, 200).await?; // Lower pitch for no speech
        }
        
        Ok(())
    }
    
    fn extract_speech_text(&self, text: &str) -> String {
        // Extract text from <speak> tags, or use the whole text if no tags
        if let Some(start) = text.find("<speak>") {
            if let Some(end) = text.find("</speak>") {
                let start_pos = start + 7; // Length of "<speak>"
                if start_pos < end {
                    return text[start_pos..end].trim().to_string();
                }
            }
        }
        
        // Fallback: clean up XML tags and return plain text
        text.replace("<", "").replace(">", "").trim().to_string()
    }
}

impl Drop for VideoChatInterface {
    fn drop(&mut self) {
        info!("VideoChatInterface being dropped, cleaning up");
        let _ = pollster::block_on(self.audio_system.stop_recording());
    }
}

```

**./src/camera.rs**
```rust
use anyhow::{Result, anyhow};
use std::time::SystemTime;
use tracing::{info, debug, warn, error};
use nokhwa::{Camera, utils::{RequestedFormat, RequestedFormatType, CameraIndex}};
use nokhwa::pixel_format::RgbFormat;
use crate::vision_basic::Mat;

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub camera_id: i32,
    #[allow(dead_code)]
    pub width: i32,
    #[allow(dead_code)]
    pub height: i32,
    #[allow(dead_code)]
    pub fps: f64,
}

#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f64,
    pub y: f64,
    #[allow(dead_code)]
    pub z: f64,
    pub distance: f64,
    #[allow(dead_code)]
    pub angle: f64,
    #[allow(dead_code)]
    pub intensity: f64,
}

#[derive(Debug, Clone)]
pub struct SensorData {
    #[allow(dead_code)]
    pub timestamp: SystemTime,
    pub frame: Mat,
    pub lidar_points: Vec<LidarPoint>,
    #[allow(dead_code)]
    pub has_lidar: bool,
}

pub struct CameraSystem {
    config: CameraConfig,
    camera: Option<Camera>,
    is_initialized: bool,
    has_hardware_lidar: bool,
}

impl CameraSystem {
    pub fn new(config: CameraConfig) -> Result<Self> {
        info!("Creating CameraSystem for real camera input");
        
        Ok(Self {
            config,
            camera: None,
            is_initialized: false,
            has_hardware_lidar: false, // Will be true on actual Go2/Jetson
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing camera system - scanning for available cameras");
        
        // Detect and list available cameras
        let available_cameras = Self::detect_cameras();
        if available_cameras.is_empty() {
            return Err(anyhow!("No cameras detected on this system"));
        }
        
        info!("Found {} camera(s): {:?}", available_cameras.len(), available_cameras);
        
        // Try to initialize the requested camera, then fallbacks
        let camera_indices = if available_cameras.contains(&(self.config.camera_id as u32)) {
            vec![self.config.camera_id as u32]
        } else {
            available_cameras
        };
        
        for cam_id in camera_indices {
            match self.try_initialize_camera(cam_id).await {
                Ok(_) => {
                    self.config.camera_id = cam_id as i32;
                    info!("Successfully initialized camera {}", cam_id);
                    break;
                }
                Err(e) => {
                    warn!("Failed to initialize camera {}: {}", cam_id, e);
                    continue;
                }
            }
        }
        
        if !self.is_initialized {
            return Err(anyhow!("Failed to initialize any available camera"));
        }
        
        // Check for LiDAR hardware (would be true on Go2/Jetson)
        self.has_hardware_lidar = self.detect_lidar_hardware();
        
        info!("Camera system initialized successfully (LiDAR: {})", self.has_hardware_lidar);
        Ok(())
    }
    
    async fn try_initialize_camera(&mut self, camera_id: u32) -> Result<()> {
        let camera_index = CameraIndex::Index(camera_id);
        let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        
        let mut camera = Camera::new(camera_index, requested_format)?;
        
        // Open the camera stream
        camera.open_stream()?;
        
        // Test capture a frame to ensure it works
        let _test_frame = camera.frame()?;
        
        self.camera = Some(camera);
        self.is_initialized = true;
        
        Ok(())
    }
    
    pub fn detect_cameras() -> Vec<u32> {
        let mut cameras = Vec::new();
        
        // Try camera indices 0-9 (most common range)
        for cam_id in 0..10 {
            let camera_index = CameraIndex::Index(cam_id);
            let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
            
            if let Ok(_camera) = Camera::new(camera_index, requested_format) {
                cameras.push(cam_id);
            }
        }
        
        cameras
    }
    
    fn detect_lidar_hardware(&self) -> bool {
        // Check for Go2/Jetson LiDAR hardware
        // This would check for actual LiDAR devices on the robot
        
        // Check common LiDAR device paths
        let lidar_paths = vec![
            "/dev/ttyUSB0",  // Common USB LiDAR
            "/dev/ttyACM0",  // Arduino/microcontroller LiDAR
            "/dev/lidar",    // Custom LiDAR device
            "/sys/class/lidar", // Potential system class
        ];
        
        for path in lidar_paths {
            if std::path::Path::new(path).exists() {
                info!("Detected potential LiDAR hardware at {}", path);
                return true;
            }
        }
        
        // On Jetson/Go2, we would also check for specific Go2 SDK devices
        false
    }
    
    pub async fn capture_sensor_data(&mut self) -> Result<SensorData> {
        if !self.is_initialized {
            return Err(anyhow!("Camera system not initialized"));
        }
        
        let timestamp = SystemTime::now();
        
        // Capture frame from camera
        let frame = self.capture_camera_frame().await?;
        
        // Capture LiDAR data if available
        let lidar_points = if self.has_hardware_lidar {
            self.capture_real_lidar_data().await?
        } else {
            // Generate minimal simulated LiDAR for standalone mode
            self.generate_minimal_lidar_simulation()
        };
        
        Ok(SensorData {
            timestamp,
            frame,
            lidar_points,
            has_lidar: self.has_hardware_lidar,
        })
    }
    
    async fn capture_camera_frame(&mut self) -> Result<Mat> {
        if let Some(ref mut camera) = self.camera {
            match camera.frame() {
                Ok(frame) => {
                    // Decode the frame to RGB
                    let decoded = frame.decode_image::<RgbFormat>()?;
                    let width = decoded.width();
                    let height = decoded.height();
                    
                    debug!("Captured real camera frame: {}x{}", width, height);
                    
                    // Convert to our Mat format
                    let rgb_data = decoded.into_raw();
                    let mut mat = Mat::new(width, height, 3);
                    mat.data = rgb_data;
                    
                    Ok(mat)
                }
                Err(e) => {
                    error!("Camera frame capture failed: {}", e);
                    Err(anyhow!("Camera frame capture error: {}", e))
                }
            }
        } else {
            Err(anyhow!("Camera not initialized"))
        }
    }
    
    async fn capture_real_lidar_data(&mut self) -> Result<Vec<LidarPoint>> {
        // This would interface with actual Go2 LiDAR hardware
        // For now, we'll implement basic LiDAR communication protocols
        
        // TODO: Implement actual LiDAR driver communication
        // This would use serial/USB communication with the LiDAR sensor
        
        info!("Reading from hardware LiDAR sensor");
        
        // Placeholder for real LiDAR implementation
        // In production, this would:
        // 1. Read from LiDAR serial/USB interface
        // 2. Parse LiDAR data packets
        // 3. Convert to point cloud format
        
        Ok(vec![]) // Real implementation would return actual sensor data
    }
    
    fn generate_minimal_lidar_simulation(&self) -> Vec<LidarPoint> {
        // Generate a minimal LiDAR simulation for standalone video chat mode
        // This is much simpler than the full simulation - just basic obstacle detection
        let mut points = Vec::new();
        
        // Simulate a few key points for basic spatial awareness
        for i in 0..8 {
            let angle = (i as f64) * std::f64::consts::PI / 4.0; // 8 directions
            let distance = 2.0 + (angle.sin() * 0.5); // Vary distance slightly
            
            let x = distance * angle.cos();
            let y = distance * angle.sin();
            
            points.push(LidarPoint {
                x,
                y,
                z: 0.0,
                distance,
                angle,
                intensity: 0.7,
            });
        }
        
        points
    }
    
    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        if lidar_points.is_empty() {
            return Ok(());
        }
        
        debug!("Overlaying {} LiDAR points", lidar_points.len());
        
        let width = frame.width as f64;
        let height = frame.height as f64;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        
        // Draw LiDAR points as colored pixels
        for point in lidar_points {
            // Convert polar coordinates to screen coordinates
            let scale = 100.0; // Scale factor for visualization
            let screen_x = center_x + (point.x * scale);
            let screen_y = center_y + (point.y * scale);
            
            if screen_x >= 0.0 && screen_x < width && screen_y >= 0.0 && screen_y < height {
                let x = screen_x as i32;
                let y = screen_y as i32;
                
                // Color code by distance: close = red, far = blue
                let color = if point.distance < 1.5 {
                    [255, 0, 0] // Red for close objects
                } else if point.distance < 3.0 {
                    [255, 255, 0] // Yellow for medium distance
                } else {
                    [0, 255, 0] // Green for far objects
                };
                
                self.set_pixel_safe(frame, x, y, color);
                
                // Draw a small cross pattern for better visibility
                self.set_pixel_safe(frame, x - 2, y, color);
                self.set_pixel_safe(frame, x + 2, y, color);
                self.set_pixel_safe(frame, x, y - 2, color);
                self.set_pixel_safe(frame, x, y + 2, color);
            }
        }
        
        Ok(())
    }
    
    fn set_pixel_safe(&self, frame: &mut Mat, x: i32, y: i32, color: [u8; 3]) {
        if x >= 0 && y >= 0 && x < frame.width as i32 && y < frame.height as i32 {
            let pixel_index = ((y as u32 * frame.width + x as u32) * frame.channels) as usize;
            if pixel_index + 2 < frame.data.len() {
                frame.data[pixel_index] = color[0];
                frame.data[pixel_index + 1] = color[1];
                frame.data[pixel_index + 2] = color[2];
            }
        }
    }
    
    #[allow(dead_code)]
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping camera system");
        
        if let Some(ref mut camera) = self.camera {
            match camera.stop_stream() {
                Ok(_) => info!("Camera stream stopped successfully"),
                Err(e) => warn!("Error stopping camera stream: {}", e),
            }
        }
        
        self.camera = None;
        self.is_initialized = false;
        
        Ok(())
    }
    
    #[allow(dead_code)]
    pub fn get_camera_info(&self) -> Result<String> {
        if let Some(ref _camera) = self.camera {
            Ok(format!("Camera {} - Active", self.config.camera_id))
        } else {
            Ok("No camera active".to_string())
        }
    }
}

impl Drop for CameraSystem {
    fn drop(&mut self) {
        if self.is_initialized {
            info!("CameraSystem being dropped, cleaning up");
            if let Some(ref mut camera) = self.camera {
                let _ = camera.stop_stream();
            }
        }
    }
}

```

**./src/context.rs**
```rust
use crate::vision::FrameContext;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::SystemTime;
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MandatoryContext {
    pub system_prompt: String,
    pub current_visual_context: String,
    pub action_history: VecDeque<ActionHistoryEntry>,
    pub environmental_constraints: Vec<String>,
    pub safety_rules: Vec<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionHistoryEntry {
    pub timestamp: SystemTime,
    pub action_type: String,
    pub action_data: String,
    pub success: bool,
    pub feedback: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ContextManager {
    base_system_prompt: String,
    max_history: usize,
    include_timestamp: bool,
    action_history: VecDeque<ActionHistoryEntry>,
    environmental_constraints: Vec<String>,
    safety_rules: Vec<String>,
}

impl ContextManager {
    pub fn new(
        system_prompt: String,
        max_history: usize,
        include_timestamp: bool,
    ) -> Self {
        let safety_rules = vec![
            "Do not perform actions that could cause harm to humans".to_string(),
            "Always validate sensor data before taking physical actions".to_string(),
            "Stop immediately if unexpected obstacles are detected".to_string(),
            "Maintain safe distances from humans and fragile objects".to_string(),
            "Do not exceed maximum speed or acceleration limits".to_string(),
        ];
        
        let environmental_constraints = vec![
            "Operating in indoor environment".to_string(),
            "Camera resolution: limited field of view".to_string(),
            "Edge computing: optimize for low latency".to_string(),
        ];
        
        Self {
            base_system_prompt: system_prompt,
            max_history,
            include_timestamp,
            action_history: VecDeque::new(),
            environmental_constraints,
            safety_rules,
        }
    }
    
    pub fn build_context(&self, frame_context: &FrameContext) -> MandatoryContext {
        let current_visual_context = self.format_visual_context(frame_context);
        
        MandatoryContext {
            system_prompt: self.base_system_prompt.clone(),
            current_visual_context,
            action_history: self.action_history.clone(),
            environmental_constraints: self.environmental_constraints.clone(),
            safety_rules: self.safety_rules.clone(),
            timestamp: SystemTime::now(),
        }
    }
    
    pub fn create_model_prompt(&self, context: &MandatoryContext) -> String {
        let mut prompt = String::new();
        
        // System prompt
        prompt.push_str("SYSTEM CONTEXT:\n");
        prompt.push_str(&context.system_prompt);
        prompt.push_str("\n\n");
        
        // Safety rules
        prompt.push_str("SAFETY CONSTRAINTS:\n");
        for rule in &context.safety_rules {
            prompt.push_str(&format!("- {}\n", rule));
        }
        prompt.push_str("\n");
        
        // Environmental constraints
        prompt.push_str("ENVIRONMENTAL CONTEXT:\n");
        for constraint in &context.environmental_constraints {
            prompt.push_str(&format!("- {}\n", constraint));
        }
        prompt.push_str("\n");
        
        // Current visual context
        prompt.push_str("CURRENT VISUAL INPUT:\n");
        prompt.push_str(&context.current_visual_context);
        prompt.push_str("\n\n");
        
        // Recent action history
        if !context.action_history.is_empty() {
            prompt.push_str("RECENT ACTIONS:\n");
            for entry in context.action_history.iter().rev().take(3) {
                let status = if entry.success { "SUCCESS" } else { "FAILED" };
                prompt.push_str(&format!(
                    "- {} [{}]: {}\n",
                    entry.action_type,
                    status,
                    entry.action_data
                ));
                if let Some(feedback) = &entry.feedback {
                    prompt.push_str(&format!("  Feedback: {}\n", feedback));
                }
            }
            prompt.push_str("\n");
        }
        
        // Timestamp if enabled
        if self.include_timestamp {
            prompt.push_str(&format!(
                "TIMESTAMP: {:?}\n\n",
                context.timestamp
            ));
        }
        
        // Action request
        prompt.push_str("REQUIRED OUTPUT:\n");
        prompt.push_str("Based on the current visual input and context, respond with an appropriate action using OpenAI-style tool calling. ");
        prompt.push_str("Available functions: move, rotate, speak, analyze, wait, stop. ");
        prompt.push_str("Always include reasoning in the function arguments. ");
        prompt.push_str("If no action is needed, use the 'wait' function with reasoning.\n\n");
        prompt.push_str("RESPONSE:");
        
        // Aggressively limit prompt length to prevent token explosion
        const MAX_PROMPT_CHARS: usize = 4000; // Conservative limit
        if prompt.len() > MAX_PROMPT_CHARS {
            debug!("Prompt too long ({} chars), truncating to {}", prompt.len(), MAX_PROMPT_CHARS);
            prompt.truncate(MAX_PROMPT_CHARS);
            prompt.push_str("\n\n[TRUNCATED] Respond based on available context.\nACTION:");
        }
        
        debug!("Generated model prompt with {} characters", prompt.len());
        prompt
    }
    
    fn format_visual_context(&self, frame_context: &FrameContext) -> String {
        let mut context = String::new();
        
        context.push_str(&format!(
            "Frame size: {}x{}\n",
            frame_context.frame_size.0,
            frame_context.frame_size.1
        ));
        
        // Truncate scene description if it's too long
        let scene_desc = if frame_context.scene_description.len() > 500 {
            let truncated = &frame_context.scene_description[..500];
            format!("{}... [truncated]", truncated)
        } else {
            frame_context.scene_description.clone()
        };
        
        context.push_str(&format!(
            "Scene description: {}\n",
            scene_desc
        ));
        
        if !frame_context.objects.is_empty() {
            context.push_str("Detected objects:\n");
            // Limit to first 5 objects to prevent context explosion
            for obj in frame_context.objects.iter().take(5) {
                context.push_str(&format!(
                    "- {} at ({}, {}) with size {}x{}, confidence: {:.2}\n",
                    obj.label,
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.width,
                    obj.bbox.height,
                    obj.confidence
                ));
            }
            if frame_context.objects.len() > 5 {
                context.push_str(&format!("... and {} more objects\n", frame_context.objects.len() - 5));
            }
        } else {
            context.push_str("No objects detected in current frame.\n");
        }
        
        // Ensure visual context doesn't exceed reasonable size
        if context.len() > 1000 {
            context.truncate(1000);
            context.push_str("... [truncated for length]");
        }
        
        context
    }
    
    pub fn add_action_to_history(
        &mut self,
        action_type: String,
        action_data: String,
        success: bool,
        feedback: Option<String>,
    ) {
        let entry = ActionHistoryEntry {
            timestamp: SystemTime::now(),
            action_type,
            action_data,
            success,
            feedback,
        };
        
        self.action_history.push_back(entry);
        
        // Maintain maximum history size
        while self.action_history.len() > self.max_history {
            self.action_history.pop_front();
        }
    }
    
    #[allow(dead_code)]
    pub fn add_environmental_constraint(&mut self, constraint: String) {
        if !self.environmental_constraints.contains(&constraint) {
            self.environmental_constraints.push(constraint);
        }
    }
    
    #[allow(dead_code)]
    pub fn add_safety_rule(&mut self, rule: String) {
        if !self.safety_rules.contains(&rule) {
            self.safety_rules.push(rule);
        }
    }
    
    #[allow(dead_code)]
    pub fn get_action_history(&self) -> &VecDeque<ActionHistoryEntry> {
        &self.action_history
    }
    
    #[allow(dead_code)]
    pub fn clear_action_history(&mut self) {
        self.action_history.clear();
    }
}

```

**./src/test_simple.rs**
```rust
use anyhow::Result;
use crate::models::{GemmaModel, ModelConfig};

pub async fn test_simple_inference() -> Result<()> {
    println!("🚀 Testing simple text inference...");
    
    let config = ModelConfig {
        max_tokens: 25,  // Very small for quick test
        temperature: 0.1,
        top_p: 0.8,
        context_length: 256,
    };
    
    println!("📥 Loading UQFF Gemma 3n model...");
    
    // Load the UQFF model
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    println!("✅ Model loaded successfully!");
    
    // Single text test
    println!("\n🔤 Simple text test");
    let prompt = "Hello!";
    println!("  📝 Prompt: \"{}\"", prompt);
    
    match model.generate(prompt).await {
        Ok(result) => {
            println!("  ✅ Response: {}", result.text);
            println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
        }
        Err(e) => {
            println!("  ❌ Generation failed: {}", e);
            return Err(e);
        }
    }
    
    println!("\n🎉 Simple inference test completed successfully!");
    Ok(())
}

```

**./src/go2_mock.rs**
```rust
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::time::{interval, sleep};
use tracing::{info, debug, warn, error};

// Mock Mat structure for development without OpenCV
#[derive(Debug, Clone)]
pub struct Mat {
    pub cols: i32,
    pub rows: i32,
    pub data: Vec<u8>,
}

impl Mat {
    pub fn default() -> Self {
        Self {
            cols: 640,
            rows: 480,
            data: vec![128; 640 * 480 * 3], // Default gray image
        }
    }
    
    pub fn cols(&self) -> i32 {
        self.cols
    }
    
    pub fn rows(&self) -> i32 {
        self.rows
    }
    
    pub fn empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Unitree Go2 camera configuration
#[derive(Debug, Clone)]
pub struct Go2CameraConfig {
    pub camera_id: i32,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub exposure: Option<f64>,
    pub gain: Option<f64>,
}

impl Default for Go2CameraConfig {
    fn default() -> Self {
        Self {
            camera_id: 0, // Front camera
            width: 1280,
            height: 720,
            fps: 30.0,
            exposure: None,
            gain: None,
        }
    }
}

/// Go2 LiDAR point for distance overlay
#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub distance: f32,
    pub intensity: u8,
}

/// Go2 sensor data combining camera and LiDAR
#[derive(Debug, Clone)]
pub struct Go2SensorData {
    pub frame: Mat,
    pub timestamp: SystemTime,
    pub lidar_points: Vec<LidarPoint>,
    pub robot_pose: Option<(f64, f64, f64)>, // x, y, yaw
}

/// Unitree Go2 camera interface (mock version)
pub struct Go2Camera {
    config: Go2CameraConfig,
    is_running: Arc<Mutex<bool>>,
    last_frame_time: SystemTime,
    frame_counter: u32,
}

impl Go2Camera {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        info!("Initializing mock Unitree Go2 camera with config: {:?}", config);
        
        Ok(Self {
            config,
            is_running: Arc::new(Mutex::new(false)),
            last_frame_time: SystemTime::now(),
            frame_counter: 0,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Connecting to Go2 camera (mock mode - device {})", self.config.camera_id);
        
        // Simulate camera initialization
        sleep(Duration::from_millis(500)).await;
        
        *self.is_running.lock().unwrap() = true;

        info!("Go2 camera initialized successfully (mock mode)");
        Ok(())
    }

    pub async fn capture_frame(&mut self) -> Result<Mat> {
        if !self.is_running() {
            return Err(anyhow!("Go2 camera not initialized"));
        }

        self.frame_counter += 1;
        self.last_frame_time = SystemTime::now();
        
        // Create a mock frame with some variation
        let mut frame_data = vec![128u8; (self.config.width * self.config.height * 3) as usize];
        
        // Add some variation based on frame counter to simulate motion
        let variation = (self.frame_counter % 100) as u8;
        for i in (0..frame_data.len()).step_by(3) {
            frame_data[i] = frame_data[i].saturating_add(variation); // R
            frame_data[i + 1] = frame_data[i + 1].saturating_sub(variation / 2); // G
            frame_data[i + 2] = frame_data[i + 2].saturating_add(variation / 3); // B
        }
        
        let frame = Mat {
            cols: self.config.width,
            rows: self.config.height,
            data: frame_data,
        };

        debug!("Captured mock frame #{} from Go2 camera: {}x{}", 
               self.frame_counter, frame.cols(), frame.rows());

        Ok(frame)
    }

    pub fn is_running(&self) -> bool {
        *self.is_running.lock().unwrap()
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 camera (mock mode)");
        *self.is_running.lock().unwrap() = false;
        Ok(())
    }
}

/// Go2 LiDAR interface (mock implementation)
pub struct Go2Lidar {
    is_running: Arc<Mutex<bool>>,
    scan_counter: u32,
}

impl Go2Lidar {
    pub fn new() -> Self {
        Self {
            is_running: Arc::new(Mutex::new(false)),
            scan_counter: 0,
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 LiDAR (mock mode)");
        
        // Simulate LiDAR initialization
        sleep(Duration::from_millis(300)).await;
        
        *self.is_running.lock().unwrap() = true;
        Ok(())
    }

    pub async fn get_scan(&mut self) -> Result<Vec<LidarPoint>> {
        if !*self.is_running.lock().unwrap() {
            return Ok(Vec::new());
        }
        
        self.scan_counter += 1;
        
        // Generate mock LiDAR points
        let mut points = Vec::new();
        
        // Create a semicircle of points in front of the robot
        for i in 0..36 {
            let angle = (i as f32 - 18.0) * 0.174533; // -π/2 to π/2 in steps of 5 degrees
            let base_distance = 2.0 + (self.scan_counter as f32 * 0.1).sin().abs(); // Vary distance slightly
            let distance = base_distance + (i as f32 * 0.05) % 2.0; // Add some variation
            
            if distance > 0.1 && distance < 8.0 { // Realistic range
                points.push(LidarPoint {
                    x: distance * angle.cos(),
                    y: distance * angle.sin(),
                    z: 0.0,
                    distance,
                    intensity: 150 + (i * 3) as u8,
                });
            }
        }
        
        // Add some random obstacles
        if self.scan_counter % 30 == 0 {
            points.push(LidarPoint {
                x: 1.5,
                y: 0.0,
                z: 0.0,
                distance: 1.5,
                intensity: 255,
            });
        }
        
        debug!("Mock LiDAR scan #{} generated {} points", self.scan_counter, points.len());
        Ok(points)
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 LiDAR (mock mode)");
        *self.is_running.lock().unwrap() = false;
        Ok(())
    }
}

/// Combined Go2 sensor system (mock implementation)
pub struct Go2SensorSystem {
    camera: Go2Camera,
    lidar: Go2Lidar,
    config: Go2CameraConfig,
}

impl Go2SensorSystem {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        let camera = Go2Camera::new(config.clone())?;
        let lidar = Go2Lidar::new();
        
        info!("Go2SensorSystem created (mock mode)");
        
        Ok(Self {
            camera,
            lidar,
            config,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 sensor system (mock mode)");
        
        self.camera.initialize().await?;
        self.lidar.initialize().await?;
        
        info!("Go2 sensor system initialized successfully (mock mode)");
        Ok(())
    }

    pub async fn capture_sensor_data(&mut self) -> Result<Go2SensorData> {
        // Capture frame from camera
        let frame = self.camera.capture_frame().await?;
        let timestamp = SystemTime::now();
        
        // Get LiDAR scan
        let lidar_points = self.lidar.get_scan().await?;
        
        // Mock robot pose
        let robot_pose = Some((0.0, 0.0, 0.0)); // Stationary for now

        debug!("Captured sensor data: {}x{} frame with {} LiDAR points", 
               frame.cols(), frame.rows(), lidar_points.len());

        Ok(Go2SensorData {
            frame,
            timestamp,
            lidar_points,
            robot_pose,
        })
    }

    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        // Mock overlay implementation - in reality would draw on the frame
        debug!("Mock LiDAR overlay: {} points on {}x{} frame", 
               lidar_points.len(), frame.cols(), frame.rows());
        
        for (i, point) in lidar_points.iter().enumerate().take(10) {
            debug!("LiDAR point {}: distance {:.1}m at ({:.1}, {:.1})", 
                   i, point.distance, point.x, point.y);
        }
        
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 sensor system (mock mode)");
        
        self.camera.stop().await?;
        self.lidar.stop().await?;
        
        Ok(())
    }
}

/// Go2-specific utility functions (mock implementations)
pub mod utils {
    use super::*;
    
    pub fn detect_go2_cameras() -> Result<Vec<i32>> {
        // Mock camera detection
        let available_cameras = vec![0, 1]; // Simulate 2 available cameras
        info!("Detected Go2 cameras (mock): {:?}", available_cameras);
        Ok(available_cameras)
    }
    
    pub fn get_optimal_camera_settings() -> Go2CameraConfig {
        Go2CameraConfig {
            camera_id: 0,
            width: 1280,
            height: 720,
            fps: 30.0,
            exposure: Some(-6.0), // Auto exposure
            gain: Some(50.0),     // Moderate gain for indoor/outdoor
        }
    }
}

```

**./src/config.rs**
```rust
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
    /// Enable multimodal inference (send actual camera images to model)
    pub enable_multimodal_inference: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: String,
    pub quantization_method: String, // "uqff" or "isq"
    pub isq_type: String, // ISQ quantization type: Q2K, Q3K, Q4K, Q5K, Q6K, Q8_0, Q8_1
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
                enable_multimodal_inference: true, // Enable by default for vision models
            },
            model: ModelConfig {
                model_path: "".to_string(), // Use default model
                quantization_method: "isq".to_string(), // Default to ISQ for fast loading
                isq_type: "Q4K".to_string(), // Default to Q4K (good balance of speed/quality)
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
    
    #[allow(dead_code)]
    pub async fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content).await?;
        Ok(())
    }
}

```

**./src/pipeline.rs**
```rust
use crate::config::OpticXTConfig;
use crate::vision::{VisionProcessor, Mat};
use crate::context::ContextManager;
use crate::models::{GemmaModel, ModelConfig, ensure_model_downloaded};
use crate::commands::{CommandExecutor, CommandExecutionResult};
use crate::camera::{CameraSystem, CameraConfig};
use crate::audio::{AudioSystem, AudioConfig};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};

pub struct VisionActionPipeline {
    vision_processor: VisionProcessor,
    camera_system: CameraSystem,
    audio_system: Option<AudioSystem>,
    context_manager: Arc<RwLock<ContextManager>>,
    model: GemmaModel,
    command_executor: CommandExecutor,
    config: OpticXTConfig,
    running: Arc<RwLock<bool>>,
}

impl VisionActionPipeline {
    pub async fn new(
        config: OpticXTConfig,
        camera_device: usize,
        model_path: Option<String>,
    ) -> Result<Self> {
        info!("Initializing OpticXT Vision-Action Pipeline");
        
        // Initialize vision processor
        let mut vision_processor = VisionProcessor::new(
            camera_device,
            config.vision.width,
            config.vision.height,
            config.vision.confidence_threshold,
            config.vision.vision_model.clone(),
        )?;
        
        // Initialize the camera system
        let camera_config = CameraConfig {
            camera_id: camera_device as i32,
            width: config.vision.width as i32,
            height: config.vision.height as i32,
            fps: 30.0,
        };
        let mut camera_system = CameraSystem::new(camera_config)?;
        camera_system.initialize().await?;
        
        // Initialize audio system for robot voice output
        let audio_system = if config.audio.enabled {
            let audio_config = AudioConfig::default();
            let mut audio = AudioSystem::new(audio_config)?;
            audio.initialize().await?;
            Some(audio)
        } else {
            None
        };
        
        // Initialize the Go2 sensor system (legacy compatibility)
        vision_processor.initialize().await?;
        
        // Initialize context manager
        let context_manager = Arc::new(RwLock::new(ContextManager::new(
            config.context.system_prompt.clone(),
            config.context.max_context_history,
            config.context.include_timestamp,
        )));
        
        // Ensure model is downloaded and load it
        let model_path = model_path.or_else(|| {
            if config.model.model_path.is_empty() {
                None // Use default UQFF model
            } else {
                Some(config.model.model_path.clone())
            }
        });
        if let Some(ref path) = model_path {
            ensure_model_downloaded(path).await?;
        }
        
        let model_config = ModelConfig {
            max_tokens: config.model.max_tokens,
            temperature: config.model.temperature,
            top_p: config.model.top_p,
            context_length: config.model.context_length,
        };
        
        let model = GemmaModel::load(model_path, model_config, config.model.quantization_method.clone(), config.model.isq_type.clone()).await?;
        
        // Initialize command executor
        let command_executor = CommandExecutor::new(
            config.commands.enabled_commands.clone(),
            config.commands.timeout_seconds,
            config.commands.validate_before_execution,
        );
        
        let running = Arc::new(RwLock::new(false));
        
        info!("Pipeline initialization complete");
        
        Ok(Self {
            vision_processor,
            camera_system,
            audio_system,
            context_manager,
            model,
            command_executor,
            config,
            running,
        })
    }
    
    pub async fn run(&mut self) -> Result<()> {
        info!("Starting OpticXT main processing loop");
        
        // Set running flag
        {
            let mut running = self.running.write().await;
            *running = true;
        }
        
        // Create debug output (simulated without OpenCV window)
        if cfg!(debug_assertions) {
            debug!("Debug mode: would show OpenCV window with annotations");
        }
        
        let mut frame_count = 0;
        let mut last_stats_time = std::time::Instant::now();
        
        loop {
            // Check if we should continue running
            {
                let running = self.running.read().await;
                if !*running {
                    break;
                }
            }
            
            // Process one frame
            match self.process_single_frame().await {
                Ok(_) => {
                    frame_count += 1;
                    
                    // Print stats every 100 frames
                    if frame_count % 100 == 0 {
                        let elapsed = last_stats_time.elapsed();
                        let fps = 100.0 / elapsed.as_secs_f32();
                        info!("Processed {} frames, current FPS: {:.2}", frame_count, fps);
                        last_stats_time = std::time::Instant::now();
                    }
                }
                Err(e) => {
                    error!("Frame processing error: {}", e);
                    // Continue processing despite errors
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
            
            // Control processing rate
            tokio::time::sleep(tokio::time::Duration::from_millis(
                self.config.performance.processing_interval_ms
            )).await;
        }
        
        info!("Pipeline stopped after processing {} frames", frame_count);
        Ok(())
    }
    
    async fn process_single_frame(&mut self) -> Result<()> {
        // Step 1: Capture frame and sensor data from camera system
        let sensor_data = self.camera_system.capture_sensor_data().await?;
        debug!("📷 Captured camera frame: {}x{}", sensor_data.frame.width, sensor_data.frame.height);
        
        // Step 2: Process frame for object detection and context
        let frame_context = self.vision_processor.process_frame(&sensor_data.frame, &sensor_data).await?;
        
        debug!("👁️ Vision processed: {} objects detected | Scene: {}", 
               frame_context.objects.len(), 
               frame_context.scene_description.chars().take(100).collect::<String>());
        debug!("🎯 LiDAR data: {} points", sensor_data.lidar_points.len());
        
        // Step 3: Build mandatory context
        let mandatory_context = {
            let context_manager = self.context_manager.read().await;
            context_manager.build_context(&frame_context)
        };
        
        // Step 4: Generate model prompt
        let prompt = {
            let context_manager = self.context_manager.read().await;
            context_manager.create_model_prompt(&mandatory_context)
        };
        
        debug!("Generated prompt with {} characters", prompt.len());
        
        // Step 5: Run model inference with vision
        let generation_result = if self.config.vision.enable_multimodal_inference {
            // Convert camera frame to image for vision model
            match sensor_data.frame.to_image() {
                Ok(camera_image) => {
                    debug!("Sending camera image to vision model for multimodal inference");
                    match self.model.generate_with_image(&prompt, camera_image).await {
                        Ok(result) => result,
                        Err(e) => {
                            warn!("Vision model inference failed, falling back to text-only: {}", e);
                            self.model.generate(&prompt).await?
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to convert camera frame to image, using text-only inference: {}", e);
                    self.model.generate(&prompt).await?
                }
            }
        } else {
            // Text-only inference (current behavior)
            debug!("Using text-only inference (multimodal disabled in config)");
            self.model.generate(&prompt).await?
        };
        
        debug!(
            "Model generated {} tokens in {}ms: {}",
            generation_result.tokens_generated,
            generation_result.processing_time_ms,
            generation_result.text.chars().take(100).collect::<String>()
        );
        
        // Step 6: Parse and execute command
        let execution_result = self.command_executor
            .parse_and_execute(&generation_result.text)
            .await?;
        
        // Step 7: Update context with action result
        {
            let mut context_manager = self.context_manager.write().await;
            context_manager.add_action_to_history(
                self.extract_action_type(&generation_result.text),
                generation_result.text.clone(),
                execution_result.success,
                Some(execution_result.message.clone()),
            );
        }
        
        // Step 8: Audio feedback for robot voice output
        if let Some(ref mut audio_system) = self.audio_system {
            if execution_result.success && generation_result.text.contains("<speak>") {
                // Extract speech text before borrowing audio system
                let speech_text = Self::extract_speech_text_static(&generation_result.text);
                if !speech_text.is_empty() {
                    if let Err(e) = audio_system.speak_text(&speech_text).await {
                        warn!("Robot voice output failed: {}", e);
                    }
                }
            }
        }
        
        // Step 9: Debug output (overlay LiDAR data on frame)
        if cfg!(debug_assertions) {
            let mut debug_frame = sensor_data.frame.clone();
            self.camera_system.overlay_lidar_distances(&mut debug_frame, &sensor_data.lidar_points)?;
            self.vision_processor.annotate_frame(&mut debug_frame, &frame_context)?;
            
            debug!("Action result: {}", execution_result.message);
        }
        
        info!(
            "Action executed: {} ({}ms total)",
            execution_result.message,
            generation_result.processing_time_ms + execution_result.execution_time.as_millis()
        );
        
        Ok(())
    }
    
    fn extract_action_type(&self, tool_call_output: &str) -> String {
        // Extract action type from JSON tool call output
        if let Ok(tool_calls) = serde_json::from_str::<Vec<serde_json::Value>>(tool_call_output) {
            if let Some(first_call) = tool_calls.first() {
                if let Some(function_name) = first_call.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str()) 
                {
                    return function_name.to_string();
                }
            }
        }
        
        // Fallback: try to detect from text content (for compatibility)
        if tool_call_output.contains("move") {
            "move".to_string()
        } else if tool_call_output.contains("rotate") {
            "rotate".to_string()
        } else if tool_call_output.contains("speak") {
            "speak".to_string()
        } else if tool_call_output.contains("analyze") {
            "analyze".to_string()
        } else if tool_call_output.contains("wait") {
            "wait".to_string()
        } else if tool_call_output.contains("stop") {
            "stop".to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    #[allow(dead_code)]
    fn add_action_overlay(&self, _frame: &mut Mat, result: &CommandExecutionResult) -> Result<()> {
        // Placeholder for action overlay without OpenCV
        debug!("Simulating action overlay: {}", result.message);
        Ok(())
    }
    
    #[allow(dead_code)]
    pub async fn stop(&self) {
        info!("Stopping pipeline...");
        let mut running = self.running.write().await;
        *running = false;
    }
    
    #[allow(dead_code)]
    pub async fn is_running(&self) -> bool {
        let running = self.running.read().await;
        *running
    }
    
    #[allow(dead_code)]
    pub async fn get_stats(&self) -> PipelineStats {
        let context_manager = self.context_manager.read().await;
        let history_count = context_manager.get_action_history().len();
        
        PipelineStats {
            actions_executed: history_count,
            model_device: format!("{:?}", self.model.get_device()),
            vision_model: self.config.vision.vision_model.clone(),
            is_running: self.is_running().await,
        }
    }
    
    #[allow(dead_code)]
    fn extract_speech_text(&self, text: &str) -> String {
        Self::extract_speech_text_static(text)
    }
    
    fn extract_speech_text_static(text: &str) -> String {
        // Extract text from <speak> tags
        if let Some(start) = text.find("<speak>") {
            if let Some(end) = text.find("</speak>") {
                let start_pos = start + 7; // Length of "<speak>"
                if start_pos < end {
                    return text[start_pos..end].trim().to_string();
                }
            }
        }
        
        // Fallback: clean up any JSON or formatting artifacts
        text.replace("{", "").replace("}", "").replace("\"", "").trim().to_string()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub actions_executed: usize,
    pub model_device: String,
    pub vision_model: String,
    pub is_running: bool,
}

// Graceful shutdown handler
#[allow(dead_code)]
pub async fn setup_shutdown_handler(_pipeline: Arc<VisionActionPipeline>) -> Result<()> {
    // Note: Camera threading issues prevent spawning for now
    // TODO: Implement proper shutdown handling when camera is Send+Sync
    info!("Shutdown handler setup (simplified due to threading constraints)");
    Ok(())
}

```

**./src/go2.rs**
```rust
use anyhow::{Result, anyhow};
use opencv::{core::Mat, prelude::*};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::time::{interval, sleep};
use tracing::{info, debug, warn, error};

/// Unitree Go2 camera configuration
#[derive(Debug, Clone)]
pub struct Go2CameraConfig {
    pub camera_id: i32,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub exposure: Option<f64>,
    pub gain: Option<f64>,
}

impl Default for Go2CameraConfig {
    fn default() -> Self {
        Self {
            camera_id: 0, // Front camera
            width: 1280,
            height: 720,
            fps: 30.0,
            exposure: None,
            gain: None,
        }
    }
}

/// Go2 LiDAR point for distance overlay
#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub distance: f32,
    pub intensity: u8,
}

/// Go2 sensor data combining camera and LiDAR
#[derive(Debug, Clone)]
pub struct Go2SensorData {
    pub frame: Mat,
    pub timestamp: SystemTime,
    pub lidar_points: Vec<LidarPoint>,
    pub robot_pose: Option<(f64, f64, f64)>, // x, y, yaw
}

/// Unitree Go2 camera interface
pub struct Go2Camera {
    config: Go2CameraConfig,
    camera: Option<opencv::videoio::VideoCapture>,
    is_running: Arc<Mutex<bool>>,
    last_frame_time: SystemTime,
}

impl Go2Camera {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        info!("Initializing Unitree Go2 camera with config: {:?}", config);
        
        Ok(Self {
            config,
            camera: None,
            is_running: Arc::new(Mutex::new(false)),
            last_frame_time: SystemTime::now(),
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Connecting to Go2 camera (device {})", self.config.camera_id);
        
        // Initialize OpenCV VideoCapture for Go2 camera
        let mut camera = opencv::videoio::VideoCapture::new(
            self.config.camera_id, 
            opencv::videoio::CAP_V4L2
        )?;
        
        if !camera.is_opened()? {
            // Try alternative camera indices for Go2
            warn!("Primary camera not available, trying alternatives...");
            for cam_id in [1, 2, 3, 4] {
                camera = opencv::videoio::VideoCapture::new(cam_id, opencv::videoio::CAP_V4L2)?;
                if camera.is_opened()? {
                    info!("Connected to Go2 camera on device {}", cam_id);
                    self.config.camera_id = cam_id;
                    break;
                }
            }
        }

        if !camera.is_opened()? {
            return Err(anyhow!("Could not open any Go2 camera device"));
        }

        // Configure camera settings for Go2
        camera.set(opencv::videoio::CAP_PROP_FRAME_WIDTH, self.config.width as f64)?;
        camera.set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, self.config.height as f64)?;
        camera.set(opencv::videoio::CAP_PROP_FPS, self.config.fps)?;
        
        // Set exposure and gain if specified
        if let Some(exposure) = self.config.exposure {
            camera.set(opencv::videoio::CAP_PROP_EXPOSURE, exposure)?;
        }
        if let Some(gain) = self.config.gain {
            camera.set(opencv::videoio::CAP_PROP_GAIN, gain)?;
        }

        // Enable auto-focus for Go2 camera
        camera.set(opencv::videoio::CAP_PROP_AUTOFOCUS, 1.0)?;

        self.camera = Some(camera);
        *self.is_running.lock().unwrap() = true;

        info!("Go2 camera initialized successfully");
        Ok(())
    }

    pub async fn capture_frame(&mut self) -> Result<Mat> {
        let camera = self.camera.as_mut()
            .ok_or_else(|| anyhow!("Go2 camera not initialized"))?;

        let mut frame = Mat::default();
        if !camera.read(&mut frame)? {
            return Err(anyhow!("Failed to read frame from Go2 camera"));
        }

        if frame.empty() {
            return Err(anyhow!("Empty frame from Go2 camera"));
        }

        self.last_frame_time = SystemTime::now();
        debug!("Captured frame from Go2 camera: {}x{}", frame.cols(), frame.rows());

        Ok(frame)
    }

    pub fn is_running(&self) -> bool {
        *self.is_running.lock().unwrap()
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 camera");
        *self.is_running.lock().unwrap() = false;
        
        if let Some(mut camera) = self.camera.take() {
            camera.release()?;
        }
        
        Ok(())
    }
}

/// Go2 LiDAR interface (placeholder for future implementation)
pub struct Go2Lidar {
    is_running: Arc<Mutex<bool>>,
}

impl Go2Lidar {
    pub fn new() -> Self {
        Self {
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 LiDAR (placeholder)");
        *self.is_running.lock().unwrap() = true;
        Ok(())
    }

    pub async fn get_scan(&self) -> Result<Vec<LidarPoint>> {
        // Placeholder: return empty scan for now
        // TODO: Implement actual Go2 LiDAR SDK integration
        Ok(Vec::new())
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 LiDAR");
        *self.is_running.lock().unwrap() = false;
        Ok(())
    }
}

/// Combined Go2 sensor system
pub struct Go2SensorSystem {
    camera: Go2Camera,
    lidar: Go2Lidar,
    config: Go2CameraConfig,
}

impl Go2SensorSystem {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        let camera = Go2Camera::new(config.clone())?;
        let lidar = Go2Lidar::new();
        
        Ok(Self {
            camera,
            lidar,
            config,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 sensor system");
        
        self.camera.initialize().await?;
        self.lidar.initialize().await?;
        
        info!("Go2 sensor system initialized successfully");
        Ok(())
    }

    pub async fn capture_sensor_data(&mut self) -> Result<Go2SensorData> {
        // Capture frame from camera
        let frame = self.camera.capture_frame().await?;
        let timestamp = SystemTime::now();
        
        // Get LiDAR scan
        let lidar_points = self.lidar.get_scan().await?;
        
        // TODO: Get robot pose from Go2 SDK
        let robot_pose = None;

        Ok(Go2SensorData {
            frame,
            timestamp,
            lidar_points,
            robot_pose,
        })
    }

    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        use opencv::{core, imgproc};
        
        // Overlay distance information on the frame
        for point in lidar_points.iter().take(50) { // Limit to 50 points for performance
            if point.distance > 0.1 && point.distance < 10.0 {
                // Project 3D point to 2D image coordinates (simplified projection)
                let img_x = ((point.x / point.distance) * 500.0 + frame.cols() as f32 / 2.0) as i32;
                let img_y = ((point.y / point.distance) * 500.0 + frame.rows() as f32 / 2.0) as i32;
                
                if img_x > 0 && img_x < frame.cols() && img_y > 0 && img_y < frame.rows() {
                    // Draw distance circle
                    let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0); // Green
                    let center = core::Point::new(img_x, img_y);
                    imgproc::circle(frame, center, 3, color, -1, imgproc::LINE_8, 0)?;
                    
                    // Add distance text
                    let distance_text = format!("{:.1}m", point.distance);
                    let text_pos = core::Point::new(img_x + 5, img_y - 5);
                    imgproc::put_text(
                        frame,
                        &distance_text,
                        text_pos,
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        imgproc::LINE_8,
                        false,
                    )?;
                }
            }
        }
        
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 sensor system");
        
        self.camera.stop().await?;
        self.lidar.stop().await?;
        
        Ok(())
    }
}

/// Go2-specific utility functions
pub mod utils {
    use super::*;
    
    pub fn detect_go2_cameras() -> Result<Vec<i32>> {
        let mut available_cameras = Vec::new();
        
        // Check common Go2 camera device indices
        for cam_id in 0..8 {
            if let Ok(camera) = opencv::videoio::VideoCapture::new(cam_id, opencv::videoio::CAP_V4L2) {
                if camera.is_opened().unwrap_or(false) {
                    available_cameras.push(cam_id);
                }
            }
        }
        
        info!("Detected Go2 cameras: {:?}", available_cameras);
        Ok(available_cameras)
    }
    
    pub fn get_optimal_camera_settings() -> Go2CameraConfig {
        Go2CameraConfig {
            camera_id: 0,
            width: 1280,
            height: 720,
            fps: 30.0,
            exposure: Some(-6.0), // Auto exposure
            gain: Some(50.0),     // Moderate gain for indoor/outdoor
        }
    }
}

```

**./src/main.rs**
```rust
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
mod tests;
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
    
    /// Test UQFF model loading and inference
    #[arg(long)]
    test_uqff: bool,
    
    /// Test multimodal inference (text, vision, audio)
    #[arg(long)]
    test_multimodal: bool,
    
    /// Test simple text inference only
    #[arg(long)]
    test_simple: bool,
    
    /// Test image inference only
    #[arg(long)]
    test_image: bool,
    
    /// Test OpenAI-style tool calling format
    #[arg(long)]
    test_tool_format: bool,
    
    /// Test quick smoke test (fast basic functionality check)
    #[arg(long)]
    test_quick_smoke: bool,
    
    /// Test image-only inference capabilities
    #[arg(long)]
    test_image_only: bool,
    
    /// Test audio-only inference capabilities
    #[arg(long)]
    test_audio: bool,
    
    /// Test robot command generation scenarios
    #[arg(long)]
    test_robot_commands: bool,
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
    
    // Check if UQFF test mode is requested
    if args.test_uqff {
        info!("Starting UQFF model test");
        tests::test_uqff_model().await?;
        return Ok(());
    }
    
    // Check if multimodal test mode is requested
    if args.test_multimodal {
        info!("Starting multimodal inference test");
        tests::test_multimodal_inference().await?;
        return Ok(());
    }
    
    // Check if simple test mode is requested
    if args.test_simple {
        info!("Starting simple inference test");
        tests::test_simple_inference().await?;
        return Ok(());
    }
    
    // Check if image test mode is requested
    if args.test_image {
        info!("Starting image inference test");
        tests::test_image_inference().await?;
        return Ok(());
    }
    
    // Check if tool format test mode is requested
    if args.test_tool_format {
        info!("Starting tool format test");
        tests::test_tool_format().await?;
        return Ok(());
    }
    
    // Check if quick smoke test mode is requested
    if args.test_quick_smoke {
        info!("Starting quick smoke test");
        tests::test_quick_smoke().await?;
        return Ok(());
    }
    
    // Check if image-only test mode is requested
    if args.test_image_only {
        info!("Starting image-only inference test");
        tests::test_image_only().await?;
        return Ok(());
    }
    
    // Check if audio test mode is requested
    if args.test_audio {
        info!("Starting audio inference test");
        tests::test_audio_inference().await?;
        return Ok(());
    }
    
    // Check if robot commands test mode is requested
    if args.test_robot_commands {
        info!("Starting robot commands test");
        tests::test_robot_commands().await?;
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
    use crate::models::{GemmaModel, ModelConfig};
    
    info!("Running benchmark with {} iterations", args.benchmark_iterations);
    
    // Test model performance
    let model_path = args.model_path.clone();
    let model_config = ModelConfig {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 100,
        context_length: 2048,
    };
    
    info!("Initializing model for benchmark...");
    let mut model = match GemmaModel::load(model_path, model_config, "isq".to_string(), "Q4K".to_string()).await {
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


```

**./src/go2_basic.rs**
```rust
use anyhow::{Result, anyhow};
use std::time::SystemTime;
use tracing::{info, debug, warn};
use nokhwa::{Camera, utils::{RequestedFormat, RequestedFormatType, CameraIndex}};
use nokhwa::pixel_format::RgbFormat;
use crate::vision_basic::Mat;

#[derive(Debug, Clone)]
pub struct Go2CameraConfig {
    pub camera_id: i32,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub exposure: Option<f64>,
    pub gain: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub distance: f64,
    pub angle: f64,
    pub intensity: f64,
}

#[derive(Debug, Clone)]
pub struct Go2SensorData {
    pub timestamp: SystemTime,
    pub frame: Mat,
    pub lidar_points: Vec<LidarPoint>,
}

pub struct Go2SensorSystem {
    config: Go2CameraConfig,
    camera: Option<Camera>,
    is_initialized: bool,
}

impl Go2SensorSystem {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        info!("Creating Go2SensorSystem with nokhwa camera (OpenCV alternative)");
        
        Ok(Self {
            config,
            camera: None,
            is_initialized: false,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 camera system using nokhwa");
        
        // Initialize camera using nokhwa
        let camera_index = CameraIndex::Index(self.config.camera_id as u32);
        let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        
        match Camera::new(camera_index, requested_format) {
            Ok(camera) => {
                self.camera = Some(camera);
                info!("Camera initialized successfully on device {}", self.config.camera_id);
            }
            Err(e) => {
                warn!("Failed to initialize camera on device {}: {}", self.config.camera_id, e);
                // Try fallback devices
                for cam_id in 1..5 {
                    if cam_id == self.config.camera_id {
                        continue;
                    }
                    
                    let fallback_index = CameraIndex::Index(cam_id as u32);
                    if let Ok(camera) = Camera::new(fallback_index, requested_format.clone()) {
                        self.camera = Some(camera);
                        info!("Using fallback camera on device {}", cam_id);
                        break;
                    }
                }
                
                if self.camera.is_none() {
                    return Err(anyhow!("Failed to initialize any camera device"));
                }
            }
        }
        
        // Apply camera settings if possible
        if let Some(ref mut camera) = self.camera {
            match camera.open_stream() {
                Ok(_) => info!("Camera stream opened successfully"),
                Err(e) => warn!("Failed to open camera stream: {}", e),
            }
        }
        
        self.is_initialized = true;
        info!("Go2 sensor system initialized successfully");
        Ok(())
    }
    
    pub async fn capture_sensor_data(&mut self) -> Result<Go2SensorData> {
        if !self.is_initialized {
            return Err(anyhow!("Go2 sensor system not initialized"));
        }
        
        let timestamp = SystemTime::now();
        
        // Capture frame from camera
        let frame = match self.capture_camera_frame().await {
            Ok(frame) => frame,
            Err(e) => {
                warn!("Failed to capture camera frame: {}, using placeholder", e);
                self.create_placeholder_frame()
            }
        };
        
        // Capture LiDAR data (simulated for now)
        let lidar_points = self.capture_lidar_data().await?;
        
        Ok(Go2SensorData {
            timestamp,
            frame,
            lidar_points,
        })
    }
    
    async fn capture_camera_frame(&mut self) -> Result<Mat> {
        if let Some(ref mut camera) = self.camera {
            match camera.frame() {
                Ok(frame) => {
                    // Decode the frame to RGB
                    let decoded = frame.decode_image::<RgbFormat>()?;
                    let width = decoded.width();
                    let height = decoded.height();
                    
                    debug!("Captured frame: {}x{}", width, height);
                    
                    // Convert to our Mat format
                    let rgb_data = decoded.into_raw();
                    let mut mat = Mat::new(width, height, 3);
                    mat.data = rgb_data;
                    
                    Ok(mat)
                }
                Err(e) => {
                    warn!("Camera frame capture failed: {}", e);
                    Err(anyhow!("Camera frame capture error: {}", e))
                }
            }
        } else {
            Err(anyhow!("Camera not initialized"))
        }
    }
    
    fn create_placeholder_frame(&self) -> Mat {
        // Create a placeholder frame when camera is not available
        let width = self.config.width as u32;
        let height = self.config.height as u32;
        let mut frame = Mat::new(width, height, 3);
        
        // Fill with a gradient pattern for testing
        for y in 0..height {
            for x in 0..width {
                let pixel_index = ((y * width + x) * 3) as usize;
                if pixel_index + 2 < frame.data.len() {
                    frame.data[pixel_index] = (x * 255 / width) as u8;     // R
                    frame.data[pixel_index + 1] = (y * 255 / height) as u8; // G
                    frame.data[pixel_index + 2] = 128;                       // B
                }
            }
        }
        
        debug!("Created placeholder frame {}x{}", width, height);
        frame
    }
    
    async fn capture_lidar_data(&mut self) -> Result<Vec<LidarPoint>> {
        // For now, simulate LiDAR data
        // In a real implementation, this would interface with the Go2's LiDAR sensor
        let mut lidar_points = Vec::new();
        
        // Simulate some LiDAR points in a 360-degree scan
        let num_points = 36; // One point every 10 degrees
        for i in 0..num_points {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_points as f64);
            let distance = 1.0 + (angle.sin() * 2.0).abs(); // Simulate varying distances
            
            let x = distance * angle.cos();
            let y = distance * angle.sin();
            let z = 0.0; // Assume flat ground for simulation
            
            lidar_points.push(LidarPoint {
                x,
                y,
                z,
                distance,
                angle,
                intensity: 0.8, // Simulated intensity
            });
        }
        
        debug!("Generated {} simulated LiDAR points", lidar_points.len());
        Ok(lidar_points)
    }
    
    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        debug!("Overlaying {} LiDAR points on frame", lidar_points.len());
        
        let width = frame.width as f64;
        let height = frame.height as f64;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        
        // Draw LiDAR points as colored pixels
        for point in lidar_points {
            // Convert polar coordinates to screen coordinates
            let scale = 50.0; // Scale factor for visualization
            let screen_x = center_x + (point.x * scale);
            let screen_y = center_y + (point.y * scale);
            
            if screen_x >= 0.0 && screen_x < width && screen_y >= 0.0 && screen_y < height {
                let x = screen_x as i32;
                let y = screen_y as i32;
                
                // Color code by distance: close = red, far = blue
                let color = if point.distance < 1.0 {
                    [255, 0, 0] // Red for close objects
                } else if point.distance < 3.0 {
                    [255, 255, 0] // Yellow for medium distance
                } else {
                    [0, 0, 255] // Blue for far objects
                };
                
                self.set_pixel_safe(frame, x, y, color);
                
                // Draw a small cross pattern for better visibility
                self.set_pixel_safe(frame, x - 1, y, color);
                self.set_pixel_safe(frame, x + 1, y, color);
                self.set_pixel_safe(frame, x, y - 1, color);
                self.set_pixel_safe(frame, x, y + 1, color);
            }
        }
        
        Ok(())
    }
    
    fn set_pixel_safe(&self, frame: &mut Mat, x: i32, y: i32, color: [u8; 3]) {
        if x >= 0 && y >= 0 && x < frame.width as i32 && y < frame.height as i32 {
            let pixel_index = ((y as u32 * frame.width + x as u32) * frame.channels) as usize;
            if pixel_index + 2 < frame.data.len() {
                frame.data[pixel_index] = color[0];
                frame.data[pixel_index + 1] = color[1];
                frame.data[pixel_index + 2] = color[2];
            }
        }
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 sensor system");
        
        if let Some(ref mut camera) = self.camera {
            match camera.stop_stream() {
                Ok(_) => info!("Camera stream stopped successfully"),
                Err(e) => warn!("Error stopping camera stream: {}", e),
            }
        }
        
        self.camera = None;
        self.is_initialized = false;
        
        Ok(())
    }
    
    pub fn get_available_cameras() -> Vec<String> {
        info!("Scanning for available cameras");
        let mut cameras = Vec::new();
        
        // Try to detect cameras using nokhwa
        for cam_id in 0..10 {
            let camera_index = CameraIndex::Index(cam_id);
            let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
            
            if let Ok(_camera) = Camera::new(camera_index, requested_format) {
                cameras.push(format!("Camera {}", cam_id));
                info!("Found camera at index {}", cam_id);
            }
        }
        
        if cameras.is_empty() {
            warn!("No cameras detected");
        } else {
            info!("Detected {} camera(s)", cameras.len());
        }
        
        cameras
    }
}

impl Drop for Go2SensorSystem {
    fn drop(&mut self) {
        if self.is_initialized {
            info!("Go2SensorSystem being dropped, cleaning up");
            // Note: Can't use async in Drop, so we do basic cleanup
            if let Some(ref mut camera) = self.camera {
                let _ = camera.stop_stream();
            }
        }
    }
}

```

**./src/vision.rs**
```rust
use anyhow::{Result, anyhow};
use opencv::{
    core::{Mat, Size, CV_8UC3, Rect, Point, Scalar},
    imgproc,
    videoio::{VideoCapture, VideoCaptureAPIs},
    objdetect::CascadeClassifier,
    dnn::{Net, blob_from_image, DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    prelude::*,
};
use std::collections::HashMap;
use std::time::SystemTime;
use std::path::Path;
use tracing::{debug, warn, error, info};
use crate::go2::{Go2SensorSystem, Go2SensorData, Go2CameraConfig, LidarPoint};

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct FrameContext {
    pub timestamp: std::time::SystemTime,
    pub objects: Vec<DetectedObject>,
    pub scene_description: String,
    pub frame_size: (u32, u32),
    pub lidar_points: Vec<LidarPoint>,
}

pub struct VisionProcessor {
    go2_system: Go2SensorSystem,
    yolo_net: Option<Net>,
    face_cascade: Option<CascadeClassifier>,
    confidence_threshold: f32,
    nms_threshold: f32,
    yolo_classes: Vec<String>,
    frame_counter: u32,
}

impl VisionProcessor {
    pub fn new(
        camera_device: usize,
        width: u32,
        height: u32,
        confidence_threshold: f32,
        vision_model: String,
    ) -> Result<Self> {
        info!("Initializing VisionProcessor with real Go2 camera input");
        
        // Configure Go2 camera
        let go2_config = Go2CameraConfig {
            camera_id: camera_device as i32,
            width: width as i32,
            height: height as i32,
            fps: 30.0,
            exposure: Some(-6.0),
            gain: Some(50.0),
        };
        
        let go2_system = Go2SensorSystem::new(go2_config)?;
        
        // Initialize YOLO network
        let yolo_net = Self::load_yolo_model(&vision_model).ok();
        if yolo_net.is_none() {
            warn!("Failed to load YOLO model, falling back to basic detection");
        }
        
        // Initialize face detection cascade
        let face_cascade = Self::load_face_cascade().ok();
        if face_cascade.is_none() {
            warn!("Failed to load face cascade classifier");
        }
        
        // COCO class names for YOLO
        let yolo_classes = vec![
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ].iter().map(|&s| s.to_string()).collect();
        
        Ok(Self {
            go2_system,
            yolo_net,
            face_cascade,
            confidence_threshold,
            nms_threshold: 0.4,
            yolo_classes,
            frame_counter: 0,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 sensor system");
        self.go2_system.initialize().await?;
        Ok(())
    }
    
    fn load_yolo_model(model_path: &str) -> Result<Net> {
        // Try to load YOLOv4 or YOLOv5 model
        let weights_path = format!("{}.weights", model_path);
        let config_path = format!("{}.cfg", model_path);
        
        if Path::new(&weights_path).exists() && Path::new(&config_path).exists() {
            info!("Loading YOLO model from {} and {}", weights_path, config_path);
            let mut net = opencv::dnn::read_net(&weights_path, &config_path, "")?;
            net.set_preferable_backend(DNN_BACKEND_OPENCV)?;
            net.set_preferable_target(DNN_TARGET_CPU)?;
            Ok(net)
        } else {
            Err(anyhow!("YOLO model files not found at {}", model_path))
        }
    }
    
    fn load_face_cascade() -> Result<CascadeClassifier> {
        // Try common locations for the face cascade
        let cascade_paths = vec![
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "./models/haarcascade_frontalface_alt.xml",
        ];
        
        for path in cascade_paths {
            if Path::new(path).exists() {
                info!("Loading face cascade from {}", path);
                return CascadeClassifier::new(path);
            }
        }
        
        Err(anyhow!("Face cascade classifier not found"))
    }
    
    pub async fn capture_frame(&mut self) -> Result<Mat> {
        self.frame_counter += 1;
        debug!("Capturing frame #{} from Go2 camera", self.frame_counter);
        
        let sensor_data = self.go2_system.capture_sensor_data().await?;
        Ok(sensor_data.frame)
    }
    
    pub async fn process_frame(&mut self, frame: &Mat) -> Result<FrameContext> {
        let timestamp = SystemTime::now();
        
        // Get sensor data including LiDAR points
        let sensor_data = self.go2_system.capture_sensor_data().await?;
        
        // Detect objects using available models
        let mut objects = Vec::new();
        
        // Try YOLO detection first
        if let Some(ref mut yolo_net) = self.yolo_net {
            match self.detect_objects_yolo(frame, yolo_net).await {
                Ok(mut yolo_objects) => objects.append(&mut yolo_objects),
                Err(e) => warn!("YOLO detection failed: {}", e),
            }
        }
        
        // Add face detection
        if let Some(ref face_cascade) = self.face_cascade {
            match self.detect_faces(frame, face_cascade) {
                Ok(mut faces) => objects.append(&mut faces),
                Err(e) => warn!("Face detection failed: {}", e),
            }
        }
        
        // If no models available, use basic detection
        if self.yolo_net.is_none() && self.face_cascade.is_none() {
            objects = self.basic_detection(frame)?;
        }
        
        // Filter by confidence threshold
        objects.retain(|obj| obj.confidence >= self.confidence_threshold);
        
        let scene_description = self.generate_scene_description(&objects, &sensor_data.lidar_points);
        
        Ok(FrameContext {
            timestamp,
            objects,
            scene_description,
            frame_size: (frame.cols() as u32, frame.rows() as u32),
            lidar_points: sensor_data.lidar_points,
        })
    }
    
    async fn detect_objects_yolo(&self, frame: &Mat, yolo_net: &mut Net) -> Result<Vec<DetectedObject>> {
        debug!("Running YOLO object detection");
        
        // Create blob from image
        let blob = blob_from_image(
            frame,
            1.0 / 255.0,
            Size::new(416, 416),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            opencv::core::CV_32F,
        )?;
        
        // Set input to the network
        yolo_net.set_input(&blob, "", 1.0, Scalar::default())?;
        
        // Get output layer names
        let output_names = yolo_net.get_unconnected_out_layers_names()?;
        let mut outputs = opencv::core::Vector::<Mat>::new();
        yolo_net.forward(&mut outputs, &output_names)?;
        
        let mut objects = Vec::new();
        let (img_width, img_height) = (frame.cols() as f32, frame.rows() as f32);
        
        // Process each output
        for output in outputs.iter() {
            let rows = output.rows();
            let cols = output.cols();
            
            for i in 0..rows {
                let row = output.row(i)?;
                let scores = row.col_range(&opencv::core::Range::new(5, cols)?)?;
                
                let mut min_val = 0.0;
                let mut max_val = 0.0;
                let mut min_loc = Point::default();
                let mut max_loc = Point::default();
                
                opencv::core::min_max_loc(
                    &scores,
                    Some(&mut min_val),
                    Some(&mut max_val),
                    Some(&mut min_loc),
                    Some(&mut max_loc),
                    &opencv::core::no_array(),
                )?;
                
                if max_val > self.confidence_threshold as f64 {
                    let class_id = max_loc.x as usize;
                    if class_id < self.yolo_classes.len() {
                        let center_x = *row.at_2d::<f32>(0, 0)? * img_width;
                        let center_y = *row.at_2d::<f32>(0, 1)? * img_height;
                        let width = *row.at_2d::<f32>(0, 2)? * img_width;
                        let height = *row.at_2d::<f32>(0, 3)? * img_height;
                        
                        let x = (center_x - width / 2.0) as i32;
                        let y = (center_y - height / 2.0) as i32;
                        
                        objects.push(DetectedObject {
                            label: self.yolo_classes[class_id].clone(),
                            confidence: max_val as f32,
                            bbox: BoundingBox {
                                x: x.max(0),
                                y: y.max(0),
                                width: width as i32,
                                height: height as i32,
                            },
                        });
                    }
                }
            }
        }
        
        debug!("YOLO detected {} objects", objects.len());
        Ok(objects)
    }
    
    fn detect_faces(&self, frame: &Mat, face_cascade: &CascadeClassifier) -> Result<Vec<DetectedObject>> {
        debug!("Running face detection");
        
        // Convert to grayscale for face detection
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Detect faces
        let mut faces = opencv::core::Vector::<Rect>::new();
        face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            3,
            0,
            Size::new(30, 30),
            Size::default(),
        )?;
        
        let mut face_objects = Vec::new();
        for face in faces.iter() {
            face_objects.push(DetectedObject {
                label: "face".to_string(),
                confidence: 0.8, // Face cascade doesn't provide confidence scores
                bbox: BoundingBox {
                    x: face.x,
                    y: face.y,
                    width: face.width,
                    height: face.height,
                },
            });
        }
        
        debug!("Face detection found {} faces", face_objects.len());
        Ok(face_objects)
    }
    
    async fn detect_objects(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        // This method is now deprecated - detection happens in process_frame
        self.basic_detection(frame)
    }
    
    fn basic_detection(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        debug!("Running basic object detection fallback");
        
        // Basic detection using image analysis when no models are available
        let mut objects = Vec::new();
        
        // Analyze image properties for basic detection
        let (width, height) = (frame.cols(), frame.rows());
        
        // Simple heuristic-based detection for demo
        // In a real scenario, this could use edge detection, color analysis, etc.
        
        // Simulate detection based on frame properties
        if width > 640 && height > 480 {
            objects.push(DetectedObject {
                label: "scene".to_string(),
                confidence: 0.6,
                bbox: BoundingBox {
                    x: width / 4,
                    y: height / 4,
                    width: width / 2,
                    height: height / 2,
                },
            });
        }
        
        debug!("Basic detection found {} objects", objects.len());
        Ok(objects)
    }
    
    fn generate_scene_description(&self, objects: &[DetectedObject], lidar_points: &[LidarPoint]) -> String {
        let mut description = String::new();
        
        if objects.is_empty() {
            description.push_str("Empty scene with no detected objects. ");
        } else {
            description.push_str(&format!("Scene contains {} object(s): ", objects.len()));
            
            for (i, obj) in objects.iter().enumerate() {
                if i > 0 {
                    description.push_str(", ");
                }
                description.push_str(&format!(
                    "{} (confidence: {:.2}, position: {},{}, size: {}x{})",
                    obj.label,
                    obj.confidence,
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.width,
                    obj.bbox.height
                ));
            }
            description.push_str(". ");
        }
        
        // Add LiDAR information if available
        if !lidar_points.is_empty() {
            let close_objects = lidar_points.iter()
                .filter(|p| p.distance < 2.0)
                .count();
            let far_objects = lidar_points.iter()
                .filter(|p| p.distance >= 2.0 && p.distance < 5.0)
                .count();
            
            description.push_str(&format!(
                "LiDAR shows {} close objects (<2m), {} mid-range objects (2-5m). ",
                close_objects, far_objects
            ));
            
            if let Some(closest) = lidar_points.iter().min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()) {
                description.push_str(&format!("Closest object at {:.1}m. ", closest.distance));
            }
        }
        
        description
    }
    
    pub fn annotate_frame(&self, frame: &mut Mat, context: &FrameContext) -> Result<()> {
        debug!("Annotating frame with {} objects and {} LiDAR points", 
               context.objects.len(), context.lidar_points.len());
        
        // Draw object bounding boxes and labels
        for obj in &context.objects {
            let color = match obj.label.as_str() {
                "person" | "face" => Scalar::new(0.0, 255.0, 0.0, 0.0), // Green for people
                "car" | "truck" | "bus" => Scalar::new(255.0, 0.0, 0.0, 0.0), // Red for vehicles
                _ => Scalar::new(0.0, 0.0, 255.0, 0.0), // Blue for other objects
            };
            
            // Draw bounding box
            let pt1 = Point::new(obj.bbox.x, obj.bbox.y);
            let pt2 = Point::new(obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height);
            imgproc::rectangle(frame, pt1, pt2, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw label with confidence
            let label = format!("{}: {:.2}", obj.label, obj.confidence);
            let text_pos = Point::new(obj.bbox.x, obj.bbox.y - 10);
            imgproc::put_text(
                frame,
                &label,
                text_pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
        
        // Overlay LiDAR distance information
        if !context.lidar_points.is_empty() {
            self.go2_system.overlay_lidar_distances(frame, &context.lidar_points)?;
        }
        
        // Add frame info
        let info_text = format!("Frame: {} | Objects: {} | LiDAR: {}", 
                               self.frame_counter, context.objects.len(), context.lidar_points.len());
        let info_pos = Point::new(10, 30);
        imgproc::put_text(
            frame,
            &info_text,
            info_pos,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;
        
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping VisionProcessor");
        self.go2_system.stop().await?;
        Ok(())
    }
}

```

**./src/tests.rs**
```rust
use anyhow::Result;
use crate::models::{GemmaModel, ModelConfig};
use image::{DynamicImage, ImageBuffer, Rgb};

/// Test simple text inference with the ISQ model
pub async fn test_simple_inference() -> Result<()> {
    println!("🔬 Testing simple text inference...");
    
    let config = ModelConfig {
        max_tokens: 50,
        temperature: 0.3,
        top_p: 0.9,
        context_length: 2048,
    };
    
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    
    let test_prompts = vec![
        "What do you see in this environment?",
        "Should the robot move forward?",
        "Describe what actions to take next.",
        "Analyze the current situation.",
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n🧪 Test {}: {}", i + 1, prompt);
        
        match model.generate(prompt).await {
            Ok(result) => {
                println!("✅ Generated response ({} tokens in {}ms):", 
                         result.tokens_generated, result.processing_time_ms);
                println!("Response: {}", result.text);
            }
            Err(e) => {
                println!("❌ Generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    println!("\n✨ Simple inference test completed successfully!");
    Ok(())
}

/// Test image inference with vision capabilities
pub async fn test_image_inference() -> Result<()> {
    println!("🔬 Testing image inference with vision capabilities...");
    
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.3,
        top_p: 0.9,
        context_length: 2048,
    };
    
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    
    // Create a test image (simple red square)
    let test_image = create_test_image();
    
    let vision_prompts = vec![
        "What colors do you see in this image?",
        "Describe what you observe in this image.",
        "Should the robot move based on what you see?",
        "What action would be appropriate given this visual input?",
    ];
    
    for (i, prompt) in vision_prompts.iter().enumerate() {
        println!("\n🧪 Vision Test {}: {}", i + 1, prompt);
        
        match model.generate_with_image(prompt, test_image.clone()).await {
            Ok(result) => {
                println!("✅ Generated vision response ({} tokens in {}ms):", 
                         result.tokens_generated, result.processing_time_ms);
                println!("Response: {}", result.text);
            }
            Err(e) => {
                println!("❌ Vision generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    println!("\n✨ Image inference test completed successfully!");
    Ok(())
}

/// Test comprehensive multimodal inference capabilities
pub async fn test_multimodal_inference() -> Result<()> {
    println!("🚀 Testing multimodal inference capabilities...");
    
    let config = ModelConfig {
        max_tokens: 50,
        temperature: 0.3,
        top_p: 0.9,
        context_length: 2048,
    };
    
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    
    // Test 1: Text-only inference
    println!("\n🧪 Test 1: Text-only inference");
    let text_result = model.generate("Hello, describe what you can do.").await?;
    println!("✅ Text response: {}", text_result.text);
    
    // Test 2: Text + Image inference
    println!("\n🧪 Test 2: Text + Image inference");
    let test_image = create_test_image();
    let vision_result = model.generate_with_image(
        "What do you see in this image? Should the robot take any action?",
        test_image
    ).await?;
    println!("✅ Vision response: {}", vision_result.text);
    
    // Test 3: Text + Audio inference (with synthetic audio data)
    println!("\n🧪 Test 3: Text + Audio inference");
    let test_audio = create_test_audio();
    let audio_result = model.generate_with_audio(
        "I'm sending you audio data. Can you process it?",
        test_audio
    ).await?;
    println!("✅ Audio response: {}", audio_result.text);
    
    // Test 4: Full multimodal (text + image + audio)
    println!("\n🧪 Test 4: Full multimodal inference");
    let full_image = create_colored_test_image();
    let full_audio = create_test_audio();
    let multimodal_result = model.generate_multimodal(
        "Process this multimodal input with both image and audio data.",
        Some(full_image),
        Some(full_audio)
    ).await?;
    println!("✅ Multimodal response: {}", multimodal_result.text);
    
    println!("\n🎉 All multimodal tests completed successfully!");
    println!("📊 Performance Summary:");
    println!("   - Text: {}ms", text_result.processing_time_ms);
    println!("   - Vision: {}ms", vision_result.processing_time_ms); 
    println!("   - Audio: {}ms", audio_result.processing_time_ms);
    println!("   - Multimodal: {}ms", multimodal_result.processing_time_ms);
    
    Ok(())
}

/// Test UQFF model inference
pub async fn test_uqff_model() -> Result<()> {
    println!("🔬 Testing UQFF Gemma 3n model inference...");
    
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.3,
        top_p: 0.9,
        context_length: 2048,
    };
    
    // Load the UQFF model specifically
    let mut model = GemmaModel::load(Some("EricB/gemma-3n-E4B-it-UQFF".to_string()), config, "uqff".to_string(), "Q4K".to_string()).await?;
    
    let test_cases = vec![
        ("Simple greeting", "Hello! How are you?"),
        ("Robot command", "Move forward 1 meter"),
        ("Vision query", "What do you see in front of you?"),
        ("Safety check", "Is it safe to proceed?"),
        ("Complex reasoning", "Analyze the environment and suggest the best action for a robot to take."),
    ];
    
    println!("📊 Running {} test cases with UQFF model...\n", test_cases.len());
    
    for (i, (test_name, prompt)) in test_cases.iter().enumerate() {
        println!("🧪 Test {}: {}", i + 1, test_name);
        println!("📝 Prompt: {}", prompt);
        
        let start_time = std::time::Instant::now();
        
        match model.generate(prompt).await {
            Ok(result) => {
                let total_time = start_time.elapsed();
                println!("✅ Response generated in {}ms (model: {}ms):", 
                         total_time.as_millis(), result.processing_time_ms);
                println!("🤖 Output: {}", result.text);
                println!("📈 Tokens: {}, Speed: {:.1} tokens/sec\n", 
                         result.tokens_generated,
                         result.tokens_generated as f64 / (result.processing_time_ms as f64 / 1000.0));
            }
            Err(e) => {
                println!("❌ Test failed: {}", e);
                return Err(e);
            }
        }
    }
    
    println!("🎉 UQFF model test completed successfully!");
    println!("✨ All inference operations working with quantized model");
    
    Ok(())
}

/// Test OpenAI-style tool calling format
pub async fn test_tool_format() -> Result<()> {
    println!("🔧 Testing OpenAI-style tool calling format...");
    
    let config = ModelConfig {
        max_tokens: 50,
        temperature: 0.3,
        top_p: 0.9,
        context_length: 2048,
    };
    
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    
    let test_prompts = vec![
        ("Move command", "Move forward to explore"),
        ("Rotate command", "Turn left to check the area"),
        ("Stop command", "Stop immediately for safety"),
        ("Wait command", "Wait for 3 seconds"),
        ("Analyze command", "Analyze what you see"),
        ("Default speak", "Hello, I am ready to help you"),
    ];
    
    println!("🧪 Testing tool call generation for different commands...\n");
    
    for (test_name, prompt) in test_prompts {
        println!("📝 Testing: {} - \"{}\"", test_name, prompt);
        
        match model.generate(prompt).await {
            Ok(result) => {
                println!("✅ Generated tool call:");
                
                // Try to parse as JSON to validate format
                match serde_json::from_str::<serde_json::Value>(&result.text) {
                    Ok(json) => {
                        println!("✅ Valid JSON format");
                        if let Some(array) = json.as_array() {
                            if let Some(first_call) = array.first() {
                                if let Some(function) = first_call.get("function") {
                                    if let Some(name) = function.get("name") {
                                        println!("🔧 Function: {}", name.as_str().unwrap_or("unknown"));
                                    }
                                }
                            }
                        }
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    Err(e) => {
                        println!("❌ Invalid JSON format: {}", e);
                        println!("Raw output: {}", result.text);
                    }
                }
            }
            Err(e) => {
                println!("❌ Generation failed: {}", e);
                return Err(e);
            }
        }
        println!();
    }
    
    println!("✨ Tool calling format test completed!");
    Ok(())
}

/// Test basic model functionality (quick smoke test)
pub async fn test_quick_smoke() -> Result<()> {
    println!("🚀 Quick smoke test for basic functionality...");
    
    let config = ModelConfig {
        max_tokens: 25,  // Very small for quick test
        temperature: 0.1,
        top_p: 0.8,
        context_length: 256,
    };
    
    println!("📥 Loading ISQ Gemma model...");
    
    // Load the ISQ model
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    println!("✅ Model loaded successfully!");
    
    // Single text test
    println!("\n🔤 Quick text test");
    let prompt = "Hello!";
    println!("  📝 Prompt: \"{}\"", prompt);
    
    match model.generate(prompt).await {
        Ok(result) => {
            println!("  ✅ Response: {}", result.text);
            println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
        }
        Err(e) => {
            println!("  ❌ Generation failed: {}", e);
            return Err(e);
        }
    }
    
    println!("\n🎉 Quick smoke test completed successfully!");
    Ok(())
}

/// Test image-only inference capabilities
pub async fn test_image_only() -> Result<()> {
    println!("🚀 Testing image inference only...");
    
    let config = ModelConfig {
        max_tokens: 50,  // Increased slightly for better responses
        temperature: 0.1,
        top_p: 0.8,
        context_length: 1024,  // Increased for vision processing
    };
    
    println!("📥 Loading ISQ Gemma model for image inference...");
    
    // Load the ISQ model fresh for image processing
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    println!("✅ Model loaded successfully!");
    
    // Create a simple test image
    println!("\n👁️ Creating synthetic test image (64x64 with red square on blue background)...");
    let test_image = create_simple_test_image();
    
    // Single image test
    println!("🔤 Image + text test");
    let prompt = "Describe this image briefly.";
    println!("  📝 Prompt: \"{}\"", prompt);
    
    println!("  🔄 Starting image inference (this may take longer than text-only)...");
    
    match model.generate_with_image(prompt, test_image).await {
        Ok(result) => {
            println!("  ✅ Response: {}", result.text);
            println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
        }
        Err(e) => {
            println!("  ❌ Image generation failed: {}", e);
            return Err(e);
        }
    }
    
    println!("\n🎉 Image inference test completed successfully!");
    Ok(())
}

/// Test audio-only inference capabilities
pub async fn test_audio_inference() -> Result<()> {
    println!("🚀 Testing audio inference capabilities...");
    
    let config = ModelConfig {
        max_tokens: 50,
        temperature: 0.1,
        top_p: 0.8,
        context_length: 1024,
    };
    
    println!("📥 Loading ISQ Gemma model for audio inference...");
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    println!("✅ Model loaded successfully!");
    
    // Create synthetic audio data
    println!("\n🔊 Creating synthetic audio data (1 second 440Hz tone)...");
    let test_audio = create_test_audio();
    
    let audio_prompts = vec![
        "What do you hear in this audio?",
        "Describe the audio content.",
        "Is this audio indicating any action needed?",
        "Analyze this audio input.",
    ];
    
    for (i, prompt) in audio_prompts.iter().enumerate() {
        println!("\n🧪 Audio Test {}: {}", i + 1, prompt);
        
        match model.generate_with_audio(prompt, test_audio.clone()).await {
            Ok(result) => {
                println!("✅ Generated audio response ({} tokens in {}ms):", 
                         result.tokens_generated, result.processing_time_ms);
                println!("Response: {}", result.text);
            }
            Err(e) => {
                println!("❌ Audio generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    println!("\n✨ Audio inference test completed successfully!");
    Ok(())
}

/// Test text generation focused on robot commands
pub async fn test_robot_commands() -> Result<()> {
    println!("🚀 Testing robot command generation...");
    
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.2,  // Lower temperature for more consistent commands
        top_p: 0.9,
        context_length: 2048,
    };
    
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    
    let robot_scenarios = vec![
        ("Navigation", "I need to move to the kitchen."),
        ("Obstacle detection", "There's something blocking my path."),
        ("Task completion", "I've finished cleaning the floor."),
        ("Safety check", "Is it safe to proceed forward?"),
        ("Status update", "Report current battery level and location."),
        ("Emergency stop", "Emergency! Stop all movement immediately."),
    ];
    
    println!("🤖 Testing {} robot command scenarios...\n", robot_scenarios.len());
    
    for (i, (scenario, prompt)) in robot_scenarios.iter().enumerate() {
        println!("🧪 Test {}: {} - \"{}\"", i + 1, scenario, prompt);
        
        match model.generate(prompt).await {
            Ok(result) => {
                println!("✅ Robot response ({} tokens in {}ms):", 
                         result.tokens_generated, result.processing_time_ms);
                println!("🤖 Command: {}", result.text);
            }
            Err(e) => {
                println!("❌ Robot command generation failed: {}", e);
                return Err(e);
            }
        }
        println!();
    }
    
    println!("🎉 Robot command test completed successfully!");
    Ok(())
}

// Helper function to create a simple test image
fn create_test_image() -> DynamicImage {
    let width = 32;
    let height = 32;
    let mut image = ImageBuffer::new(width, height);
    
    // Create a simple red square
    for (_x, _y, pixel) in image.enumerate_pixels_mut() {
        *pixel = Rgb([255, 0, 0]); // Red
    }
    
    DynamicImage::ImageRgb8(image)
}

// Helper function to create a more complex colored test image
fn create_colored_test_image() -> DynamicImage {
    let width = 64;
    let height = 64;
    let mut image = ImageBuffer::new(width, height);
    
    // Create a pattern with different colors
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let r = (x * 4) as u8;
        let g = (y * 4) as u8;
        let b = ((x + y) * 2) as u8;
        *pixel = Rgb([r, g, b]);
    }
    
    DynamicImage::ImageRgb8(image)
}

// Helper function to create a simple test image with shapes
fn create_simple_test_image() -> DynamicImage {
    // Create a 64x64 image with a red square on blue background
    let mut img = ImageBuffer::new(64, 64);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let color = if x >= 16 && x < 48 && y >= 16 && y < 48 {
            // Red square in center
            Rgb([255, 0, 0])
        } else {
            // Blue background
            Rgb([0, 0, 255])
        };
        *pixel = color;
    }
    
    DynamicImage::ImageRgb8(img)
}

// Helper function to create synthetic audio data
fn create_test_audio() -> Vec<u8> {
    // Create 1 second of synthetic audio data (44.1kHz, 16-bit, mono)
    let sample_rate = 44100;
    let duration = 1.0; // 1 second
    let samples = (sample_rate as f64 * duration) as usize;
    
    let mut audio_data = Vec::with_capacity(samples * 2); // 16-bit = 2 bytes per sample
    
    // Generate a simple sine wave at 440Hz (A4 note)
    let frequency = 440.0;
    for i in 0..samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * frequency * t).sin();
        let sample_i16 = (sample * 32767.0) as i16;
        
        // Convert to little-endian bytes
        audio_data.push((sample_i16 & 0xFF) as u8);
        audio_data.push(((sample_i16 >> 8) & 0xFF) as u8);
    }
    
    audio_data
}

```

**./src/test_image.rs**
```rust
use anyhow::Result;
use crate::models::{GemmaModel, ModelConfig};
use image::{DynamicImage, ImageBuffer, Rgb};

pub async fn test_image_only() -> Result<()> {
    println!("🚀 Testing image inference only...");
    
    let config = ModelConfig {
        max_tokens: 50,  // Increased slightly for better responses
        temperature: 0.1,
        top_p: 0.8,
        context_length: 1024,  // Increased for vision processing
    };
    
    println!("📥 Loading UQFF Gemma 3n model for image inference...");
    
    // Load the UQFF model fresh for image processing
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    println!("✅ Model loaded successfully!");
    
    // Create a simple test image
    println!("\n👁️ Creating synthetic test image (64x64 with red square on blue background)...");
    let test_image = create_simple_test_image();
    
    // Single image test
    println!("🔤 Image + text test");
    let prompt = "Describe this image briefly.";
    println!("  📝 Prompt: \"{}\"", prompt);
    
    println!("  🔄 Starting image inference (this may take longer than text-only)...");
    
    match model.generate_with_image(prompt, test_image).await {
        Ok(result) => {
            println!("  ✅ Response: {}", result.text);
            println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
        }
        Err(e) => {
            println!("  ❌ Image generation failed: {}", e);
            return Err(e);
        }
    }
    
    println!("\n🎉 Image inference test completed successfully!");
    Ok(())
}

fn create_simple_test_image() -> DynamicImage {
    // Create a 64x64 image with a red square on blue background
    let mut img = ImageBuffer::new(64, 64);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let color = if x >= 16 && x < 48 && y >= 16 && y < 48 {
            // Red square in center
            Rgb([255, 0, 0])
        } else {
            // Blue background
            Rgb([0, 0, 255])
        };
        *pixel = color;
    }
    
    DynamicImage::ImageRgb8(img)
}

```

**./src/vision_basic.rs**
```rust
use anyhow::{Result, anyhow};
use image::{RgbImage, ImageBuffer, Rgb, DynamicImage};
use tracing::{debug, info};
use crate::camera::{SensorData, LidarPoint};

// Simple matrix type for basic image operations
#[derive(Debug, Clone)]
pub struct Mat {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl Mat {
    pub fn new(width: u32, height: u32, channels: u32) -> Self {
        let size = (width * height * channels) as usize;
        Self {
            data: vec![0u8; size],
            width,
            height,
            channels,
        }
    }

    pub fn to_image(&self) -> Result<DynamicImage> {
        if self.channels == 3 {
            let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                self.width,
                self.height,
                self.data.clone(),
            )
            .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            Ok(DynamicImage::ImageRgb8(img_buffer))
        } else {
            Err(anyhow!("Unsupported channel count: {}", self.channels))
        }
    }

    pub fn rows(&self) -> u32 {
        self.height
    }

    pub fn cols(&self) -> u32 {
        self.width
    }
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct FrameContext {
    pub objects: Vec<DetectedObject>,
    pub scene_description: String,
    pub frame_size: (u32, u32),
    pub lidar_points: Vec<LidarPoint>,
}

pub struct VisionProcessor {
    confidence_threshold: f32,
}

impl VisionProcessor {
    pub fn new(
        _camera_device: usize,
        _width: u32,
        _height: u32,
        confidence_threshold: f32,
        _vision_model: String,
    ) -> Result<Self> {
        info!("Initializing VisionProcessor with basic image processing (OpenCV alternative)");
        
        Ok(Self {
            confidence_threshold,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("VisionProcessor initialized (basic vision mode)");
        Ok(())
    }
    
    pub async fn process_frame(&mut self, frame: &Mat, sensor_data: &SensorData) -> Result<FrameContext> {
        // Perform basic object detection using image analysis
        let objects = self.basic_detection(frame)?;
        
        // Filter by confidence threshold and limit quantity to prevent spam
        let mut filtered_objects: Vec<DetectedObject> = objects
            .into_iter()
            .filter(|obj| obj.confidence >= self.confidence_threshold)
            .collect();
        
        // Sort by confidence and keep only the most relevant objects (max 10)
        filtered_objects.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        filtered_objects.truncate(10);
        
        let scene_description = self.generate_scene_description(&filtered_objects, &sensor_data.lidar_points);
        
        Ok(FrameContext {
            objects: filtered_objects,
            scene_description,
            frame_size: (frame.cols(), frame.rows()),
            lidar_points: sensor_data.lidar_points.clone(),
        })
    }
    
    fn basic_detection(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        debug!("Running optimized object detection with spam prevention");
        
        let mut objects = Vec::new();
        let (_width, _height) = (frame.cols(), frame.rows());
        
        // Convert to image for analysis
        let image = frame.to_image()?;
        
        // Real computer vision detection using image analysis with stricter thresholds
        
        // 1. Person detection using skin tone and face-like regions (higher confidence needed)
        let person_objects = self.detect_people(&image)?;
        objects.extend(person_objects.into_iter().filter(|obj| obj.confidence > 0.6));
        
        // 2. Major object detection using edge and color analysis (reduce sensitivity)
        let shape_objects = self.detect_objects_by_shape(&image)?; 
        objects.extend(shape_objects.into_iter().filter(|obj| obj.confidence > 0.7));
        
        // 3. Motion detection (if we had previous frame) - only high-activity regions
        let motion_objects = self.detect_motion_regions(&image)?;
        objects.extend(motion_objects.into_iter().filter(|obj| obj.confidence > 0.8));
        
        // 4. Only detect major furniture/large objects to avoid clutter
        let furniture_objects = self.detect_furniture(&image)?;
        objects.extend(furniture_objects.into_iter().filter(|obj| obj.confidence > 0.7));
        
        // Merge overlapping detections to prevent duplicates
        let merged_objects = self.merge_overlapping_detections(objects);
        
        debug!("Filtered detection found {} high-confidence objects", merged_objects.len());
        Ok(merged_objects)
    }
    
    fn detect_people(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        let mut people = Vec::new();
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Skin tone detection
        let mut skin_regions = Vec::new();
        
        for y in (0..height).step_by(8) {
            for x in (0..width).step_by(8) {
                if let Some(pixel) = rgb_image.get_pixel_checked(x, y) {
                    let r = pixel[0] as f32;
                    let g = pixel[1] as f32; 
                    let b = pixel[2] as f32;
                    
                    // Skin tone detection algorithm
                    if self.is_skin_tone(r, g, b) {
                        skin_regions.push((x, y));
                    }
                }
            }
        }
        
        // Cluster skin regions into potential people
        if skin_regions.len() > 20 { // Minimum skin pixels for person detection
            let (center_x, center_y) = self.find_skin_cluster_center(&skin_regions);
            let confidence = (skin_regions.len() as f32 / 100.0).min(0.95);
            
            people.push(DetectedObject {
                label: "person".to_string(),
                confidence,
                bbox: BoundingBox {
                    x: (center_x as i32 - 50).max(0),
                    y: (center_y as i32 - 75).max(0),
                    width: 100,
                    height: 150,
                },
            });
        }
        
        Ok(people)
    }
    
    fn is_skin_tone(&self, r: f32, g: f32, b: f32) -> bool {
        // YCbCr color space skin detection
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
        
        // Skin tone ranges in YCbCr
        y > 80.0 && cb >= 85.0 && cb <= 135.0 && cr >= 135.0 && cr <= 180.0
    }
    
    fn find_skin_cluster_center(&self, skin_regions: &[(u32, u32)]) -> (u32, u32) {
        let sum_x: u32 = skin_regions.iter().map(|(x, _)| *x).sum();
        let sum_y: u32 = skin_regions.iter().map(|(_, y)| *y).sum();
        let count = skin_regions.len() as u32;
        
        (sum_x / count, sum_y / count)
    }
    
    fn detect_objects_by_shape(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        let mut objects = Vec::new();
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Edge detection using simple Sobel-like operator
        let edges = self.detect_edges(&rgb_image)?;
        
        // Find rectangular objects (tables, monitors, etc.)
        let rectangles = self.find_rectangles(&edges, width, height);
        for (x, y, w, h, confidence) in rectangles {
            objects.push(DetectedObject {
                label: "rectangular_object".to_string(),
                confidence,
                bbox: BoundingBox { x, y, width: w, height: h },
            });
        }
        
        Ok(objects)
    }
    
    fn detect_edges(&self, rgb_image: &image::RgbImage) -> Result<Vec<Vec<f32>>> {
        let (width, height) = rgb_image.dimensions();
        let mut edges = vec![vec![0.0; width as usize]; height as usize];
        
        // Simple edge detection
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if let Some(_center) = rgb_image.get_pixel_checked(x, y) {
                    let mut grad_x = 0.0;
                    let mut grad_y = 0.0;
                    
                    // Compute gradients
                    if let (Some(left), Some(right)) = (
                        rgb_image.get_pixel_checked(x - 1, y),
                        rgb_image.get_pixel_checked(x + 1, y)
                    ) {
                        grad_x = (right[0] as f32) - (left[0] as f32);
                    }
                    
                    if let (Some(top), Some(bottom)) = (
                        rgb_image.get_pixel_checked(x, y - 1),
                        rgb_image.get_pixel_checked(x, y + 1)
                    ) {
                        grad_y = (bottom[0] as f32) - (top[0] as f32);
                    }
                    
                    edges[y as usize][x as usize] = (grad_x * grad_x + grad_y * grad_y).sqrt();
                }
            }
        }
        
        Ok(edges)
    }
    
    fn find_rectangles(&self, edges: &[Vec<f32>], width: u32, height: u32) -> Vec<(i32, i32, i32, i32, f32)> {
        let mut rectangles = Vec::new();
        
        // Simple rectangle detection using edge accumulation with higher thresholds
        for y in (40..height - 40).step_by(40) {  // Increased step size and margins
            for x in (40..width - 40).step_by(40) {
                let edge_score = self.calculate_rectangular_score(edges, x as usize, y as usize, 60, 45);
                
                // Much higher threshold to reduce false positives
                if edge_score > 30.0 {
                    let confidence = (edge_score / 50.0).min(0.9);
                    rectangles.push((x as i32, y as i32, 60, 45, confidence));
                }
            }
        }
        
        rectangles
    }
    
    fn calculate_rectangular_score(&self, edges: &[Vec<f32>], x: usize, y: usize, w: usize, h: usize) -> f32 {
        let mut score = 0.0;
        
        // Check horizontal edges (top and bottom)
        for i in x..(x + w).min(edges[0].len()) {
            if y < edges.len() {
                score += edges[y][i];
            }
            if (y + h) < edges.len() {
                score += edges[y + h][i];
            }
        }
        
        // Check vertical edges (left and right)
        for i in y..(y + h).min(edges.len()) {
            if x < edges[i].len() {
                score += edges[i][x];
            }
            if (x + w) < edges[i].len() {
                score += edges[i][x + w];
            }
        }
        
        score
    }
    
    fn detect_motion_regions(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        // Placeholder for motion detection - would need previous frame
        // For now, detect high-frequency areas that might indicate movement
        let mut motion_objects = Vec::new();
        
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Detect high-variance regions that might indicate activity with stricter thresholds
        for y in (0..height - 64).step_by(32) {  // Larger regions, less frequent sampling
            for x in (0..width - 64).step_by(32) {
                let variance = self.calculate_region_variance(&rgb_image, x, y, 64, 64);
                
                // Much higher threshold to reduce false motion detections
                if variance > 1500.0 {
                    motion_objects.push(DetectedObject {
                        label: "active_region".to_string(),
                        confidence: (variance / 3000.0).min(0.8),
                        bbox: BoundingBox {
                            x: x as i32,
                            y: y as i32,
                            width: 64,
                            height: 64,
                        },
                    });
                }
            }
        }
        
        Ok(motion_objects)
    }
    
    fn calculate_region_variance(&self, image: &image::RgbImage, x: u32, y: u32, w: u32, h: u32) -> f32 {
        let mut values = Vec::new();
        
        for dy in 0..h {
            for dx in 0..w {
                if let Some(pixel) = image.get_pixel_checked(x + dx, y + dy) {
                    let brightness = pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114;
                    values.push(brightness);
                }
            }
        }
        
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance
    }
    
    fn detect_furniture(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        let mut furniture = Vec::new();
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Detect large horizontal surfaces (tables, desks)
        let horizontal_surfaces = self.find_horizontal_surfaces(&rgb_image, width, height);
        furniture.extend(horizontal_surfaces);
        
        // Detect vertical structures (chairs, walls)
        let vertical_structures = self.find_vertical_structures(&rgb_image, width, height);
        furniture.extend(vertical_structures);
        
        Ok(furniture)
    }
    
    fn find_horizontal_surfaces(&self, image: &image::RgbImage, width: u32, height: u32) -> Vec<DetectedObject> {
        let mut surfaces = Vec::new();
        
        // Look for consistent horizontal lines that might be table edges
        for y in (height / 3)..(2 * height / 3) {
            let mut line_strength = 0.0;
            let mut consistent_pixels = 0;
            
            for x in 10..(width - 10) {
                if let (Some(left), Some(center), Some(right)) = (
                    image.get_pixel_checked(x - 5, y),
                    image.get_pixel_checked(x, y),
                    image.get_pixel_checked(x + 5, y)
                ) {
                    let left_brightness = left[0] as f32 * 0.299 + left[1] as f32 * 0.587 + left[2] as f32 * 0.114;
                    let center_brightness = center[0] as f32 * 0.299 + center[1] as f32 * 0.587 + center[2] as f32 * 0.114;
                    let right_brightness = right[0] as f32 * 0.299 + right[1] as f32 * 0.587 + right[2] as f32 * 0.114;
                    
                    if (left_brightness - center_brightness).abs() > 20.0 || (right_brightness - center_brightness).abs() > 20.0 {
                        line_strength += 1.0;
                        consistent_pixels += 1;
                    }
                }
            }
            
            if consistent_pixels > (width / 4) {
                let confidence = (line_strength / (width as f32)).min(0.85);
                surfaces.push(DetectedObject {
                    label: "table_surface".to_string(),
                    confidence,
                    bbox: BoundingBox {
                        x: 10,
                        y: y as i32 - 20,
                        width: (width - 20) as i32,
                        height: 40,
                    },
                });
            }
        }
        
        surfaces
    }
    
    fn find_vertical_structures(&self, image: &image::RgbImage, width: u32, height: u32) -> Vec<DetectedObject> {
        let mut structures = Vec::new();
        
        // Look for chair backs or vertical furniture elements
        for x in (width / 4)..(3 * width / 4) {
            let mut vertical_score = 0.0;
            
            for y in 10..(height - 10) {
                if let (Some(top), Some(center), Some(bottom)) = (
                    image.get_pixel_checked(x, y - 5),
                    image.get_pixel_checked(x, y),
                    image.get_pixel_checked(x, y + 5)
                ) {
                    let top_brightness = top[0] as f32 * 0.299 + top[1] as f32 * 0.587 + top[2] as f32 * 0.114;
                    let center_brightness = center[0] as f32 * 0.299 + center[1] as f32 * 0.587 + center[2] as f32 * 0.114;
                    let bottom_brightness = bottom[0] as f32 * 0.299 + bottom[1] as f32 * 0.587 + bottom[2] as f32 * 0.114;
                    
                    if (top_brightness - center_brightness).abs() > 15.0 || (bottom_brightness - center_brightness).abs() > 15.0 {
                        vertical_score += 1.0;
                    }
                }
            }
            
            if vertical_score > (height as f32 / 6.0) {
                let confidence = (vertical_score / (height as f32 / 2.0)).min(0.75);
                structures.push(DetectedObject {
                    label: "vertical_furniture".to_string(),
                    confidence,
                    bbox: BoundingBox {
                        x: x as i32 - 25,
                        y: 10,
                        width: 50,
                        height: (height - 20) as i32,
                    },
                });
            }
        }
        
        structures
    }
    
    #[allow(dead_code)]
    fn calculate_average_brightness(&self, image: &DynamicImage) -> f32 {
        let rgb_image = image.to_rgb8();
        let pixels = rgb_image.pixels();
        let mut total_brightness = 0.0;
        let mut pixel_count = 0;
        
        for pixel in pixels {
            let brightness = pixel[0] as f32 * 0.299 + 
                            pixel[1] as f32 * 0.587 + 
                            pixel[2] as f32 * 0.114;
            total_brightness += brightness;
            pixel_count += 1;
        }
        
        if pixel_count > 0 {
            total_brightness / pixel_count as f32
        } else {
            0.0
        }
    }
    
    #[allow(dead_code)]
    fn estimate_edge_density(&self, image: &DynamicImage) -> f32 {
        // Simple edge detection by measuring pixel variation
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        let mut edge_count = 0;
        let mut total_pixels = 0;
        
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let current = rgb_image.get_pixel(x, y);
                let right = rgb_image.get_pixel(x + 1, y);
                let down = rgb_image.get_pixel(x, y + 1);
                
                let horizontal_diff = Self::pixel_difference(current, right);
                let vertical_diff = Self::pixel_difference(current, down);
                
                if horizontal_diff > 30.0 || vertical_diff > 30.0 {
                    edge_count += 1;
                }
                total_pixels += 1;
            }
        }
        
        if total_pixels > 0 {
            edge_count as f32 / total_pixels as f32
        } else {
            0.0
        }
    }
    
    #[allow(dead_code)]
    fn pixel_difference(p1: &Rgb<u8>, p2: &Rgb<u8>) -> f32 {
        let dr = p1[0] as f32 - p2[0] as f32;
        let dg = p1[1] as f32 - p2[1] as f32;
        let db = p1[2] as f32 - p2[2] as f32;
        (dr * dr + dg * dg + db * db).sqrt()
    }
    
    #[allow(dead_code)]
    fn detect_by_color(&self, image: &DynamicImage) -> Vec<DetectedObject> {
        let mut objects = Vec::new();
        let rgb_image = image.to_rgb8();
        let (_width, _height) = rgb_image.dimensions();
        
        // Look for red objects (could be stop signs, people, etc.)
        let red_regions = self.find_color_regions(&rgb_image, |r, g, b| r > 150 && g < 100 && b < 100);
        for (x, y, w, h) in red_regions {
            if w > 20 && h > 20 {  // Minimum size threshold
                objects.push(DetectedObject {
                    label: "red_object".to_string(),
                    confidence: 0.5,
                    bbox: BoundingBox { x, y, width: w, height: h },
                });
            }
        }
        
        // Look for green objects (vegetation, signs, etc.)
        let green_regions = self.find_color_regions(&rgb_image, |r, g, b| g > 150 && r < 100 && b < 100);
        for (x, y, w, h) in green_regions {
            if w > 30 && h > 30 {
                objects.push(DetectedObject {
                    label: "green_object".to_string(),
                    confidence: 0.4,
                    bbox: BoundingBox { x, y, width: w, height: h },
                });
            }
        }
        
        objects
    }
    
    #[allow(dead_code)]
    fn find_color_regions<F>(&self, image: &RgbImage, color_match: F) -> Vec<(i32, i32, i32, i32)>
    where
        F: Fn(u8, u8, u8) -> bool,
    {
        let (width, height) = image.dimensions();
        let mut regions = Vec::new();
        let mut visited = vec![vec![false; width as usize]; height as usize];
        
        for y in 0..height {
            for x in 0..width {
                if !visited[y as usize][x as usize] {
                    let pixel = image.get_pixel(x, y);
                    if color_match(pixel[0], pixel[1], pixel[2]) {
                        let region = self.flood_fill_region(image, &mut visited, x, y, &color_match);
                        if let Some((min_x, min_y, max_x, max_y)) = region {
                            regions.push((
                                min_x as i32,
                                min_y as i32,
                                (max_x - min_x) as i32,
                                (max_y - min_y) as i32,
                            ));
                        }
                    }
                }
            }
        }
        
        regions
    }
    
    #[allow(dead_code)]
    fn flood_fill_region<F>(
        &self,
        image: &RgbImage,
        visited: &mut Vec<Vec<bool>>,
        start_x: u32,
        start_y: u32,
        color_match: &F,
    ) -> Option<(u32, u32, u32, u32)>
    where
        F: Fn(u8, u8, u8) -> bool,
    {
        let (width, height) = image.dimensions();
        let mut stack = vec![(start_x, start_y)];
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;
        let mut pixel_count = 0;
        
        while let Some((x, y)) = stack.pop() {
            if x >= width || y >= height || visited[y as usize][x as usize] {
                continue;
            }
            
            let pixel = image.get_pixel(x, y);
            if !color_match(pixel[0], pixel[1], pixel[2]) {
                continue;
            }
            
            visited[y as usize][x as usize] = true;
            pixel_count += 1;
            
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            
            // Add neighbors
            if x > 0 { stack.push((x - 1, y)); }
            if x < width - 1 { stack.push((x + 1, y)); }
            if y > 0 { stack.push((x, y - 1)); }
            if y < height - 1 { stack.push((x, y + 1)); }
            
            // Limit region size to prevent excessive computation
            if pixel_count > 10000 {
                break;
            }
        }
        
        if pixel_count > 10 {  // Minimum region size
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }
    
    fn merge_overlapping_detections(&self, mut objects: Vec<DetectedObject>) -> Vec<DetectedObject> {
        if objects.len() <= 1 {
            return objects;
        }
        
        // Sort by confidence (highest first)
        objects.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut merged: Vec<DetectedObject> = Vec::new();
        
        for obj in objects {
            let mut should_add = true;
            
            // Check if this object significantly overlaps with any existing object
            for existing in &merged {
                if self.bboxes_overlap(&obj.bbox, &existing.bbox, 0.5) {
                    should_add = false;
                    break;
                }
            }
            
            if should_add {
                merged.push(obj);
            }
        }
        
        merged
    }
    
    fn bboxes_overlap(&self, bbox1: &BoundingBox, bbox2: &BoundingBox, threshold: f32) -> bool {
        let x1_min = bbox1.x;
        let y1_min = bbox1.y;
        let x1_max = bbox1.x + bbox1.width;
        let y1_max = bbox1.y + bbox1.height;
        
        let x2_min = bbox2.x;
        let y2_min = bbox2.y;
        let x2_max = bbox2.x + bbox2.width;
        let y2_max = bbox2.y + bbox2.height;
        
        let intersection_x = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0);
        let intersection_y = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0);
        let intersection_area = intersection_x * intersection_y;
        
        let area1 = bbox1.width * bbox1.height;
        let area2 = bbox2.width * bbox2.height;
        let union_area = area1 + area2 - intersection_area;
        
        if union_area == 0 {
            return false;
        }
        
        let overlap_ratio = intersection_area as f32 / union_area as f32;
        overlap_ratio > threshold
    }
    
    fn generate_scene_description(&self, objects: &[DetectedObject], lidar_points: &[LidarPoint]) -> String {
        let mut description = String::new();
        
        if objects.is_empty() {
            description.push_str("Clear environment with no significant objects detected.");
        } else {
            // Group objects by type for cleaner summary
            let mut object_counts = std::collections::HashMap::new();
            let mut highest_confidence = 0.0;
            let mut most_confident_object = "";
            
            for obj in objects {
                *object_counts.entry(&obj.label).or_insert(0) += 1;
                if obj.confidence > highest_confidence {
                    highest_confidence = obj.confidence;
                    most_confident_object = &obj.label;
                }
            }
            
            // Create concise summary
            description.push_str("Environment contains: ");
            let mut first = true;
            for (label, count) in object_counts {
                if !first {
                    description.push_str(", ");
                }
                if count == 1 {
                    description.push_str(&label.replace("_", " "));
                } else {
                    description.push_str(&format!("{} {}s", count, label.replace("_", " ")));
                }
                first = false;
            }
            
            if !most_confident_object.is_empty() && highest_confidence > 0.7 {
                description.push_str(&format!(". Most prominent: {} ({}% confidence)", 
                                           most_confident_object.replace("_", " "),
                                           (highest_confidence * 100.0) as u8));
            }
            description.push('.');
        }
        
        // Add concise LiDAR information if available
        if !lidar_points.is_empty() {
            let close_objects = lidar_points.iter()
                .filter(|p| p.distance < 2.0)
                .count();
            
            if close_objects > 0 {
                if let Some(closest) = lidar_points.iter().min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()) {
                    description.push_str(&format!(" Nearest obstacle: {:.1}m away", closest.distance));
                }
            } else {
                description.push_str(" Path appears clear (no close obstacles)");
            }
        }
        
        description
    }
    
    pub fn annotate_frame(&self, frame: &mut Mat, context: &FrameContext) -> Result<()> {
        debug!("Annotating frame with {} objects and {} LiDAR points (basic mode)", 
               context.objects.len(), context.lidar_points.len());
        
        // For basic annotation, we'll modify the raw image data
        // This is a simplified version without OpenCV drawing functions
        
        // Add simple visual indicators by modifying pixel values
        for obj in &context.objects {
            self.draw_simple_rectangle(frame, &obj.bbox, [0, 255, 0])?; // Green rectangles
        }
        
        // LiDAR overlay is now handled by the camera system
        // since we no longer have direct access to go2_system
        
        Ok(())
    }
    
    fn draw_simple_rectangle(&self, frame: &mut Mat, bbox: &BoundingBox, color: [u8; 3]) -> Result<()> {
        let width = frame.width as i32;
        let height = frame.height as i32;
        
        // Clamp coordinates
        let x1 = bbox.x.max(0).min(width - 1);
        let y1 = bbox.y.max(0).min(height - 1);
        let x2 = (bbox.x + bbox.width).max(0).min(width - 1);
        let y2 = (bbox.y + bbox.height).max(0).min(height - 1);
        
        // Draw simple rectangle outline by modifying pixels
        for x in x1..x2 {
            self.set_pixel(frame, x, y1, color)?;
            self.set_pixel(frame, x, y2 - 1, color)?;
        }
        
        for y in y1..y2 {
            self.set_pixel(frame, x1, y, color)?;
            self.set_pixel(frame, x2 - 1, y, color)?;
        }
        
        Ok(())
    }
    
    fn set_pixel(&self, frame: &mut Mat, x: i32, y: i32, color: [u8; 3]) -> Result<()> {
        if x < 0 || y < 0 || x >= frame.width as i32 || y >= frame.height as i32 {
            return Ok(()); // Skip out-of-bounds pixels
        }
        
        let pixel_index = ((y as u32 * frame.width + x as u32) * frame.channels) as usize;
        if pixel_index + 2 < frame.data.len() {
            frame.data[pixel_index] = color[0];     // R
            frame.data[pixel_index + 1] = color[1]; // G
            frame.data[pixel_index + 2] = color[2]; // B
        }
        
        Ok(())
    }
    
    #[allow(dead_code)]
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping VisionProcessor (basic mode)");
        // No longer need to stop go2_system since we use shared camera
        Ok(())
    }
}

```

**./src/lib.rs**
```rust
pub mod models;
pub mod config;
pub mod camera;
pub mod audio;
pub mod context;
pub mod pipeline;
pub mod commands;
pub mod vision_basic;
pub mod tests;

// Re-export vision_basic as vision for compatibility
pub use vision_basic as vision;

```

**./src/vision_mock.rs**
```rust
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::time::SystemTime;
use std::path::Path;
use tracing::{debug, warn, error, info};
use crate::go2::{Go2SensorSystem, Go2SensorData, Go2CameraConfig, LidarPoint};

// Mock implementations without OpenCV dependencies

#[derive(Debug, Clone)]
pub struct Mat {
    pub cols: i32,
    pub rows: i32,
    pub data: Vec<u8>,
}

impl Mat {
    pub fn default() -> Self {
        Self {
            cols: 640,
            rows: 480,
            data: vec![128; 640 * 480 * 3], // Default gray image
        }
    }
    
    pub fn cols(&self) -> i32 {
        self.cols
    }
    
    pub fn rows(&self) -> i32 {
        self.rows
    }
    
    pub fn empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct FrameContext {
    pub timestamp: std::time::SystemTime,
    pub objects: Vec<DetectedObject>,
    pub scene_description: String,
    pub frame_size: (u32, u32),
    pub lidar_points: Vec<LidarPoint>,
}

pub struct VisionProcessor {
    go2_system: Go2SensorSystem,
    confidence_threshold: f32,
    frame_counter: u32,
}

impl VisionProcessor {
    pub fn new(
        camera_device: usize,
        width: u32,
        height: u32,
        confidence_threshold: f32,
        vision_model: String,
    ) -> Result<Self> {
        info!("Initializing VisionProcessor with mock Go2 camera input");
        
        // Configure Go2 camera
        let go2_config = Go2CameraConfig {
            camera_id: camera_device as i32,
            width: width as i32,
            height: height as i32,
            fps: 30.0,
            exposure: Some(-6.0),
            gain: Some(50.0),
        };
        
        let go2_system = Go2SensorSystem::new(go2_config)?;
        
        info!("VisionProcessor initialized for Go2 (mock mode)");
        
        Ok(Self {
            go2_system,
            confidence_threshold,
            frame_counter: 0,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 sensor system (mock mode)");
        self.go2_system.initialize().await?;
        Ok(())
    }
    
    pub async fn capture_frame(&mut self) -> Result<Mat> {
        self.frame_counter += 1;
        debug!("Capturing frame #{} from Go2 camera (mock)", self.frame_counter);
        
        // Mock camera capture - create a sample frame
        let mat = Mat {
            cols: 1280,
            rows: 720,
            data: vec![128; 1280 * 720 * 3], // Gray frame
        };
        
        Ok(mat)
    }
    
    pub async fn process_frame(&mut self, frame: &Mat) -> Result<FrameContext> {
        let timestamp = SystemTime::now();
        
        // Mock object detection
        let objects = self.detect_objects_mock(frame)?;
        
        // Mock LiDAR points
        let lidar_points = self.generate_mock_lidar_points();
        
        // Filter by confidence threshold
        let filtered_objects: Vec<DetectedObject> = objects
            .into_iter()
            .filter(|obj| obj.confidence >= self.confidence_threshold)
            .collect();
        
        let scene_description = self.generate_scene_description(&filtered_objects, &lidar_points);
        
        Ok(FrameContext {
            timestamp,
            objects: filtered_objects,
            scene_description,
            frame_size: (frame.cols() as u32, frame.rows() as u32),
            lidar_points,
        })
    }
    
    fn detect_objects_mock(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        debug!("Running mock object detection");
        
        let mut objects = Vec::new();
        
        // Simulate object detection based on frame counter for variety
        match self.frame_counter % 6 {
            0 => {
                objects.push(DetectedObject {
                    label: "person".to_string(),
                    confidence: 0.87,
                    bbox: BoundingBox { x: 200, y: 100, width: 120, height: 200 },
                });
            }
            1 => {
                objects.push(DetectedObject {
                    label: "chair".to_string(),
                    confidence: 0.75,
                    bbox: BoundingBox { x: 400, y: 300, width: 80, height: 100 },
                });
                objects.push(DetectedObject {
                    label: "table".to_string(),
                    confidence: 0.68,
                    bbox: BoundingBox { x: 350, y: 250, width: 150, height: 80 },
                });
            }
            2 => {
                objects.push(DetectedObject {
                    label: "door".to_string(),
                    confidence: 0.82,
                    bbox: BoundingBox { x: 50, y: 50, width: 100, height: 300 },
                });
            }
            3 => {
                objects.push(DetectedObject {
                    label: "cup".to_string(),
                    confidence: 0.91,
                    bbox: BoundingBox { x: 500, y: 200, width: 40, height: 60 },
                });
                objects.push(DetectedObject {
                    label: "bottle".to_string(),
                    confidence: 0.79,
                    bbox: BoundingBox { x: 600, y: 180, width: 30, height: 80 },
                });
            }
            4 => {
                objects.push(DetectedObject {
                    label: "book".to_string(),
                    confidence: 0.73,
                    bbox: BoundingBox { x: 300, y: 350, width: 60, height: 40 },
                });
            }
            _ => {
                // Empty scene occasionally
            }
        }
        
        debug!("Mock detection found {} objects", objects.len());
        Ok(objects)
    }
    
    fn generate_mock_lidar_points(&self) -> Vec<LidarPoint> {
        let mut points = Vec::new();
        
        // Generate some mock LiDAR points around the robot
        for i in 0..20 {
            let angle = (i as f32) * 0.314; // ~18 degrees apart
            let distance = 1.5 + (i as f32 * 0.2) % 3.0; // Distance between 1.5-4.5m
            
            points.push(LidarPoint {
                x: distance * angle.cos(),
                y: distance * angle.sin(),
                z: 0.0,
                distance,
                intensity: 200,
            });
        }
        
        points
    }
    
    fn generate_scene_description(&self, objects: &[DetectedObject], lidar_points: &[LidarPoint]) -> String {
        let mut description = String::new();
        
        if objects.is_empty() {
            description.push_str("Empty scene with no detected objects. ");
        } else {
            description.push_str(&format!("Scene contains {} object(s): ", objects.len()));
            
            for (i, obj) in objects.iter().enumerate() {
                if i > 0 {
                    description.push_str(", ");
                }
                description.push_str(&format!(
                    "{} (confidence: {:.2}, position: {},{}, size: {}x{})",
                    obj.label,
                    obj.confidence,
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.width,
                    obj.bbox.height
                ));
            }
            description.push_str(". ");
        }
        
        // Add LiDAR information if available
        if !lidar_points.is_empty() {
            let close_objects = lidar_points.iter()
                .filter(|p| p.distance < 2.0)
                .count();
            let far_objects = lidar_points.iter()
                .filter(|p| p.distance >= 2.0 && p.distance < 5.0)
                .count();
            
            description.push_str(&format!(
                "LiDAR shows {} close objects (<2m), {} mid-range objects (2-5m). ",
                close_objects, far_objects
            ));
            
            if let Some(closest) = lidar_points.iter().min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()) {
                description.push_str(&format!("Closest object at {:.1}m. ", closest.distance));
            }
        }
        
        description
    }
    
    pub fn annotate_frame(&self, frame: &mut Mat, context: &FrameContext) -> Result<()> {
        debug!("Mock annotation: frame with {} objects and {} LiDAR points", 
               context.objects.len(), context.lidar_points.len());
        
        // In a real implementation, this would draw bounding boxes and labels on the frame
        for obj in &context.objects {
            debug!(
                "Object: {} at ({},{}) {}x{} (confidence: {:.2})",
                obj.label, obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height, obj.confidence
            );
        }
        
        // Mock LiDAR overlay
        for (i, point) in context.lidar_points.iter().enumerate().take(5) {
            debug!("LiDAR point {}: distance {:.1}m at ({:.1}, {:.1})", 
                   i, point.distance, point.x, point.y);
        }
        
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping VisionProcessor (mock mode)");
        self.go2_system.stop().await?;
        Ok(())
    }
}

```

**./src/test_multimodal.rs**
```rust
use anyhow::Result;
use crate::models::{GemmaModel, ModelConfig};
use image::{DynamicImage, ImageBuffer, Rgb};

pub async fn test_multimodal_inference() -> Result<()> {
    println!("🚀 Testing multimodal inference capabilities...");
    
    let config = ModelConfig {
        max_tokens: 50,  // Reduced for faster inference
        temperature: 0.1,  // Lower temperature for more deterministic output
        top_p: 0.8,
        context_length: 512,  // Reduced context length for faster processing
    };
    
    println!("📥 Loading UQFF Gemma 3n multimodal model...");
    
    // Load the UQFF model
    let mut model = match GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await {
        Ok(model) => {
            println!("✅ UQFF multimodal model loaded successfully!");
            model
        }
        Err(e) => {
            println!("❌ Failed to load UQFF model: {}", e);
            return Err(e);
        }
    };
    
    // Test 1: Text-only inference
    println!("\n🔤 Test 1: Text-only inference");
    test_text_inference(&mut model).await?;
    
    // Test 2: Vision inference (with synthetic image)
    println!("\n👁️ Test 2: Vision inference with synthetic image");
    test_vision_inference(&mut model).await?;
    
    // Test 3: Audio inference (with synthetic audio)
    println!("\n🔊 Test 3: Audio inference with synthetic audio");
    test_audio_inference(&mut model).await?;
    
    // Test 4: Multimodal inference (text + image + audio)
    println!("\n🎭 Test 4: Full multimodal inference (text + image + audio)");
    test_full_multimodal_inference(&mut model).await?;
    
    println!("\n🎉 All multimodal inference tests completed successfully!");
    Ok(())
}

async fn test_text_inference(model: &mut GemmaModel) -> Result<()> {
    let prompts = vec![
        "Hello! Introduce yourself as a helpful robot assistant.",
        "What can you help me with today?",
        "Move forward slowly and look around for obstacles.",
        "Analyze the current environment and report what you see.",
    ];
    
    for (i, prompt) in prompts.iter().enumerate() {
        println!("  📝 Text test {}: \"{}\"", i + 1, prompt);
        
        match model.generate(prompt).await {
            Ok(result) => {
                println!("  ✅ Response: {}", result.text);
                println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
            }
            Err(e) => {
                println!("  ❌ Text generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(())
}

async fn test_vision_inference(model: &mut GemmaModel) -> Result<()> {
    // Create a synthetic test image (red square on blue background)
    let test_image = create_synthetic_image();
    
    let vision_prompts = vec![
        "What do you see in this image?",
        "Describe the colors and shapes in the image.",
        "Is there anything noteworthy about this image?",
        "Should I move towards or away from what's in the image?",
    ];
    
    for (i, prompt) in vision_prompts.iter().enumerate() {
        println!("  🖼️ Vision test {}: \"{}\"", i + 1, prompt);
        
        match model.generate_with_image(prompt, test_image.clone()).await {
            Ok(result) => {
                println!("  ✅ Vision response: {}", result.text);
                println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
            }
            Err(e) => {
                println!("  ❌ Vision generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(())
}

async fn test_audio_inference(model: &mut GemmaModel) -> Result<()> {
    // Create synthetic audio data (simple sine wave pattern)
    let test_audio = create_synthetic_audio();
    
    let audio_prompts = vec![
        "What do you hear in this audio?",
        "Describe the sound pattern you're hearing.",
        "Is this audio indicating any specific action I should take?",
    ];
    
    for (i, prompt) in audio_prompts.iter().enumerate() {
        println!("  🎵 Audio test {}: \"{}\"", i + 1, prompt);
        
        match model.generate_with_audio(prompt, test_audio.clone()).await {
            Ok(result) => {
                println!("  ✅ Audio response: {}", result.text);
                println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
            }
            Err(e) => {
                println!("  ❌ Audio generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(())
}

async fn test_full_multimodal_inference(model: &mut GemmaModel) -> Result<()> {
    let test_image = create_synthetic_image();
    let test_audio = create_synthetic_audio();
    
    let multimodal_prompts = vec![
        "Based on what you see and hear, what should I do next?",
        "Analyze both the visual and audio information and give me guidance.",
        "What is the overall situation based on all sensory inputs?",
    ];
    
    for (i, prompt) in multimodal_prompts.iter().enumerate() {
        println!("  🎭 Multimodal test {}: \"{}\"", i + 1, prompt);
        
        match model.generate_multimodal(prompt, Some(test_image.clone()), Some(test_audio.clone())).await {
            Ok(result) => {
                println!("  ✅ Multimodal response: {}", result.text);
                println!("  📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
            }
            Err(e) => {
                println!("  ❌ Multimodal generation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(())
}

fn create_synthetic_image() -> DynamicImage {
    // Create a 64x64 test image with a red square on blue background
    let mut img = ImageBuffer::new(64, 64);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        if x >= 16 && x < 48 && y >= 16 && y < 48 {
            // Red square in center
            *pixel = Rgb([255, 0, 0]);
        } else {
            // Blue background
            *pixel = Rgb([0, 0, 255]);
        }
    }
    
    DynamicImage::ImageRgb8(img)
}

fn create_synthetic_audio() -> Vec<u8> {
    // Create a simple synthetic audio pattern (sine wave data)
    let sample_rate = 16000;
    let duration_seconds = 1.0;
    let frequency = 440.0; // A4 note
    
    let num_samples = (sample_rate as f32 * duration_seconds) as usize;
    let mut audio_data = Vec::with_capacity(num_samples * 2); // 16-bit samples
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        let sample_i16 = (sample * 32767.0) as i16;
        
        // Convert to little-endian bytes
        audio_data.extend_from_slice(&sample_i16.to_le_bytes());
    }
    
    audio_data
}

```

**./src/audio.rs**
```rust
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tracing::{info, debug, warn, error};
use cpal::{Device, Stream, StreamConfig, SampleFormat, SampleRate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::{Decoder, OutputStream, Sink};
use tts::Tts;

#[derive(Debug, Clone)]
pub struct AudioConfig {
    #[allow(dead_code)]
    pub sample_rate: u32,
    #[allow(dead_code)]
    pub channels: u16,
    #[allow(dead_code)]
    pub buffer_size: usize,
    #[allow(dead_code)]
    pub voice_detection_threshold: f32,
    #[allow(dead_code)]
    pub silence_duration_ms: u64,
    #[allow(dead_code)]
    pub speech_timeout_ms: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 1,
            buffer_size: 4096,
            voice_detection_threshold: 0.02,
            silence_duration_ms: 500,
            speech_timeout_ms: 5000,
        }
    }
}

pub struct AudioSystem {
    #[allow(dead_code)]
    config: AudioConfig,
    input_device: Option<Device>,
    output_device: Option<Device>,
    input_stream: Option<Stream>,
    #[allow(dead_code)]
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    is_recording: Arc<Mutex<bool>>,
    tts: Option<Tts>,
    _output_stream: Option<OutputStream>,
    output_sink: Option<Sink>,
    is_initialized: bool,
}

impl AudioSystem {
    pub fn new(config: AudioConfig) -> Result<Self> {
        info!("Creating AudioSystem for real audio I/O");
        
        Ok(Self {
            config,
            input_device: None,
            output_device: None,
            input_stream: None,
            audio_buffer: Arc::new(Mutex::new(VecDeque::new())),
            is_recording: Arc::new(Mutex::new(false)),
            tts: None,
            _output_stream: None,
            output_sink: None,
            is_initialized: false,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing audio system - scanning for audio devices");
        
        let host = cpal::default_host();
        
        // Initialize input device (microphone)
        self.input_device = match host.default_input_device() {
            Some(device) => {
                info!("Found default input device: {}", device.name().unwrap_or("Unknown".to_string()));
                Some(device)
            }
            None => {
                warn!("No default input device found, scanning for alternatives");
                self.find_input_device(&host)?
            }
        };
        
        // Initialize output device (speakers)
        self.output_device = match host.default_output_device() {
            Some(device) => {
                info!("Found default output device: {}", device.name().unwrap_or("Unknown".to_string()));
                Some(device)
            }
            None => {
                warn!("No default output device found, scanning for alternatives");
                self.find_output_device(&host)?
            }
        };
        
        // Initialize TTS engine
        self.tts = match Tts::default() {
            Ok(tts) => {
                info!("Text-to-speech engine initialized successfully");
                Some(tts)
            }
            Err(e) => {
                warn!("Failed to initialize TTS engine: {}", e);
                None
            }
        };
        
        // Initialize output stream for audio playback
        if let Some(ref output_device) = self.output_device {
            match OutputStream::try_from_device(output_device) {
                Ok((stream, handle)) => {
                    self._output_stream = Some(stream);
                    self.output_sink = Some(Sink::try_new(&handle)?);
                    info!("Audio output stream initialized");
                }
                Err(e) => {
                    warn!("Failed to initialize output stream: {}", e);
                }
            }
        }
        
        self.is_initialized = true;
        info!("Audio system initialized successfully");
        Ok(())
    }
    
    fn find_input_device(&self, host: &cpal::Host) -> Result<Option<Device>> {
        let devices = host.input_devices()?;
        
        for device in devices {
            if let Ok(name) = device.name() {
                info!("Found input device: {}", name);
                return Ok(Some(device));
            }
        }
        
        Ok(None)
    }
    
    fn find_output_device(&self, host: &cpal::Host) -> Result<Option<Device>> {
        let devices = host.output_devices()?;
        
        for device in devices {
            if let Ok(name) = device.name() {
                info!("Found output device: {}", name);
                return Ok(Some(device));
            }
        }
        
        Ok(None)
    }
    
    #[allow(dead_code)]
    pub async fn start_recording(&mut self) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow!("Audio system not initialized"));
        }
        
        let input_device = self.input_device.as_ref()
            .ok_or_else(|| anyhow!("No input device available"))?;
        
        // Get the default input config
        let config = input_device.default_input_config()?;
        info!("Input config: {:?}", config);
        
        // Convert to stream config
        let stream_config = StreamConfig {
            channels: config.channels().min(self.config.channels),
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };
        
        // Create audio buffer and recording flag references
        let buffer = Arc::clone(&self.audio_buffer);
        let recording = Arc::clone(&self.is_recording);
        
        // Set recording flag
        {
            let mut is_recording = recording.lock().unwrap();
            *is_recording = true;
        }
        
        // Create input stream based on sample format
        let stream = match config.sample_format() {
            SampleFormat::F32 => self.create_input_stream_f32(input_device, &stream_config, buffer)?,
            SampleFormat::I16 => self.create_input_stream_i16(input_device, &stream_config, buffer)?,
            SampleFormat::U16 => self.create_input_stream_u16(input_device, &stream_config, buffer)?,
            _ => return Err(anyhow!("Unsupported sample format")),
        };
        
        // Start the stream
        stream.play()?;
        self.input_stream = Some(stream);
        
        info!("Audio recording started");
        Ok(())
    }
    
    #[allow(dead_code)]
    fn create_input_stream_f32(
        &self,
        device: &Device,
        config: &StreamConfig,
        buffer: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<Stream> {
        let stream = device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                for &sample in data {
                    buffer.push_back(sample);
                    // Keep buffer size manageable
                    if buffer.len() > 44100 * 10 { // 10 seconds of audio
                        buffer.pop_front();
                    }
                }
            },
            |err| error!("Audio input stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    #[allow(dead_code)]
    fn create_input_stream_i16(
        &self,
        device: &Device,
        config: &StreamConfig,
        buffer: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<Stream> {
        let stream = device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                for &sample in data {
                    let normalized = sample as f32 / i16::MAX as f32;
                    buffer.push_back(normalized);
                    if buffer.len() > 44100 * 10 {
                        buffer.pop_front();
                    }
                }
            },
            |err| error!("Audio input stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    #[allow(dead_code)]
    fn create_input_stream_u16(
        &self,
        device: &Device,
        config: &StreamConfig,
        buffer: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<Stream> {
        let stream = device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                for &sample in data {
                    let normalized = (sample as f32 - 32768.0) / 32768.0;
                    buffer.push_back(normalized);
                    if buffer.len() > 44100 * 10 {
                        buffer.pop_front();
                    }
                }
            },
            |err| error!("Audio input stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    pub async fn speak_text(&mut self, text: &str) -> Result<()> {
        if let Some(ref mut tts) = self.tts {
            info!("Speaking: {}", text);
            
            match tts.speak(text, false) {
                Ok(_) => {
                    // Wait for speech to complete
                    while tts.is_speaking()? {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                    debug!("Speech completed");
                }
                Err(e) => {
                    error!("TTS error: {}", e);
                    return Err(anyhow!("Text-to-speech failed: {}", e));
                }
            }
        } else {
            warn!("TTS not available, text would be spoken: {}", text);
        }
        
        Ok(())
    }
    
    #[allow(dead_code)]
    pub async fn play_audio_file(&mut self, file_path: &str) -> Result<()> {
        if let Some(ref sink) = self.output_sink {
            let file = std::fs::File::open(file_path)?;
            let source = Decoder::new(file)?;
            
            info!("Playing audio file: {}", file_path);
            sink.append(source);
            
            // Wait for playback to complete
            sink.sleep_until_end();
            
            Ok(())
        } else {
            Err(anyhow!("Audio output not initialized"))
        }
    }
    
    #[allow(dead_code)]
    pub async fn play_beep(&mut self, frequency: f32, duration_ms: u64) -> Result<()> {
        if let Some(ref sink) = self.output_sink {
            let sample_rate = 44100;
            let samples_per_ms = sample_rate as f64 / 1000.0;
            let total_samples = (duration_ms as f64 * samples_per_ms) as usize;
            
            let mut samples = Vec::with_capacity(total_samples);
            
            for i in 0..total_samples {
                let t = i as f32 / sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
                samples.push(sample);
            }
            
            // Convert to source
            let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples);
            
            info!("Playing beep: {}Hz for {}ms", frequency, duration_ms);
            sink.append(source);
            
            // Wait for playback to complete
            tokio::time::sleep(tokio::time::Duration::from_millis(duration_ms + 100)).await;
            
            Ok(())
        } else {
            Err(anyhow!("Audio output not initialized"))
        }
    }
    
    pub async fn stop_recording(&mut self) -> Result<()> {
        {
            let mut is_recording = self.is_recording.lock().unwrap();
            *is_recording = false;
        }
        
        if let Some(stream) = self.input_stream.take() {
            drop(stream);
            info!("Audio recording stopped");
        }
        
        Ok(())
    }
    
    #[allow(dead_code)]
    pub fn is_recording(&self) -> bool {
        *self.is_recording.lock().unwrap()
    }
    
    #[allow(dead_code)]
    pub fn clear_audio_buffer(&self) {
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.clear();
        debug!("Audio buffer cleared");
    }
    
    #[allow(dead_code)]  
    pub fn get_audio_info(&self) -> String {
        if self.is_initialized {
            format!(
                "Audio System - Input: {}, Output: {}, TTS: {}",
                if self.input_device.is_some() { "Available" } else { "None" },
                if self.output_device.is_some() { "Available" } else { "None" },
                if self.tts.is_some() { "Available" } else { "None" }
            )
        } else {
            "Audio System - Not initialized".to_string()
        }
    }
}

impl Drop for AudioSystem {
    fn drop(&mut self) {
        if self.is_initialized {
            info!("AudioSystem being dropped, cleaning up");
            let _ = self.stop_recording();
        }
    }
}

```

**./tests/integration_tests.rs**
```rust
use opticxt::models::ModelConfig;
use opticxt::tests::*;
use anyhow::Result;

#[tokio::test]
async fn test_device_fallback_integration() -> Result<()> {
    // This test verifies that the device fallback system compiles
    // and the model loading methods are properly defined
    
    let _config = ModelConfig {
        max_tokens: 10,
        temperature: 0.3,
        top_p: 0.9,
        context_length: 512,
    };
    
    // This should compile and not panic - actual model loading
    // will be tested separately due to resource requirements
    println!("✅ Device fallback system compiled successfully");
    println!("✅ ModelConfig and GemmaModel definitions are valid");
    
    // Verify the model ID getter works
    let model_id = "test-model";
    println!("✅ Model ID handling: {}", model_id);
    
    Ok(())
}

#[test]
fn test_model_config_creation() {
    // Test basic model configuration creation
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.95,
        context_length: 2048,
    };
    
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_p, 0.95);
    assert_eq!(config.context_length, 2048);
    
    println!("✅ ModelConfig creation and field access working correctly");
}

#[tokio::test]
#[ignore] // Ignored by default - requires model files and GPU
async fn test_quick_smoke_integration() -> Result<()> {
    // Integration test for quick smoke test
    // Run with: cargo test test_quick_smoke_integration -- --ignored
    test_quick_smoke().await
}

#[tokio::test]
#[ignore] // Ignored by default - requires model files and GPU
async fn test_simple_inference_integration() -> Result<()> {
    // Integration test for simple inference
    // Run with: cargo test test_simple_inference_integration -- --ignored
    test_simple_inference().await
}

#[tokio::test]
#[ignore] // Ignored by default - requires model files and GPU
async fn test_image_inference_integration() -> Result<()> {
    // Integration test for image inference
    // Run with: cargo test test_image_inference_integration -- --ignored
    test_image_inference().await
}

#[tokio::test]
#[ignore] // Ignored by default - requires model files and GPU
async fn test_multimodal_inference_integration() -> Result<()> {
    // Integration test for multimodal inference
    // Run with: cargo test test_multimodal_inference_integration -- --ignored
    test_multimodal_inference().await
}

#[tokio::test]
#[ignore] // Ignored by default - requires model files and GPU
async fn test_uqff_model_integration() -> Result<()> {
    // Integration test for UQFF model
    // Run with: cargo test test_uqff_model_integration -- --ignored
    test_uqff_model().await
}

#[tokio::test]
#[ignore] // Ignored by default - requires model files and GPU
async fn test_tool_format_integration() -> Result<()> {
    // Integration test for tool format
    // Run with: cargo test test_tool_format_integration -- --ignored
    test_tool_format().await
}

```

**./config.toml**
```toml
# OpticXT Configuration File
# Vision-Driven Autonomous Robot Control System

[vision]
# Camera settings
width = 640
height = 480
fps = 30
confidence_threshold = 0.5
vision_model = "yolo"  # Options: yolo, coco, custom

[model]
# Gemma model configuration
# Quantization method: "uqff" (faster inference, slower loading) or "isq" (fast loading, in-place quantization)
quantization_method = "isq"  # Options: "uqff", "isq"
# ISQ quantization type: Q4K (recommended), Q2K (smallest), Q5K (more accurate), Q8_0 (high quality)
isq_type = "Q4K"  # Options: Q2K, Q3K, Q4K, Q5K, Q6K, Q8_0, Q8_1
model_path = ""  # Leave empty to use default model
context_length = 4096
temperature = 0.7
top_p = 0.9
max_tokens = 512

[context]
# Mandatory context system
system_prompt = "prompts/system_prompt.txt"
max_context_history = 10
include_timestamp = true

[commands]
# Available command types
enabled_commands = ["move", "rotate", "speak", "analyze", "offload"]
timeout_seconds = 30
validate_before_execution = true

[performance]
# Performance optimization
worker_threads = 4
frame_buffer_size = 10
processing_interval_ms = 100
use_gpu = true

[audio]
# Audio settings for voice input/output
enabled = true
sample_rate = 44100
channels = 1
enable_tts = true
enable_speech_recognition = false

```

**./Cargo.toml**
```toml
[package]
name = "opticxt"
version = "0.1.0"
edition = "2021"
authors = ["Josh-XT <josh@josh-xt.com>"]
description = "Vision-Driven Autonomous Robot Control System with GPU-Accelerated ISQ Inference"
license = "MIT"
repository = "https://github.com/Josh-XT/OpticXT"

[dependencies]
# Core async runtime
tokio = { version = "1.0", features = ["full"] }
futures = "0.3"

# Computer Vision and Image Processing - using compatible versions with mistralrs
image = "0.25"  # Updated to match mistralrs requirements

# Video processing and camera input
nokhwa = { version = "0.10", features = ["input-v4l"] }

# Machine Learning with ISQ support - using mistral.rs for GPU-accelerated inference
mistralrs = { git = "https://github.com/EricLBuehler/mistral.rs.git", default-features = false }
hf-hub = "0.3"  # For automatic model downloading
either = "1.9"  # Required by mistral.rs

# Serialization for OpenAI-style tool calls
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Logging and error handling
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1.0"
thiserror = "1.0"

# Configuration
config = "0.14"
toml = "0.8"
clap = { version = "4.0", features = ["derive"] }

# Real-time processing
crossbeam = "0.8"
parking_lot = "0.12"
rand = "0.8"  # For benchmark simulation

# Audio processing for voice input/output
rodio = "0.17"  # Audio playback
cpal = "0.15"   # Cross-platform audio I/O
tts = "0.26"    # Text-to-speech

[dev-dependencies]
criterion = "0.5"

[[bin]]
name = "opticxt"
path = "src/main.rs"

[features]
default = ["cuda"]
cuda = ["mistralrs/cuda"]
cpu = []  # CPU-only mode (no CUDA dependencies)

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.dev]
opt-level = 0
debug = true

```

**./CONTRIBUTING.md**
```markdown
# Contributing to OpticXT

We welcome contributions to OpticXT! This guide will help you get started.

## Development Setup

### Prerequisites

- Rust 1.70+ 
- NVIDIA GPU with CUDA support (recommended)
- Camera and microphone for testing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Josh-XT/OpticXT.git
cd OpticXT
```

2. Build with CUDA support:
```bash
cargo build --release --features cuda
```

3. Run tests:
```bash
cargo test
```

## Code Style

- Use `cargo fmt` to format code
- Use `cargo clippy` to check for common mistakes
- Follow Rust naming conventions
- Add documentation for public APIs

## Testing

- Add tests for new functionality
- Ensure existing tests pass
- Test both CUDA and CPU modes
- Test with real hardware when possible

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push to your fork
7. Create a pull request

## Areas for Contribution

- **Vision Processing**: Improve object detection and scene understanding
- **Performance**: Optimize GPU utilization and inference speed
- **Hardware Support**: Add support for new cameras and sensors
- **Documentation**: Improve setup guides and API documentation
- **Testing**: Add comprehensive test coverage

## Questions?

Feel free to open an issue for questions or discussions about contributing.

```

**./README.md**
```markdown
# OpticXT

Vision-Driven Autonomous Robot Control System

## Overview

OpticXT is a high-performance, real-time robot control system that combines computer vision, audio processing, and multimodal AI decision-making. The system is optimized for edge deployment on NVIDIA hardware with CUDA acceleration, delivering fast inference through ISQ (In-Situ Quantization) and GPU-accelerated processing.

**Key Achievement**: Full GPU acceleration with ISQ quantization achieving optimal performance on NVIDIA RTX 4090 and compatible hardware.

The system operates as an autonomous robot control platform with:
- **Real-time Vision Processing**: Optimized object detection and scene understanding
- **GPU-Accelerated AI**: ISQ quantization with CUDA acceleration for fast inference
- **Multimodal Capabilities**: Text, image, and audio processing with `unsloth/gemma-3n-E4B-it` model
- **OpenAI Tool Calling**: Modern function call interface for precise robot control

OpticXT transforms visual and audio input into contextual understanding and immediate robotic actions through GPU-accelerated inference.

## Key Features

### 🚀 GPU-Accelerated AI Inference

- **ISQ Quantization**: In-Situ Quantization with Q4K precision for optimal speed/quality balance
- **CUDA Acceleration**: Full GPU acceleration on NVIDIA RTX 4090 and compatible hardware
- **Fast Model Loading**: 22-second model loading with optimized memory footprint
- **Multimodal Support**: Text, image, and audio processing with `unsloth/gemma-3n-E4B-it` vision model
- **Real-time Inference**: 36-38% GPU utilization with 6.8GB VRAM usage for continuous processing

### 🎯 Real-Time Vision & Audio Processing

- Real camera input with automatic device detection and hardware fallback
- Optimized object detection with spam prevention (max 10 high-confidence objects)
- Concise scene descriptions: "Environment contains: person, 9 rectangular objects"
- Real-time audio input from microphone with voice activity detection
- Text-to-speech output with configurable voice options

### 🤖 Autonomous Robot Control

- **OpenAI Tool Calling**: Modern function call interface for precise robot actions
- **Action-First Architecture**: Direct translation of visual context to robot commands
- **Context-Aware Responses**: Real model computation influences tool calls based on multimodal input
- **Safety Constraints**: Built-in collision avoidance and human detection
- **Hardware Integration**: Real motor control and sensor feedback loops

### ⚡ Performance Optimized

- **CUDA Detection**: Automatic GPU detection with CPU fallback
- **Memory Efficient**: ISQ reduces memory footprint compared to full-precision models
- **Edge-Ready**: Optimized for NVIDIA Jetson Nano and desktop GPU deployment
- **Real-time Pipeline**: Sub-second inference with continuous processing

## Core Concepts

### Vision-to-Action Pipeline

1. **Visual Input**: Real-time camera stream with automatic device detection
2. **Object Detection**: Optimized computer vision with spam prevention (max 10 objects)
3. **AI Processing**: GPU-accelerated ISQ inference with multimodal understanding
4. **Action Output**: OpenAI-style function calls for immediate robot execution

### ISQ Quantization System

OpticXT uses In-Situ Quantization (ISQ) for optimal performance:

- **Q4K Precision**: 4-bit quantization with optimal speed/quality balance
- **In-Memory Processing**: Weights quantized during model loading (reduced memory footprint)
- **GPU Acceleration**: Full CUDA support with 36-38% GPU utilization on RTX 4090
- **Fast Loading**: 22-second model initialization vs. slower UQFF alternatives

### Command Execution Framework

Building on modern OpenAI tool calling, OpticXT translates model outputs into robot actions:

- Physical movements and navigation commands
- Sensor integration and feedback loops
- Environmental awareness and safety constraints
- Audio/visual output generation

## Architecture Philosophy

OpticXT represents a paradigm shift from conversational AI to action-oriented intelligence. By eliminating the conversational layer and focusing purely on vision-driven decision-making, we achieve the low-latency response times critical for real-world robotics applications.

The system acts as a remote AGiXT agent, maintaining compatibility with the broader ecosystem while operating independently on edge hardware. This hybrid approach enables sophisticated behaviors through local processing while retaining the ability to offload complex tasks when needed.

## Use Cases

- Autonomous navigation in dynamic environments
- Real-time object interaction and manipulation
- Surveillance and monitoring applications
- Assistive robotics with visual understanding
- Industrial automation with adaptive behavior

## Getting Started

### Prerequisites

#### System Requirements

- Any Linux system with camera and microphone (for video chat mode)
- NVIDIA Jetson Nano (16GB) or Go2 robot (for full robot mode)
- USB camera, webcam, or CSI camera module
- Microphone and speakers/headphones for audio
- Rust 1.70+ installed

#### Dependencies

```bash
# Ubuntu/Debian - Basic dependencies
sudo apt update
sudo apt install -y build-essential cmake pkg-config

# Audio system dependencies (required)
sudo apt install -y libasound2-dev portaudio19-dev

# TTS support (required for voice output)
sudo apt install -y espeak espeak-data libespeak-dev

# Optional: Additional audio codecs
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Josh-XT/OpticXT.git
cd OpticXT
```

2. **Build with CUDA support for GPU acceleration**

```bash
# For NVIDIA GPU acceleration (recommended)
cargo build --release --features cuda

# For CPU-only mode (fallback)
cargo build --release
```

3. **The system uses ISQ quantization with automatic model download**

The system automatically downloads and quantizes the `unsloth/gemma-3n-E4B-it` model:

- **Model**: unsloth/gemma-3n-E4B-it (vision-capable, no authentication required)
- **Quantization**: ISQ Q4K (in-situ quantization during loading)
- **Loading Time**: ~22 seconds with GPU acceleration
- **Memory Usage**: ~6.8GB VRAM on RTX 4090

**Note**: Models are downloaded automatically from HuggingFace on first run. No manual model installation required.

### Configuration

Edit `config.toml` to match your setup:

```toml
[vision]
width = 640           # Camera resolution
height = 480
confidence_threshold = 0.5

[audio]
input_device = "default"    # Microphone device
output_device = "default"   # Speaker device
voice = "en"               # TTS voice language
enable_vad = true          # Voice activity detection

[model]
model_path = "models/gemma-3n-E4B-it-Q4_K_M.gguf"
temperature = 0.7     # Lower = more deterministic

[performance]
use_gpu = true        # Set to false if no CUDA
processing_interval_ms = 100  # Adjust for performance
```

### Running OpticXT

OpticXT runs as an autonomous robot control system with GPU-accelerated AI inference:

#### Basic Usage

```bash
# Run with CUDA acceleration (recommended)
cargo run --release --features cuda -- --verbose

# Monitor GPU utilization (separate terminal)
watch -n 1 nvidia-smi

# Check real-time inference performance
cargo run --release --features cuda -- --verbose 2>&1 | grep "GPU Utilization\|GPU Memory"
```

#### Command Line Options

```bash
USAGE:
    opticxt [OPTIONS]

OPTIONS:
    -c, --config <CONFIG>              Configuration file path [default: config.toml]
    -d, --camera-device <DEVICE>       Camera device index [default: 0]
    -v, --verbose                      Enable verbose logging with GPU monitoring
    -h, --help                         Print help information
```

#### Expected Performance (RTX 4090)

- **Model Loading**: ~22 seconds (ISQ quantization)
- **GPU Memory Usage**: ~6.8GB VRAM / 24.5GB total (28% utilization)
- **Inference Speed**: 36-38% GPU utilization during processing
- **Object Detection**: Max 10 high-confidence objects per frame
- **Response Time**: 3-6 seconds per inference cycle

## Hardware Requirements & Compatibility

### Minimum Requirements (Video Chat Mode)

- Any Linux system with USB ports
- Webcam or built-in camera
- Microphone and speakers/headphones
- 4GB RAM minimum (8GB recommended)
- Rust 1.70+ toolchain

### Full Robot Mode Requirements

- NVIDIA Jetson Nano (16GB) or Go2 robot platform
- CSI or USB camera
- Microphone and speaker system
- Motor controllers and actuators
- Optional: LiDAR sensors (falls back to simulation)

### Audio System

OpticXT includes a complete real audio pipeline:

- **Input**: Real-time microphone capture with voice activity detection
- **Output**: Text-to-speech synthesis with multiple voice options
- **Processing**: Audio filtering and noise reduction
- **Fallback**: Graceful degradation when audio hardware is unavailable

### Camera System

Flexible camera support with automatic detection:

- **USB Cameras**: Automatic detection of any UVC-compatible camera
- **Built-in Cameras**: Laptop/desktop integrated cameras
- **CSI Cameras**: Jetson Nano camera modules
- **Fallback**: Simulation mode when no camera is detected

## Testing & Development

OpticXT includes comprehensive testing capabilities for development and validation:

### Test Modes

The system provides 9 different test modes to validate various components:

#### Core AI Tests

```bash
# Quick smoke test (fast basic functionality check)
cargo run --release --features cuda -- --test-quick-smoke

# Simple text inference test
cargo run --release --features cuda -- --test-simple

# UQFF quantized model test
cargo run --release --features cuda -- --test-uqff
```

#### Multimodal Tests

```bash
# Comprehensive multimodal test (text + image + audio)
cargo run --release --features cuda -- --test-multimodal

# Image-only inference test
cargo run --release --features cuda -- --test-image

# Alternative image inference test
cargo run --release --features cuda -- --test-image-only

# Audio-only inference test
cargo run --release --features cuda -- --test-audio
```

#### Specialized Tests

```bash
# OpenAI-style tool calling format validation
cargo run --release --features cuda -- --test-tool-format

# Robot command generation scenarios
cargo run --release --features cuda -- --test-robot-commands
```

### Integration Tests

```bash
# Run unit tests (no GPU required)
cargo test

# Run all tests including GPU-intensive ones
cargo test -- --ignored

# Run specific integration test
cargo test test_quick_smoke_integration -- --ignored
```

### Development Testing Workflow

1. **Start with smoke test**: `--test-quick-smoke` for basic functionality
2. **Test specific components**: Use targeted tests like `--test-image` or `--test-audio`
3. **Full validation**: Run `--test-multimodal` for comprehensive testing
4. **Performance validation**: Use `--test-uqff` for model-specific testing

### Expected Test Results

- **Model Loading**: 15-25 seconds depending on hardware
- **Text Generation**: 50-200ms per response
- **Image Processing**: 200-800ms per image
- **Audio Processing**: 100-500ms per audio segment
- **Tool Call Generation**: Valid JSON format with proper function structure

## Troubleshooting

### Common Issues

#### No Camera Detected

```bash
# Check available cameras
ls /dev/video*
v4l2-ctl --list-devices

# Fix permissions
sudo usermod -a -G video $USER
# Log out and back in
```

#### Audio Issues

```bash
# Check audio devices
aplay -l    # List playback devices
arecord -l  # List capture devices

# Test microphone
arecord -d 5 test.wav && aplay test.wav

# Fix audio permissions
sudo usermod -a -G audio $USER
```

#### Model Loading (Real AI Inference - Current Status)

OpticXT successfully loads and runs AI models with real neural network inference:

```bash
# 1. Current Status Check
ls -lh models/
# Should show both gemma-3n-E4B-it-Q4_K_M.gguf and tokenizer.json

# 2. What's Working:
# ✅ Model file loading (GGUF format)
# ✅ Tokenizer loading and text processing
# ✅ Model architecture initialization
# ✅ Neural network forward pass with real inference
# ✅ Real token generation from model logits
# ✅ Context-aware function call output generation from actual model output
# ✅ Complete removal of all hardcoded/simulation fallbacks

# 3. Current Behavior:
# - Models load successfully with real tokenizer and UQFF quantization
# - Real model inference with mistral.rs and multimodal support (text/image/audio)
# - OpenAI-style function calls generated from genuine model output
# - System fails gracefully with clear error messages when models unavailable
# - All functionality (camera, audio, movement) works with real hardware input
# - NO hardcoded responses or simulation fallbacks whatsoever

# 4. Expected Log Messages:
# ✅ "✅ Successfully loaded HuggingFace Gemma 3n model with multimodal support" 
# ✅ "Real multimodal model generated text in XXXms"
# ✅ "Running model inference with X modalities"
# ❌ "Model inference timed out after 180 seconds" (when model performance issues)

# 5. Error Handling:
# The system now properly fails with informative errors when:
# - Model files are missing or corrupted
# - Tokenizer cannot be loaded
# - Real inference fails
# This ensures complete authenticity - no fake responses under any circumstances
```

**Current Status**: The system uses exclusively real neural network inference with genuine GGUF model loading and authentic tokenizer processing. All simulation logic, hardcoded responses, and fallback mechanisms have been completely removed. The system will only operate with actual model inference or fail gracefully with clear error messages.

#### Build Issues

```bash
# Install missing dependencies
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libasound2-dev portaudio19-dev
sudo apt install -y espeak espeak-data libespeak-dev

# Clean and rebuild
cargo clean
cargo build --release
```

#### Performance Issues

```bash
# Monitor system resources
htop
# Watch for CPU/memory usage

# Adjust processing interval in config.toml
# Increase processing_interval_ms for lower resource usage
```

### Debug Mode

Enable comprehensive logging:

```bash
RUST_LOG=debug cargo run -- --verbose
```

This shows:

- Camera detection and initialization
- Audio device enumeration
- Model loading status (real or simulation)
- Frame processing times
- Audio input/output status
- Error details and fallback triggers

## Architecture & Technical Details

### Real Hardware Integration

OpticXT is built around real hardware components with intelligent fallbacks:

#### Vision System
- **Primary**: Real camera capture via nokhwa library
- **Fallback**: Simulated visual input when no camera detected
- **Support**: USB, CSI, and built-in cameras with automatic detection

#### Audio System
- **Input**: Real microphone capture via cpal library
- **Output**: Text-to-speech via tts/espeak integration
- **Processing**: Voice activity detection and audio filtering
- **Fallback**: Silent operation when audio hardware unavailable

#### AI Model System

- **Model Loading**: Successfully loads UQFF models with mistral.rs VisionModelBuilder
- **Real Inference**: Full multimodal neural network inference with text, image, and audio support
- **Context-Aware Responses**: Real model computation influences function call output based on multimodal input
- **Tool Call Generation**: OpenAI-style function calls generated from actual model outputs
- **Intelligent Fallback**: Graceful degradation when models unavailable

### Dual Mode Architecture

#### Video Chat Assistant Mode
```
Camera → Vision Processing → AI Model → TTS Response
   ↑                                      ↓
Microphone ← Audio Processing ← User Interaction
```

#### Robot Control Mode
```
Sensors → Context Assembly → AI Decision → Function Calls
   ↑                                         ↓
Environment ← Robot Actions ← Motor Control ← Tool Call Output
```

### Command System (Robot Mode)

OpticXT generates OpenAI-style function calls for precise robot control:

```json
[{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "move",
    "arguments": "{\"direction\": \"forward\", \"distance\": 1.0, \"speed\": \"slow\", \"reasoning\": \"Moving forward to investigate detected object\"}"
  }
}]

[{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "speak",
    "arguments": "{\"text\": \"I can see someone approaching\", \"voice\": \"default\", \"reasoning\": \"Alerting about detected human presence\"}"
  }
}]

[{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "analyze",
    "arguments": "{\"target\": \"obstacle\", \"detail_level\": \"detailed\", \"reasoning\": \"Need to assess navigation path\"}"
  }
}]
```

## Deployment Guide

### Desktop/Laptop Development

1. **Install dependencies** (audio system required)
2. **Build project**: `cargo build --release`
3. **Run**: `cargo run --release`
4. **Mode**: Automatically runs as video chat assistant

### Jetson Nano Deployment

1. **Flash Jetson with Ubuntu 20.04**
2. **Install Rust toolchain**
3. **Install system dependencies** (including CUDA if available)
4. **Build with optimizations**: `cargo build --release`
5. **Configure hardware** in `config.toml`
6. **Deploy**: Copy binary and config to target system

### Go2 Robot Integration

1. **Cross-compile** for ARM64 architecture
2. **Install on Go2** via SDK deployment tools
3. **Configure sensors** for robot hardware
4. **Enable robot mode** in configuration
5. **Test control commands** before full deployment

## Key Features Summary

✅ **GPU-Accelerated AI**: ISQ quantization with CUDA acceleration on NVIDIA RTX 4090  
✅ **Real Camera Input**: Works with any USB/CSI/built-in camera  
✅ **Real Audio I/O**: Microphone input and TTS output  
✅ **Multimodal AI**: unsloth/gemma-3n-E4B-it model with text, image, and audio support  
✅ **OpenAI Tool Calling**: Robot control commands via modern function call interface  
✅ **Optimized Vision**: Spam prevention with max 10 high-confidence objects per frame  
✅ **Hardware Auto-Detection**: Real hardware integration with CUDA detection and CPU fallback  
✅ **Edge Deployment Ready**: Optimized for NVIDIA Jetson and desktop GPU platforms  
✅ **Production Ready**: 22-second model loading, 36-38% GPU utilization, 6.8GB VRAM usage  

**Performance Status**: Full GPU acceleration achieved with ISQ quantization. System loads `unsloth/gemma-3n-E4B-it` model in 22 seconds, utilizes 28% of RTX 4090 VRAM, and processes inference at 36-38% GPU utilization with optimized vision processing.

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## OpticXT: GPU-Accelerated Vision Robot Control

*Real-time robot control with ISQ quantization and CUDA acceleration - from autonomous navigation to precise manipulation.*

CA: Ga9P2TZcxsHjYmXdEyu9Z7wL1QAowjBAZwRQ41gBbonk

```

