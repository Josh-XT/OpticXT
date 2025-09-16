use crate::config::OpticXTConfig;
use crate::vision::{VisionProcessor, Mat};
use crate::context::ContextManager;
use crate::models::{GemmaModel, ensure_model_downloaded};
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
        
        let model = GemmaModel::load(model_path, config.model.clone(), config.model.quantization_method.clone(), config.model.isq_type.clone()).await?;
        
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
    
    // API-specific methods for external access
    #[allow(dead_code)]
    pub async fn get_current_camera_frame(&mut self) -> Result<image::DynamicImage> {
        let sensor_data = self.camera_system.capture_sensor_data().await?;
        sensor_data.frame.to_image()
    }
    
    #[allow(dead_code)]
    pub async fn process_inference_with_context(&mut self, prompt: &str, image: Option<image::DynamicImage>) -> Result<crate::models::GenerationResult> {
        // Build context from current sensor data if no image provided
        let generation_result = if let Some(provided_image) = image {
            // Use provided image
            self.model.generate_with_image(prompt, provided_image).await?
        } else {
            // Try to get current camera frame
            match self.get_current_camera_frame().await {
                Ok(camera_image) => {
                    debug!("Using current camera frame for inference");
                    self.model.generate_with_image(prompt, camera_image).await?
                }
                Err(e) => {
                    warn!("Failed to get camera frame, using text-only inference: {}", e);
                    self.model.generate(prompt).await?
                }
            }
        };
        
        Ok(generation_result)
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
