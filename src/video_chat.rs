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
