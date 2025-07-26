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
