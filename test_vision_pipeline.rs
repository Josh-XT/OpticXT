#!/usr/bin/env cargo
/*
[dependencies]
opticxt = { path = "." }
tokio = { version = "1.0", features = ["full"] }
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"
*/

// Vision Test - Verify camera → vision → model pipeline
use anyhow::Result;
use tracing::{info, debug, error};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();
    
    println!("🔍 OpticXT Vision Pipeline Test");
    println!("==============================");
    
    // Test 1: Camera System
    println!("\n📷 Test 1: Camera System Initialization");
    let camera_result = test_camera_system().await;
    match camera_result {
        Ok(_) => println!("✅ Camera system working"),
        Err(e) => {
            println!("❌ Camera system failed: {}", e);
            println!("   This may be expected if no camera is available");
        }
    }
    
    // Test 2: Vision Processing  
    println!("\n👁️ Test 2: Vision Processing");
    let vision_result = test_vision_processing().await;
    match vision_result {
        Ok(_) => println!("✅ Vision processing working"),
        Err(e) => println!("❌ Vision processing failed: {}", e),
    }
    
    // Test 3: Configuration Check
    println!("\n⚙️ Test 3: Configuration Check");
    test_configuration().await?;
    
    println!("\n🎉 Vision pipeline test complete!");
    println!("If camera and vision tests passed, the system is ready for full testing.");
    
    Ok(())
}

async fn test_camera_system() -> Result<()> {
    use opticxt::camera::{CameraSystem, CameraConfig};
    
    let camera_config = CameraConfig {
        camera_id: 0,
        width: 640,
        height: 480,
        fps: 30.0,
    };
    
    info!("Initializing camera system...");
    let mut camera_system = CameraSystem::new(camera_config)?;
    camera_system.initialize().await?;
    
    info!("Capturing test frame...");
    let sensor_data = camera_system.capture_sensor_data().await?;
    
    println!("   📊 Frame: {}x{} pixels", sensor_data.frame.width, sensor_data.frame.height);
    println!("   🎯 LiDAR: {} points", sensor_data.lidar_points.len());
    println!("   💾 Data size: {} bytes", sensor_data.frame.data.len());
    
    Ok(())
}

async fn test_vision_processing() -> Result<()> {
    use opticxt::vision_basic::{VisionProcessor, Mat};
    use opticxt::camera::SensorData;
    use std::time::SystemTime;
    
    let mut vision_processor = VisionProcessor::new(0, 640, 480, 0.5, "basic".to_string())?;
    vision_processor.initialize().await?;
    
    // Create a test frame (simulated camera data)
    let mut test_frame = Mat::new(640, 480, 3);
    // Fill with a simple pattern to simulate camera input
    for i in 0..test_frame.data.len() {
        test_frame.data[i] = ((i % 256) as u8);
    }
    
    let sensor_data = SensorData {
        timestamp: SystemTime::now(),
        frame: test_frame,
        lidar_points: vec![],
        has_lidar: false,
    };
    
    info!("Processing test frame...");
    let frame_context = vision_processor.process_frame(&sensor_data.frame, &sensor_data).await?;
    
    println!("   🔍 Objects detected: {}", frame_context.objects.len());
    println!("   📝 Scene: {}", frame_context.scene_description.chars().take(100).collect::<String>());
    println!("   📐 Frame size: {}x{}", frame_context.frame_size.0, frame_context.frame_size.1);
    
    Ok(())
}

async fn test_configuration() -> Result<()> {
    use opticxt::config::OpticXTConfig;
    
    let config = OpticXTConfig::default();
    
    println!("   🎯 Vision config:");
    println!("      - Resolution: {}x{}", config.vision.width, config.vision.height);
    println!("      - Confidence threshold: {}", config.vision.confidence_threshold);
    println!("      - Multimodal inference: {}", config.vision.enable_multimodal_inference);
    
    println!("   🧠 Model config:");
    println!("      - Quantization: {}", config.model.quantization_method);
    println!("      - ISQ type: {}", config.model.isq_type);
    println!("      - Max tokens: {}", config.model.max_tokens);
    
    if config.vision.enable_multimodal_inference {
        println!("   ✅ Multimodal inference enabled - camera images will be sent to model");
    } else {
        println!("   ⚠️ Multimodal inference disabled - only text descriptions will be sent");
    }
    
    Ok(())
}
