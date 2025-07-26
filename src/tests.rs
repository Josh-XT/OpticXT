use anyhow::Result;
use crate::models::{GemmaModel, ModelConfig};
use crate::test_camera_vision::{test_real_camera_vision_description, test_vision_consistency_with_main, test_vision_integration_quick};
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

/// Test that confirms real camera input is used for vision description
/// This validates that the system properly uses camera input and matches main branch behavior
pub async fn test_camera_vision_description() -> Result<()> {
    test_real_camera_vision_description().await
}

/// Test vision consistency with main branch behavior
pub async fn test_vision_main_consistency() -> Result<()> {
    test_vision_consistency_with_main().await
}

/// Quick test to verify camera vision integration works
pub async fn test_camera_integration_quick() -> Result<()> {
    test_vision_integration_quick().await
}
