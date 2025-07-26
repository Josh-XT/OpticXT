use anyhow::Result;
use crate::models::{GemmaModel, ModelConfig};
use image::{DynamicImage, ImageBuffer, Rgb};

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
