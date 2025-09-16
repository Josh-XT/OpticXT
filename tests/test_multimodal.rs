use anyhow::Result;
use opticxt::models::GemmaModel;
use opticxt::config::ModelConfig;
use image::{DynamicImage, ImageBuffer, Rgb};

pub async fn test_multimodal_inference() -> Result<()> {
    println!("🚀 Testing multimodal inference capabilities...");
    
    let config = ModelConfig {
        model_path: "".to_string(),
        quantization_method: "isq".to_string(),
        isq_type: "Q4K".to_string(),
        max_tokens: 50,  // Reduced for faster inference
        temperature: 0.1,  // Lower temperature for more deterministic output
        top_p: 0.8,
        context_length: 512,  // Reduced context length for faster processing
        remote: None,
    };
    
    println!("📥 Loading UQFF Gemma 3n multimodal model...");
    
    // Load the UQFF model
    let mut model = match GemmaModel::load(None, config.clone(), config.quantization_method.clone(), config.isq_type.clone()).await {
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
