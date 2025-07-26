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
