use anyhow::Result;
use opticxt::models::GemmaModel;
use opticxt::config::ModelConfig;

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
