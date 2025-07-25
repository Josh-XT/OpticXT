use opticxt::models::{GemmaModel, ModelConfig};
use anyhow::Result;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let _ = tracing_subscriber::fmt::try_init(); // Use try_init to avoid panic if already initialized
    
    println!("🔬 Testing UQFF Gemma 3n model inference...");
    
    // Create model config
    let config = ModelConfig {
        max_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
        context_length: 2048,
    };
    
    // Load the UQFF model (should use EricB/gemma-3n-E4B-it-UQFF by default)
    println!("📥 Loading UQFF model...");
    let mut model = GemmaModel::load(None, config).await?;
    
    // Test basic text generation
    println!("💬 Testing text generation...");
    let response1 = model.generate("Hello! Please introduce yourself.").await?;
    println!("✅ Text response: {}", response1.text);
    println!("📊 Tokens: {}, Time: {}ms", response1.tokens_generated, response1.processing_time_ms);
    
    // Test another prompt
    println!("\n🤖 Testing instruction following...");
    let response2 = model.generate("Explain what you can do in one sentence.").await?;
    println!("✅ Instruction response: {}", response2.text);
    println!("📊 Tokens: {}, Time: {}ms", response2.tokens_generated, response2.processing_time_ms);
    
    // Test command-style prompt
    println!("\n⚡ Testing command generation...");
    let response3 = model.generate("Move forward and analyze the environment ahead.").await?;
    println!("✅ Command response: {}", response3.text);
    println!("📊 Tokens: {}, Time: {}ms", response3.tokens_generated, response3.processing_time_ms);
    
    println!("\n🎉 UQFF model test completed successfully!");
    Ok(())
}
