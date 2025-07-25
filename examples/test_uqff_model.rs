use anyhow::Result;

// Import the models module directly since we're in an example
mod models;
use models::{GemmaModel, ModelConfig};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing UQFF Gemma 3n model...");
    
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        context_length: 2048,
    };
    
    println!("📥 Loading UQFF model (EricB/gemma-3n-E4B-it-UQFF)...");
    
    // This should use the UQFF model by default
    let mut model = match GemmaModel::load(None, config).await {
        Ok(model) => {
            println!("✅ UQFF model loaded successfully!");
            model
        }
        Err(e) => {
            println!("❌ Failed to load UQFF model: {}", e);
            return Err(e);
        }
    };
    
    println!("💭 Testing text generation...");
    let test_prompts = vec![
        "Hello! Introduce yourself briefly.",
        "What can you help me with?",
        "Move forward slowly and look around.",
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n🔍 Test {}: \"{}\"", i + 1, prompt);
        
        match model.generate(prompt).await {
            Ok(result) => {
                println!("✅ Generated: {}", result.text);
                println!("📊 Tokens: {}, Time: {}ms", result.tokens_generated, result.processing_time_ms);
            }
            Err(e) => {
                println!("❌ Generation failed: {}", e);
            }
        }
    }
    
    println!("\n🎉 UQFF model test completed!");
    Ok(())
}
