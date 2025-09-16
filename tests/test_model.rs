use opticxt::models::GemmaModel;
use opticxt::config::ModelConfig;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::set_var("RUST_LOG", "info");
    tracing_subscriber::fmt::init();
    
    println!("Testing OpticXT Gemma Model...");
    
    let config = ModelConfig {
        model_path: "models/gemma-3n-E4B-it-Q4_K_M.gguf".to_string(),
        quantization_method: "isq".to_string(),
        isq_type: "Q4K".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        context_length: 512,
        remote: None,
    };
    
    let mut model = GemmaModel::load(None, config, "isq".to_string(), "Q4K".to_string()).await?;
    
    let test_prompts = vec![
        "Hello, how are you?",
        "What can you see in this image?",
        "Move forward",
        "Tell me about robotics",
    ];
    
    for prompt in test_prompts {
        println!("\nPrompt: {}", prompt);
        match model.generate(prompt).await {
            Ok(result) => println!("Response: {}", result.text),
            Err(e) => println!("Error: {}", e),
        }
    }
    
    Ok(())
}
