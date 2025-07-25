mod models;

use models::GemmaModel;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::set_var("RUST_LOG", "info");
    let _ = tracing_subscriber::fmt::try_init(); // Use try_init to avoid panic if already initialized
    
    println!("Testing OpticXT Gemma Model...");
    
    let model = GemmaModel::new("models/gemma-3n-E4B-it-Q4_K_M.gguf", "models/tokenizer.json").await?;
    
    let test_prompts = vec![
        "Hello, how are you?",
        "What can you see in this image?",
        "Move forward",
        "Tell me about robotics",
    ];
    
    for prompt in test_prompts {
        println!("\nPrompt: {}", prompt);
        match model.generate_response(prompt, 50).await {
            Ok(response) => println!("Response: {}", response),
            Err(e) => println!("Error: {}", e),
        }
    }
    
    Ok(())
}
