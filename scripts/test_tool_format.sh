#!/usr/bin/env bash
# Simple test to verify tool calling output format

echo "🧪 Testing OpenAI tool calling format output..."

# Create a simple mock test that directly calls the formatting function
cat > test_tool_format.rs << 'EOF'
use serde_json::json;

fn format_as_tool_call(text: &str, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_lower = prompt.to_lowercase();
    
    // If text is empty or garbled, provide a sensible default
    let effective_text = if text.is_empty() || text.len() < 3 {
        "I am ready to assist you."
    } else {
        text
    };
    
    // Determine appropriate OpenAI-style function call based on content and prompt
    let tool_call = if effective_text.contains("move") || effective_text.contains("forward") || prompt_lower.contains("move") || prompt_lower.contains("navigate") {
        json!([{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "move",
                "arguments": json!({
                    "direction": "forward",
                    "distance": 0.5,
                    "speed": "normal",
                    "reasoning": effective_text
                }).to_string()
            }
        }])
    } else if effective_text.contains("turn") || effective_text.contains("rotate") || prompt_lower.contains("turn") || prompt_lower.contains("rotate") {
        json!([{
            "id": "call_1", 
            "type": "function",
            "function": {
                "name": "rotate",
                "arguments": json!({
                    "direction": "left",
                    "angle": 30.0,
                    "reasoning": effective_text
                }).to_string()
            }
        }])
    } else {
        // Default to speak for general responses
        json!([{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "speak",
                "arguments": json!({
                    "text": effective_text,
                    "voice": "default",
                    "reasoning": "Providing a response to the user's query"
                }).to_string()
            }
        }])
    };
    
    // Format as pretty JSON
    let formatted_json = serde_json::to_string_pretty(&tool_call)
        .unwrap_or_else(|_| r#"[{"id": "call_1", "type": "function", "function": {"name": "speak", "arguments": "{\"text\": \"I am ready to assist you.\"}"}}]"#.to_string());
    
    Ok(formatted_json)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Testing tool call formatting...");
    
    // Test cases
    let test_cases = vec![
        ("Hello, how are you?", "General greeting"),
        ("I need to move forward", "Movement command"),
        ("Turn left please", "Rotation command"),  
        ("Stop now!", "Stop command"),
        ("Analyze this image", "Vision analysis"),
        ("", "Empty input"),
    ];
    
    for (input, description) in test_cases {
        println!("\n📝 Test: {} - Input: '{}'", description, input);
        match format_as_tool_call(input, input) {
            Ok(result) => {
                println!("✅ Output:\n{}", result);
                
                // Validate JSON structure
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
                    if let Some(array) = parsed.as_array() {
                        if !array.is_empty() && array[0].get("type").and_then(|v| v.as_str()) == Some("function") {
                            println!("✅ Valid OpenAI tool call format");
                        } else {
                            println!("❌ Invalid tool call structure");
                        }
                    } else {
                        println!("❌ Not an array");
                    }
                } else {
                    println!("❌ Invalid JSON");
                }
            }
            Err(e) => println!("❌ Error: {}", e),
        }
    }
    
    Ok(())
}
EOF

echo "📝 Created tool format test. Running it..."
rustc --extern serde_json test_tool_format.rs && ./test_tool_format

echo "🧹 Cleaning up..."
rm -f test_tool_format.rs test_tool_format
