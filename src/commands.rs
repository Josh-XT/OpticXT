use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionCommand {
    #[serde(rename = "move")]
    Move {
        direction: String,
        distance: f32,
        speed: String,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "rotate")]
    Rotate {
        direction: String,
        angle: f32,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "speak")]
    Speak {
        text: String,
        #[serde(default)]
        voice: String,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "analyze")]
    Analyze {
        target: Option<String>,
        detail_level: Option<String>,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "offload")]
    Offload {
        task_description: String,
        target_agent: Option<String>,
        priority: Option<String>,
        #[serde(default)]
        reasoning: String,
    },
    #[serde(rename = "wait")]
    Wait {
        duration: Option<f32>,
        #[serde(default)]
        reasoning: String,
    },
}

#[derive(Debug, Clone)]
pub struct CommandExecutionResult {
    pub success: bool,
    pub message: String,
    pub execution_time: Duration,
    pub side_effects: Vec<String>,
}

pub struct CommandExecutor {
    enabled_commands: Vec<String>,
    timeout_seconds: u64,
    validate_before_execution: bool,
    tts_engine: Option<TtsEngine>,
}

#[derive(Debug)]
struct TtsEngine {
    // Placeholder for TTS integration
    voice: String,
}

impl CommandExecutor {
    pub fn new(
        enabled_commands: Vec<String>,
        timeout_seconds: u64,
        validate_before_execution: bool,
    ) -> Self {
        let tts_engine = Some(TtsEngine {
            voice: "default".to_string(),
        });
        
        Self {
            enabled_commands,
            timeout_seconds,
            validate_before_execution,
            tts_engine,
        }
    }
    
    pub async fn parse_and_execute(&self, tool_call_output: &str) -> Result<CommandExecutionResult> {
        debug!("Parsing tool call command: {}", tool_call_output);
        
        // Extract the tool call command from the model output
        let command = self.parse_tool_call_command(tool_call_output)?;
        
        // Validate command if enabled
        if self.validate_before_execution {
            self.validate_command(&command)?;
        }
        
        // Execute the command
        self.execute_command(command).await
    }
    
    fn parse_tool_call_command(&self, tool_call_output: &str) -> Result<ActionCommand> {
        // Parse the JSON tool call output
        let tool_calls: Vec<serde_json::Value> = serde_json::from_str(tool_call_output)
            .map_err(|e| anyhow!("Failed to parse tool call JSON: {}", e))?;
        
        if tool_calls.is_empty() {
            return Err(anyhow!("No tool calls found in output"));
        }
        
        // Get the first tool call
        let tool_call = &tool_calls[0];
        let function = tool_call.get("function")
            .ok_or_else(|| anyhow!("No function found in tool call"))?;
        
        let function_name = function.get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| anyhow!("No function name found"))?;
        
        let arguments_str = function.get("arguments")
            .and_then(|a| a.as_str())
            .ok_or_else(|| anyhow!("No function arguments found"))?;
        
        let arguments: serde_json::Value = serde_json::from_str(arguments_str)
            .map_err(|e| anyhow!("Failed to parse function arguments: {}", e))?;
        
        // Parse based on function name
        match function_name {
            "move" => self.parse_move_from_json(&arguments),
            "rotate" => self.parse_rotate_from_json(&arguments),
            "speak" => self.parse_speak_from_json(&arguments),
            "analyze" => self.parse_analyze_from_json(&arguments),
            "wait" => self.parse_wait_from_json(&arguments),
            "stop" => self.parse_stop_from_json(&arguments),
            _ => Err(anyhow!("Unknown function name: {}", function_name))
        }
    }
    
    
    // JSON-based parsing functions for OpenAI tool calls
    fn parse_move_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let direction = args.get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("forward")
            .to_string();
        let distance = args.get("distance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let speed = args.get("speed")
            .and_then(|v| v.as_str())
            .unwrap_or("normal")
            .to_string();
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Move {
            direction,
            distance,
            speed,
            reasoning,
        })
    }
    
    fn parse_rotate_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let direction = args.get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("left")
            .to_string();
        let angle = args.get("angle")
            .and_then(|v| v.as_f64())
            .unwrap_or(30.0) as f32;
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Rotate {
            direction,
            angle,
            reasoning,
        })
    }
    
    fn parse_speak_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let text = args.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Text is required for speak command"))?
            .to_string();
        let voice = args.get("voice")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Speak {
            text,
            voice,
            reasoning,
        })
    }
    
    fn parse_analyze_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let target = args.get("target")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let detail_level = args.get("detail_level")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Analyze {
            target,
            detail_level,
            reasoning,
        })
    }
    
    fn parse_wait_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let duration = args.get("duration")
            .and_then(|v| v.as_f64())
            .map(|d| d as f32);
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        Ok(ActionCommand::Wait {
            duration,
            reasoning,
        })
    }
    
    fn parse_stop_from_json(&self, args: &serde_json::Value) -> Result<ActionCommand> {
        let _immediate = args.get("immediate")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let reasoning = args.get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        
        // Map to Wait with duration 0 for immediate stop
        Ok(ActionCommand::Wait {
            duration: Some(0.0),
            reasoning: format!("STOP - {}", reasoning),
        })
    }
    
    fn validate_command(&self, command: &ActionCommand) -> Result<()> {
        match command {
            ActionCommand::Move { direction, distance, speed, .. } => {
                if !["forward", "backward", "left", "right"].contains(&direction.as_str()) {
                    return Err(anyhow!("Invalid move direction: {}", direction));
                }
                if *distance <= 0.0 || *distance > 10.0 {
                    return Err(anyhow!("Invalid move distance: {}", distance));
                }
                if !["slow", "normal", "fast"].contains(&speed.as_str()) {
                    return Err(anyhow!("Invalid speed: {}", speed));
                }
            }
            ActionCommand::Rotate { direction, angle, .. } => {
                if !["left", "right", "clockwise", "counterclockwise"].contains(&direction.as_str()) {
                    return Err(anyhow!("Invalid rotate direction: {}", direction));
                }
                if *angle <= 0.0 || *angle > 360.0 {
                    return Err(anyhow!("Invalid rotation angle: {}", angle));
                }
            }
            ActionCommand::Speak { text, .. } => {
                if text.is_empty() {
                    return Err(anyhow!("Speech text cannot be empty"));
                }
                if text.len() > 500 {
                    return Err(anyhow!("Speech text too long"));
                }
            }
            _ => {} // Other commands have minimal validation requirements
        }
        
        Ok(())
    }
    
    async fn execute_command(&self, command: ActionCommand) -> Result<CommandExecutionResult> {
        let start_time = std::time::Instant::now();
        
        info!("Executing command: {:?}", command);
        
        let result = match command {
            ActionCommand::Move { direction, distance, speed, reasoning } => {
                self.execute_move(direction, distance, speed, reasoning).await
            }
            ActionCommand::Rotate { direction, angle, reasoning } => {
                self.execute_rotate(direction, angle, reasoning).await
            }
            ActionCommand::Speak { text, voice, reasoning } => {
                self.execute_speak(text, voice, reasoning).await
            }
            ActionCommand::Analyze { target, detail_level, reasoning } => {
                self.execute_analyze(target, detail_level, reasoning).await
            }
            ActionCommand::Offload { task_description, target_agent, priority, reasoning } => {
                self.execute_offload(task_description, target_agent, priority, reasoning).await
            }
            ActionCommand::Wait { duration, reasoning } => {
                self.execute_wait(duration, reasoning).await
            }
        };
        
        let execution_time = start_time.elapsed();
        
        match result {
            Ok(mut cmd_result) => {
                cmd_result.execution_time = execution_time;
                Ok(cmd_result)
            }
            Err(e) => Ok(CommandExecutionResult {
                success: false,
                message: format!("Command failed: {}", e),
                execution_time,
                side_effects: vec![],
            })
        }
    }
    
    async fn execute_move(&self, direction: String, distance: f32, speed: String, reasoning: String) -> Result<CommandExecutionResult> {
        // Placeholder implementation - would interface with actual robot hardware
        debug!("Moving {} {} meters at {} speed. Reasoning: {}", direction, distance, speed, reasoning);
        
        // Simulate movement time based on distance and speed
        let movement_time = match speed.as_str() {
            "slow" => distance * 2.0,
            "normal" => distance * 1.0,
            "fast" => distance * 0.5,
            _ => distance * 1.0,
        };
        
        tokio::time::sleep(Duration::from_secs_f32(movement_time)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Moved {} {} meters at {} speed", direction, distance, speed),
            execution_time: Duration::default(),
            side_effects: vec![format!("Position changed by {} meters", distance)],
        })
    }
    
    async fn execute_rotate(&self, direction: String, angle: f32, reasoning: String) -> Result<CommandExecutionResult> {
        debug!("Rotating {} {} degrees. Reasoning: {}", direction, angle, reasoning);
        
        // Simulate rotation time
        let rotation_time = angle / 90.0; // 1 second per 90 degrees
        tokio::time::sleep(Duration::from_secs_f32(rotation_time)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Rotated {} {} degrees", direction, angle),
            execution_time: Duration::default(),
            side_effects: vec![format!("Orientation changed by {} degrees", angle)],
        })
    }
    
    async fn execute_speak(&self, text: String, _voice: String, reasoning: String) -> Result<CommandExecutionResult> {
        info!("Speaking: '{}'. Reasoning: {}", text, reasoning);
        
        // Placeholder for TTS - would use actual TTS engine
        println!("🤖 Robot says: {}", text);
        
        // Simulate speech duration
        let speech_duration = text.len() as f32 * 0.1; // ~100ms per character
        tokio::time::sleep(Duration::from_secs_f32(speech_duration)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Spoke: '{}'", text),
            execution_time: Duration::default(),
            side_effects: vec!["Audio output generated".to_string()],
        })
    }
    
    async fn execute_analyze(&self, target: Option<String>, _detail_level: Option<String>, reasoning: String) -> Result<CommandExecutionResult> {
        let target_desc = target.unwrap_or_else(|| "current scene".to_string());
        debug!("Analyzing: {}. Reasoning: {}", target_desc, reasoning);
        
        // Simulate analysis time
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Analysis completed for: {}", target_desc),
            execution_time: Duration::default(),
            side_effects: vec!["Additional sensor data collected".to_string()],
        })
    }
    
    async fn execute_offload(&self, task_description: String, target_agent: Option<String>, _priority: Option<String>, reasoning: String) -> Result<CommandExecutionResult> {
        let agent = target_agent.unwrap_or_else(|| "default_agent".to_string());
        info!("Offloading task to {}: '{}'. Reasoning: {}", agent, task_description, reasoning);
        
        // Placeholder for task offloading - would communicate with AGiXT agents
        println!("📤 Offloading to {}: {}", agent, task_description);
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Task offloaded to {}: {}", agent, task_description),
            execution_time: Duration::default(),
            side_effects: vec!["External task queued".to_string()],
        })
    }
    
    async fn execute_wait(&self, duration: Option<f32>, reasoning: String) -> Result<CommandExecutionResult> {
        let wait_time = duration.unwrap_or(1.0);
        debug!("Waiting for {} seconds. Reasoning: {}", wait_time, reasoning);
        
        tokio::time::sleep(Duration::from_secs_f32(wait_time)).await;
        
        Ok(CommandExecutionResult {
            success: true,
            message: format!("Waited for {} seconds", wait_time),
            execution_time: Duration::default(),
            side_effects: vec![],
        })
    }
}
