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
    
    pub async fn parse_and_execute(&self, xml_output: &str) -> Result<CommandExecutionResult> {
        debug!("Parsing XML command: {}", xml_output);
        
        // Extract the XML command from the model output
        let command = self.parse_xml_command(xml_output)?;
        
        // Validate command if enabled
        if self.validate_before_execution {
            self.validate_command(&command)?;
        }
        
        // Execute the command
        self.execute_command(command).await
    }
    
    fn parse_xml_command(&self, xml_output: &str) -> Result<ActionCommand> {
        // Clean up the XML output - model might include extra text
        let xml_content = self.extract_xml_from_text(xml_output)?;
        
        // Try to parse different command types
        if let Ok(command) = self.try_parse_move(&xml_content) {
            return Ok(command);
        }
        if let Ok(command) = self.try_parse_rotate(&xml_content) {
            return Ok(command);
        }
        if let Ok(command) = self.try_parse_speak(&xml_content) {
            return Ok(command);
        }
        if let Ok(command) = self.try_parse_analyze(&xml_content) {
            return Ok(command);
        }
        if let Ok(command) = self.try_parse_offload(&xml_content) {
            return Ok(command);
        }
        if let Ok(command) = self.try_parse_wait(&xml_content) {
            return Ok(command);
        }
        
        Err(anyhow!("Failed to parse XML command: {}", xml_content))
    }
    
    fn extract_xml_from_text(&self, text: &str) -> Result<String> {
        // Look for XML tags in the text
        let xml_patterns = vec![
            r"<move[^>]*>.*?</move>",
            r"<rotate[^>]*>.*?</rotate>",
            r"<speak[^>]*>.*?</speak>",
            r"<analyze[^>]*>.*?</analyze>",
            r"<offload[^>]*>.*?</offload>",
            r"<wait[^>]*>.*?</wait>",
            r"<move[^>]*/>",
            r"<rotate[^>]*/>",
            r"<analyze[^>]*/>",
            r"<wait[^>]*/>",
        ];
        
        for pattern in xml_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if let Some(captures) = regex.find(text) {
                    return Ok(captures.as_str().to_string());
                }
            }
        }
        
        // If no XML found, check if it's a simple tag
        if text.contains('<') && text.contains('>') {
            return Ok(text.trim().to_string());
        }
        
        Err(anyhow!("No valid XML command found in text: {}", text))
    }
    
    fn try_parse_move(&self, xml: &str) -> Result<ActionCommand> {
        // Parse move command attributes
        let direction = self.extract_attribute(xml, "direction")?;
        let distance = self.extract_attribute(xml, "distance")?
            .parse::<f32>()
            .map_err(|_| anyhow!("Invalid distance value"))?;
        let speed = self.extract_attribute(xml, "speed").unwrap_or_else(|_| "normal".to_string());
        let reasoning = self.extract_text_content(xml).unwrap_or_default();
        
        Ok(ActionCommand::Move {
            direction,
            distance,
            speed,
            reasoning,
        })
    }
    
    fn try_parse_rotate(&self, xml: &str) -> Result<ActionCommand> {
        let direction = self.extract_attribute(xml, "direction")?;
        let angle = self.extract_attribute(xml, "angle")?
            .parse::<f32>()
            .map_err(|_| anyhow!("Invalid angle value"))?;
        let reasoning = self.extract_text_content(xml).unwrap_or_default();
        
        Ok(ActionCommand::Rotate {
            direction,
            angle,
            reasoning,
        })
    }
    
    fn try_parse_speak(&self, xml: &str) -> Result<ActionCommand> {
        let text = self.extract_text_content(xml)?;
        let voice = self.extract_attribute(xml, "voice").unwrap_or_else(|_| "default".to_string());
        let reasoning = self.extract_attribute(xml, "reasoning").unwrap_or_default();
        
        Ok(ActionCommand::Speak {
            text,
            voice,
            reasoning,
        })
    }
    
    fn try_parse_analyze(&self, xml: &str) -> Result<ActionCommand> {
        let target = self.extract_attribute(xml, "target").ok();
        let detail_level = self.extract_attribute(xml, "detail_level").ok();
        let reasoning = self.extract_text_content(xml).unwrap_or_default();
        
        Ok(ActionCommand::Analyze {
            target,
            detail_level,
            reasoning,
        })
    }
    
    fn try_parse_offload(&self, xml: &str) -> Result<ActionCommand> {
        let task_description = self.extract_text_content(xml)?;
        let target_agent = self.extract_attribute(xml, "target_agent").ok();
        let priority = self.extract_attribute(xml, "priority").ok();
        let reasoning = self.extract_attribute(xml, "reasoning").unwrap_or_default();
        
        Ok(ActionCommand::Offload {
            task_description,
            target_agent,
            priority,
            reasoning,
        })
    }
    
    fn try_parse_wait(&self, xml: &str) -> Result<ActionCommand> {
        let duration = self.extract_attribute(xml, "duration")
            .ok()
            .and_then(|d| d.parse::<f32>().ok());
        let reasoning = self.extract_text_content(xml).unwrap_or_default();
        
        Ok(ActionCommand::Wait {
            duration,
            reasoning,
        })
    }
    
    fn extract_attribute(&self, xml: &str, attr_name: &str) -> Result<String> {
        let pattern = format!(r#"{}="([^"]*)""#, attr_name);
        if let Ok(regex) = regex::Regex::new(&pattern) {
            if let Some(captures) = regex.captures(xml) {
                return Ok(captures[1].to_string());
            }
        }
        
        Err(anyhow!("Attribute {} not found", attr_name))
    }
    
    fn extract_text_content(&self, xml: &str) -> Result<String> {
        if let Some(start) = xml.find('>') {
            if let Some(end) = xml.rfind('<') {
                if start < end {
                    return Ok(xml[start + 1..end].trim().to_string());
                }
            }
        }
        
        Err(anyhow!("No text content found"))
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
