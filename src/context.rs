use crate::vision::FrameContext;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::SystemTime;
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MandatoryContext {
    pub system_prompt: String,
    pub current_visual_context: String,
    pub action_history: VecDeque<ActionHistoryEntry>,
    pub environmental_constraints: Vec<String>,
    pub safety_rules: Vec<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionHistoryEntry {
    pub timestamp: SystemTime,
    pub action_type: String,
    pub action_data: String,
    pub success: bool,
    pub feedback: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ContextManager {
    base_system_prompt: String,
    max_history: usize,
    include_timestamp: bool,
    action_history: VecDeque<ActionHistoryEntry>,
    environmental_constraints: Vec<String>,
    safety_rules: Vec<String>,
}

impl ContextManager {
    pub fn new(
        system_prompt: String,
        max_history: usize,
        include_timestamp: bool,
    ) -> Self {
        let mut safety_rules = vec![
            "Do not perform actions that could cause harm to humans".to_string(),
            "Always validate sensor data before taking physical actions".to_string(),
            "Stop immediately if unexpected obstacles are detected".to_string(),
            "Maintain safe distances from humans and fragile objects".to_string(),
            "Do not exceed maximum speed or acceleration limits".to_string(),
        ];
        
        let environmental_constraints = vec![
            "Operating in indoor environment".to_string(),
            "Camera resolution: limited field of view".to_string(),
            "Edge computing: optimize for low latency".to_string(),
        ];
        
        Self {
            base_system_prompt: system_prompt,
            max_history,
            include_timestamp,
            action_history: VecDeque::new(),
            environmental_constraints,
            safety_rules,
        }
    }
    
    pub fn build_context(&self, frame_context: &FrameContext) -> MandatoryContext {
        let current_visual_context = self.format_visual_context(frame_context);
        
        MandatoryContext {
            system_prompt: self.base_system_prompt.clone(),
            current_visual_context,
            action_history: self.action_history.clone(),
            environmental_constraints: self.environmental_constraints.clone(),
            safety_rules: self.safety_rules.clone(),
            timestamp: SystemTime::now(),
        }
    }
    
    pub fn create_model_prompt(&self, context: &MandatoryContext) -> String {
        let mut prompt = String::new();
        
        // System prompt
        prompt.push_str("SYSTEM CONTEXT:\n");
        prompt.push_str(&context.system_prompt);
        prompt.push_str("\n\n");
        
        // Safety rules
        prompt.push_str("SAFETY CONSTRAINTS:\n");
        for rule in &context.safety_rules {
            prompt.push_str(&format!("- {}\n", rule));
        }
        prompt.push_str("\n");
        
        // Environmental constraints
        prompt.push_str("ENVIRONMENTAL CONTEXT:\n");
        for constraint in &context.environmental_constraints {
            prompt.push_str(&format!("- {}\n", constraint));
        }
        prompt.push_str("\n");
        
        // Current visual context
        prompt.push_str("CURRENT VISUAL INPUT:\n");
        prompt.push_str(&context.current_visual_context);
        prompt.push_str("\n\n");
        
        // Recent action history
        if !context.action_history.is_empty() {
            prompt.push_str("RECENT ACTIONS:\n");
            for entry in context.action_history.iter().rev().take(3) {
                let status = if entry.success { "SUCCESS" } else { "FAILED" };
                prompt.push_str(&format!(
                    "- {} [{}]: {}\n",
                    entry.action_type,
                    status,
                    entry.action_data
                ));
                if let Some(feedback) = &entry.feedback {
                    prompt.push_str(&format!("  Feedback: {}\n", feedback));
                }
            }
            prompt.push_str("\n");
        }
        
        // Timestamp if enabled
        if self.include_timestamp {
            prompt.push_str(&format!(
                "TIMESTAMP: {:?}\n\n",
                context.timestamp
            ));
        }
        
        // Action request
        prompt.push_str("REQUIRED OUTPUT:\n");
        prompt.push_str("Based on the current visual input and context, generate the appropriate action command in XML format. ");
        prompt.push_str("Available commands: <move>, <rotate>, <speak>, <analyze>, <offload>. ");
        prompt.push_str("Always include reasoning for your decision. ");
        prompt.push_str("If no action is needed, output <wait> with reasoning.\n\n");
        prompt.push_str("ACTION:");
        
        debug!("Generated model prompt with {} characters", prompt.len());
        prompt
    }
    
    fn format_visual_context(&self, frame_context: &FrameContext) -> String {
        let mut context = String::new();
        
        context.push_str(&format!(
            "Frame size: {}x{}\n",
            frame_context.frame_size.0,
            frame_context.frame_size.1
        ));
        
        context.push_str(&format!(
            "Scene description: {}\n",
            frame_context.scene_description
        ));
        
        if !frame_context.objects.is_empty() {
            context.push_str("Detected objects:\n");
            for obj in &frame_context.objects {
                context.push_str(&format!(
                    "- {} at ({}, {}) with size {}x{}, confidence: {:.2}\n",
                    obj.label,
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.width,
                    obj.bbox.height,
                    obj.confidence
                ));
            }
        } else {
            context.push_str("No objects detected in current frame.\n");
        }
        
        context
    }
    
    pub fn add_action_to_history(
        &mut self,
        action_type: String,
        action_data: String,
        success: bool,
        feedback: Option<String>,
    ) {
        let entry = ActionHistoryEntry {
            timestamp: SystemTime::now(),
            action_type,
            action_data,
            success,
            feedback,
        };
        
        self.action_history.push_back(entry);
        
        // Maintain maximum history size
        while self.action_history.len() > self.max_history {
            self.action_history.pop_front();
        }
    }
    
    pub fn add_environmental_constraint(&mut self, constraint: String) {
        if !self.environmental_constraints.contains(&constraint) {
            self.environmental_constraints.push(constraint);
        }
    }
    
    pub fn add_safety_rule(&mut self, rule: String) {
        if !self.safety_rules.contains(&rule) {
            self.safety_rules.push(rule);
        }
    }
    
    pub fn get_action_history(&self) -> &VecDeque<ActionHistoryEntry> {
        &self.action_history
    }
    
    pub fn clear_action_history(&mut self) {
        self.action_history.clear();
    }
}
