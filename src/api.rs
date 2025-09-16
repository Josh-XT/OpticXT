use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
    routing::post,
    Router,
};
use bytes::Bytes;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::models::GemmaModel;

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub text: Option<String>,
    pub task_context: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub text: String,
    pub processing_time_ms: u128,
    pub tokens_generated: usize,
    pub status: String,
    pub current_task: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

pub struct ApiState {
    pub model: Arc<Mutex<GemmaModel>>,
    pub current_task: Arc<Mutex<Option<String>>>,
}

impl ApiState {
    pub fn new(model: GemmaModel) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            current_task: Arc::new(Mutex::new(None)),
        }
    }
}

pub fn create_api_router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/inference", post(inference_handler))
        .with_state(Arc::new(state))
        .layer(CorsLayer::permissive())
}

async fn inference_handler(
    State(state): State<Arc<ApiState>>,
    mut multipart: Multipart,
) -> Result<Json<InferenceResponse>, (StatusCode, Json<ErrorResponse>)> {
    let request_id = Uuid::new_v4().to_string();
    info!("Processing inference request: {}", request_id);

    let mut text_input: Option<String> = None;
    let mut image_data: Option<DynamicImage> = None;
    let mut video_data: Option<Vec<u8>> = None;
    let mut task_context: Option<String> = None;

    // Parse multipart form data
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        error!("Failed to parse multipart field: {}", e);
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid multipart data".to_string(),
                code: "INVALID_MULTIPART".to_string(),
            }),
        )
    })? {
        let field_name = field.name().unwrap_or("unknown").to_string();
        debug!("Processing field: {}", field_name);

        match field_name.as_str() {
            "text" => {
                text_input = Some(field.text().await.map_err(|e| {
                    error!("Failed to read text field: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "Failed to read text input".to_string(),
                            code: "TEXT_READ_ERROR".to_string(),
                        }),
                    )
                })?);
            }
            "task_context" => {
                task_context = Some(field.text().await.map_err(|e| {
                    error!("Failed to read task_context field: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "Failed to read task context".to_string(),
                            code: "CONTEXT_READ_ERROR".to_string(),
                        }),
                    )
                })?);
            }
            "image" => {
                let content_type = field.content_type().unwrap_or("").to_string();
                let image_bytes = field.bytes().await.map_err(|e| {
                    error!("Failed to read image bytes: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "Failed to read image data".to_string(),
                            code: "IMAGE_READ_ERROR".to_string(),
                        }),
                    )
                })?;

                if is_image_content_type(&content_type) {
                    image_data = Some(parse_image_bytes(image_bytes).map_err(|e| {
                        error!("Failed to parse image: {}", e);
                        (
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: "Invalid image format".to_string(),
                                code: "IMAGE_PARSE_ERROR".to_string(),
                            }),
                        )
                    })?);
                }
            }
            "video" => {
                let content_type = field.content_type().unwrap_or("").to_string();
                let video_bytes = field.bytes().await.map_err(|e| {
                    error!("Failed to read video bytes: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: "Failed to read video data".to_string(),
                            code: "VIDEO_READ_ERROR".to_string(),
                        }),
                    )
                })?;

                if is_video_content_type(&content_type) {
                    video_data = Some(video_bytes.to_vec());
                    // For now, extract first frame from video as image
                    // In a full implementation, you might want to process the entire video
                    if let Ok(first_frame) = extract_first_frame_from_video(&video_bytes) {
                        image_data = Some(first_frame);
                    }
                }
            }
            _ => {
                warn!("Unknown field in multipart data: {}", field_name);
            }
        }
    }

    // Check current task status
    let current_task = {
        let task_lock = state.current_task.lock().await;
        task_lock.clone()
    };

    // If there's a current task and no new input, report current status
    if let Some(ref current_task_desc) = current_task {
        if text_input.is_none() && image_data.is_none() && video_data.is_none() {
            // Get current camera feed and ask model what robot is doing
            let status_response = get_current_robot_status(&state, current_task_desc, &request_id).await?;
            return Ok(Json(status_response));
        }
    }

    // Determine the prompt based on available inputs
    let prompt = build_inference_prompt(&text_input, &task_context, &current_task);

    // Generate response using the model directly
    let generation_result = {
        let mut model = state.model.lock().await;
        
        if let Some(image) = image_data {
            debug!("Generating multimodal response with image");
            model.generate_with_image(&prompt, image).await
        } else {
            debug!("Generating text-only response");
            model.generate(&prompt).await
        }
    }.map_err(|e| {
        error!("Model inference failed: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Model inference failed".to_string(),
                code: "INFERENCE_ERROR".to_string(),
            }),
        )
    })?;

    // Update current task if this looks like a task command
    if is_task_command(&generation_result.text) {
        let mut task_lock = state.current_task.lock().await;
        *task_lock = Some(generation_result.text.clone());
    }

    // Check if this indicates task completion
    if is_task_completion(&generation_result.text) {
        let mut task_lock = state.current_task.lock().await;
        *task_lock = None;
    }

    let response = InferenceResponse {
        id: request_id,
        text: generation_result.text,
        processing_time_ms: generation_result.processing_time_ms,
        tokens_generated: generation_result.tokens_generated,
        status: "completed".to_string(),
        current_task: current_task,
    };

    info!("Inference request {} completed successfully", response.id);
    Ok(Json(response))
}

async fn get_current_robot_status(
    state: &Arc<ApiState>,
    current_task: &str,
    request_id: &str,
) -> Result<InferenceResponse, (StatusCode, Json<ErrorResponse>)> {
    info!("Getting current robot status for task: {}", current_task);

    // Try to get current camera frame from pipeline
    let prompt = format!(
        "The robot is currently executing this task: {}. Based on the current camera feed, briefly describe what the robot is currently doing. Be specific about its current action or position.",
        current_task
    );

    // Get current status using model directly
    let generation_result = {
        let mut model = state.model.lock().await;
        model.generate(&prompt).await
    }.map_err(|e| {
        error!("Failed to get robot status: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get robot status".to_string(),
                code: "STATUS_ERROR".to_string(),
            }),
        )
    })?;

    Ok(InferenceResponse {
        id: request_id.to_string(),
        text: generation_result.text,
        processing_time_ms: generation_result.processing_time_ms,
        tokens_generated: generation_result.tokens_generated,
        status: "current_task".to_string(),
        current_task: Some(current_task.to_string()),
    })
}

fn build_inference_prompt(
    text_input: &Option<String>,
    task_context: &Option<String>,
    current_task: &Option<String>,
) -> String {
    let mut prompt = String::new();

    if let Some(context) = task_context {
        prompt.push_str(&format!("Context: {}\n\n", context));
    }

    if let Some(task) = current_task {
        prompt.push_str(&format!("Current ongoing task: {}\n\n", task));
    }

    if let Some(text) = text_input {
        prompt.push_str(text);
    } else {
        prompt.push_str("Analyze the provided image/video and describe what you see. If there's a current task, provide an update on the robot's progress.");
    }

    prompt
}

fn is_task_command(text: &str) -> bool {
    let task_keywords = ["move", "navigate", "go to", "find", "search", "execute", "perform", "start"];
    let text_lower = text.to_lowercase();
    task_keywords.iter().any(|keyword| text_lower.contains(keyword))
}

fn is_task_completion(text: &str) -> bool {
    let completion_keywords = ["completed", "finished", "done", "arrived", "success", "accomplished"];
    let text_lower = text.to_lowercase();
    completion_keywords.iter().any(|keyword| text_lower.contains(keyword))
}

fn is_image_content_type(content_type: &str) -> bool {
    content_type.starts_with("image/")
}

fn is_video_content_type(content_type: &str) -> bool {
    content_type.starts_with("video/")
}

fn parse_image_bytes(bytes: Bytes) -> Result<DynamicImage, image::ImageError> {
    image::load_from_memory(&bytes)
}

fn extract_first_frame_from_video(video_bytes: &[u8]) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    // This is a simplified implementation
    // In a real application, you'd use a video processing library like ffmpeg-rs
    // For now, we'll try to parse it as an image (which will work for single-frame formats)
    match image::load_from_memory(video_bytes) {
        Ok(image) => Ok(image),
        Err(_) => {
            // If it's not a static image, we'd need proper video processing
            // For now, return an error
            Err("Video processing not implemented - use image files or implement video frame extraction".into())
        }
    }
}

pub async fn start_api_server(
    model: GemmaModel,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = ApiState::new(model);
    let app = create_api_router(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
