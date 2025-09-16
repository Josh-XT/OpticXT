# OpticXT API Documentation

## REST API Endpoint

OpticXT exposes a single REST API endpoint for inference operations.

### Endpoint

**POST** `/v1/inference`

### Description

This endpoint accepts multipart form data with optional image/video files and text input, and returns text responses from the vision-language model. If the robot is currently executing a task, it can provide status updates based on the current camera feed.

### Request Format

The endpoint accepts `multipart/form-data` with the following optional fields:

- `text` (string, optional): Text input/prompt for the model
- `image` (file, optional): Image file (JPEG, PNG, etc.)
- `video` (file, optional): Video file (currently extracts first frame)
- `task_context` (string, optional): Additional context about the current task

### Response Format

```json
{
  "id": "uuid-string",
  "text": "Model response text",
  "processing_time_ms": 1250,
  "tokens_generated": 45,
  "status": "completed|current_task",
  "current_task": "optional description of ongoing task"
}
```

### Usage Examples

#### 1. Text-only inference
```bash
curl -X POST http://localhost:8080/v1/inference \
  -F "text=What can you see in the camera feed?"
```

#### 2. Image analysis
```bash
curl -X POST http://localhost:8080/v1/inference \
  -F "text=Describe what you see in this image" \
  -F "image=@/path/to/image.jpg"
```

#### 3. Robot task command
```bash
curl -X POST http://localhost:8080/v1/inference \
  -F "text=Move forward and find a red object" \
  -F "task_context=Navigation task in living room"
```

#### 4. Check current task status
```bash
# If a task is running, send request without input to get status
curl -X POST http://localhost:8080/v1/inference
```

### Starting the API Server

To start OpticXT in API server mode:

```bash
# Start the API server on default port 8080
./opticxt --api-server

# Start on custom port
./opticxt --api-server --api-port 3000

# With custom model
./opticxt --api-server --model-path /path/to/model.gguf

# CPU-only mode (no CUDA)
cargo run --no-default-features -- --api-server
```

### Features

- **Multimodal Input**: Supports text, images, and video files
- **Task Tracking**: Automatically tracks ongoing robot tasks
- **Status Updates**: Provides real-time status of current operations
- **Vision Integration**: Uses camera feed when no image is provided
- **CORS Enabled**: Ready for web frontend integration

### Error Responses

```json
{
  "error": "Error description",
  "code": "ERROR_CODE"
}
```

Common error codes:
- `INVALID_MULTIPART`: Malformed request data
- `IMAGE_PARSE_ERROR`: Invalid image format
- `INFERENCE_ERROR`: Model processing failed
- `STATUS_ERROR`: Failed to get robot status

### Integration Notes

- The API is designed for integration with web frontends, mobile apps, or other services
- Supports real-time robot control and monitoring
- Can be used alongside the regular robot control mode
- Thread-safe for concurrent requests
