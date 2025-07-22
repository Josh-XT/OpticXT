use anyhow::{Result, anyhow};
use opencv::{
    core::{Mat, Size, CV_8UC3, Rect, Point, Scalar},
    imgproc,
    videoio::{VideoCapture, VideoCaptureAPIs},
    objdetect::CascadeClassifier,
    dnn::{Net, blob_from_image, DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    prelude::*,
};
use std::collections::HashMap;
use std::time::SystemTime;
use std::path::Path;
use tracing::{debug, warn, error, info};
use crate::go2::{Go2SensorSystem, Go2SensorData, Go2CameraConfig, LidarPoint};

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct FrameContext {
    pub timestamp: std::time::SystemTime,
    pub objects: Vec<DetectedObject>,
    pub scene_description: String,
    pub frame_size: (u32, u32),
    pub lidar_points: Vec<LidarPoint>,
}

pub struct VisionProcessor {
    go2_system: Go2SensorSystem,
    yolo_net: Option<Net>,
    face_cascade: Option<CascadeClassifier>,
    confidence_threshold: f32,
    nms_threshold: f32,
    yolo_classes: Vec<String>,
    frame_counter: u32,
}

impl VisionProcessor {
    pub fn new(
        camera_device: usize,
        width: u32,
        height: u32,
        confidence_threshold: f32,
        vision_model: String,
    ) -> Result<Self> {
        info!("Initializing VisionProcessor with real Go2 camera input");
        
        // Configure Go2 camera
        let go2_config = Go2CameraConfig {
            camera_id: camera_device as i32,
            width: width as i32,
            height: height as i32,
            fps: 30.0,
            exposure: Some(-6.0),
            gain: Some(50.0),
        };
        
        let go2_system = Go2SensorSystem::new(go2_config)?;
        
        // Initialize YOLO network
        let yolo_net = Self::load_yolo_model(&vision_model).ok();
        if yolo_net.is_none() {
            warn!("Failed to load YOLO model, falling back to basic detection");
        }
        
        // Initialize face detection cascade
        let face_cascade = Self::load_face_cascade().ok();
        if face_cascade.is_none() {
            warn!("Failed to load face cascade classifier");
        }
        
        // COCO class names for YOLO
        let yolo_classes = vec![
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ].iter().map(|&s| s.to_string()).collect();
        
        Ok(Self {
            go2_system,
            yolo_net,
            face_cascade,
            confidence_threshold,
            nms_threshold: 0.4,
            yolo_classes,
            frame_counter: 0,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 sensor system");
        self.go2_system.initialize().await?;
        Ok(())
    }
    
    fn load_yolo_model(model_path: &str) -> Result<Net> {
        // Try to load YOLOv4 or YOLOv5 model
        let weights_path = format!("{}.weights", model_path);
        let config_path = format!("{}.cfg", model_path);
        
        if Path::new(&weights_path).exists() && Path::new(&config_path).exists() {
            info!("Loading YOLO model from {} and {}", weights_path, config_path);
            let mut net = opencv::dnn::read_net(&weights_path, &config_path, "")?;
            net.set_preferable_backend(DNN_BACKEND_OPENCV)?;
            net.set_preferable_target(DNN_TARGET_CPU)?;
            Ok(net)
        } else {
            Err(anyhow!("YOLO model files not found at {}", model_path))
        }
    }
    
    fn load_face_cascade() -> Result<CascadeClassifier> {
        // Try common locations for the face cascade
        let cascade_paths = vec![
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "./models/haarcascade_frontalface_alt.xml",
        ];
        
        for path in cascade_paths {
            if Path::new(path).exists() {
                info!("Loading face cascade from {}", path);
                return CascadeClassifier::new(path);
            }
        }
        
        Err(anyhow!("Face cascade classifier not found"))
    }
    
    pub async fn capture_frame(&mut self) -> Result<Mat> {
        self.frame_counter += 1;
        debug!("Capturing frame #{} from Go2 camera", self.frame_counter);
        
        let sensor_data = self.go2_system.capture_sensor_data().await?;
        Ok(sensor_data.frame)
    }
    
    pub async fn process_frame(&mut self, frame: &Mat) -> Result<FrameContext> {
        let timestamp = SystemTime::now();
        
        // Get sensor data including LiDAR points
        let sensor_data = self.go2_system.capture_sensor_data().await?;
        
        // Detect objects using available models
        let mut objects = Vec::new();
        
        // Try YOLO detection first
        if let Some(ref mut yolo_net) = self.yolo_net {
            match self.detect_objects_yolo(frame, yolo_net).await {
                Ok(mut yolo_objects) => objects.append(&mut yolo_objects),
                Err(e) => warn!("YOLO detection failed: {}", e),
            }
        }
        
        // Add face detection
        if let Some(ref face_cascade) = self.face_cascade {
            match self.detect_faces(frame, face_cascade) {
                Ok(mut faces) => objects.append(&mut faces),
                Err(e) => warn!("Face detection failed: {}", e),
            }
        }
        
        // If no models available, use basic detection
        if self.yolo_net.is_none() && self.face_cascade.is_none() {
            objects = self.basic_detection(frame)?;
        }
        
        // Filter by confidence threshold
        objects.retain(|obj| obj.confidence >= self.confidence_threshold);
        
        let scene_description = self.generate_scene_description(&objects, &sensor_data.lidar_points);
        
        Ok(FrameContext {
            timestamp,
            objects,
            scene_description,
            frame_size: (frame.cols() as u32, frame.rows() as u32),
            lidar_points: sensor_data.lidar_points,
        })
    }
    
    async fn detect_objects_yolo(&self, frame: &Mat, yolo_net: &mut Net) -> Result<Vec<DetectedObject>> {
        debug!("Running YOLO object detection");
        
        // Create blob from image
        let blob = blob_from_image(
            frame,
            1.0 / 255.0,
            Size::new(416, 416),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            opencv::core::CV_32F,
        )?;
        
        // Set input to the network
        yolo_net.set_input(&blob, "", 1.0, Scalar::default())?;
        
        // Get output layer names
        let output_names = yolo_net.get_unconnected_out_layers_names()?;
        let mut outputs = opencv::core::Vector::<Mat>::new();
        yolo_net.forward(&mut outputs, &output_names)?;
        
        let mut objects = Vec::new();
        let (img_width, img_height) = (frame.cols() as f32, frame.rows() as f32);
        
        // Process each output
        for output in outputs.iter() {
            let rows = output.rows();
            let cols = output.cols();
            
            for i in 0..rows {
                let row = output.row(i)?;
                let scores = row.col_range(&opencv::core::Range::new(5, cols)?)?;
                
                let mut min_val = 0.0;
                let mut max_val = 0.0;
                let mut min_loc = Point::default();
                let mut max_loc = Point::default();
                
                opencv::core::min_max_loc(
                    &scores,
                    Some(&mut min_val),
                    Some(&mut max_val),
                    Some(&mut min_loc),
                    Some(&mut max_loc),
                    &opencv::core::no_array(),
                )?;
                
                if max_val > self.confidence_threshold as f64 {
                    let class_id = max_loc.x as usize;
                    if class_id < self.yolo_classes.len() {
                        let center_x = *row.at_2d::<f32>(0, 0)? * img_width;
                        let center_y = *row.at_2d::<f32>(0, 1)? * img_height;
                        let width = *row.at_2d::<f32>(0, 2)? * img_width;
                        let height = *row.at_2d::<f32>(0, 3)? * img_height;
                        
                        let x = (center_x - width / 2.0) as i32;
                        let y = (center_y - height / 2.0) as i32;
                        
                        objects.push(DetectedObject {
                            label: self.yolo_classes[class_id].clone(),
                            confidence: max_val as f32,
                            bbox: BoundingBox {
                                x: x.max(0),
                                y: y.max(0),
                                width: width as i32,
                                height: height as i32,
                            },
                        });
                    }
                }
            }
        }
        
        debug!("YOLO detected {} objects", objects.len());
        Ok(objects)
    }
    
    fn detect_faces(&self, frame: &Mat, face_cascade: &CascadeClassifier) -> Result<Vec<DetectedObject>> {
        debug!("Running face detection");
        
        // Convert to grayscale for face detection
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Detect faces
        let mut faces = opencv::core::Vector::<Rect>::new();
        face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            3,
            0,
            Size::new(30, 30),
            Size::default(),
        )?;
        
        let mut face_objects = Vec::new();
        for face in faces.iter() {
            face_objects.push(DetectedObject {
                label: "face".to_string(),
                confidence: 0.8, // Face cascade doesn't provide confidence scores
                bbox: BoundingBox {
                    x: face.x,
                    y: face.y,
                    width: face.width,
                    height: face.height,
                },
            });
        }
        
        debug!("Face detection found {} faces", face_objects.len());
        Ok(face_objects)
    }
    
    async fn detect_objects(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        // This method is now deprecated - detection happens in process_frame
        self.basic_detection(frame)
    }
    
    fn basic_detection(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        debug!("Running basic object detection fallback");
        
        // Basic detection using image analysis when no models are available
        let mut objects = Vec::new();
        
        // Analyze image properties for basic detection
        let (width, height) = (frame.cols(), frame.rows());
        
        // Simple heuristic-based detection for demo
        // In a real scenario, this could use edge detection, color analysis, etc.
        
        // Simulate detection based on frame properties
        if width > 640 && height > 480 {
            objects.push(DetectedObject {
                label: "scene".to_string(),
                confidence: 0.6,
                bbox: BoundingBox {
                    x: width / 4,
                    y: height / 4,
                    width: width / 2,
                    height: height / 2,
                },
            });
        }
        
        debug!("Basic detection found {} objects", objects.len());
        Ok(objects)
    }
    
    fn generate_scene_description(&self, objects: &[DetectedObject], lidar_points: &[LidarPoint]) -> String {
        let mut description = String::new();
        
        if objects.is_empty() {
            description.push_str("Empty scene with no detected objects. ");
        } else {
            description.push_str(&format!("Scene contains {} object(s): ", objects.len()));
            
            for (i, obj) in objects.iter().enumerate() {
                if i > 0 {
                    description.push_str(", ");
                }
                description.push_str(&format!(
                    "{} (confidence: {:.2}, position: {},{}, size: {}x{})",
                    obj.label,
                    obj.confidence,
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.width,
                    obj.bbox.height
                ));
            }
            description.push_str(". ");
        }
        
        // Add LiDAR information if available
        if !lidar_points.is_empty() {
            let close_objects = lidar_points.iter()
                .filter(|p| p.distance < 2.0)
                .count();
            let far_objects = lidar_points.iter()
                .filter(|p| p.distance >= 2.0 && p.distance < 5.0)
                .count();
            
            description.push_str(&format!(
                "LiDAR shows {} close objects (<2m), {} mid-range objects (2-5m). ",
                close_objects, far_objects
            ));
            
            if let Some(closest) = lidar_points.iter().min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()) {
                description.push_str(&format!("Closest object at {:.1}m. ", closest.distance));
            }
        }
        
        description
    }
    
    pub fn annotate_frame(&self, frame: &mut Mat, context: &FrameContext) -> Result<()> {
        debug!("Annotating frame with {} objects and {} LiDAR points", 
               context.objects.len(), context.lidar_points.len());
        
        // Draw object bounding boxes and labels
        for obj in &context.objects {
            let color = match obj.label.as_str() {
                "person" | "face" => Scalar::new(0.0, 255.0, 0.0, 0.0), // Green for people
                "car" | "truck" | "bus" => Scalar::new(255.0, 0.0, 0.0, 0.0), // Red for vehicles
                _ => Scalar::new(0.0, 0.0, 255.0, 0.0), // Blue for other objects
            };
            
            // Draw bounding box
            let pt1 = Point::new(obj.bbox.x, obj.bbox.y);
            let pt2 = Point::new(obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height);
            imgproc::rectangle(frame, pt1, pt2, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw label with confidence
            let label = format!("{}: {:.2}", obj.label, obj.confidence);
            let text_pos = Point::new(obj.bbox.x, obj.bbox.y - 10);
            imgproc::put_text(
                frame,
                &label,
                text_pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
        
        // Overlay LiDAR distance information
        if !context.lidar_points.is_empty() {
            self.go2_system.overlay_lidar_distances(frame, &context.lidar_points)?;
        }
        
        // Add frame info
        let info_text = format!("Frame: {} | Objects: {} | LiDAR: {}", 
                               self.frame_counter, context.objects.len(), context.lidar_points.len());
        let info_pos = Point::new(10, 30);
        imgproc::put_text(
            frame,
            &info_text,
            info_pos,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;
        
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping VisionProcessor");
        self.go2_system.stop().await?;
        Ok(())
    }
}
