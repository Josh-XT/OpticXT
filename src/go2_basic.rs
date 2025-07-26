use anyhow::{Result, anyhow};
use std::time::SystemTime;
use tracing::{info, debug, warn};
use nokhwa::{Camera, utils::{RequestedFormat, RequestedFormatType, CameraIndex}};
use nokhwa::pixel_format::RgbFormat;
use crate::vision_basic::Mat;

#[derive(Debug, Clone)]
pub struct Go2CameraConfig {
    pub camera_id: i32,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub exposure: Option<f64>,
    pub gain: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub distance: f64,
    pub angle: f64,
    pub intensity: f64,
}

#[derive(Debug, Clone)]
pub struct Go2SensorData {
    pub timestamp: SystemTime,
    pub frame: Mat,
    pub lidar_points: Vec<LidarPoint>,
}

pub struct Go2SensorSystem {
    config: Go2CameraConfig,
    camera: Option<Camera>,
    is_initialized: bool,
}

impl Go2SensorSystem {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        info!("Creating Go2SensorSystem with nokhwa camera (OpenCV alternative)");
        
        Ok(Self {
            config,
            camera: None,
            is_initialized: false,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 camera system using nokhwa");
        
        // Initialize camera using nokhwa
        let camera_index = CameraIndex::Index(self.config.camera_id as u32);
        let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        
        match Camera::new(camera_index, requested_format) {
            Ok(camera) => {
                self.camera = Some(camera);
                info!("Camera initialized successfully on device {}", self.config.camera_id);
            }
            Err(e) => {
                warn!("Failed to initialize camera on device {}: {}", self.config.camera_id, e);
                // Try fallback devices
                for cam_id in 1..5 {
                    if cam_id == self.config.camera_id {
                        continue;
                    }
                    
                    let fallback_index = CameraIndex::Index(cam_id as u32);
                    if let Ok(camera) = Camera::new(fallback_index, requested_format.clone()) {
                        self.camera = Some(camera);
                        info!("Using fallback camera on device {}", cam_id);
                        break;
                    }
                }
                
                if self.camera.is_none() {
                    return Err(anyhow!("Failed to initialize any camera device"));
                }
            }
        }
        
        // Apply camera settings if possible
        if let Some(ref mut camera) = self.camera {
            match camera.open_stream() {
                Ok(_) => info!("Camera stream opened successfully"),
                Err(e) => warn!("Failed to open camera stream: {}", e),
            }
        }
        
        self.is_initialized = true;
        info!("Go2 sensor system initialized successfully");
        Ok(())
    }
    
    pub async fn capture_sensor_data(&mut self) -> Result<Go2SensorData> {
        if !self.is_initialized {
            return Err(anyhow!("Go2 sensor system not initialized"));
        }
        
        let timestamp = SystemTime::now();
        
        // Capture frame from camera
        let frame = match self.capture_camera_frame().await {
            Ok(frame) => frame,
            Err(e) => {
                warn!("Failed to capture camera frame: {}, using placeholder", e);
                self.create_placeholder_frame()
            }
        };
        
        // Capture LiDAR data (simulated for now)
        let lidar_points = self.capture_lidar_data().await?;
        
        Ok(Go2SensorData {
            timestamp,
            frame,
            lidar_points,
        })
    }
    
    async fn capture_camera_frame(&mut self) -> Result<Mat> {
        if let Some(ref mut camera) = self.camera {
            match camera.frame() {
                Ok(frame) => {
                    // Decode the frame to RGB
                    let decoded = frame.decode_image::<RgbFormat>()?;
                    let width = decoded.width();
                    let height = decoded.height();
                    
                    debug!("Captured frame: {}x{}", width, height);
                    
                    // Convert to our Mat format
                    let rgb_data = decoded.into_raw();
                    let mut mat = Mat::new(width, height, 3);
                    mat.data = rgb_data;
                    
                    Ok(mat)
                }
                Err(e) => {
                    warn!("Camera frame capture failed: {}", e);
                    Err(anyhow!("Camera frame capture error: {}", e))
                }
            }
        } else {
            Err(anyhow!("Camera not initialized"))
        }
    }
    
    fn create_placeholder_frame(&self) -> Mat {
        // Create a placeholder frame when camera is not available
        let width = self.config.width as u32;
        let height = self.config.height as u32;
        let mut frame = Mat::new(width, height, 3);
        
        // Fill with a gradient pattern for testing
        for y in 0..height {
            for x in 0..width {
                let pixel_index = ((y * width + x) * 3) as usize;
                if pixel_index + 2 < frame.data.len() {
                    frame.data[pixel_index] = (x * 255 / width) as u8;     // R
                    frame.data[pixel_index + 1] = (y * 255 / height) as u8; // G
                    frame.data[pixel_index + 2] = 128;                       // B
                }
            }
        }
        
        debug!("Created placeholder frame {}x{}", width, height);
        frame
    }
    
    async fn capture_lidar_data(&mut self) -> Result<Vec<LidarPoint>> {
        // For now, simulate LiDAR data
        // In a real implementation, this would interface with the Go2's LiDAR sensor
        let mut lidar_points = Vec::new();
        
        // Simulate some LiDAR points in a 360-degree scan
        let num_points = 36; // One point every 10 degrees
        for i in 0..num_points {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_points as f64);
            let distance = 1.0 + (angle.sin() * 2.0).abs(); // Simulate varying distances
            
            let x = distance * angle.cos();
            let y = distance * angle.sin();
            let z = 0.0; // Assume flat ground for simulation
            
            lidar_points.push(LidarPoint {
                x,
                y,
                z,
                distance,
                angle,
                intensity: 0.8, // Simulated intensity
            });
        }
        
        debug!("Generated {} simulated LiDAR points", lidar_points.len());
        Ok(lidar_points)
    }
    
    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        debug!("Overlaying {} LiDAR points on frame", lidar_points.len());
        
        let width = frame.width as f64;
        let height = frame.height as f64;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        
        // Draw LiDAR points as colored pixels
        for point in lidar_points {
            // Convert polar coordinates to screen coordinates
            let scale = 50.0; // Scale factor for visualization
            let screen_x = center_x + (point.x * scale);
            let screen_y = center_y + (point.y * scale);
            
            if screen_x >= 0.0 && screen_x < width && screen_y >= 0.0 && screen_y < height {
                let x = screen_x as i32;
                let y = screen_y as i32;
                
                // Color code by distance: close = red, far = blue
                let color = if point.distance < 1.0 {
                    [255, 0, 0] // Red for close objects
                } else if point.distance < 3.0 {
                    [255, 255, 0] // Yellow for medium distance
                } else {
                    [0, 0, 255] // Blue for far objects
                };
                
                self.set_pixel_safe(frame, x, y, color);
                
                // Draw a small cross pattern for better visibility
                self.set_pixel_safe(frame, x - 1, y, color);
                self.set_pixel_safe(frame, x + 1, y, color);
                self.set_pixel_safe(frame, x, y - 1, color);
                self.set_pixel_safe(frame, x, y + 1, color);
            }
        }
        
        Ok(())
    }
    
    fn set_pixel_safe(&self, frame: &mut Mat, x: i32, y: i32, color: [u8; 3]) {
        if x >= 0 && y >= 0 && x < frame.width as i32 && y < frame.height as i32 {
            let pixel_index = ((y as u32 * frame.width + x as u32) * frame.channels) as usize;
            if pixel_index + 2 < frame.data.len() {
                frame.data[pixel_index] = color[0];
                frame.data[pixel_index + 1] = color[1];
                frame.data[pixel_index + 2] = color[2];
            }
        }
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 sensor system");
        
        if let Some(ref mut camera) = self.camera {
            match camera.stop_stream() {
                Ok(_) => info!("Camera stream stopped successfully"),
                Err(e) => warn!("Error stopping camera stream: {}", e),
            }
        }
        
        self.camera = None;
        self.is_initialized = false;
        
        Ok(())
    }
    
    pub fn get_available_cameras() -> Vec<String> {
        info!("Scanning for available cameras");
        let mut cameras = Vec::new();
        
        // Try to detect cameras using nokhwa
        for cam_id in 0..10 {
            let camera_index = CameraIndex::Index(cam_id);
            let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
            
            if let Ok(_camera) = Camera::new(camera_index, requested_format) {
                cameras.push(format!("Camera {}", cam_id));
                info!("Found camera at index {}", cam_id);
            }
        }
        
        if cameras.is_empty() {
            warn!("No cameras detected");
        } else {
            info!("Detected {} camera(s)", cameras.len());
        }
        
        cameras
    }
}

impl Drop for Go2SensorSystem {
    fn drop(&mut self) {
        if self.is_initialized {
            info!("Go2SensorSystem being dropped, cleaning up");
            // Note: Can't use async in Drop, so we do basic cleanup
            if let Some(ref mut camera) = self.camera {
                let _ = camera.stop_stream();
            }
        }
    }
}
