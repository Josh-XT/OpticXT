use anyhow::{Result, anyhow};
use std::time::SystemTime;
use tracing::{info, debug, warn, error};
use nokhwa::{Camera, utils::{RequestedFormat, RequestedFormatType, CameraIndex}};
use nokhwa::pixel_format::RgbFormat;
use crate::vision_basic::Mat;

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub camera_id: i32,
    #[allow(dead_code)]
    pub width: i32,
    #[allow(dead_code)]
    pub height: i32,
    #[allow(dead_code)]
    pub fps: f64,
}

#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f64,
    pub y: f64,
    #[allow(dead_code)]
    pub z: f64,
    pub distance: f64,
    #[allow(dead_code)]
    pub angle: f64,
    #[allow(dead_code)]
    pub intensity: f64,
}

#[derive(Debug, Clone)]
pub struct SensorData {
    #[allow(dead_code)]
    pub timestamp: SystemTime,
    pub frame: Mat,
    pub lidar_points: Vec<LidarPoint>,
    #[allow(dead_code)]
    pub has_lidar: bool,
}

pub struct CameraSystem {
    config: CameraConfig,
    camera: Option<Camera>,
    is_initialized: bool,
    has_hardware_lidar: bool,
}

impl CameraSystem {
    pub fn new(config: CameraConfig) -> Result<Self> {
        info!("Creating CameraSystem for real camera input");
        
        Ok(Self {
            config,
            camera: None,
            is_initialized: false,
            has_hardware_lidar: false, // Will be true on actual Go2/Jetson
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing camera system - scanning for available cameras");
        
        // Detect and list available cameras
        let available_cameras = Self::detect_cameras();
        if available_cameras.is_empty() {
            return Err(anyhow!("No cameras detected on this system"));
        }
        
        info!("Found {} camera(s): {:?}", available_cameras.len(), available_cameras);
        
        // Try to initialize the requested camera, then fallbacks
        let camera_indices = if available_cameras.contains(&(self.config.camera_id as u32)) {
            vec![self.config.camera_id as u32]
        } else {
            available_cameras
        };
        
        for cam_id in camera_indices {
            match self.try_initialize_camera(cam_id).await {
                Ok(_) => {
                    self.config.camera_id = cam_id as i32;
                    info!("Successfully initialized camera {}", cam_id);
                    break;
                }
                Err(e) => {
                    warn!("Failed to initialize camera {}: {}", cam_id, e);
                    continue;
                }
            }
        }
        
        if !self.is_initialized {
            return Err(anyhow!("Failed to initialize any available camera"));
        }
        
        // Check for LiDAR hardware (would be true on Go2/Jetson)
        self.has_hardware_lidar = self.detect_lidar_hardware();
        
        info!("Camera system initialized successfully (LiDAR: {})", self.has_hardware_lidar);
        Ok(())
    }
    
    async fn try_initialize_camera(&mut self, camera_id: u32) -> Result<()> {
        let camera_index = CameraIndex::Index(camera_id);
        let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        
        let mut camera = Camera::new(camera_index, requested_format)?;
        
        // Open the camera stream
        camera.open_stream()?;
        
        // Test capture a frame to ensure it works
        let _test_frame = camera.frame()?;
        
        self.camera = Some(camera);
        self.is_initialized = true;
        
        Ok(())
    }
    
    pub fn detect_cameras() -> Vec<u32> {
        let mut cameras = Vec::new();
        
        // Try camera indices 0-9 (most common range)
        for cam_id in 0..10 {
            let camera_index = CameraIndex::Index(cam_id);
            let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
            
            if let Ok(_camera) = Camera::new(camera_index, requested_format) {
                cameras.push(cam_id);
            }
        }
        
        cameras
    }
    
    fn detect_lidar_hardware(&self) -> bool {
        // Check for Go2/Jetson LiDAR hardware
        // This would check for actual LiDAR devices on the robot
        
        // Check common LiDAR device paths
        let lidar_paths = vec![
            "/dev/ttyUSB0",  // Common USB LiDAR
            "/dev/ttyACM0",  // Arduino/microcontroller LiDAR
            "/dev/lidar",    // Custom LiDAR device
            "/sys/class/lidar", // Potential system class
        ];
        
        for path in lidar_paths {
            if std::path::Path::new(path).exists() {
                info!("Detected potential LiDAR hardware at {}", path);
                return true;
            }
        }
        
        // On Jetson/Go2, we would also check for specific Go2 SDK devices
        false
    }
    
    pub async fn capture_sensor_data(&mut self) -> Result<SensorData> {
        if !self.is_initialized {
            return Err(anyhow!("Camera system not initialized"));
        }
        
        let timestamp = SystemTime::now();
        
        // Capture frame from camera
        let frame = self.capture_camera_frame().await?;
        
        // Capture LiDAR data if available
        let lidar_points = if self.has_hardware_lidar {
            self.capture_real_lidar_data().await?
        } else {
            // Generate minimal simulated LiDAR for standalone mode
            self.generate_minimal_lidar_simulation()
        };
        
        Ok(SensorData {
            timestamp,
            frame,
            lidar_points,
            has_lidar: self.has_hardware_lidar,
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
                    
                    debug!("Captured real camera frame: {}x{}", width, height);
                    
                    // Convert to our Mat format
                    let rgb_data = decoded.into_raw();
                    let mut mat = Mat::new(width, height, 3);
                    mat.data = rgb_data;
                    
                    Ok(mat)
                }
                Err(e) => {
                    error!("Camera frame capture failed: {}", e);
                    Err(anyhow!("Camera frame capture error: {}", e))
                }
            }
        } else {
            Err(anyhow!("Camera not initialized"))
        }
    }
    
    async fn capture_real_lidar_data(&mut self) -> Result<Vec<LidarPoint>> {
        // This would interface with actual Go2 LiDAR hardware
        // For now, we'll implement basic LiDAR communication protocols
        
        // TODO: Implement actual LiDAR driver communication
        // This would use serial/USB communication with the LiDAR sensor
        
        info!("Reading from hardware LiDAR sensor");
        
        // Placeholder for real LiDAR implementation
        // In production, this would:
        // 1. Read from LiDAR serial/USB interface
        // 2. Parse LiDAR data packets
        // 3. Convert to point cloud format
        
        Ok(vec![]) // Real implementation would return actual sensor data
    }
    
    fn generate_minimal_lidar_simulation(&self) -> Vec<LidarPoint> {
        // Generate a minimal LiDAR simulation for standalone video chat mode
        // This is much simpler than the full simulation - just basic obstacle detection
        let mut points = Vec::new();
        
        // Simulate a few key points for basic spatial awareness
        for i in 0..8 {
            let angle = (i as f64) * std::f64::consts::PI / 4.0; // 8 directions
            let distance = 2.0 + (angle.sin() * 0.5); // Vary distance slightly
            
            let x = distance * angle.cos();
            let y = distance * angle.sin();
            
            points.push(LidarPoint {
                x,
                y,
                z: 0.0,
                distance,
                angle,
                intensity: 0.7,
            });
        }
        
        points
    }
    
    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        if lidar_points.is_empty() {
            return Ok(());
        }
        
        debug!("Overlaying {} LiDAR points", lidar_points.len());
        
        let width = frame.width as f64;
        let height = frame.height as f64;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        
        // Draw LiDAR points as colored pixels
        for point in lidar_points {
            // Convert polar coordinates to screen coordinates
            let scale = 100.0; // Scale factor for visualization
            let screen_x = center_x + (point.x * scale);
            let screen_y = center_y + (point.y * scale);
            
            if screen_x >= 0.0 && screen_x < width && screen_y >= 0.0 && screen_y < height {
                let x = screen_x as i32;
                let y = screen_y as i32;
                
                // Color code by distance: close = red, far = blue
                let color = if point.distance < 1.5 {
                    [255, 0, 0] // Red for close objects
                } else if point.distance < 3.0 {
                    [255, 255, 0] // Yellow for medium distance
                } else {
                    [0, 255, 0] // Green for far objects
                };
                
                self.set_pixel_safe(frame, x, y, color);
                
                // Draw a small cross pattern for better visibility
                self.set_pixel_safe(frame, x - 2, y, color);
                self.set_pixel_safe(frame, x + 2, y, color);
                self.set_pixel_safe(frame, x, y - 2, color);
                self.set_pixel_safe(frame, x, y + 2, color);
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
    
    #[allow(dead_code)]
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping camera system");
        
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
    
    #[allow(dead_code)]
    pub fn get_camera_info(&self) -> Result<String> {
        if let Some(ref _camera) = self.camera {
            Ok(format!("Camera {} - Active", self.config.camera_id))
        } else {
            Ok("No camera active".to_string())
        }
    }
}

impl Drop for CameraSystem {
    fn drop(&mut self) {
        if self.is_initialized {
            info!("CameraSystem being dropped, cleaning up");
            if let Some(ref mut camera) = self.camera {
                let _ = camera.stop_stream();
            }
        }
    }
}
