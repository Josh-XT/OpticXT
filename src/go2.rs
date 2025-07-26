use anyhow::{Result, anyhow};
use opencv::{core::Mat, prelude::*};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::time::{interval, sleep};
use tracing::{info, debug, warn, error};

/// Unitree Go2 camera configuration
#[derive(Debug, Clone)]
pub struct Go2CameraConfig {
    pub camera_id: i32,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub exposure: Option<f64>,
    pub gain: Option<f64>,
}

impl Default for Go2CameraConfig {
    fn default() -> Self {
        Self {
            camera_id: 0, // Front camera
            width: 1280,
            height: 720,
            fps: 30.0,
            exposure: None,
            gain: None,
        }
    }
}

/// Go2 LiDAR point for distance overlay
#[derive(Debug, Clone)]
pub struct LidarPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub distance: f32,
    pub intensity: u8,
}

/// Go2 sensor data combining camera and LiDAR
#[derive(Debug, Clone)]
pub struct Go2SensorData {
    pub frame: Mat,
    pub timestamp: SystemTime,
    pub lidar_points: Vec<LidarPoint>,
    pub robot_pose: Option<(f64, f64, f64)>, // x, y, yaw
}

/// Unitree Go2 camera interface
pub struct Go2Camera {
    config: Go2CameraConfig,
    camera: Option<opencv::videoio::VideoCapture>,
    is_running: Arc<Mutex<bool>>,
    last_frame_time: SystemTime,
}

impl Go2Camera {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        info!("Initializing Unitree Go2 camera with config: {:?}", config);
        
        Ok(Self {
            config,
            camera: None,
            is_running: Arc::new(Mutex::new(false)),
            last_frame_time: SystemTime::now(),
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Connecting to Go2 camera (device {})", self.config.camera_id);
        
        // Initialize OpenCV VideoCapture for Go2 camera
        let mut camera = opencv::videoio::VideoCapture::new(
            self.config.camera_id, 
            opencv::videoio::CAP_V4L2
        )?;
        
        if !camera.is_opened()? {
            // Try alternative camera indices for Go2
            warn!("Primary camera not available, trying alternatives...");
            for cam_id in [1, 2, 3, 4] {
                camera = opencv::videoio::VideoCapture::new(cam_id, opencv::videoio::CAP_V4L2)?;
                if camera.is_opened()? {
                    info!("Connected to Go2 camera on device {}", cam_id);
                    self.config.camera_id = cam_id;
                    break;
                }
            }
        }

        if !camera.is_opened()? {
            return Err(anyhow!("Could not open any Go2 camera device"));
        }

        // Configure camera settings for Go2
        camera.set(opencv::videoio::CAP_PROP_FRAME_WIDTH, self.config.width as f64)?;
        camera.set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, self.config.height as f64)?;
        camera.set(opencv::videoio::CAP_PROP_FPS, self.config.fps)?;
        
        // Set exposure and gain if specified
        if let Some(exposure) = self.config.exposure {
            camera.set(opencv::videoio::CAP_PROP_EXPOSURE, exposure)?;
        }
        if let Some(gain) = self.config.gain {
            camera.set(opencv::videoio::CAP_PROP_GAIN, gain)?;
        }

        // Enable auto-focus for Go2 camera
        camera.set(opencv::videoio::CAP_PROP_AUTOFOCUS, 1.0)?;

        self.camera = Some(camera);
        *self.is_running.lock().unwrap() = true;

        info!("Go2 camera initialized successfully");
        Ok(())
    }

    pub async fn capture_frame(&mut self) -> Result<Mat> {
        let camera = self.camera.as_mut()
            .ok_or_else(|| anyhow!("Go2 camera not initialized"))?;

        let mut frame = Mat::default();
        if !camera.read(&mut frame)? {
            return Err(anyhow!("Failed to read frame from Go2 camera"));
        }

        if frame.empty() {
            return Err(anyhow!("Empty frame from Go2 camera"));
        }

        self.last_frame_time = SystemTime::now();
        debug!("Captured frame from Go2 camera: {}x{}", frame.cols(), frame.rows());

        Ok(frame)
    }

    pub fn is_running(&self) -> bool {
        *self.is_running.lock().unwrap()
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 camera");
        *self.is_running.lock().unwrap() = false;
        
        if let Some(mut camera) = self.camera.take() {
            camera.release()?;
        }
        
        Ok(())
    }
}

/// Go2 LiDAR interface (placeholder for future implementation)
pub struct Go2Lidar {
    is_running: Arc<Mutex<bool>>,
}

impl Go2Lidar {
    pub fn new() -> Self {
        Self {
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 LiDAR (placeholder)");
        *self.is_running.lock().unwrap() = true;
        Ok(())
    }

    pub async fn get_scan(&self) -> Result<Vec<LidarPoint>> {
        // Placeholder: return empty scan for now
        // TODO: Implement actual Go2 LiDAR SDK integration
        Ok(Vec::new())
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 LiDAR");
        *self.is_running.lock().unwrap() = false;
        Ok(())
    }
}

/// Combined Go2 sensor system
pub struct Go2SensorSystem {
    camera: Go2Camera,
    lidar: Go2Lidar,
    config: Go2CameraConfig,
}

impl Go2SensorSystem {
    pub fn new(config: Go2CameraConfig) -> Result<Self> {
        let camera = Go2Camera::new(config.clone())?;
        let lidar = Go2Lidar::new();
        
        Ok(Self {
            camera,
            lidar,
            config,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Go2 sensor system");
        
        self.camera.initialize().await?;
        self.lidar.initialize().await?;
        
        info!("Go2 sensor system initialized successfully");
        Ok(())
    }

    pub async fn capture_sensor_data(&mut self) -> Result<Go2SensorData> {
        // Capture frame from camera
        let frame = self.camera.capture_frame().await?;
        let timestamp = SystemTime::now();
        
        // Get LiDAR scan
        let lidar_points = self.lidar.get_scan().await?;
        
        // TODO: Get robot pose from Go2 SDK
        let robot_pose = None;

        Ok(Go2SensorData {
            frame,
            timestamp,
            lidar_points,
            robot_pose,
        })
    }

    pub fn overlay_lidar_distances(&self, frame: &mut Mat, lidar_points: &[LidarPoint]) -> Result<()> {
        use opencv::{core, imgproc};
        
        // Overlay distance information on the frame
        for point in lidar_points.iter().take(50) { // Limit to 50 points for performance
            if point.distance > 0.1 && point.distance < 10.0 {
                // Project 3D point to 2D image coordinates (simplified projection)
                let img_x = ((point.x / point.distance) * 500.0 + frame.cols() as f32 / 2.0) as i32;
                let img_y = ((point.y / point.distance) * 500.0 + frame.rows() as f32 / 2.0) as i32;
                
                if img_x > 0 && img_x < frame.cols() && img_y > 0 && img_y < frame.rows() {
                    // Draw distance circle
                    let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0); // Green
                    let center = core::Point::new(img_x, img_y);
                    imgproc::circle(frame, center, 3, color, -1, imgproc::LINE_8, 0)?;
                    
                    // Add distance text
                    let distance_text = format!("{:.1}m", point.distance);
                    let text_pos = core::Point::new(img_x + 5, img_y - 5);
                    imgproc::put_text(
                        frame,
                        &distance_text,
                        text_pos,
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        imgproc::LINE_8,
                        false,
                    )?;
                }
            }
        }
        
        Ok(())
    }

    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Go2 sensor system");
        
        self.camera.stop().await?;
        self.lidar.stop().await?;
        
        Ok(())
    }
}

/// Go2-specific utility functions
pub mod utils {
    use super::*;
    
    pub fn detect_go2_cameras() -> Result<Vec<i32>> {
        let mut available_cameras = Vec::new();
        
        // Check common Go2 camera device indices
        for cam_id in 0..8 {
            if let Ok(camera) = opencv::videoio::VideoCapture::new(cam_id, opencv::videoio::CAP_V4L2) {
                if camera.is_opened().unwrap_or(false) {
                    available_cameras.push(cam_id);
                }
            }
        }
        
        info!("Detected Go2 cameras: {:?}", available_cameras);
        Ok(available_cameras)
    }
    
    pub fn get_optimal_camera_settings() -> Go2CameraConfig {
        Go2CameraConfig {
            camera_id: 0,
            width: 1280,
            height: 720,
            fps: 30.0,
            exposure: Some(-6.0), // Auto exposure
            gain: Some(50.0),     // Moderate gain for indoor/outdoor
        }
    }
}
