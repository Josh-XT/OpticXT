use anyhow::{Result, anyhow};
use image::{RgbImage, ImageBuffer, Rgb, DynamicImage};
use std::time::SystemTime;
use tracing::{debug, info};
use crate::camera::{SensorData, LidarPoint};

// Simple matrix type for basic image operations
#[derive(Debug, Clone)]
pub struct Mat {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl Mat {
    pub fn new(width: u32, height: u32, channels: u32) -> Self {
        let size = (width * height * channels) as usize;
        Self {
            data: vec![0u8; size],
            width,
            height,
            channels,
        }
    }

    pub fn from_image(img: &DynamicImage) -> Self {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        Self {
            data: rgb_img.into_raw(),
            width,
            height,
            channels: 3,
        }
    }

    pub fn to_image(&self) -> Result<DynamicImage> {
        if self.channels == 3 {
            let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                self.width,
                self.height,
                self.data.clone(),
            )
            .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            Ok(DynamicImage::ImageRgb8(img_buffer))
        } else {
            Err(anyhow!("Unsupported channel count: {}", self.channels))
        }
    }

    pub fn rows(&self) -> u32 {
        self.height
    }

    pub fn cols(&self) -> u32 {
        self.width
    }
}

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
    confidence_threshold: f32,
    frame_counter: usize,
}

impl VisionProcessor {
    pub fn new(
        _camera_device: usize,
        _width: u32,
        _height: u32,
        confidence_threshold: f32,
        _vision_model: String,
    ) -> Result<Self> {
        info!("Initializing VisionProcessor with basic image processing (OpenCV alternative)");
        
        Ok(Self {
            confidence_threshold,
            frame_counter: 0,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("VisionProcessor initialized (basic vision mode)");
        Ok(())
    }
    
    pub async fn process_frame(&mut self, frame: &Mat, sensor_data: &SensorData) -> Result<FrameContext> {
        let timestamp = SystemTime::now();
        
        // Perform basic object detection using image analysis
        let objects = self.basic_detection(frame)?;
        
        // Filter by confidence threshold
        let filtered_objects: Vec<DetectedObject> = objects
            .into_iter()
            .filter(|obj| obj.confidence >= self.confidence_threshold)
            .collect();
        
        let scene_description = self.generate_scene_description(&filtered_objects, &sensor_data.lidar_points);
        
        Ok(FrameContext {
            timestamp,
            objects: filtered_objects,
            scene_description,
            frame_size: (frame.cols(), frame.rows()),
            lidar_points: sensor_data.lidar_points.clone(),
        })
    }
    
    fn basic_detection(&self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        debug!("Running real object detection using image analysis");
        
        let mut objects = Vec::new();
        let (width, height) = (frame.cols(), frame.rows());
        
        // Convert to image for analysis
        let image = frame.to_image()?;
        
        // Real computer vision detection using image analysis
        
        // 1. Person detection using skin tone and face-like regions
        let person_objects = self.detect_people(&image)?;
        objects.extend(person_objects);
        
        // 2. Object detection using edge and color analysis
        let shape_objects = self.detect_objects_by_shape(&image)?; 
        objects.extend(shape_objects);
        
        // 3. Motion detection (if we had previous frame)
        let motion_objects = self.detect_motion_regions(&image)?;
        objects.extend(motion_objects);
        
        // 4. Scene analysis for furniture/large objects
        let furniture_objects = self.detect_furniture(&image)?;
        objects.extend(furniture_objects);
        
        debug!("Real detection found {} objects", objects.len());
        Ok(objects)
    }
    
    fn detect_people(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        let mut people = Vec::new();
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Skin tone detection
        let mut skin_regions = Vec::new();
        
        for y in (0..height).step_by(8) {
            for x in (0..width).step_by(8) {
                if let Some(pixel) = rgb_image.get_pixel_checked(x, y) {
                    let r = pixel[0] as f32;
                    let g = pixel[1] as f32; 
                    let b = pixel[2] as f32;
                    
                    // Skin tone detection algorithm
                    if self.is_skin_tone(r, g, b) {
                        skin_regions.push((x, y));
                    }
                }
            }
        }
        
        // Cluster skin regions into potential people
        if skin_regions.len() > 20 { // Minimum skin pixels for person detection
            let (center_x, center_y) = self.find_skin_cluster_center(&skin_regions);
            let confidence = (skin_regions.len() as f32 / 100.0).min(0.95);
            
            people.push(DetectedObject {
                label: "person".to_string(),
                confidence,
                bbox: BoundingBox {
                    x: (center_x as i32 - 50).max(0),
                    y: (center_y as i32 - 75).max(0),
                    width: 100,
                    height: 150,
                },
            });
        }
        
        Ok(people)
    }
    
    fn is_skin_tone(&self, r: f32, g: f32, b: f32) -> bool {
        // YCbCr color space skin detection
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
        
        // Skin tone ranges in YCbCr
        y > 80.0 && cb >= 85.0 && cb <= 135.0 && cr >= 135.0 && cr <= 180.0
    }
    
    fn find_skin_cluster_center(&self, skin_regions: &[(u32, u32)]) -> (u32, u32) {
        let sum_x: u32 = skin_regions.iter().map(|(x, _)| *x).sum();
        let sum_y: u32 = skin_regions.iter().map(|(_, y)| *y).sum();
        let count = skin_regions.len() as u32;
        
        (sum_x / count, sum_y / count)
    }
    
    fn detect_objects_by_shape(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        let mut objects = Vec::new();
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Edge detection using simple Sobel-like operator
        let edges = self.detect_edges(&rgb_image)?;
        
        // Find rectangular objects (tables, monitors, etc.)
        let rectangles = self.find_rectangles(&edges, width, height);
        for (x, y, w, h, confidence) in rectangles {
            objects.push(DetectedObject {
                label: "rectangular_object".to_string(),
                confidence,
                bbox: BoundingBox { x, y, width: w, height: h },
            });
        }
        
        Ok(objects)
    }
    
    fn detect_edges(&self, rgb_image: &image::RgbImage) -> Result<Vec<Vec<f32>>> {
        let (width, height) = rgb_image.dimensions();
        let mut edges = vec![vec![0.0; width as usize]; height as usize];
        
        // Simple edge detection
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                if let Some(center) = rgb_image.get_pixel_checked(x, y) {
                    let mut grad_x = 0.0;
                    let mut grad_y = 0.0;
                    
                    // Compute gradients
                    if let (Some(left), Some(right)) = (
                        rgb_image.get_pixel_checked(x - 1, y),
                        rgb_image.get_pixel_checked(x + 1, y)
                    ) {
                        grad_x = (right[0] as f32) - (left[0] as f32);
                    }
                    
                    if let (Some(top), Some(bottom)) = (
                        rgb_image.get_pixel_checked(x, y - 1),
                        rgb_image.get_pixel_checked(x, y + 1)
                    ) {
                        grad_y = (bottom[0] as f32) - (top[0] as f32);
                    }
                    
                    edges[y as usize][x as usize] = (grad_x * grad_x + grad_y * grad_y).sqrt();
                }
            }
        }
        
        Ok(edges)
    }
    
    fn find_rectangles(&self, edges: &[Vec<f32>], width: u32, height: u32) -> Vec<(i32, i32, i32, i32, f32)> {
        let mut rectangles = Vec::new();
        
        // Simple rectangle detection using edge accumulation
        for y in (20..height - 20).step_by(20) {
            for x in (20..width - 20).step_by(20) {
                let edge_score = self.calculate_rectangular_score(edges, x as usize, y as usize, 40, 30);
                
                if edge_score > 15.0 {
                    let confidence = (edge_score / 30.0).min(0.9);
                    rectangles.push((x as i32, y as i32, 40, 30, confidence));
                }
            }
        }
        
        rectangles
    }
    
    fn calculate_rectangular_score(&self, edges: &[Vec<f32>], x: usize, y: usize, w: usize, h: usize) -> f32 {
        let mut score = 0.0;
        
        // Check horizontal edges (top and bottom)
        for i in x..(x + w).min(edges[0].len()) {
            if y < edges.len() {
                score += edges[y][i];
            }
            if (y + h) < edges.len() {
                score += edges[y + h][i];
            }
        }
        
        // Check vertical edges (left and right)
        for i in y..(y + h).min(edges.len()) {
            if x < edges[i].len() {
                score += edges[i][x];
            }
            if (x + w) < edges[i].len() {
                score += edges[i][x + w];
            }
        }
        
        score
    }
    
    fn detect_motion_regions(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        // Placeholder for motion detection - would need previous frame
        // For now, detect high-frequency areas that might indicate movement
        let mut motion_objects = Vec::new();
        
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Detect high-variance regions that might indicate activity
        for y in (0..height - 32).step_by(16) {
            for x in (0..width - 32).step_by(16) {
                let variance = self.calculate_region_variance(&rgb_image, x, y, 32, 32);
                
                if variance > 800.0 {
                    motion_objects.push(DetectedObject {
                        label: "active_region".to_string(),
                        confidence: (variance / 2000.0).min(0.8),
                        bbox: BoundingBox {
                            x: x as i32,
                            y: y as i32,
                            width: 32,
                            height: 32,
                        },
                    });
                }
            }
        }
        
        Ok(motion_objects)
    }
    
    fn calculate_region_variance(&self, image: &image::RgbImage, x: u32, y: u32, w: u32, h: u32) -> f32 {
        let mut values = Vec::new();
        
        for dy in 0..h {
            for dx in 0..w {
                if let Some(pixel) = image.get_pixel_checked(x + dx, y + dy) {
                    let brightness = pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114;
                    values.push(brightness);
                }
            }
        }
        
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance
    }
    
    fn detect_furniture(&self, image: &DynamicImage) -> Result<Vec<DetectedObject>> {
        let mut furniture = Vec::new();
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Detect large horizontal surfaces (tables, desks)
        let horizontal_surfaces = self.find_horizontal_surfaces(&rgb_image, width, height);
        furniture.extend(horizontal_surfaces);
        
        // Detect vertical structures (chairs, walls)
        let vertical_structures = self.find_vertical_structures(&rgb_image, width, height);
        furniture.extend(vertical_structures);
        
        Ok(furniture)
    }
    
    fn find_horizontal_surfaces(&self, image: &image::RgbImage, width: u32, height: u32) -> Vec<DetectedObject> {
        let mut surfaces = Vec::new();
        
        // Look for consistent horizontal lines that might be table edges
        for y in (height / 3)..(2 * height / 3) {
            let mut line_strength = 0.0;
            let mut consistent_pixels = 0;
            
            for x in 10..(width - 10) {
                if let (Some(left), Some(center), Some(right)) = (
                    image.get_pixel_checked(x - 5, y),
                    image.get_pixel_checked(x, y),
                    image.get_pixel_checked(x + 5, y)
                ) {
                    let left_brightness = left[0] as f32 * 0.299 + left[1] as f32 * 0.587 + left[2] as f32 * 0.114;
                    let center_brightness = center[0] as f32 * 0.299 + center[1] as f32 * 0.587 + center[2] as f32 * 0.114;
                    let right_brightness = right[0] as f32 * 0.299 + right[1] as f32 * 0.587 + right[2] as f32 * 0.114;
                    
                    if (left_brightness - center_brightness).abs() > 20.0 || (right_brightness - center_brightness).abs() > 20.0 {
                        line_strength += 1.0;
                        consistent_pixels += 1;
                    }
                }
            }
            
            if consistent_pixels > (width / 4) {
                let confidence = (line_strength / (width as f32)).min(0.85);
                surfaces.push(DetectedObject {
                    label: "table_surface".to_string(),
                    confidence,
                    bbox: BoundingBox {
                        x: 10,
                        y: y as i32 - 20,
                        width: (width - 20) as i32,
                        height: 40,
                    },
                });
            }
        }
        
        surfaces
    }
    
    fn find_vertical_structures(&self, image: &image::RgbImage, width: u32, height: u32) -> Vec<DetectedObject> {
        let mut structures = Vec::new();
        
        // Look for chair backs or vertical furniture elements
        for x in (width / 4)..(3 * width / 4) {
            let mut vertical_score = 0.0;
            
            for y in 10..(height - 10) {
                if let (Some(top), Some(center), Some(bottom)) = (
                    image.get_pixel_checked(x, y - 5),
                    image.get_pixel_checked(x, y),
                    image.get_pixel_checked(x, y + 5)
                ) {
                    let top_brightness = top[0] as f32 * 0.299 + top[1] as f32 * 0.587 + top[2] as f32 * 0.114;
                    let center_brightness = center[0] as f32 * 0.299 + center[1] as f32 * 0.587 + center[2] as f32 * 0.114;
                    let bottom_brightness = bottom[0] as f32 * 0.299 + bottom[1] as f32 * 0.587 + bottom[2] as f32 * 0.114;
                    
                    if (top_brightness - center_brightness).abs() > 15.0 || (bottom_brightness - center_brightness).abs() > 15.0 {
                        vertical_score += 1.0;
                    }
                }
            }
            
            if vertical_score > (height as f32 / 6.0) {
                let confidence = (vertical_score / (height as f32 / 2.0)).min(0.75);
                structures.push(DetectedObject {
                    label: "vertical_furniture".to_string(),
                    confidence,
                    bbox: BoundingBox {
                        x: x as i32 - 25,
                        y: 10,
                        width: 50,
                        height: (height - 20) as i32,
                    },
                });
            }
        }
        
        structures
    }
    
    fn calculate_average_brightness(&self, image: &DynamicImage) -> f32 {
        let rgb_image = image.to_rgb8();
        let pixels = rgb_image.pixels();
        let mut total_brightness = 0.0;
        let mut pixel_count = 0;
        
        for pixel in pixels {
            let brightness = pixel[0] as f32 * 0.299 + 
                            pixel[1] as f32 * 0.587 + 
                            pixel[2] as f32 * 0.114;
            total_brightness += brightness;
            pixel_count += 1;
        }
        
        if pixel_count > 0 {
            total_brightness / pixel_count as f32
        } else {
            0.0
        }
    }
    
    fn estimate_edge_density(&self, image: &DynamicImage) -> f32 {
        // Simple edge detection by measuring pixel variation
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        let mut edge_count = 0;
        let mut total_pixels = 0;
        
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let current = rgb_image.get_pixel(x, y);
                let right = rgb_image.get_pixel(x + 1, y);
                let down = rgb_image.get_pixel(x, y + 1);
                
                let horizontal_diff = Self::pixel_difference(current, right);
                let vertical_diff = Self::pixel_difference(current, down);
                
                if horizontal_diff > 30.0 || vertical_diff > 30.0 {
                    edge_count += 1;
                }
                total_pixels += 1;
            }
        }
        
        if total_pixels > 0 {
            edge_count as f32 / total_pixels as f32
        } else {
            0.0
        }
    }
    
    fn pixel_difference(p1: &Rgb<u8>, p2: &Rgb<u8>) -> f32 {
        let dr = p1[0] as f32 - p2[0] as f32;
        let dg = p1[1] as f32 - p2[1] as f32;
        let db = p1[2] as f32 - p2[2] as f32;
        (dr * dr + dg * dg + db * db).sqrt()
    }
    
    fn detect_by_color(&self, image: &DynamicImage) -> Vec<DetectedObject> {
        let mut objects = Vec::new();
        let rgb_image = image.to_rgb8();
        let (_width, _height) = rgb_image.dimensions();
        
        // Look for red objects (could be stop signs, people, etc.)
        let red_regions = self.find_color_regions(&rgb_image, |r, g, b| r > 150 && g < 100 && b < 100);
        for (x, y, w, h) in red_regions {
            if w > 20 && h > 20 {  // Minimum size threshold
                objects.push(DetectedObject {
                    label: "red_object".to_string(),
                    confidence: 0.5,
                    bbox: BoundingBox { x, y, width: w, height: h },
                });
            }
        }
        
        // Look for green objects (vegetation, signs, etc.)
        let green_regions = self.find_color_regions(&rgb_image, |r, g, b| g > 150 && r < 100 && b < 100);
        for (x, y, w, h) in green_regions {
            if w > 30 && h > 30 {
                objects.push(DetectedObject {
                    label: "green_object".to_string(),
                    confidence: 0.4,
                    bbox: BoundingBox { x, y, width: w, height: h },
                });
            }
        }
        
        objects
    }
    
    fn find_color_regions<F>(&self, image: &RgbImage, color_match: F) -> Vec<(i32, i32, i32, i32)>
    where
        F: Fn(u8, u8, u8) -> bool,
    {
        let (width, height) = image.dimensions();
        let mut regions = Vec::new();
        let mut visited = vec![vec![false; width as usize]; height as usize];
        
        for y in 0..height {
            for x in 0..width {
                if !visited[y as usize][x as usize] {
                    let pixel = image.get_pixel(x, y);
                    if color_match(pixel[0], pixel[1], pixel[2]) {
                        let region = self.flood_fill_region(image, &mut visited, x, y, &color_match);
                        if let Some((min_x, min_y, max_x, max_y)) = region {
                            regions.push((
                                min_x as i32,
                                min_y as i32,
                                (max_x - min_x) as i32,
                                (max_y - min_y) as i32,
                            ));
                        }
                    }
                }
            }
        }
        
        regions
    }
    
    fn flood_fill_region<F>(
        &self,
        image: &RgbImage,
        visited: &mut Vec<Vec<bool>>,
        start_x: u32,
        start_y: u32,
        color_match: &F,
    ) -> Option<(u32, u32, u32, u32)>
    where
        F: Fn(u8, u8, u8) -> bool,
    {
        let (width, height) = image.dimensions();
        let mut stack = vec![(start_x, start_y)];
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;
        let mut pixel_count = 0;
        
        while let Some((x, y)) = stack.pop() {
            if x >= width || y >= height || visited[y as usize][x as usize] {
                continue;
            }
            
            let pixel = image.get_pixel(x, y);
            if !color_match(pixel[0], pixel[1], pixel[2]) {
                continue;
            }
            
            visited[y as usize][x as usize] = true;
            pixel_count += 1;
            
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            
            // Add neighbors
            if x > 0 { stack.push((x - 1, y)); }
            if x < width - 1 { stack.push((x + 1, y)); }
            if y > 0 { stack.push((x, y - 1)); }
            if y < height - 1 { stack.push((x, y + 1)); }
            
            // Limit region size to prevent excessive computation
            if pixel_count > 10000 {
                break;
            }
        }
        
        if pixel_count > 10 {  // Minimum region size
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }
    
    fn generate_scene_description(&self, objects: &[DetectedObject], lidar_points: &[LidarPoint]) -> String {
        let mut description = String::new();
        
        if objects.is_empty() {
            description.push_str("Basic scene analysis: No significant objects detected. ");
        } else {
            description.push_str(&format!("Basic scene analysis found {} object(s): ", objects.len()));
            
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
        debug!("Annotating frame with {} objects and {} LiDAR points (basic mode)", 
               context.objects.len(), context.lidar_points.len());
        
        // For basic annotation, we'll modify the raw image data
        // This is a simplified version without OpenCV drawing functions
        
        // Add simple visual indicators by modifying pixel values
        for obj in &context.objects {
            self.draw_simple_rectangle(frame, &obj.bbox, [0, 255, 0])?; // Green rectangles
        }
        
        // LiDAR overlay is now handled by the camera system
        // since we no longer have direct access to go2_system
        
        Ok(())
    }
    
    fn draw_simple_rectangle(&self, frame: &mut Mat, bbox: &BoundingBox, color: [u8; 3]) -> Result<()> {
        let width = frame.width as i32;
        let height = frame.height as i32;
        
        // Clamp coordinates
        let x1 = bbox.x.max(0).min(width - 1);
        let y1 = bbox.y.max(0).min(height - 1);
        let x2 = (bbox.x + bbox.width).max(0).min(width - 1);
        let y2 = (bbox.y + bbox.height).max(0).min(height - 1);
        
        // Draw simple rectangle outline by modifying pixels
        for x in x1..x2 {
            self.set_pixel(frame, x, y1, color)?;
            self.set_pixel(frame, x, y2 - 1, color)?;
        }
        
        for y in y1..y2 {
            self.set_pixel(frame, x1, y, color)?;
            self.set_pixel(frame, x2 - 1, y, color)?;
        }
        
        Ok(())
    }
    
    fn set_pixel(&self, frame: &mut Mat, x: i32, y: i32, color: [u8; 3]) -> Result<()> {
        if x < 0 || y < 0 || x >= frame.width as i32 || y >= frame.height as i32 {
            return Ok(()); // Skip out-of-bounds pixels
        }
        
        let pixel_index = ((y as u32 * frame.width + x as u32) * frame.channels) as usize;
        if pixel_index + 2 < frame.data.len() {
            frame.data[pixel_index] = color[0];     // R
            frame.data[pixel_index + 1] = color[1]; // G
            frame.data[pixel_index + 2] = color[2]; // B
        }
        
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping VisionProcessor (basic mode)");
        // No longer need to stop go2_system since we use shared camera
        Ok(())
    }
}
