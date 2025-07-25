use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tracing::{info, debug, warn, error};
use cpal::{Device, Stream, StreamConfig, SampleFormat, SampleRate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::{Decoder, OutputStream, Sink};
use tts::Tts;

#[derive(Debug, Clone)]
pub struct AudioConfig {
    #[allow(dead_code)]
    pub sample_rate: u32,
    #[allow(dead_code)]
    pub channels: u16,
    #[allow(dead_code)]
    pub buffer_size: usize,
    #[allow(dead_code)]
    pub voice_detection_threshold: f32,
    #[allow(dead_code)]
    pub silence_duration_ms: u64,
    #[allow(dead_code)]
    pub speech_timeout_ms: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 1,
            buffer_size: 4096,
            voice_detection_threshold: 0.02,
            silence_duration_ms: 500,
            speech_timeout_ms: 5000,
        }
    }
}

pub struct AudioSystem {
    #[allow(dead_code)]
    config: AudioConfig,
    input_device: Option<Device>,
    output_device: Option<Device>,
    input_stream: Option<Stream>,
    #[allow(dead_code)]
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    is_recording: Arc<Mutex<bool>>,
    tts: Option<Tts>,
    _output_stream: Option<OutputStream>,
    output_sink: Option<Sink>,
    is_initialized: bool,
}

impl AudioSystem {
    pub fn new(config: AudioConfig) -> Result<Self> {
        info!("Creating AudioSystem for real audio I/O");
        
        Ok(Self {
            config,
            input_device: None,
            output_device: None,
            input_stream: None,
            audio_buffer: Arc::new(Mutex::new(VecDeque::new())),
            is_recording: Arc::new(Mutex::new(false)),
            tts: None,
            _output_stream: None,
            output_sink: None,
            is_initialized: false,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing audio system - scanning for audio devices");
        
        let host = cpal::default_host();
        
        // Initialize input device (microphone)
        self.input_device = match host.default_input_device() {
            Some(device) => {
                info!("Found default input device: {}", device.name().unwrap_or("Unknown".to_string()));
                Some(device)
            }
            None => {
                warn!("No default input device found, scanning for alternatives");
                self.find_input_device(&host)?
            }
        };
        
        // Initialize output device (speakers)
        self.output_device = match host.default_output_device() {
            Some(device) => {
                info!("Found default output device: {}", device.name().unwrap_or("Unknown".to_string()));
                Some(device)
            }
            None => {
                warn!("No default output device found, scanning for alternatives");
                self.find_output_device(&host)?
            }
        };
        
        // Initialize TTS engine
        self.tts = match Tts::default() {
            Ok(tts) => {
                info!("Text-to-speech engine initialized successfully");
                Some(tts)
            }
            Err(e) => {
                warn!("Failed to initialize TTS engine: {}", e);
                None
            }
        };
        
        // Initialize output stream for audio playback
        if let Some(ref output_device) = self.output_device {
            match OutputStream::try_from_device(output_device) {
                Ok((stream, handle)) => {
                    self._output_stream = Some(stream);
                    self.output_sink = Some(Sink::try_new(&handle)?);
                    info!("Audio output stream initialized");
                }
                Err(e) => {
                    warn!("Failed to initialize output stream: {}", e);
                }
            }
        }
        
        self.is_initialized = true;
        info!("Audio system initialized successfully");
        Ok(())
    }
    
    fn find_input_device(&self, host: &cpal::Host) -> Result<Option<Device>> {
        let devices = host.input_devices()?;
        
        for device in devices {
            if let Ok(name) = device.name() {
                info!("Found input device: {}", name);
                return Ok(Some(device));
            }
        }
        
        Ok(None)
    }
    
    fn find_output_device(&self, host: &cpal::Host) -> Result<Option<Device>> {
        let devices = host.output_devices()?;
        
        for device in devices {
            if let Ok(name) = device.name() {
                info!("Found output device: {}", name);
                return Ok(Some(device));
            }
        }
        
        Ok(None)
    }
    
    #[allow(dead_code)]
    pub async fn start_recording(&mut self) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow!("Audio system not initialized"));
        }
        
        let input_device = self.input_device.as_ref()
            .ok_or_else(|| anyhow!("No input device available"))?;
        
        // Get the default input config
        let config = input_device.default_input_config()?;
        info!("Input config: {:?}", config);
        
        // Convert to stream config
        let stream_config = StreamConfig {
            channels: config.channels().min(self.config.channels),
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };
        
        // Create audio buffer and recording flag references
        let buffer = Arc::clone(&self.audio_buffer);
        let recording = Arc::clone(&self.is_recording);
        
        // Set recording flag
        {
            let mut is_recording = recording.lock().unwrap();
            *is_recording = true;
        }
        
        // Create input stream based on sample format
        let stream = match config.sample_format() {
            SampleFormat::F32 => self.create_input_stream_f32(input_device, &stream_config, buffer)?,
            SampleFormat::I16 => self.create_input_stream_i16(input_device, &stream_config, buffer)?,
            SampleFormat::U16 => self.create_input_stream_u16(input_device, &stream_config, buffer)?,
            _ => return Err(anyhow!("Unsupported sample format")),
        };
        
        // Start the stream
        stream.play()?;
        self.input_stream = Some(stream);
        
        info!("Audio recording started");
        Ok(())
    }
    
    #[allow(dead_code)]
    fn create_input_stream_f32(
        &self,
        device: &Device,
        config: &StreamConfig,
        buffer: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<Stream> {
        let stream = device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                for &sample in data {
                    buffer.push_back(sample);
                    // Keep buffer size manageable
                    if buffer.len() > 44100 * 10 { // 10 seconds of audio
                        buffer.pop_front();
                    }
                }
            },
            |err| error!("Audio input stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    #[allow(dead_code)]
    fn create_input_stream_i16(
        &self,
        device: &Device,
        config: &StreamConfig,
        buffer: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<Stream> {
        let stream = device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                for &sample in data {
                    let normalized = sample as f32 / i16::MAX as f32;
                    buffer.push_back(normalized);
                    if buffer.len() > 44100 * 10 {
                        buffer.pop_front();
                    }
                }
            },
            |err| error!("Audio input stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    #[allow(dead_code)]
    fn create_input_stream_u16(
        &self,
        device: &Device,
        config: &StreamConfig,
        buffer: Arc<Mutex<VecDeque<f32>>>,
    ) -> Result<Stream> {
        let stream = device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                for &sample in data {
                    let normalized = (sample as f32 - 32768.0) / 32768.0;
                    buffer.push_back(normalized);
                    if buffer.len() > 44100 * 10 {
                        buffer.pop_front();
                    }
                }
            },
            |err| error!("Audio input stream error: {}", err),
            None,
        )?;
        
        Ok(stream)
    }
    
    pub async fn speak_text(&mut self, text: &str) -> Result<()> {
        if let Some(ref mut tts) = self.tts {
            info!("Speaking: {}", text);
            
            match tts.speak(text, false) {
                Ok(_) => {
                    // Wait for speech to complete
                    while tts.is_speaking()? {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                    debug!("Speech completed");
                }
                Err(e) => {
                    error!("TTS error: {}", e);
                    return Err(anyhow!("Text-to-speech failed: {}", e));
                }
            }
        } else {
            warn!("TTS not available, text would be spoken: {}", text);
        }
        
        Ok(())
    }
    
    #[allow(dead_code)]
    pub async fn play_audio_file(&mut self, file_path: &str) -> Result<()> {
        if let Some(ref sink) = self.output_sink {
            let file = std::fs::File::open(file_path)?;
            let source = Decoder::new(file)?;
            
            info!("Playing audio file: {}", file_path);
            sink.append(source);
            
            // Wait for playback to complete
            sink.sleep_until_end();
            
            Ok(())
        } else {
            Err(anyhow!("Audio output not initialized"))
        }
    }
    
    #[allow(dead_code)]
    pub async fn play_beep(&mut self, frequency: f32, duration_ms: u64) -> Result<()> {
        if let Some(ref sink) = self.output_sink {
            let sample_rate = 44100;
            let samples_per_ms = sample_rate as f64 / 1000.0;
            let total_samples = (duration_ms as f64 * samples_per_ms) as usize;
            
            let mut samples = Vec::with_capacity(total_samples);
            
            for i in 0..total_samples {
                let t = i as f32 / sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
                samples.push(sample);
            }
            
            // Convert to source
            let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples);
            
            info!("Playing beep: {}Hz for {}ms", frequency, duration_ms);
            sink.append(source);
            
            // Wait for playback to complete
            tokio::time::sleep(tokio::time::Duration::from_millis(duration_ms + 100)).await;
            
            Ok(())
        } else {
            Err(anyhow!("Audio output not initialized"))
        }
    }
    
    pub async fn stop_recording(&mut self) -> Result<()> {
        {
            let mut is_recording = self.is_recording.lock().unwrap();
            *is_recording = false;
        }
        
        if let Some(stream) = self.input_stream.take() {
            drop(stream);
            info!("Audio recording stopped");
        }
        
        Ok(())
    }
    
    #[allow(dead_code)]
    pub fn is_recording(&self) -> bool {
        *self.is_recording.lock().unwrap()
    }
    
    #[allow(dead_code)]
    pub fn clear_audio_buffer(&self) {
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.clear();
        debug!("Audio buffer cleared");
    }
    
    #[allow(dead_code)]  
    pub fn get_audio_info(&self) -> String {
        if self.is_initialized {
            format!(
                "Audio System - Input: {}, Output: {}, TTS: {}",
                if self.input_device.is_some() { "Available" } else { "None" },
                if self.output_device.is_some() { "Available" } else { "None" },
                if self.tts.is_some() { "Available" } else { "None" }
            )
        } else {
            "Audio System - Not initialized".to_string()
        }
    }
}

impl Drop for AudioSystem {
    fn drop(&mut self) {
        if self.is_initialized {
            info!("AudioSystem being dropped, cleaning up");
            let _ = self.stop_recording();
        }
    }
}
