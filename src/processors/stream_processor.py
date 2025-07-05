import cv2
import threading
import queue
import time
import logging
import subprocess
import signal
import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.config import AppConfig
from src.api.fastapi_server import FastAPIWebSocketServer   
from src.models.stream import FrameData

logger = logging.getLogger(__name__)

class StreamStatus(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

@dataclass
class StreamInfo:
    """Information about a stream"""
    stream_id: str
    url: str
    name: str
    status: StreamStatus
    last_frame_time: float
    error_count: int
    reconnect_count: int
    fps: float = 0.0
    resolution: tuple = (0, 0)

@dataclass
class DisplayFrame:
    """Frame prepared for display"""
    stream_id: str
    frame: np.ndarray
    window_name: str
    timestamp: float

class StreamReader:
    """FFmpeg-based stream reader - ROBUST RTSP HANDLING"""
    
    def __init__(self, stream_id: str, url: str, config: AppConfig):
        self.stream_id = stream_id
        self.url = url
        self.config = config
        
        # FFmpeg process instead of OpenCV VideoCapture
        self.ffmpeg_process = None
        self.use_ffmpeg = True  # Toggle to fallback to OpenCV if needed
        self.cap = None  # Fallback OpenCV capture
        
        self.running = False
        self.thread = None
        
        # Frame communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.info = StreamInfo(
            stream_id=stream_id,
            url=url,
            name=f"Camera-{stream_id}",
            status=StreamStatus.IDLE,
            last_frame_time=0,
            error_count=0,
            reconnect_count=0
        )
        self._frame_count = 0
        self._lock = threading.Lock()
        
        # FPS calculation
        self._fps_window = []
        self._fps_window_size = 10
        self._color_format_warned = False
        self.face_detector = None
        
        # FFmpeg specific settings
        self.frame_width = 1280
        self.frame_height = 720
        self.frame_size = self.frame_width * self.frame_height * 3  # BGR24
        self.target_fps = 25
        
        logger.info(f"StreamReader {stream_id} initialized with FFmpeg backend")
        
    def set_face_detector(self, face_detector):
        """Set the face detection processor"""
        self.face_detector = face_detector
    
    def _create_ffmpeg_command(self) -> List[str]:
        """Create FFmpeg command for robust RTSP streaming"""
        cmd = [
            'ffmpeg',
            '-i', self.url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-r', str(self.target_fps),
            '-s', f'{self.frame_width}x{self.frame_height}',
            '-loglevel', 'error',  # Only show errors
            '-fflags', '+genpts',  # Generate presentation timestamps
            '-avoid_negative_ts', 'make_zero',
            '-'  # Output to stdout
        ]
        return cmd
    
    def _start_ffmpeg_process(self) -> bool:
        """Start FFmpeg subprocess"""
        try:
            if self.ffmpeg_process:
                self._stop_ffmpeg_process()
            
            cmd = self._create_ffmpeg_command()
            logger.info(f"Starting FFmpeg for {self.stream_id}: {' '.join(cmd[:3])}...")
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            # Test read one frame to verify connection
            test_data = self.ffmpeg_process.stdout.read(self.frame_size)
            if len(test_data) == self.frame_size:
                logger.info(f"FFmpeg process started successfully for {self.stream_id}")
                return True
            else:
                logger.error(f"FFmpeg test frame failed for {self.stream_id}: got {len(test_data)} bytes, expected {self.frame_size}")
                self._stop_ffmpeg_process()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process for {self.stream_id}: {e}")
            self._stop_ffmpeg_process()
            return False
    
    def _stop_ffmpeg_process(self):
        """Stop FFmpeg subprocess cleanly"""
        if self.ffmpeg_process:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGKILL)
                    self.ffmpeg_process.wait()
                    
            except Exception as e:
                logger.debug(f"Error stopping FFmpeg process: {e}")
            finally:
                self.ffmpeg_process = None
    
    def _read_frames(self):
        """Frame reading loop using FFmpeg subprocess"""
        consecutive_failures = 0
        last_frame_time = time.time()
        max_failures = 20  # More tolerant than OpenCV

        while self.running:
            try:
                # Ensure FFmpeg process is running
                if not self._ensure_connection():
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        if not self._reconnect():
                            time.sleep(1.0)
                            continue
                        consecutive_failures = 0
                    continue
                
                frame_start = time.time()
                
                # Read frame from FFmpeg
                if self.use_ffmpeg and self.ffmpeg_process:
                    ret, frame = self._read_ffmpeg_frame()
                else:
                    # Fallback to OpenCV
                    ret, frame = self._read_opencv_frame()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    
                    # Only log occasionally to avoid spam
                    if consecutive_failures <= 5 or consecutive_failures % 10 == 0:
                        logger.warning(f"Failed to read frame from stream {self.stream_id} "
                                     f"(attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error(f"Too many consecutive failures for stream {self.stream_id}")
                        if not self._reconnect():
                            time.sleep(1.0)
                        consecutive_failures = 0
                    continue
                
                # SUCCESS: Reset counters
                consecutive_failures = 0
                if self.info.reconnect_count > 0:
                    logger.info(f"Stream {self.stream_id} recovered after {self.info.reconnect_count} attempts")
                    self.info.reconnect_count = 0
                
                # Update timing and stats
                current_time = time.time()
                frame_interval = current_time - last_frame_time
                
                if frame_interval > 0:
                    instant_fps = 1.0 / frame_interval
                    self._fps_window.append(instant_fps)
                    if len(self._fps_window) > self._fps_window_size:
                        self._fps_window.pop(0)
                    
                    with self._lock:
                        self.info.last_frame_time = current_time
                        self.info.status = StreamStatus.ACTIVE
                        self._frame_count += 1
                        self.info.fps = sum(self._fps_window) / len(self._fps_window)
                
                last_frame_time = current_time
                
                # Create frame data
                frame_data = FrameData(
                    stream_id=self.stream_id,
                    frame=frame,
                    timestamp=current_time,
                    frame_number=self._frame_count,
                    stream_info=self.info
                )
                
                # Add to queue (drop old frames)
                try:
                    while not self.frame_queue.empty():
                        try:
                            old_frame = self.frame_queue.get_nowait()
                            del old_frame
                        except queue.Empty:
                            break
                    
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    pass
                
                # Minimal delay
                processing_time = time.time() - frame_start
                if processing_time < 0.01:  # 10ms max frame rate
                    time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in frame reading for {self.stream_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    if not self._reconnect():
                        time.sleep(1.0)
                    consecutive_failures = 0
        
        # Cleanup
        self._stop_ffmpeg_process()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info(f"Stream reader {self.stream_id} thread ended")
    
    def _read_ffmpeg_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read frame from FFmpeg subprocess"""
        try:
            if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                return False, None
            
            # Read raw frame data
            frame_data = self.ffmpeg_process.stdout.read(self.frame_size)
            
            if len(frame_data) != self.frame_size:
                if len(frame_data) == 0:
                    # EOF - process ended
                    return False, None
                else:
                    # Partial frame - skip it
                    logger.debug(f"Partial frame received: {len(frame_data)}/{self.frame_size} bytes")
                    return False, None
            
            # Convert raw bytes to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((self.frame_height, self.frame_width, 3))
            
            return True, frame
            
        except Exception as e:
            logger.debug(f"FFmpeg frame read error: {e}")
            return False, None
    
    def _read_opencv_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Fallback: Read frame using OpenCV"""
        try:
            if not self.cap or not self.cap.isOpened():
                return False, None
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resize to expected dimensions
                if frame.shape[:2] != (self.frame_height, self.frame_width):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            return ret, frame
            
        except Exception as e:
            logger.debug(f"OpenCV frame read error: {e}")
            return False, None
    
    def _ensure_connection(self) -> bool:
        """Ensure connection is active"""
        if self.use_ffmpeg:
            return (self.ffmpeg_process is not None and 
                   self.ffmpeg_process.poll() is None)
        else:
            return self.cap is not None and self.cap.isOpened()
    
    def connect(self) -> bool:
        """Connect to the stream"""
        try:
            self.info.status = StreamStatus.CONNECTING
            logger.info(f"Connecting to stream {self.stream_id}: {self.url}")
            
            if not self.url or self.url in ["0", ""]:
                raise Exception("Invalid stream URL")
            
            # Try FFmpeg first
            if self.use_ffmpeg:
                if self._start_ffmpeg_process():
                    self.info.resolution = (self.frame_width, self.frame_height)
                    self.info.status = StreamStatus.ACTIVE
                    logger.info(f"Stream {self.stream_id} connected via FFmpeg. Resolution: {self.frame_width}x{self.frame_height}")
                    return True
                else:
                    logger.warning(f"FFmpeg failed for {self.stream_id}, trying OpenCV fallback")
                    self.use_ffmpeg = False
            
            # Fallback to OpenCV
            if not self.use_ffmpeg:
                return self._connect_opencv()
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to stream {self.stream_id}: {e}")
            self.info.status = StreamStatus.ERROR
            self.info.error_count += 1
            return False
    
    def _connect_opencv(self) -> bool:
        """Fallback OpenCV connection"""
        try:
            backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends_to_try:
                try:
                    self.cap = cv2.VideoCapture(self.url, backend)
                    if self.cap.isOpened():
                        # Test frame read
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            self.info.resolution = (width, height)
                            self.info.status = StreamStatus.ACTIVE
                            logger.info(f"Stream {self.stream_id} connected via OpenCV backend {backend}")
                            return True
                    
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                        
                except Exception as e:
                    logger.debug(f"OpenCV backend {backend} failed: {e}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            
            return False
            
        except Exception as e:
            logger.error(f"OpenCV fallback failed: {e}")
            return False
    
    def start(self):
        """Start reading frames"""
        if self.running:
            return
        
        if not self.connect():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        logger.info(f"Stream reader {self.stream_id} started")
    
    def stop(self):
        """Stop reading frames"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        self._stop_ffmpeg_process()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.info.status = StreamStatus.STOPPED
        logger.info(f"Stream reader {self.stream_id} stopped")
    
    def get_frame(self) -> Optional[FrameData]:
        """Get the latest frame (non-blocking)"""
        try:
            latest_frame = None
            while True:
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            return latest_frame
        except:
            return None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the stream"""
        self.info.status = StreamStatus.RECONNECTING
        self.info.reconnect_count += 1
        
        logger.info(f"Attempting to reconnect stream {self.stream_id} (attempt {self.info.reconnect_count})")
        
        # Clean up current connection
        self._stop_ffmpeg_process()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Progressive backoff
        if self.info.reconnect_count <= 5:
            delay = 1.0
        elif self.info.reconnect_count <= 10:
            delay = 3.0
        else:
            delay = 5.0
        
        logger.info(f"Waiting {delay}s before reconnection attempt...")
        time.sleep(delay)
        
        # Reset to try FFmpeg first again
        self.use_ffmpeg = True
        
        return self.connect()

class CentralizedDisplayManager:
    """Manages all camera displays in a single thread - THREAD SAFE"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False
        self.display_thread = None
        self.display_queue = queue.Queue(maxsize=50)
        self.windows_created = set()
        self.face_detector = None
        # ADD: FastAPI WebSocket server
        self.fastapi_server = None
        if config.ENABLE_WEBSOCKET:
            self.fastapi_server = FastAPIWebSocketServer(
                config, 
                host="0.0.0.0",
                port=config.WEBSOCKET_PORT
            )
            logger.info("FastAPI WebSocket server initialized")
    
    async def start_fastapi_server(self):
        """Start FastAPI server"""
        if self.fastapi_server:
            await self.fastapi_server.start_server()

    async def stop_fastapi_server(self):
        """Stop FastAPI server"""
        if self.fastapi_server:
            await self.fastapi_server.stop_server()

    def set_camera_manager(self, camera_manager):
        """Set camera manager for FastAPI server"""
        if self.fastapi_server:
            self.fastapi_server.set_camera_manager(camera_manager)
          
    def set_face_detector(self, face_detector):
        """Set face detector for drawing annotations"""
        self.face_detector = face_detector
        
        if self.fastapi_server:
            self.fastapi_server.set_face_detector(face_detector)
            
    def add_frame_for_display(self, frame_data: FrameData):
        """Add frame to display queue (called from processing thread)"""
        
        # CV2 Display Logic - ONLY if GUI is enabled
        if self.config.enable_gui:
            try:
                # Prepare display frame with annotations
                display_frame = self._prepare_display_frame(frame_data)
                
                # Add to display queue, drop old frames if full
                try:
                    self.display_queue.put_nowait(display_frame)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(display_frame)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error preparing frame for display: {e}")
        
        # WebSocket Streaming Logic - COMPLETELY SEPARATE from CV2
        if self.fastapi_server:
            try:
                # Throttle WebSocket frames
                current_time = time.time()
                if not hasattr(self, '_last_websocket_frame_time'):
                    self._last_websocket_frame_time = {}
                
                stream_id = frame_data.stream_id
                last_time = self._last_websocket_frame_time.get(stream_id, 0)
                min_interval = 1.0 / getattr(self.config, 'WEBSOCKET_MAX_FPS', 15)
                
                if current_time - last_time < min_interval:
                    return
                
                self._last_websocket_frame_time[stream_id] = current_time
                
                # ALWAYS use WebSocket-specific frame preparation (no CV2 windows)
                annotated_frame = self._prepare_websocket_frame(frame_data)
                
                # Send frame to WebSocket clients (thread-safe)
                self.fastapi_server.add_frame_threadsafe(
                    frame_data.stream_id,
                    annotated_frame
                )
                
            except Exception as e:
                logger.error(f"Error adding frame to FastAPI stream: {e}")
            
    def _prepare_websocket_frame(self, frame_data: FrameData):
        """Prepare frame for WebSocket streaming WITHOUT creating CV2 windows"""
        frame = frame_data.frame.copy()
        
        # Apply face detection annotations if available
        if self.face_detector:
            try:
                detections = self.face_detector.get_stream_detections(frame_data.stream_id)
                if detections and detections.faces:
                    # Draw face annotations directly on frame
                    frame = self._draw_face_annotations_directly(frame, detections.faces)
            except Exception as e:
                logger.debug(f"Error drawing detections for WebSocket: {e}")
        
        # Resize frame for WebSocket (no CV2 display sizing)
        height, width = frame.shape[:2]
        if width > 640:  # WebSocket-specific sizing
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Add minimal stream info overlay for WebSocket
        info_text = f"Stream: {frame_data.stream_id} | FPS: {frame_data.stream_info.fps:.1f}"
        cv2.putText(frame, info_text, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _draw_face_annotations_directly(self, frame, faces):
        """Draw face annotations directly on frame with proper names and emotions"""
        from .emotion_recognizer import normalize_emotion
        
        for face in faces:
            x, y, w, h = int(face.x), int(face.y), int(face.width), int(face.height)
            
            # Choose color and thickness based on recognition status
            if face.is_recognized:
                color = (0, 255, 0)  # Green for recognized faces
                thickness = 3
            else:
                color = self._get_emotion_color_simple(face.emotion)
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare labels (same logic as draw_detections)
            name_parts = []
            emotion_parts = []
            
            # Build name label
            if face.is_recognized and face.human_name:
                name_parts.append(f"{face.human_name}")
                if face.recognition_confidence and face.recognition_confidence > 0:
                    name_parts.append(f"({face.recognition_confidence:.1%})")
            else:
                if face.face_id:
                    name_parts.append(f"Unknown #{face.face_id.split('_')[-1]}")
            
            # Build emotion label with normalize_emotion
            if face.emotion:
                try:
                    emotion_text = f"{normalize_emotion(face.emotion).title()}"
                    if face.emotion_confidence and face.emotion_confidence > 0:
                        emotion_text += f" {face.emotion_confidence:.0%}"
                    emotion_parts.append(emotion_text)
                except Exception:
                    # Fallback if normalize_emotion is not available
                    emotion_text = f"{face.emotion.title()}"
                    if face.emotion_confidence and face.emotion_confidence > 0:
                        emotion_text += f" {face.emotion_confidence:.0%}"
                    emotion_parts.append(emotion_text)
            
            # Draw labels
            if name_parts or emotion_parts:
                if face.is_recognized:
                    # For recognized faces: name on top, emotion below
                    if name_parts:
                        name_label = " ".join(name_parts)
                        self._draw_websocket_label(frame, name_label, (x, y - 10), color, 0.8, 2)
                    
                    if emotion_parts:
                        emotion_label = " ".join(emotion_parts)
                        emotion_color = self._get_emotion_color_simple(face.emotion)
                        self._draw_websocket_label(frame, emotion_label, (x, y + h + 25), emotion_color, 0.6, 1)
                else:
                    # For unknown faces: show all info in one label
                    all_parts = name_parts + emotion_parts
                    main_label = " | ".join(all_parts)
                    self._draw_websocket_label(frame, main_label, (x, y - 10), color, 0.6, 2)
        
        return frame

    def _draw_websocket_label(self, frame, text, position, color, font_scale, thickness):
        """Helper method to draw text with background for WebSocket"""
        x, y = position
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Draw background rectangle
        cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y + 5), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Also update _get_emotion_color_simple to match the main emotion colors:
    def _get_emotion_color_simple(self, emotion):
        """Get BGR color for emotion (matching main draw_detections method)"""
        emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue  
            'angry': (0, 0, 255),      # Red
            'surprise': (0, 255, 255), # Yellow
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 0),    # Dark Green
            'neutral': (128, 128, 128), # Gray
            'unknown': (255, 255, 255)  # White
        }
        return emotion_colors.get(emotion.lower() if emotion else 'neutral', (255, 255, 255))
        
    def _get_emotion_color(self, emotion):
        """Get emotion color (add this if not present)"""
        colors = {
            'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
            'surprise': (0, 255, 255), 'fear': (128, 0, 128),
            'disgust': (0, 128, 0), 'neutral': (128, 128, 128)
        }
        return colors.get(emotion.lower() if emotion else 'neutral', (255, 255, 255))
    
        
    def _prepare_annotated_frame(self, frame_data: FrameData):
        """Prepare frame with annotations for FastAPI streaming"""
        frame = frame_data.frame.copy()
        
        # if self.face_detector:
        #     try:
        #         detections = self.face_detector.get_stream_detections(frame_data.stream_id)
        #         if detections and detections.faces:
        #             # Use same drawing logic as CV2 display
        #             frame = self._prepare_display_frame(frame_data).frame
        #     except Exception as e:
        #         logger.debug(f"Error drawing detections: {e}")
        
        return frame   
         
    def _prepare_display_frame(self, frame_data: FrameData) -> DisplayFrame:
        """Prepare frame for display with annotations"""
        frame = frame_data.frame.copy()
        
        # Apply face detection annotations if available
        if self.face_detector:
            try:
                detections = self.face_detector.get_stream_detections(frame_data.stream_id)
                if detections and detections.faces:
                    frame = self.face_detector.draw_detections(
                        frame, 
                        detections.faces,
                        show_emotions=True,
                        show_probabilities=False
                    )
            except Exception as e:
                logger.debug(f"Error drawing detections for {frame_data.stream_id}: {e}")
        
        # Resize frame to display dimensions
        frame = self._resize_frame_for_display(frame)
        
        # Add stream info overlay
        if hasattr(self.config, 'show_frame_info') and getattr(self.config.display, 'show_frame_info', True):
            info_text = f"Stream: {frame_data.stream_id} | FPS: {frame_data.stream_info.fps:.1f} | Frame: {frame_data.frame_number}"
            cv2.putText(frame, info_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        window_name = f"Camera {frame_data.stream_id}"
        
        return DisplayFrame(
            stream_id=frame_data.stream_id,
            frame=frame,
            window_name=window_name,
            timestamp=frame_data.timestamp
        )
    
    def _resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to match display configuration"""
        display_width = getattr(self.config.display, 'display_width', 960)
        display_height = getattr(self.config.display, 'display_height', 540)
        
        h, w = frame.shape[:2]
        scale_w = display_width / w
        scale_h = display_height / h
        scale = min(scale_w, scale_h)
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        y_offset = (display_height - new_height) // 2
        x_offset = (display_width - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return canvas
    
    def _display_loop(self):
        """Main display loop - ONLY THREAD THAT CALLS OpenCV GUI"""
        logger.info("Centralized display thread started")
        
        while self.running:
            try:
                # Collect all available frames
                frames_to_display = []
                try:
                    # Get frames with timeout
                    frame = self.display_queue.get(timeout=0.1)
                    frames_to_display.append(frame)
                    
                    # Collect additional frames without blocking
                    while True:
                        try:
                            frame = self.display_queue.get_nowait()
                            frames_to_display.append(frame)
                        except queue.Empty:
                            break
                            
                except queue.Empty:
                    # No frames to display, just handle events
                    pass
                
                # Display all collected frames
                for display_frame in frames_to_display:
                    try:
                        # Create window if not exists
                        if display_frame.window_name not in self.windows_created:
                            cv2.namedWindow(display_frame.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            display_width = getattr(self.config.display, 'display_width', 960)
                            display_height = getattr(self.config.display, 'display_height', 540)
                            cv2.resizeWindow(display_frame.window_name, display_width, display_height)
                            self.windows_created.add(display_frame.window_name)
                            logger.info(f"Created display window: {display_frame.window_name}")
                        
                        # Display frame
                        cv2.imshow(display_frame.window_name, display_frame.frame)
                        
                    except Exception as e:
                        logger.error(f"Error displaying frame for {display_frame.window_name}: {e}")
                
                # Handle keyboard events - CRITICAL: Only this thread calls waitKey
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit key pressed - stopping display")
                    self.running = False
                    break
                elif key == ord('c'):
                    logger.info("Clearing all windows")
                    cv2.destroyAllWindows()
                    self.windows_created.clear()
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                time.sleep(0.01)
        
        # Cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("Centralized display thread ended")
    
    def start(self):
        """Start the display manager"""
        # Convert string to boolean if needed
        gui_enabled = self.config.enable_gui
        if isinstance(gui_enabled, str):
            gui_enabled = gui_enabled.lower() in ('true', '1', 'yes', 'on')
        
        logger.critical(f"GUI enabled check: {gui_enabled} (original: {self.config.enable_gui}, type: {type(self.config.enable_gui)})")
        
        # ONLY start if GUI is enabled - USE THE CONVERTED VARIABLE
        if not gui_enabled:  # ✅ Fixed: use gui_enabled instead of self.config.enable_gui
            logger.info("🚫 GUI DISABLED - CV2 display manager will NOT start")
            logger.info(f"🔧 enable_gui setting: {self.config.enable_gui}")
            return
            
        if self.running:
            logger.info("⚠️  Display manager already running")
            return
            
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        logger.info("✅ CV2 display manager started (GUI enabled)")
    
    def stop(self):
        """Stop the display manager"""
        self.running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=5)
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("Centralized display manager stopped")