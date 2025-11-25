# src/services/camera_detection_service.py
import subprocess
import logging
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import threading
import shutil
from enum import Enum

logger = logging.getLogger(__name__)

class CameraStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    ERROR = "error"
    STREAMING = "streaming"

@dataclass
class CameraInfo:
    """Information about a detected camera"""
    device_path: str  # e.g., /dev/video0
    name: str
    description: str
    resolution: tuple  # (width, height)
    supported_formats: List[str]
    status: CameraStatus
    capabilities: Dict
    is_streaming: bool = False
    rtsp_url: Optional[str] = None
    ffmpeg_process: Optional[subprocess.Popen] = None

class CameraDetectionService:
    """Service for detecting and managing local web cameras on Linux"""
    
    def __init__(self):
        self.detected_cameras: Dict[str, CameraInfo] = {}
        self.mediamtx_process: Optional[subprocess.Popen] = None
        self.mediamtx_port = 8554
        self.mediamtx_config_path = "/app/data/mediamtx.yml"
        self.mediamtx_binary_path = self._find_mediamtx_binary()
        self.config_file = Path("data/camera_streaming_config.json")
        self.auto_start_cameras = self._load_auto_start_config()
        self._lock = threading.Lock()
    
    def _find_mediamtx_binary(self) -> Optional[str]:
        """Find MediaMTX binary in various locations"""
        # Check common locations
        possible_paths = [
            '/home/javokhir/go/src/gitlab.com/udevs/mediamtx',
            '/usr/local/bin/mediamtx',
            '/usr/bin/mediamtx',
            './mediamtx',
            'mediamtx'  # In PATH
        ]
        
        for path in possible_paths:
            if shutil.which(path):
                logger.info(f"Found MediaMTX at: {path}")
                return path
        
        logger.warning("MediaMTX not found in any common locations")
        return None
    
    def _install_mediamtx(self) -> bool:
        """Try to install MediaMTX"""
        try:
            logger.info("Attempting to install MediaMTX...")
            
            # Try to download and install MediaMTX
            install_dir = Path("/usr/local/bin")
            if not install_dir.exists():
                install_dir = Path.home() / "bin"
                install_dir.mkdir(exist_ok=True)
            
            # Download MediaMTX
            download_url = "https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.0.0_linux_amd64.tar.gz"
            temp_file = "/tmp/mediamtx.tar.gz"
            
            # Use curl or wget to download
            download_cmd = None
            if shutil.which("curl"):
                download_cmd = ["curl", "-L", "-o", temp_file, download_url]
            elif shutil.which("wget"):
                download_cmd = ["wget", "-O", temp_file, download_url]
            
            if not download_cmd:
                logger.error("Neither curl nor wget found for downloading MediaMTX")
                return False
            
            # Download
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Failed to download MediaMTX: {result.stderr}")
                return False
            
            # Extract and install
            extract_cmd = ["tar", "-xzf", temp_file, "-C", "/tmp"]
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to extract MediaMTX: {result.stderr}")
                return False
            
            # Move to install directory
            binary_path = install_dir / "mediamtx"
            shutil.move("/tmp/mediamtx", str(binary_path))
            os.chmod(binary_path, 0o755)
            
            # Clean up
            os.remove(temp_file)
            
            self.mediamtx_binary_path = str(binary_path)
            logger.info(f"MediaMTX installed successfully at: {binary_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install MediaMTX: {e}")
            return False
    
    def _load_auto_start_config(self) -> Dict[str, bool]:
        """Load auto-start configuration for cameras"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('auto_start_cameras', {})
        except Exception as e:
            logger.warning(f"Could not load auto-start config: {e}")
        return {}
    
    def _save_auto_start_config(self):
        """Save auto-start configuration"""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            config = {
                'auto_start_cameras': self.auto_start_cameras,
                'last_updated': time.time()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Auto-start configuration saved")
        except Exception as e:
            logger.error(f"Could not save auto-start config: {e}")
    
    def set_auto_start_camera(self, device_path: str, auto_start: bool):
        """Set whether a camera should auto-start"""
        self.auto_start_cameras[device_path] = auto_start
        self._save_auto_start_config()
        logger.info(f"Auto-start for {device_path} set to: {auto_start}")
    
    def start_stream_monitoring(self):
        """Start monitoring streams for failures and auto-restart"""
        def monitor_streams():
            while True:
                try:
                    time.sleep(10)  # Check every 10 seconds
                    
                    with self._lock:
                        for device_path, camera in self.detected_cameras.items():
                            if camera.is_streaming and camera.ffmpeg_process:
                                # Check if FFmpeg process is still running
                                if camera.ffmpeg_process.poll() is not None:
                                    logger.warning(f"FFmpeg process for {device_path} has stopped, attempting restart...")
                                    
                                    # Clear the old process reference
                                    camera.ffmpeg_process = None
                                    camera.is_streaming = False
                                    camera.status = CameraStatus.AVAILABLE
                                    
                                    # Restart the stream if it was auto-started
                                    if self.auto_start_cameras.get(device_path, False):
                                        logger.info(f"Auto-restarting stream for {device_path}")
                                        time.sleep(2)  # Brief delay before restart
                                        self.start_camera_stream(device_path, auto_start=True)
                                        
                except Exception as e:
                    logger.error(f"Error in stream monitoring: {e}")
                    time.sleep(5)
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_streams, daemon=True)
        monitor_thread.start()
        logger.info("Stream monitoring started")
    
    def _kill_existing_mediamtx_processes(self):
        """Kill any existing MediaMTX processes"""
        try:
            # Find MediaMTX processes
            result = subprocess.run(['pgrep', '-f', 'mediamtx'], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        try:
                            subprocess.run(['kill', pid.strip()], check=False)
                            logger.info(f"Killed existing MediaMTX process {pid}")
                        except Exception as e:
                            logger.debug(f"Could not kill process {pid}: {e}")
            
            # Wait a moment for processes to die
            time.sleep(2)
            
        except Exception as e:
            logger.debug(f"Error killing existing MediaMTX processes: {e}")
        
    def detect_cameras(self) -> List[CameraInfo]:
        """Detect all available video devices"""
        cameras = []
        
        try:
            # Method 1: Check /dev/video* devices
            video_devices = list(Path("/dev").glob("video*"))
            
            for device_path in video_devices:
                if device_path.is_char_device():
                    camera_info = self._probe_camera(device_path)
                    if camera_info:
                        cameras.append(camera_info)
                        
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
            
        # Method 2: Use v4l2-ctl if available
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                cameras.extend(self._parse_v4l2_output(result.stdout))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("v4l2-ctl not available, using basic detection")
            
        # Method 3: Use ffmpeg to list devices
        try:
            cameras.extend(self._detect_with_ffmpeg())
        except Exception as e:
            logger.debug(f"FFmpeg detection failed: {e}")
            
        # Store detected cameras for later use
        for camera in cameras:
            self.detected_cameras[camera.device_path] = camera
            
        # Group cameras by physical device (same name and description)
        grouped_cameras = self._group_cameras_by_device(cameras)
        
        logger.info(f"Detected {len(grouped_cameras)} physical cameras: {[cam.name for cam in grouped_cameras]}")
        
        # Auto-start cameras that were previously configured for auto-start
        self._auto_start_cameras()
        
        # Auto-start the working camera (/dev/video0) if no cameras are auto-started
        if not any(cam.is_streaming for cam in self.detected_cameras.values()):
            working_camera = self.detected_cameras.get('/dev/video0')
            if working_camera and working_camera.status == CameraStatus.AVAILABLE:
                logger.info("Auto-starting working camera /dev/video0 for main application")
                self.start_camera_stream('/dev/video0', 'camera__dev_video0')
        
        return grouped_cameras
    
    def _group_cameras_by_device(self, cameras: List[CameraInfo]) -> List[CameraInfo]:
        """Group cameras by physical device and combine their resolutions"""
        device_groups = {}
        
        for camera in cameras:
            # Use name and description as grouping key
            device_key = f"{camera.name}:{camera.description}"
            
            if device_key not in device_groups:
                device_groups[device_key] = []
            device_groups[device_key].append(camera)
        
        grouped_cameras = []
        for device_key, device_cameras in device_groups.items():
            if len(device_cameras) == 1:
                # Single camera, no grouping needed
                grouped_cameras.append(device_cameras[0])
            else:
                # Multiple cameras for same device, group them
                primary_camera = device_cameras[0]
                
                # Collect all resolutions and formats
                all_resolutions = []
                all_formats = set()
                
                for cam in device_cameras:
                    all_resolutions.append(cam.resolution)
                    all_formats.update(cam.supported_formats)
                
                # Use the highest resolution as default
                primary_resolution = max(all_resolutions, key=lambda r: r[0] * r[1])
                
                # Create grouped camera info
                grouped_camera = CameraInfo(
                    device_path=primary_camera.device_path,
                    name=primary_camera.name,
                    description=primary_camera.description,
                    resolution=primary_resolution,
                    supported_formats=list(all_formats),
                    status=primary_camera.status,
                    capabilities={
                        'available_resolutions': all_resolutions,
                        'available_devices': [cam.device_path for cam in device_cameras],
                        'grouped_cameras': device_cameras
                    }
                )
                
                grouped_cameras.append(grouped_camera)
                
                # Store all individual cameras for streaming purposes
                for cam in device_cameras:
                    self.detected_cameras[cam.device_path] = cam
        
        return grouped_cameras
    
    def get_cameras_for_api(self) -> List[CameraInfo]:
        """Get cameras in a format suitable for API responses (no thread locks)"""
        cameras = []
        for camera in self.detected_cameras.values():
            # Create a clean copy without problematic fields
            clean_camera = CameraInfo(
                device_path=camera.device_path,
                name=camera.name,
                description=camera.description,
                resolution=camera.resolution,
                supported_formats=camera.supported_formats,
                status=camera.status,
                capabilities=camera.capabilities,
                is_streaming=camera.is_streaming,
                rtsp_url=camera.rtsp_url,
                ffmpeg_process=None  # Don't include the process object
            )
            cameras.append(clean_camera)
        return cameras
    
    def _probe_camera(self, device_path: Path) -> Optional[CameraInfo]:
        """Probe a specific camera device"""
        try:
            device_str = str(device_path)
            
            # Test if device is accessible
            with open(device_path, 'rb') as f:
                pass  # Just test if we can open it
                
            # Get basic info using v4l2-ctl
            name = f"Camera {device_path.name}"
            description = f"Video device {device_path.name}"
            resolution = (640, 480)  # Default
            supported_formats = ["MJPG", "YUYV"]
            capabilities = {}
            
            try:
                # Try to get more detailed info
                result = subprocess.run([
                    'v4l2-ctl', '--device', device_str, '--info'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    info = self._parse_v4l2_info(result.stdout)
                    name = info.get('name', name)
                    description = info.get('description', description)
                    capabilities = info.get('capabilities', {})
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug(f"Could not get detailed info for {device_str}")
                
            # Test resolution with ffmpeg
            try:
                resolution = self._test_camera_resolution(device_str)
            except Exception as e:
                logger.debug(f"Could not test resolution for {device_str}: {e}")
                
            return CameraInfo(
                device_path=device_str,
                name=name,
                description=description,
                resolution=resolution,
                supported_formats=supported_formats,
                status=CameraStatus.AVAILABLE,
                capabilities=capabilities
            )
            
        except Exception as e:
            logger.debug(f"Could not probe camera {device_path}: {e}")
            return None
    
    def _parse_v4l2_output(self, output: str) -> List[CameraInfo]:
        """Parse v4l2-ctl --list-devices output"""
        cameras = []
        lines = output.split('\n')
        current_name = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '\t' not in line and line.startswith('/dev/video'):
                # This is a device path
                device_path = line
                camera = self._probe_camera(Path(device_path))
                if camera:
                    camera.name = current_name or camera.name
                    cameras.append(camera)
            elif '\t' not in line:
                # This is likely a camera name
                current_name = line
                
        return cameras
    
    def _parse_v4l2_info(self, output: str) -> Dict:
        """Parse v4l2-ctl --info output"""
        info = {}
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'card_name':
                    info['name'] = value
                elif key == 'bus_info':
                    info['description'] = value
                elif key == 'driver_name':
                    info['driver'] = value
                    
        return info
    
    def _detect_with_ffmpeg(self) -> List[CameraInfo]:
        """Detect cameras using ffmpeg"""
        cameras = []
        
        try:
            # Try to list V4L2 devices
            cmd = [
                'ffmpeg', '-f', 'v4l2', '-list_formats', 'all', '-i', '/dev/video0'
            ]
            
            # Test common video devices
            for i in range(10):  # Check /dev/video0 to /dev/video9
                device_path = f"/dev/video{i}"
                if Path(device_path).exists():
                    try:
                        result = subprocess.run([
                            'ffmpeg', '-f', 'v4l2', '-list_formats', 'all', '-i', device_path
                        ], capture_output=True, text=True, timeout=5)
                        
                        if result.returncode == 0:
                            camera = self._parse_ffmpeg_output(device_path, result.stderr)
                            if camera:
                                cameras.append(camera)
                                
                    except subprocess.TimeoutExpired:
                        continue
                        
        except Exception as e:
            logger.debug(f"FFmpeg detection error: {e}")
            
        return cameras
    
    def _parse_ffmpeg_output(self, device_path: str, output: str) -> Optional[CameraInfo]:
        """Parse ffmpeg device listing output"""
        try:
            lines = output.split('\n')
            formats = []
            resolution = (640, 480)
            
            for line in lines:
                if '[v4l2 @' in line and 'Raw' in line:
                    # Extract format info
                    if 'yuyv422' in line.lower():
                        formats.append('YUYV')
                    elif 'mjpeg' in line.lower():
                        formats.append('MJPG')
                        
                elif 'Size:' in line:
                    # Extract resolution
                    try:
                        size_part = line.split('Size:')[1].strip()
                        if 'x' in size_part:
                            parts = size_part.split()
                            for part in parts:
                                if 'x' in part and part.replace('x', '').isdigit():
                                    w, h = map(int, part.split('x'))
                                    if w > resolution[0] and h > resolution[1]:
                                        resolution = (w, h)
                                    break
                    except:
                        pass
                        
            if formats:
                return CameraInfo(
                    device_path=device_path,
                    name=f"Camera {device_path}",
                    description=f"FFmpeg detected camera {device_path}",
                    resolution=resolution,
                    supported_formats=formats,
                    status=CameraStatus.AVAILABLE,
                    capabilities={}
                )
                
        except Exception as e:
            logger.debug(f"Error parsing ffmpeg output: {e}")
            
        return None
    
    def _test_camera_resolution(self, device_path: str) -> tuple:
        """Test camera resolution by attempting to capture"""
        try:
            # Try common resolutions
            test_resolutions = [(1920, 1080), (1280, 720), (640, 480)]
            
            for width, height in test_resolutions:
                try:
                    cmd = [
                        'ffmpeg', '-f', 'v4l2', '-video_size', f'{width}x{height}',
                        '-i', device_path, '-frames', '1', '-f', 'null', '-'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, timeout=3)
                    if result.returncode == 0:
                        return (width, height)
                        
                except subprocess.TimeoutExpired:
                    continue
                    
        except Exception as e:
            logger.debug(f"Resolution test failed for {device_path}: {e}")
            
        return (640, 480)  # Fallback
    
    def start_mediamtx(self) -> bool:
        """Start MediaMTX server for RTSP streaming"""
        try:
            # Check if MediaMTX is already running
            if self.mediamtx_process and self.mediamtx_process.poll() is None:
                logger.info("MediaMTX is already running")
                return True
            
            # Kill any existing MediaMTX processes
            self._kill_existing_mediamtx_processes()
            
            # Check if MediaMTX binary is available
            if not self.mediamtx_binary_path:
                logger.info("MediaMTX not found, attempting to install...")
                if not self._install_mediamtx():
                    logger.error("Failed to install MediaMTX")
                    return False
            
            # Try different ports if default is in use
            for port_offset in range(5):  # Try ports 8554-8558
                self.mediamtx_port = 8554 + port_offset
                
                # Create MediaMTX config with current port
                self._create_mediamtx_config()
                
                # Start MediaMTX
                cmd = [self.mediamtx_binary_path, self.mediamtx_config_path]
                logger.info(f"Starting MediaMTX with command: {' '.join(cmd)}")
                
                self.mediamtx_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                
                # Wait a moment for startup
                time.sleep(3)
                
                if self.mediamtx_process.poll() is None:
                    logger.info(f"MediaMTX started successfully on port {self.mediamtx_port}")
                    
                    # Auto-start cameras that were previously configured
                    self._auto_start_cameras()
                    return True
                else:
                    stdout, stderr = self.mediamtx_process.communicate()
                    logger.warning(f"MediaMTX failed on port {self.mediamtx_port}. Exit code: {self.mediamtx_process.returncode}")
                    logger.debug(f"MediaMTX stdout: {stdout}")
                    logger.debug(f"MediaMTX stderr: {stderr}")
                    
                    # If port is in use, try next port
                    if "address already in use" in stdout or "address already in use" in stderr:
                        logger.info(f"Port {self.mediamtx_port} is in use, trying next port...")
                        continue
                    else:
                        # Other error, try minimal config
                        logger.info("Attempting to start MediaMTX with minimal configuration...")
                        return self._start_mediamtx_minimal()
            
            logger.error("Failed to start MediaMTX on any port")
            return False
                
        except Exception as e:
            logger.error(f"Error starting MediaMTX: {e}")
            return False
    
    def _auto_start_cameras(self):
        """Auto-start cameras that were previously configured"""
        for device_path, should_auto_start in self.auto_start_cameras.items():
            if should_auto_start:
                logger.info(f"Auto-starting camera: {device_path}")
                self.start_camera_stream(device_path)
    
    def _create_mediamtx_config(self):
        """Create MediaMTX configuration file"""
        # Use different UDP ports to avoid conflicts
        rtp_port = 8000 + (self.mediamtx_port - 8554)
        rtcp_port = 8001 + (self.mediamtx_port - 8554)
        
        config = f"""# MediaMTX configuration for camera streaming
rtspAddress: :{self.mediamtx_port}
rtspEncryption: "no"
rtpAddress: :{rtp_port}
rtcpAddress: :{rtcp_port}
api: yes
apiAddress: :{9997 + (self.mediamtx_port - 8554)}
logLevel: info

paths:
  ~^.*$:
    source: publisher
"""
        
        with open(self.mediamtx_config_path, 'w') as f:
            f.write(config)
            
        logger.info(f"MediaMTX config created at {self.mediamtx_config_path}")
    
    def _start_mediamtx_minimal(self) -> bool:
        """Start MediaMTX with minimal configuration"""
        try:
            # Create minimal config (same as main config now)
            minimal_config = f"""# MediaMTX minimal configuration
rtspAddress: :{self.mediamtx_port}
api: yes
apiAddress: :9997
logLevel: info

paths:
  ~^.*$:
    source: publisher
"""
            
            minimal_config_path = "/tmp/mediamtx_minimal.yml"
            with open(minimal_config_path, 'w') as f:
                f.write(minimal_config)
            
            logger.info(f"Created minimal MediaMTX config at {minimal_config_path}")
            
            # Start with minimal config
            cmd = [self.mediamtx_binary_path, minimal_config_path]
            logger.info(f"Starting MediaMTX with minimal config: {' '.join(cmd)}")
            
            self.mediamtx_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            time.sleep(3)
            
            if self.mediamtx_process.poll() is None:
                logger.info(f"MediaMTX started successfully with minimal config on port {self.mediamtx_port}")
                return True
            else:
                stdout, stderr = self.mediamtx_process.communicate()
                logger.error(f"MediaMTX minimal config also failed. Exit code: {self.mediamtx_process.returncode}")
                logger.error(f"Minimal config stdout: {stdout}")
                logger.error(f"Minimal config stderr: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting MediaMTX with minimal config: {e}")
            return False
    
    def start_camera_stream(self, device_path: str, stream_name: str = None, auto_start: bool = False, resolution: tuple = None) -> bool:
        """Start streaming a camera to RTSP"""
        logger.info(f"Starting camera stream for {device_path}, stream_name={stream_name}, auto_start={auto_start}")
        
        if device_path not in self.detected_cameras:
            logger.error(f"Camera {device_path} not found in detected cameras: {list(self.detected_cameras.keys())}")
            return False
            
        camera = self.detected_cameras[device_path]
        
        if camera.is_streaming:
            logger.warning(f"Camera {device_path} is already streaming")
            return True
            
        if not stream_name:
            stream_name = f"camera_{device_path.replace('/', '_')}"
            
        # Ensure MediaMTX is running
        if not self.mediamtx_process or self.mediamtx_process.poll() is not None:
            if not self.start_mediamtx():
                return False
        
        # Start FFmpeg stream with reconnection logic
        try:
            rtsp_url = f"rtsp://localhost:{self.mediamtx_port}/{stream_name}"
            
            # Test camera access first
            test_cmd = ['ffmpeg', '-f', 'v4l2', '-i', device_path, '-frames', '1', '-f', 'null', '-']
            logger.info(f"Testing camera access: {' '.join(test_cmd)}")
            
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            if test_result.returncode != 0:
                logger.error(f"Camera test failed for {device_path}: {test_result.stderr}")
                camera.status = CameraStatus.ERROR
                # Update the camera in detected_cameras
                self.detected_cameras[device_path] = camera
                return False
            
            logger.info(f"Camera test successful for {device_path}")
            
            # Determine best input format
            input_format = 'mjpeg' if 'MJPG' in camera.supported_formats else 'yuyv422'
            
            # Use specified resolution or camera's default
            stream_resolution = resolution if resolution else camera.resolution
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'v4l2',
                '-input_format', input_format,
                '-video_size', f'{stream_resolution[0]}x{stream_resolution[1]}',
                '-i', device_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '2',
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                rtsp_url
            ]
            
            logger.info(f"Starting FFmpeg stream: {' '.join(ffmpeg_cmd)}")
            
            camera.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            # Wait a moment and check if FFmpeg started successfully
            time.sleep(2)
            
            if camera.ffmpeg_process.poll() is not None:
                stdout, stderr = camera.ffmpeg_process.communicate()
                logger.error(f"FFmpeg failed to start for {device_path}")
                logger.error(f"FFmpeg stdout: {stdout}")
                logger.error(f"FFmpeg stderr: {stderr}")
                camera.status = CameraStatus.ERROR
                return False
            
            # Update camera status
            camera.is_streaming = True
            camera.rtsp_url = rtsp_url
            camera.status = CameraStatus.STREAMING
            
            # Save auto-start preference if requested
            if auto_start:
                self.set_auto_start_camera(device_path, True)
            
            logger.info(f"Started streaming {device_path} to {rtsp_url}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Camera test timeout for {device_path}")
            camera.status = CameraStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Error starting stream for {device_path}: {e}")
            camera.status = CameraStatus.ERROR
            return False
    
    def stop_camera_stream(self, device_path: str) -> bool:
        """Stop streaming a camera"""
        if device_path not in self.detected_cameras:
            return False
            
        camera = self.detected_cameras[device_path]
        
        if camera.ffmpeg_process:
            try:
                camera.ffmpeg_process.terminate()
                camera.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                camera.ffmpeg_process.kill()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg for {device_path}: {e}")
            
            camera.ffmpeg_process = None
            
        camera.is_streaming = False
        camera.rtsp_url = None
        camera.status = CameraStatus.AVAILABLE
        
        logger.info(f"Stopped streaming {device_path}")
        return True
    
    def get_camera_list(self) -> List[Dict]:
        """Get list of detected cameras as dictionaries"""
        with self._lock:
            cameras = []
            for camera in self.detected_cameras.values():
                cam_dict = {
                    'device_path': camera.device_path,
                    'name': camera.name,
                    'description': camera.description,
                    'resolution': camera.resolution,
                    'supported_formats': camera.supported_formats,
                    'status': camera.status,
                    'capabilities': camera.capabilities,
                    'is_streaming': camera.is_streaming,
                    'rtsp_url': camera.rtsp_url
                }
                cameras.append(cam_dict)
            return cameras
    
    def get_camera(self, device_path: str) -> Optional[CameraInfo]:
        """Get specific camera information"""
        with self._lock:
            return self.detected_cameras.get(device_path)
    
    def stop_all_streams(self):
        """Stop all camera streams"""
        with self._lock:
            for device_path in list(self.detected_cameras.keys()):
                self.stop_camera_stream(device_path)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_all_streams()
        
        if self.mediamtx_process:
            try:
                self.mediamtx_process.terminate()
                self.mediamtx_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mediamtx_process.kill()
            except Exception as e:
                logger.error(f"Error stopping MediaMTX: {e}")
                
        # Clean up config file
        try:
            if os.path.exists(self.mediamtx_config_path):
                os.remove(self.mediamtx_config_path)
        except Exception as e:
            logger.debug(f"Could not remove config file: {e}")
