# src/api/camera_management_api.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from ..services.camera_detection_service import CameraDetectionService, CameraInfo, CameraStatus
from ..config.settings import AppConfig

logger = logging.getLogger(__name__)

class CameraResponse(BaseModel):
    """Camera information response model"""
    device_path: str
    name: str
    description: str
    resolution: tuple
    supported_formats: List[str]
    status: str
    capabilities: Dict
    is_streaming: bool
    rtsp_url: Optional[str] = None

class StreamRequest(BaseModel):
    """Request model for starting a stream"""
    device_path: str
    stream_name: Optional[str] = None
    resolution: Optional[tuple] = None
    fps: Optional[int] = 30
    auto_start: Optional[bool] = False

class StreamResponse(BaseModel):
    """Response model for stream operations"""
    success: bool
    message: str
    rtsp_url: Optional[str] = None
    stream_name: Optional[str] = None

class CameraManagementAPI:
    """API endpoints for managing local cameras and RTSP streaming"""
    
    def __init__(self, camera_service: CameraDetectionService):
        self.camera_service = camera_service
        self.router = APIRouter(prefix="", tags=["camera-management"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.router.get("/api/cameras/", response_model=List[CameraResponse])
        async def get_cameras():
            """Get list of all detected cameras"""
            try:
                cameras = self.camera_service.get_camera_list()
                return [
                    CameraResponse(
                        device_path=cam['device_path'],
                        name=cam['name'],
                        description=cam['description'],
                        resolution=cam['resolution'],
                        supported_formats=cam['supported_formats'],
                        status=cam['status'].value if isinstance(cam['status'], CameraStatus) else cam['status'],
                        capabilities=cam['capabilities'],
                        is_streaming=cam['is_streaming'],
                        rtsp_url=cam['rtsp_url']
                    )
                    for cam in cameras
                ]
            except Exception as e:
                logger.error(f"Error getting cameras: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/detect", response_model=List[CameraResponse])
        async def detect_cameras():
            """Force camera detection"""
            try:
                cameras = self.camera_service.detect_cameras()
                return [
                    CameraResponse(
                        device_path=cam.device_path,
                        name=cam.name,
                        description=cam.description,
                        resolution=cam.resolution,
                        supported_formats=cam.supported_formats,
                        status=cam.status.value,
                        capabilities=cam.capabilities,
                        is_streaming=cam.is_streaming,
                        rtsp_url=cam.rtsp_url
                    )
                    for cam in cameras
                ]
            except Exception as e:
                logger.error(f"Error detecting cameras: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/api/cameras/by-path/{device_path:path}", response_model=CameraResponse)
        async def get_camera(device_path: str):
            """Get specific camera information"""
            try:
                camera = self.camera_service.get_camera(device_path)
                if not camera:
                    raise HTTPException(status_code=404, detail="Camera not found")
                
                return CameraResponse(
                    device_path=camera.device_path,
                    name=camera.name,
                    description=camera.description,
                    resolution=camera.resolution,
                    supported_formats=camera.supported_formats,
                    status=camera.status.value,
                    capabilities=camera.capabilities,
                    is_streaming=camera.is_streaming,
                    rtsp_url=camera.rtsp_url
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting camera {device_path}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/by-path/{device_path:path}/stream", response_model=StreamResponse)
        async def start_stream(device_path: str, request: StreamRequest):
            """Start streaming a camera"""
            try:
                camera = self.camera_service.get_camera(device_path)
                if not camera:
                    raise HTTPException(status_code=404, detail="Camera not found")
                
                if camera.is_streaming:
                    return StreamResponse(
                        success=True,
                        message="Camera is already streaming",
                        rtsp_url=camera.rtsp_url,
                        stream_name=request.stream_name or f"camera_{device_path.replace('/', '_')}"
                    )
                
                # Generate stream name if not provided
                stream_name = request.stream_name or f"camera_{device_path.replace('/', '_')}"
                
                success = self.camera_service.start_camera_stream(
                    device_path, 
                    stream_name, 
                    auto_start=request.auto_start,
                    resolution=request.resolution
                )
                
                if success:
                    # Get updated camera info
                    updated_camera = self.camera_service.get_camera(device_path)
                    return StreamResponse(
                        success=True,
                        message="Stream started successfully",
                        rtsp_url=updated_camera.rtsp_url if updated_camera else None,
                        stream_name=stream_name
                    )
                else:
                    # Get camera info to provide more specific error
                    camera = self.camera_service.get_camera(device_path)
                    error_msg = "Failed to start stream"
                    if camera and camera.status == CameraStatus.ERROR:
                        error_msg = f"Camera error: {camera.status.value}"
                    return StreamResponse(
                        success=False,
                        message=error_msg
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error starting stream for {device_path}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/by-path/{device_path:path}/stop", response_model=StreamResponse)
        async def stop_stream(device_path: str):
            """Stop streaming a camera"""
            try:
                camera = self.camera_service.get_camera(device_path)
                if not camera:
                    raise HTTPException(status_code=404, detail="Camera not found")
                
                if not camera.is_streaming:
                    return StreamResponse(
                        success=True,
                        message="Camera is not streaming"
                    )
                
                success = self.camera_service.stop_camera_stream(device_path)
                
                if success:
                    return StreamResponse(
                        success=True,
                        message="Stream stopped successfully"
                    )
                else:
                    return StreamResponse(
                        success=False,
                        message="Failed to stop stream"
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error stopping stream for {device_path}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/stop-all", response_model=StreamResponse)
        async def stop_all_streams():
            """Stop all camera streams"""
            try:
                self.camera_service.stop_all_streams()
                return StreamResponse(
                    success=True,
                    message="All streams stopped successfully"
                )
            except Exception as e:
                logger.error(f"Error stopping all streams: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/api/cameras/status/mediamtx")
        async def get_mediamtx_status():
            """Get MediaMTX server status"""
            try:
                mediamtx_running = (
                    self.camera_service.mediamtx_process is not None and
                    self.camera_service.mediamtx_process.poll() is None
                )
                
                # Test MediaMTX binary
                mediamtx_binary_available = self.camera_service.mediamtx_binary_path is not None
                
                return {
                    "running": mediamtx_running,
                    "port": self.camera_service.mediamtx_port if mediamtx_running else None,
                    "config_path": self.camera_service.mediamtx_config_path,
                    "binary_available": mediamtx_binary_available,
                    "binary_path": self.camera_service.mediamtx_binary_path
                }
            except Exception as e:
                logger.error(f"Error getting MediaMTX status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/status/mediamtx/start")
        async def start_mediamtx():
            """Start MediaMTX server"""
            try:
                success = self.camera_service.start_mediamtx()
                
                if success:
                    return {
                        "success": True,
                        "message": "MediaMTX started successfully",
                        "port": self.camera_service.mediamtx_port
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to start MediaMTX"
                    }
                    
            except Exception as e:
                logger.error(f"Error starting MediaMTX: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/api/cameras/test/by-path/{device_path:path}")
        async def test_camera(device_path: str):
            """Test camera access"""
            try:
                camera = self.camera_service.get_camera(device_path)
                if not camera:
                    return {"error": f"Camera {device_path} not found"}
                
                # Test camera access
                import subprocess
                test_cmd = ['ffmpeg', '-f', 'v4l2', '-i', device_path, '-frames', '1', '-f', 'null', '-']
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                
                return {
                    "device_path": device_path,
                    "camera_info": {
                        "name": camera.name,
                        "resolution": camera.resolution,
                        "supported_formats": camera.supported_formats,
                        "status": camera.status.value
                    },
                    "test_result": {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.router.get("/api/cameras/streams/active", response_model=List[CameraResponse])
        async def get_active_streams():
            """Get list of currently streaming cameras"""
            try:
                cameras = self.camera_service.get_camera_list()
                active_cameras = [
                    cam for cam in cameras 
                    if cam.get('is_streaming', False)
                ]
                
                return [
                    CameraResponse(
                        device_path=cam['device_path'],
                        name=cam['name'],
                        description=cam['description'],
                        resolution=cam['resolution'],
                        supported_formats=cam['supported_formats'],
                        status=cam['status'].value if isinstance(cam['status'], CameraStatus) else cam['status'],
                        capabilities=cam['capabilities'],
                        is_streaming=cam['is_streaming'],
                        rtsp_url=cam['rtsp_url']
                    )
                    for cam in active_cameras
                ]
            except Exception as e:
                logger.error(f"Error getting active streams: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/by-path/{device_path:path}/auto-start")
        async def set_auto_start(device_path: str, auto_start: bool):
            """Set auto-start preference for a camera"""
            try:
                self.camera_service.set_auto_start_camera(device_path, auto_start)
                return {
                    "success": True,
                    "message": f"Auto-start for {device_path} set to {auto_start}",
                    "auto_start": auto_start
                }
            except Exception as e:
                logger.error(f"Error setting auto-start for {device_path}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/api/cameras/by-path/{device_path:path}/stream/auto-start")
        async def start_stream_with_auto_start(device_path: str, request: StreamRequest):
            """Start streaming a camera with auto-start enabled"""
            try:
                camera = self.camera_service.get_camera(device_path)
                if not camera:
                    raise HTTPException(status_code=404, detail="Camera not found")
                
                if camera.is_streaming:
                    return StreamResponse(
                        success=True,
                        message="Camera is already streaming",
                        rtsp_url=camera.rtsp_url,
                        stream_name=request.stream_name or f"camera_{device_path.replace('/', '_')}"
                    )
                
                # Generate stream name if not provided
                stream_name = request.stream_name or f"camera_{device_path.replace('/', '_')}"
                
                success = self.camera_service.start_camera_stream(device_path, stream_name, auto_start=True)
                
                if success:
                    # Get updated camera info
                    updated_camera = self.camera_service.get_camera(device_path)
                    return StreamResponse(
                        success=True,
                        message="Stream started successfully with auto-start enabled",
                        rtsp_url=updated_camera.rtsp_url if updated_camera else None,
                        stream_name=stream_name
                    )
                else:
                    return StreamResponse(
                        success=False,
                        message="Failed to start stream"
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error starting auto-start stream for {device_path}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

# Factory function to create the API router
def create_camera_management_api(camera_service: CameraDetectionService) -> APIRouter:
    """Create camera management API router"""
    api = CameraManagementAPI(camera_service)
    return api.router
