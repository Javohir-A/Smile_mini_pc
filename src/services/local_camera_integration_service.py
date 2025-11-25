"""
Local Camera Integration Service

This service integrates local camera detection with the existing database-driven camera discovery system.
It doesn't create cameras in the database - instead, it provides local camera streams that can be used
by cameras that are already assigned to the Mini PC in the database.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from src.models.camera import Camera
from src.models.mini_pc import MiniPC
from src.services.camera_detection_service import CameraDetectionService, CameraInfo, CameraStatus
from src.di.dependencies import DependencyContainer

logger = logging.getLogger(__name__)

class LocalCameraIntegrationService:
    """Provides local camera streams for database-assigned cameras"""
    
    def __init__(self, dependency_container: DependencyContainer):
        self.container = dependency_container
        self.camera_detection_service = CameraDetectionService()
        self._mini_pc_info: Optional[MiniPC] = None
        self._local_cameras: Dict[str, CameraInfo] = {}  # device_path -> camera_info
        self._stream_mappings: Dict[str, str] = {}  # camera_guid -> stream_name
        
    def set_mini_pc_info(self, mini_pc_info: MiniPC):
        """Set the current Mini PC information"""
        self._mini_pc_info = mini_pc_info
        logger.info(f"Local camera integration set for Mini PC: {mini_pc_info.device_name}")
    
    def detect_and_start_local_cameras(self) -> Dict[str, Any]:
        """Detect local cameras and start streaming for database-assigned cameras"""
        if not self._mini_pc_info:
            logger.warning("No Mini PC info available - cannot start local cameras")
            return {"error": "No Mini PC info"}
        
        try:
            logger.info("🔍 Detecting local cameras...")
            
            # Detect local cameras
            local_cameras = self.camera_detection_service.detect_cameras()
            logger.info(f"Found {len(local_cameras)} local cameras")
            
            # Store local camera info
            self._local_cameras = {
                cam.device_path: cam for cam in local_cameras 
                if cam.status == CameraStatus.AVAILABLE
            }
            
            # Get cameras assigned to this Mini PC from database
            mini_pc_usecase = self.container.get_mini_pc_usecase()
            assigned_cameras = mini_pc_usecase.get_mini_pc_cameras(mini_pc_id=self._mini_pc_info.guid)
            
            logger.info(f"Found {len(assigned_cameras)} cameras assigned to Mini PC in database")
            
            # Start streaming for local cameras that can serve database cameras
            stream_results = {}
            local_streams_started = 0
            
            for camera in assigned_cameras:
                # Check if this database camera can be served by a local camera
                if camera.ip_address == "localhost" and camera.port == 8554:
                    # This is a local camera assignment
                    device_path = self._extract_device_path_from_rtsp(camera.rtsp_url)
                    
                    if device_path and device_path in self._local_cameras:
                        # Start streaming for this local camera
                        stream_name = f"camera__dev_{device_path.replace('/', '_')}"
                        success = self.camera_detection_service.start_camera_stream(
                            device_path=device_path,
                            stream_name=stream_name,
                            auto_start=True
                        )
                        
                        stream_results[device_path] = success
                        self._stream_mappings[str(camera.guid)] = stream_name
                        
                        if success:
                            local_streams_started += 1
                            logger.info(f"✅ Started stream for database camera {camera.guid}: {stream_name}")
                        else:
                            logger.error(f"❌ Failed to start stream for {device_path}")
                    else:
                        logger.warning(f"⚠️  Database camera {camera.guid} references unavailable local device: {device_path}")
            
            # Also start streaming for any available local cameras (even if not in database)
            for device_path, camera_info in self._local_cameras.items():
                if device_path not in stream_results:
                    stream_name = f"camera__dev_{device_path.replace('/', '_')}"
                    success = self.camera_detection_service.start_camera_stream(
                        device_path=device_path,
                        stream_name=stream_name,
                        auto_start=True
                    )
                    
                    stream_results[device_path] = success
                    if success:
                        local_streams_started += 1
                        logger.info(f"✅ Started stream for available local camera: {device_path}")
            
            result = {
                "local_cameras_detected": len(self._local_cameras),
                "database_cameras_assigned": len(assigned_cameras),
                "local_streams_started": local_streams_started,
                "total_streams": len(stream_results),
                "successful_streams": sum(1 for success in stream_results.values() if success),
                "stream_results": stream_results
            }
            
            logger.info(f"🎬 Local camera integration complete: {local_streams_started} streams started")
            return result
            
        except Exception as e:
            logger.error(f"Error in local camera integration: {e}")
            return {"error": str(e)}
    
    def _extract_device_path_from_rtsp(self, rtsp_url: str) -> Optional[str]:
        """Extract device path from RTSP URL"""
        if not rtsp_url or "camera__dev_" not in rtsp_url:
            return None
        
        try:
            # rtsp://localhost:8554/camera__dev_video0 -> /dev/video0
            device_part = rtsp_url.split("camera__dev_")[1]
            device_path = f"/dev/{device_part.replace('_', '/')}"
            return device_path
        except Exception as e:
            logger.error(f"Error extracting device path from RTSP URL {rtsp_url}: {e}")
            return None
    
    def get_available_local_cameras(self) -> Dict[str, CameraInfo]:
        """Get currently available local cameras"""
        return self._local_cameras.copy()
    
    def get_stream_mappings(self) -> Dict[str, str]:
        """Get mappings from database camera GUID to stream name"""
        return self._stream_mappings.copy()
    
    def update_database_camera_rtsp_urls(self, cameras: List[Camera]) -> List[Camera]:
        """Update RTSP URLs for database cameras to point to local streams"""
        updated_cameras = []
        
        for camera in cameras:
            if camera.ip_address == "localhost" and camera.port == 8554:
                # This is a local camera - ensure RTSP URL points to our stream
                device_path = self._extract_device_path_from_rtsp(camera.rtsp_url)
                
                if device_path and device_path in self._local_cameras:
                    # Update RTSP URL to match our stream
                    expected_stream_name = f"camera__dev_{device_path.replace('/', '_')}"
                    expected_rtsp_url = f"rtsp://localhost:8554/{expected_stream_name}"
                    
                    if camera.rtsp_url != expected_rtsp_url:
                        camera.rtsp_url = expected_rtsp_url
                        logger.info(f"Updated RTSP URL for camera {camera.guid}: {expected_rtsp_url}")
                
                updated_cameras.append(camera)
            else:
                # Remote camera - keep as is
                updated_cameras.append(camera)
        
        return updated_cameras
