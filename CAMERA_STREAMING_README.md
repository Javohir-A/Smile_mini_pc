# 🎥 Camera Streaming Feature

This document describes the new camera streaming feature that allows your Mini PC application to detect local web cameras and expose them as RTSP streams.

## 🚀 Features

- **Automatic Camera Detection**: Detects available web cameras connected to the Mini PC
- **RTSP Streaming**: Exposes cameras as RTSP streams using MediaMTX and FFmpeg
- **Web Interface**: Beautiful web interface for camera management (port 8765)
- **API Endpoints**: RESTful API for programmatic camera control
- **Docker Integration**: Seamlessly integrated with your existing Docker setup
- **Real-time Monitoring**: Live camera status and streaming statistics

## 📋 Prerequisites

- Linux-based Mini PC
- Docker and Docker Compose installed
- Web cameras connected via USB
- FFmpeg and v4l-utils (automatically installed by Docker)

## 🛠️ Quick Setup

1. **Run the setup script**:
   ```bash
   ./setup_camera_streaming.sh
   ```

2. **Test camera detection**:
   ```bash
   ./test_cameras.sh
   ```

3. **Start the services**:
   ```bash
   ./start_camera_streaming.sh
   ```

4. **Access the web interface**:
   - Camera Management: http://localhost:8765/camera_management.html
   - Main Streams: http://localhost:8765

## 🏗️ Architecture

```
Mini PC
├── Local Web Cameras (/dev/video0, /dev/video1, etc.)
├── Camera Detection Service (V4L2)
├── MediaMTX Server (RTSP server on port 8554)
├── FFmpeg Streamers (per camera)
└── FastAPI Web Interface (port 8765)
```

## 🔧 API Endpoints

### Camera Management

- `GET /api/cameras/` - List all detected cameras
- `POST /api/cameras/detect` - Force camera detection
- `GET /api/cameras/{device_path}` - Get specific camera info
- `POST /api/cameras/{device_path}/stream` - Start streaming a camera
- `POST /api/cameras/{device_path}/stop` - Stop streaming a camera
- `POST /api/cameras/stop-all` - Stop all streams

### MediaMTX Management

- `GET /api/cameras/status/mediamtx` - Get MediaMTX server status
- `POST /api/cameras/status/mediamtx/start` - Start MediaMTX server

### Active Streams

- `GET /api/cameras/streams/active` - Get list of active streams

## 📱 Web Interface

The web interface provides:

- **Camera Detection**: Scan for available cameras
- **Stream Management**: Start/stop individual camera streams
- **Real-time Status**: Live status of cameras and streams
- **RTSP URLs**: Copy RTSP URLs for external access
- **MediaMTX Control**: Start/stop MediaMTX server

### Features:
- Responsive design for mobile and desktop
- Real-time updates every 30 seconds
- One-click stream management
- Copy RTSP URLs to clipboard
- Visual status indicators

## 🐳 Docker Configuration

### Updated Dockerfile
- Added MediaMTX binary
- Added v4l-utils for camera detection
- Added FFmpeg for streaming

### Updated docker-compose.yaml
- Added device access for `/dev/video*`
- Added MediaMTX ports (8554, 9997, 9998)
- Added environment variables for camera management

## 🔍 Camera Detection Methods

The system uses multiple methods to detect cameras:

1. **Device File Scanning**: Scans `/dev/video*` devices
2. **v4l2-ctl**: Uses Video4Linux2 utilities for detailed info
3. **FFmpeg Detection**: Tests camera accessibility with FFmpeg
4. **Resolution Testing**: Tests different resolutions to find optimal settings

## 📊 Streaming Configuration

### Default Settings
- **Resolution**: Auto-detected (up to 1920x1080)
- **Format**: MJPG (preferred) or YUYV
- **FPS**: 30 FPS
- **Codec**: H.264 with ultrafast preset
- **Transport**: TCP

### RTSP URLs
Cameras are exposed as RTSP streams:
```
rtsp://localhost:8554/camera_/dev/video0
rtsp://localhost:8554/camera_/dev/video1
```

## 🔧 Configuration

### Environment Variables
```bash
ENABLE_LOCAL_CAMERA_DETECTION=true
MEDIAMTX_PORT=8554
MEDIAMTX_API_PORT=9997
MEDIAMTX_METRICS_PORT=9998
```

### MediaMTX Configuration
The system creates a MediaMTX configuration file at `/tmp/mediamtx.yml` with optimized settings for camera streaming.

## 🚨 Troubleshooting

### Camera Not Detected
1. Check USB connection
2. Verify device appears in `/dev/video*`
3. Check user permissions (should be in `video` group)
4. Run `./test_cameras.sh` for diagnostics

### Permission Issues
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and log back in
```

### FFmpeg Errors
1. Check camera format support
2. Verify camera is not in use by another application
3. Try different input formats (MJPG vs YUYV)

### MediaMTX Issues
1. Check if port 8554 is available
2. Verify MediaMTX binary is installed
3. Check MediaMTX logs in container

## 📈 Monitoring

### Logs
```bash
# View application logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f camera-app
```

### Metrics
- MediaMTX metrics: http://localhost:9998/metrics
- MediaMTX API: http://localhost:9997

### WebSocket Connections
The system maintains WebSocket connections for real-time updates:
- Single stream: `ws://localhost:8765/ws/stream/{stream_id}`
- All streams: `ws://localhost:8765/ws/streams/all`

## 🔒 Security Considerations

- Docker containers run as non-root user
- Device access is limited to video devices only
- MediaMTX server is bound to localhost by default
- No external RTSP access by default (can be configured)

## 🎯 Usage Examples

### Start Streaming a Camera
```bash
curl -X POST http://localhost:8765/api/cameras/\/dev\/video0/stream \
  -H "Content-Type: application/json" \
  -d '{"stream_name": "front_camera"}'
```

### Stop All Streams
```bash
curl -X POST http://localhost:8765/api/cameras/stop-all
```

### Get Camera List
```bash
curl http://localhost:8765/api/cameras/
```

## 🔄 Integration with Existing System

The camera streaming feature integrates seamlessly with your existing system:

- Uses the same FastAPI server (port 8765)
- Extends existing camera management
- Maintains compatibility with current RTSP streams
- Adds local camera detection alongside external cameras

## 📝 Development Notes

### File Structure
```
src/
├── services/
│   └── camera_detection_service.py    # Core camera detection logic
├── api/
│   └── camera_management_api.py       # API endpoints
└── api/
    └── fastapi_server.py              # Updated main server

frontend/
├── index.html                         # Updated main interface
└── camera_management.html             # New camera management interface
```

### Key Components
- `CameraDetectionService`: Handles camera detection and streaming
- `CameraManagementAPI`: RESTful API endpoints
- `FastAPIWebSocketServer`: Extended with camera management
- Web interfaces for user interaction

## 🎉 Success!

Your Mini PC now has the ability to:
- ✅ Detect local web cameras automatically
- ✅ Stream cameras via RTSP using MediaMTX
- ✅ Manage cameras through a beautiful web interface
- ✅ Provide APIs for programmatic control
- ✅ Integrate seamlessly with your existing system

Enjoy your enhanced camera streaming capabilities! 🎥✨

