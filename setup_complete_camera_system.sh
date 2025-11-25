#!/bin/bash

# Complete Camera Streaming Setup Script
# This script sets up the entire camera streaming system with auto-startup

set -e

echo "🎥 Setting up Complete Camera Streaming System"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Get current user and directory
USER=$(whoami)
CURRENT_DIR=$(pwd)
SERVICE_FILE="smile-camera-streaming.service"

print_status "Setting up camera streaming for user: $USER"
print_status "Working directory: $CURRENT_DIR"

# 1. Install MediaMTX if not found
print_status "Checking for MediaMTX..."
if ! command -v mediamtx &> /dev/null; then
    print_warning "MediaMTX not found, installing..."
    
    # Try to install MediaMTX
    MEDIAMTX_URL="https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.0.0_linux_amd64.tar.gz"
    TEMP_FILE="/tmp/mediamtx.tar.gz"
    
    if command -v wget &> /dev/null; then
        wget -O "$TEMP_FILE" "$MEDIAMTX_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$TEMP_FILE" "$MEDIAMTX_URL"
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    # Extract and install
    tar -xzf "$TEMP_FILE" -C /tmp
    sudo mv /tmp/mediamtx /usr/local/bin/mediamtx
    sudo chmod +x /usr/local/bin/mediamtx
    rm "$TEMP_FILE"
    
    print_success "MediaMTX installed successfully"
else
    print_success "MediaMTX already installed"
fi

# 2. Install system dependencies
print_status "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y v4l-utils ffmpeg python3-pip

# 3. Add user to video group
print_status "Adding user to video group..."
sudo usermod -a -G video "$USER"
print_warning "You may need to log out and log back in for group changes to take effect"

# 4. Create necessary directories
print_status "Creating directories..."
mkdir -p data logs temp_videos videos
chmod 755 data logs temp_videos videos

# 5. Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install fastapi uvicorn websockets

# 6. Test camera detection
print_status "Testing camera detection..."
if ls /dev/video* 1> /dev/null 2>&1; then
    print_success "Found video devices:"
    ls -la /dev/video* | while read line; do
        echo "  $line"
    done
else
    print_warning "No video devices found. Make sure cameras are connected."
fi

# 7. Setup systemd service for auto-startup
print_status "Setting up systemd service for auto-startup..."

# Update service file with current user and directory
sed -i "s|User=javokhir|User=$USER|g" "$SERVICE_FILE"
sed -i "s|Group=javokhir|Group=$USER|g" "$SERVICE_FILE"
sed -i "s|WorkingDirectory=/home/javokhir/go/src/gitlab.com/udevs/smile/smile_mini_pc_app|WorkingDirectory=$CURRENT_DIR|g" "$SERVICE_FILE"
sed -i "s|Environment=PYTHONPATH=/home/javokhir/go/src/gitlab.com/udevs/smile/smile_mini_pc_app|Environment=PYTHONPATH=$CURRENT_DIR|g" "$SERVICE_FILE"

# Copy service file to systemd
sudo cp "$SERVICE_FILE" /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable smile-camera-streaming.service

print_success "Systemd service installed and enabled"

# 8. Create startup scripts
print_status "Creating startup scripts..."

# Create start script
cat > start_camera_system.sh << EOF
#!/bin/bash
echo "🚀 Starting Smile Camera Streaming System"
echo "========================================="

# Start the service
sudo systemctl start smile-camera-streaming.service

# Check status
sudo systemctl status smile-camera-streaming.service --no-pager

echo ""
echo "✅ Camera streaming system started!"
echo ""
echo "📱 Access the system:"
echo "   Main Interface: http://localhost:8765"
echo "   Camera Management: http://localhost:8765/camera_management.html"
echo "   Test Page: http://localhost:8765/test_camera_streaming.html"
echo ""
echo "📊 Check status: sudo systemctl status smile-camera-streaming.service"
echo "📋 View logs: sudo journalctl -u smile-camera-streaming.service -f"
EOF

# Create stop script
cat > stop_camera_system.sh << EOF
#!/bin/bash
echo "⏹️  Stopping Smile Camera Streaming System"
echo "=========================================="

sudo systemctl stop smile-camera-streaming.service

echo "✅ Camera streaming system stopped!"
EOF

# Create status script
cat > status_camera_system.sh << EOF
#!/bin/bash
echo "📊 Smile Camera Streaming System Status"
echo "======================================="

echo "Service Status:"
sudo systemctl status smile-camera-streaming.service --no-pager

echo ""
echo "Recent Logs:"
sudo journalctl -u smile-camera-streaming.service --no-pager -n 20

echo ""
echo "Active Camera Streams:"
curl -s http://localhost:8765/api/cameras/streams/active | jq '.' 2>/dev/null || echo "API not available or jq not installed"
EOF

chmod +x start_camera_system.sh stop_camera_system.sh status_camera_system.sh

print_success "Startup scripts created"

# 9. Create configuration file
print_status "Creating initial configuration..."
cat > data/camera_streaming_config.json << EOF
{
  "auto_start_cameras": {},
  "last_updated": $(date +%s),
  "settings": {
    "auto_install_mediamtx": true,
    "auto_start_on_boot": true,
    "stream_monitoring": true,
    "reconnect_attempts": 5
  }
}
EOF

# 10. Test the system
print_status "Testing the system..."
if python3 -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
    print_success "Python dependencies are working"
else
    print_error "Python dependencies test failed"
    exit 1
fi

# 11. Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "📋 What was installed:"
echo "✅ MediaMTX server for RTSP streaming"
echo "✅ System dependencies (v4l-utils, ffmpeg)"
echo "✅ Python dependencies"
echo "✅ Systemd service for auto-startup"
echo "✅ Camera detection and streaming system"
echo ""
echo "🚀 To start the system:"
echo "   ./start_camera_system.sh"
echo ""
echo "⏹️  To stop the system:"
echo "   ./stop_camera_system.sh"
echo ""
echo "📊 To check status:"
echo "   ./status_camera_system.sh"
echo ""
echo "🌐 Access the web interface:"
echo "   http://localhost:8765/camera_management.html"
echo ""
echo "🔄 Auto-startup:"
echo "   The system will automatically start on boot"
echo "   Cameras configured for auto-start will begin streaming automatically"
echo ""
echo "📝 Configuration:"
echo "   Camera preferences are saved in: data/camera_streaming_config.json"
echo ""
print_warning "Note: You may need to log out and log back in for the video group changes to take effect"
echo ""
print_success "Setup completed! Your camera streaming system is ready! 🎥✨"

