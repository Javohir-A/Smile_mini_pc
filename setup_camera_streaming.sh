#!/bin/bash

# Setup script for camera streaming on Mini PC
# This script helps install and configure the necessary tools for local camera streaming

set -e

echo "🎥 Setting up Camera Streaming for Smile Mini PC"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Check for video devices
print_status "Checking for available video devices..."
if ls /dev/video* 1> /dev/null 2>&1; then
    print_success "Found video devices:"
    ls -la /dev/video* | while read line; do
        echo "  $line"
    done
else
    print_warning "No video devices found at /dev/video*"
    print_status "Make sure your web cameras are connected and drivers are installed"
fi

# Check if v4l2-ctl is available
if command -v v4l2-ctl &> /dev/null; then
    print_success "v4l2-ctl is available"
    print_status "Testing camera detection with v4l2-ctl..."
    v4l2-ctl --list-devices 2>/dev/null || print_warning "v4l2-ctl failed to list devices"
else
    print_warning "v4l2-ctl not found. Installing v4l-utils..."
    sudo apt-get update
    sudo apt-get install -y v4l-utils
fi

# Check if FFmpeg is available
if command -v ffmpeg &> /dev/null; then
    print_success "FFmpeg is available: $(ffmpeg -version | head -n1)"
else
    print_warning "FFmpeg not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs data temp_videos videos

# Set proper permissions
print_status "Setting up permissions..."
sudo chown -R $USER:$USER logs data temp_videos videos
chmod -R 755 logs data temp_videos videos

# Check if user is in video group
if ! groups $USER | grep -q video; then
    print_warning "User $USER is not in the 'video' group"
    print_status "Adding user to video group (requires sudo)..."
    sudo usermod -a -G video $USER
    print_warning "Please log out and log back in for group changes to take effect"
else
    print_success "User is in video group"
fi

# Test camera access
print_status "Testing camera access..."
if ls /dev/video* 1> /dev/null 2>&1; then
    for device in /dev/video*; do
        if [[ -r "$device" ]]; then
            print_success "Can read $device"
        else
            print_warning "Cannot read $device - check permissions"
        fi
    done
fi

# Create a test script for camera detection
print_status "Creating camera test script..."
cat > test_cameras.sh << 'EOF'
#!/bin/bash
echo "🎥 Testing Camera Detection"
echo "=========================="

echo "Available video devices:"
ls -la /dev/video* 2>/dev/null || echo "No video devices found"

echo ""
echo "v4l2-ctl device list:"
v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available or failed"

echo ""
echo "Testing FFmpeg with first available camera:"
if ls /dev/video0 1> /dev/null 2>&1; then
    echo "Testing /dev/video0 with FFmpeg..."
    timeout 5 ffmpeg -f v4l2 -i /dev/video0 -frames 1 -f null - 2>&1 | head -10 || echo "FFmpeg test failed"
else
    echo "No /dev/video0 found"
fi

echo ""
echo "Camera detection test completed!"
EOF

chmod +x test_cameras.sh
print_success "Created test_cameras.sh script"

# Create MediaMTX configuration
print_status "Creating MediaMTX configuration..."
mkdir -p config
cat > config/mediamtx.yml << 'EOF'
# MediaMTX configuration for camera streaming
rtspAddress: :8554
api: yes
apiAddress: :9997
metrics: yes
metricsAddress: :9998
logLevel: info
logDestinations: [stdout]
logFile: /tmp/mediamtx.log

# Paths configuration
paths:
  ~^.*$:
    source: publisher
    sourceProtocol: automatic
    sourceAnyPortEnable: yes
    sourceFingerprint: ""
    sourceOnDemand: yes
    sourceOnDemandStartTimeout: 10s
    sourceOnDemandCloseAfter: 10s
    sourceRedirect: ""
    disablePublisherOverride: no
    fallback: ""
    mux: gortsplib
    record: no
    recordPath: /tmp/recordings/%path/%Y-%m-%d_%H-%M-%S
    recordFormat: mp4
    recordPartDuration: 1h
    recordSegmentDuration: 1h
    recordDeleteAfter: 24h
    runOnInit: ""
    runOnInitRestart: no
    runOnDemand: ""
    runOnDemandRestart: no
    runOnDemandStartTimeout: 10s
    runOnDemandCloseAfter: 10s
    runOnReady: ""
    runOnReadyRestart: no
    runOnRead: ""
    runOnReadRestart: no
EOF

print_success "Created MediaMTX configuration"

# Create startup script
print_status "Creating startup script..."
cat > start_camera_streaming.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Smile Mini PC Camera Streaming"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker-compose build

# Start the services
echo "🚀 Starting services..."
docker-compose up -d

echo "✅ Services started!"
echo ""
echo "📱 Access the camera management interface at:"
echo "   http://localhost:8765/camera_management.html"
echo ""
echo "📺 Access the main camera streams at:"
echo "   http://localhost:8765"
echo ""
echo "🔧 MediaMTX API (if needed):"
echo "   http://localhost:9997"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f"
echo ""
echo "⏹️  To stop services:"
echo "   docker-compose down"
EOF

chmod +x start_camera_streaming.sh
print_success "Created start_camera_streaming.sh script"

# Create stop script
cat > stop_camera_streaming.sh << 'EOF'
#!/bin/bash
echo "⏹️  Stopping Smile Mini PC Camera Streaming"
echo "==========================================="

docker-compose down

echo "✅ Services stopped!"
EOF

chmod +x stop_camera_streaming.sh
print_success "Created stop_camera_streaming.sh script"

# Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "📋 Next steps:"
echo "1. Make sure your web cameras are connected"
echo "2. Run './test_cameras.sh' to test camera detection"
echo "3. Run './start_camera_streaming.sh' to start the services"
echo "4. Open http://localhost:8765/camera_management.html to manage cameras"
echo ""
echo "🔧 Troubleshooting:"
echo "- If cameras are not detected, check USB connections"
echo "- If permission errors occur, make sure you're in the 'video' group"
echo "- Check logs with 'docker-compose logs -f'"
echo ""
echo "📚 Available scripts:"
echo "- test_cameras.sh: Test camera detection"
echo "- start_camera_streaming.sh: Start all services"
echo "- stop_camera_streaming.sh: Stop all services"
echo ""
print_success "Setup completed! 🎥✨"

