#!/usr/bin/env python3
"""
Minimal upload test - just print raw response
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.video_storage import VideoRecord
from src.services.video_upload_service import VideoUploadService

class TestConfig:
    def __init__(self):
        self.ucode_app_id = "P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL"
        self.ucode_base_url = "https://api.client.u-code.io"

# Disable all logging
import logging
logging.disable(logging.CRITICAL)

def test_upload():
    # Use available video
    test_video = "temp_videos/Javohir_normal_20250828_181216_dev_00.mp4"
    
    # Create service
    config = TestConfig()
    upload_service = VideoUploadService(config)
    
    # Create video record
    video_record = VideoRecord(
        file_path=test_video,
        human_name="TestUser",
        emotion_type="test",
        camera_id="test_camera"
    )
    
    # Upload and get response
    result = upload_service.sdk.files().upload(video_record.file_path).exec()
    
    # Print raw response
    if result and len(result) >= 1:
        create_response = result[0]
        print("RAW UPLOAD RESPONSE:")
        print("=" * 50)
        print(f"Status: {create_response.status}")
        print(f"Description: {create_response.description}")
        print(f"Data: {create_response.data}")
        print(f"Custom Message: {create_response.custom_message}")
        print("=" * 50)
        
        # Show constructed URL
        if create_response.data and create_response.data.get('link'):
            file_url = f"https://cdn.u-code.io/{create_response.data['link']}"
            print(f"Final URL: {file_url}")
            
            # Extract video ID
            filename = create_response.data['link'].split('/')[-1]
            import re
            video_id_match = re.match(r'^([a-f0-9-]+)_', filename)
            if video_id_match:
                print(f"Video ID: {video_id_match.group(1)}")
    else:
        print("No response received")

if __name__ == "__main__":
    test_upload()
