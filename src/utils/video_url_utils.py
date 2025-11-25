"""
Video URL Utilities

Utility functions for extracting video IDs and S3 keys from video URLs and upload responses.
"""

import re
from typing import Optional, Dict, Any

def extract_video_id_from_url(url: str) -> Optional[str]:
    """
    Extract video ID from database URL.
    
    Args:
        url: Complete video URL from database (e.g., https://cdn.u-code.io/...)
    
    Returns:
        Video ID (UUID string) or None if not found
        
    Example:
        >>> url = "https://cdn.u-code.io/efba2b71-f75f-482f-9b4e-6538961864b7/Media/c00ca252-ea4e-4931-adba-0b0449ce649e_Barra_Hotdog_client1756502296_upset_20250917_102428_stream_00.mp4"
        >>> extract_video_id_from_url(url)
        "c00ca252-ea4e-4931-adba-0b0449ce649e"
    """
    try:
        filename = url.split('/')[-1]
        video_id_match = re.match(r'^([a-f0-9-]+)_', filename)
        return video_id_match.group(1) if video_id_match else None
    except Exception:
        return None

def extract_video_id_from_upload_response(response_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract video ID from ucode-sdk upload response.
    
    Args:
        response_data: Upload response data from ucode-sdk
    
    Returns:
        Video ID (UUID string) or None if not found
        
    Example:
        >>> response = {"link": "f3752975-d396-4baa-a4fc-54c8cd5bd959/Media/ec597180-5b51-4ee0-9138-c23577488d37_7a22249b-9abe-444d-b847-61d40e9e9c12.FILE"}
        >>> extract_video_id_from_upload_response(response)
        "ec597180-5b51-4ee0-9138-c23577488d37"
    """
    try:
        link = response_data.get('link', '')
        filename = link.split('/')[-1]
        video_id_match = re.match(r'^([a-f0-9-]+)_', filename)
        return video_id_match.group(1) if video_id_match else None
    except Exception:
        return None

def get_s3_key_from_url(url: str) -> Optional[str]:
    """
    Get S3 key from database URL for direct S3 access.
    
    Args:
        url: Complete video URL from database
    
    Returns:
        S3 key path or None if invalid
        
    Example:
        >>> url = "https://cdn.u-code.io/efba2b71-f75f-482f-9b4e-6538961864b7/Media/c00ca252-ea4e-4931-adba-0b0449ce649e_Barra_Hotdog_client1756502296_upset_20250917_102428_stream_00.mp4"
        >>> get_s3_key_from_url(url)
        "efba2b71-f75f-482f-9b4e-6538961864b7/Media/c00ca252-ea4e-4931-adba-0b0449ce649e_Barra_Hotdog_client1756502296_upset_20250917_102428_stream_00.mp4"
    """
    try:
        # Remove https://cdn.u-code.io/ prefix
        s3_key = url.replace('https://cdn.u-code.io/', '')
        return s3_key
    except Exception:
        return None

def get_app_id_from_url(url: str) -> Optional[str]:
    """
    Extract app ID from video URL.
    
    Args:
        url: Complete video URL from database
    
    Returns:
        App ID (UUID string) or None if not found
    """
    try:
        parts = url.split('/')
        if len(parts) >= 5 and 'cdn.u-code.io' in url:
            return parts[4]  # App ID is the 5th part
        return None
    except Exception:
        return None

def parse_video_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse video filename to extract all metadata.
    
    Args:
        filename: Video filename (e.g., "c00ca252-ea4e-4931-adba-0b0449ce649e_Barra_Hotdog_client1756502296_upset_20250917_102428_stream_00.mp4")
    
    Returns:
        Dictionary with parsed metadata
        
    Example:
        >>> filename = "c00ca252-ea4e-4931-adba-0b0449ce649e_Barra_Hotdog_client1756502296_upset_20250917_102428_stream_00.mp4"
        >>> parse_video_filename(filename)
        {
            'video_id': 'c00ca252-ea4e-4931-adba-0b0449ce649e',
            'human_name': 'Barra_Hotdog',
            'client_id': '1756502296',
            'emotion': 'upset',
            'timestamp': '20250917_102428',
            'stream_num': '00'
        }
    """
    try:
        # Remove .mp4 extension
        name_without_ext = filename.replace('.mp4', '')
        
        # Pattern: {video_id}_{human_name}_client{client_id}_{emotion}_{timestamp}_stream_{stream_num}
        pattern = r'^([a-f0-9-]+)_(.+?)_client(\d+)_(\w+)_(\d{8}_\d{6})_stream_(\d+)$'
        match = re.match(pattern, name_without_ext)
        
        if match:
            return {
                'video_id': match.group(1),
                'human_name': match.group(2),
                'client_id': match.group(3),
                'emotion': match.group(4),
                'timestamp': match.group(5),
                'stream_num': match.group(6)
            }
        else:
            return {
                'video_id': None,
                'human_name': None,
                'client_id': None,
                'emotion': None,
                'timestamp': None,
                'stream_num': None
            }
    except Exception:
        return {
            'video_id': None,
            'human_name': None,
            'client_id': None,
            'emotion': None,
            'timestamp': None,
            'stream_num': None
        }

def get_video_metadata_from_url(url: str) -> Dict[str, Optional[str]]:
    """
    Get complete video metadata from database URL.
    
    Args:
        url: Complete video URL from database
    
    Returns:
        Dictionary with all metadata including app_id, video_id, etc.
    """
    try:
        filename = url.split('/')[-1]
        app_id = get_app_id_from_url(url)
        video_metadata = parse_video_filename(filename)
        
        return {
            'app_id': app_id,
            's3_key': get_s3_key_from_url(url),
            **video_metadata
        }
    except Exception:
        return {
            'app_id': None,
            's3_key': None,
            'video_id': None,
            'human_name': None,
            'client_id': None,
            'emotion': None,
            'timestamp': None,
            'stream_num': None
        }

# Example usage and testing
if __name__ == "__main__":
    # Test with real data
    test_url = "https://cdn.u-code.io/efba2b71-f75f-482f-9b4e-6538961864b7/Media/c00ca252-ea4e-4931-adba-0b0449ce649e_Barra_Hotdog_client1756502296_upset_20250917_102428_stream_00.mp4"
    
    print("🧪 Testing Video URL Utilities")
    print("=" * 50)
    
    print(f"Test URL: {test_url}")
    print(f"Video ID: {extract_video_id_from_url(test_url)}")
    print(f"S3 Key: {get_s3_key_from_url(test_url)}")
    print(f"App ID: {get_app_id_from_url(test_url)}")
    
    metadata = get_video_metadata_from_url(test_url)
    print(f"Complete Metadata: {metadata}")

