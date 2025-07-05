from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

@dataclass
class FaceRecognition:
    """Face recognition result"""
    guid: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None

@dataclass
class BoundingBox:
    """Represents a bounding box for detected faces"""
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    confidence: float = 0.0

@dataclass
class FaceDetection:
    """Face detection result"""
    bounding_box: BoundingBox
                # (top, right, bottom, left)
    face_location: Tuple[int, int, int, int]  

@dataclass
class EmotionResult:
    """Emotion detection result"""
    dominant_emotion: str = None
    emotion_probabilities: Dict[str, float] = None
    emotion_category: str = None
    starting_time: float = None
    ending_time: float = None
    percentage: float = None
    type: list = None

@dataclass
class ProcessedFace:
    """Complete face processing result"""
    detection: FaceDetection
    recognition: FaceRecognition
    emotion: EmotionResult
    
    @property
    def bounding_box(self) -> BoundingBox:
        return self.detection.bounding_box
    
    @property
    def name(self) -> str:
        return self.recognition.name
    
    @property
    def dominant_emotion(self) -> str:
        return self.emotion.dominant_emotion


@dataclass
class FaceDetection:
    """Represents a detected face with recognition info"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Optional[List[tuple]] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    face_id: Optional[str] = None
    recognition_confidence: Optional[float] = None

    human_guid: Optional[str] = None
    human_name: Optional[str] = None
    human_type: Optional[str] = None
    is_recognized: bool = False
    face_embedding: Optional[List[float]] = None
    pose_info: Optional[dict] = None  # ADD THIS LINE
    
    

@dataclass
class DetectionResult:
    """Result of face detection processing"""
    stream_id: str
    timestamp: float
    frame_number: int
    faces: List[FaceDetection]
    processing_time: float    

@dataclass
class CachedFaceData:
    """Cached face recognition and emotion data"""
    human_guid: Optional[str] = None
    human_name: Optional[str] = None
    human_type: Optional[str] = None
    recognition_confidence: float = 0.0
    is_recognized: bool = False
    emotion: str = "neutral"
    emotion_confidence: float = 0.5
    last_update: float = 0.0
    update_count: int = 0
    pose_info: Optional[dict] = None  
