import logging
import time
import numpy as np
from typing import List

from src.config.settings import AppConfig

logger = logging.getLogger(__name__)

class UnknownPersonManager:
    """Manages unknown face detection and quality assessment"""
    
    def __init__(self, config: AppConfig):
        self.unknown_faces_buffer = {}  # face_id -> list of quality assessments
        self.capture_attempts = {}      # face_id -> attempt count
        self.unknown_face_images = {}   # face_id -> list of face images
        self.face_first_seen = {}  # face_id -> timestamp
        self.recognition_grace_period = 1.5  # Seconds to wait before processing unknown faces
        
        # ENHANCED: Duplicate prevention
        self.processing_faces = set()  # Currently being processed
        self.recently_saved = {}       # face_id -> timestamp of save
        self.save_cooldown = 300       # 5 minutes cooldown between saves
        
        # Configuration
        self.max_attempts = getattr(config.detection, 'UNKNOWN_FACE_MAX_ATTEMPTS', 5)
        self.min_good_captures = getattr(config.detection, 'UNKNOWN_FACE_MIN_CAPTURES', 2)
        self.min_quality_score = getattr(config.detection, 'UNKNOWN_FACE_MIN_QUALITY', 0.65)
        self.require_frontal = getattr(config.detection, 'UNKNOWN_FACE_REQUIRE_FRONTAL', True)
        self.saved_faces = set()  
        
        logger.info(f"UnknownPersonManager initialized with enhanced duplicate prevention: "
                f"max_attempts={self.max_attempts}, min_captures={self.min_good_captures}, "
                f"min_quality={self.min_quality_score}, grace_period={self.recognition_grace_period}s, "
                f"save_cooldown={self.save_cooldown}s")
    
    
    def get_face_recognition_status(self, face_id: str) -> dict:
        """Debug method to check face recognition status"""
        cached_data = self.face_cache.get(face_id)
        if not cached_data:
            return {"error": "Face not found in cache"}
        
        return {
            "face_id": face_id,
            "is_recognized": cached_data.is_recognized,
            "human_name": cached_data.human_name,
            "human_guid": cached_data.human_guid,
            "recognition_confidence": cached_data.recognition_confidence,
            "last_update": cached_data.last_update,
            "update_count": cached_data.update_count,
            "in_saved_faces": face_id in self.unknown_person_manager.saved_faces
        }

    def should_capture_unknown_face(self, face_id: str, quality_assessment: dict, is_currently_recognized: bool = False) -> bool:
        """Enhanced duplicate prevention for unknown face capture"""
        current_time = time.time()
        
        logger.info(f"📋 CHECKING should_capture for {face_id}: recognized={is_currently_recognized}")
        
        # ENHANCED: Multiple duplicate checks
        if is_currently_recognized:
            logger.debug(f"❌ Skipping {face_id} - currently recognized")
            return False
        
        # Check if already saved
        if face_id in self.saved_faces:
            logger.info(f"❌ Skipping {face_id} - already saved to database")
            return False
            
        # ENHANCED: Check recent save cooldown
        if face_id in self.recently_saved:
            time_since_save = current_time - self.recently_saved[face_id]
            if time_since_save < self.save_cooldown:
                logger.info(f"❌ Skipping {face_id} - recent save cooldown ({time_since_save:.1f}s)")
                return False
        
        # ENHANCED: Check if currently being processed
        if face_id in self.processing_faces:
            logger.info(f"❌ Skipping {face_id} - currently being processed")
            return False

        # Track when this face was first seen
        if face_id not in self.face_first_seen:
            self.face_first_seen[face_id] = current_time
            logger.info(f"🆕 First time seeing {face_id}")
        
        # Give recognition system time to work
        time_since_first_seen = current_time - self.face_first_seen[face_id]
        logger.info(f"⏰ {face_id} time since first seen: {time_since_first_seen:.1f}s (grace: {self.recognition_grace_period}s)")
        
        if time_since_first_seen < self.recognition_grace_period:
            logger.info(f"⏳ {face_id} still in grace period")
            return False
        
        logger.info(f"✅ {face_id} passed grace period, proceeding with capture checks")
        
        # Initialize tracking for new face
        if face_id not in self.unknown_faces_buffer:
            self.unknown_faces_buffer[face_id] = []
            self.capture_attempts[face_id] = 0
            self.unknown_face_images[face_id] = []
            logger.info(f"🆕 Initialized tracking for {face_id}")
        
        # Check limits
        if self.capture_attempts[face_id] >= self.max_attempts:
            logger.info(f"❌ {face_id} exceeded max attempts ({self.capture_attempts[face_id]}/{self.max_attempts})")
            return False
        
        good_captures = len([q for q in self.unknown_faces_buffer[face_id] if q['is_good_quality']])
        if good_captures >= self.min_good_captures:
            logger.info(f"❌ {face_id} has enough good captures ({good_captures}/{self.min_good_captures})")
            return False
        
        # Check pose requirement
        pose_info = quality_assessment.get('pose_info', {})
        if self.require_frontal and not pose_info.get('is_frontal', False):
            logger.info(f"❌ {face_id} not frontal pose")
            return False
        
        # Increment attempt counter
        self.capture_attempts[face_id] += 1
        self.unknown_faces_buffer[face_id].append(quality_assessment)
        
        # Check if this capture has good quality
        is_good_quality = quality_assessment.get('is_good_quality', False)
        logger.info(f"✅ {face_id} should capture: good_quality={is_good_quality}, attempt={self.capture_attempts[face_id]}")
        
        return is_good_quality    

    def mark_processing_start(self, face_id: str):
        """Mark face as being processed"""
        self.processing_faces.add(face_id)
        logger.debug(f"Marked {face_id} as being processed")

    def mark_processing_end(self, face_id: str, success: bool = False):
        """Mark face processing as completed"""
        self.processing_faces.discard(face_id)
        if success:
            self.saved_faces.add(face_id)
            self.recently_saved[face_id] = time.time()
            logger.info(f"Marked {face_id} as successfully saved")
        else:
            logger.debug(f"Marked {face_id} processing as completed (failed)")
            
    def add_face_image(self, face_id: str, face_image: np.ndarray, quality_assessment: dict):
        """Store a high-quality face image"""
        if face_id in self.unknown_face_images:
            self.unknown_face_images[face_id].append({
                'image': face_image.copy(),
                'quality': quality_assessment,
                'timestamp': time.time()
            })
            logger.debug(f"Stored face image for {face_id} "
                        f"(quality: {quality_assessment['quality_score']:.2f})")
    
    # def is_ready_for_database(self, face_id: str) -> bool:
    #     """Check if we have enough good quality captures to save to database"""
    #     if face_id not in self.unknown_faces_buffer:
    #         return False
        
    #     good_captures = len([q for q in self.unknown_faces_buffer[face_id] if q['is_good_quality']])
    #     return good_captures >= self.min_good_captures
    
    def is_ready_for_database(self, face_id: str) -> bool:
        """Check if we have enough good quality captures to save to database"""
        if face_id in self.saved_faces:
            return False  # Already saved this face
            
        if face_id not in self.unknown_faces_buffer:
            return False
    
        good_captures = len([q for q in self.unknown_faces_buffer[face_id] if q['is_good_quality']])
        return good_captures >= self.min_good_captures
    
    def get_best_images(self, face_id: str, count: int = 3) -> List[dict]:
        """Get the best quality images for a face"""
        if face_id not in self.unknown_face_images:
            return []
        
        # Sort by quality score
        images = self.unknown_face_images[face_id]
        sorted_images = sorted(images, key=lambda x: x['quality']['quality_score'], reverse=True)
        return sorted_images[:count]
    
    def cleanup_face(self, face_id: str):
        """Clean up data for a face that's been processed"""
        self.unknown_faces_buffer.pop(face_id, None)
        self.capture_attempts.pop(face_id, None)
        self.unknown_face_images.pop(face_id, None)
        logger.debug(f"Cleaned up unknown face data for {face_id}")
        
    def cleanup_old_first_seen(self):
        """Clean up old first_seen entries to prevent memory leaks"""
        current_time = time.time()
        to_remove = []
        
        for face_id, first_seen_time in self.face_first_seen.items():
            if current_time - first_seen_time > (self.recognition_grace_period + 30):  # Extra buffer
                to_remove.append(face_id)
        
        for face_id in to_remove:
            del self.face_first_seen[face_id]
            logger.debug(f"Cleaned up old first_seen entry for {face_id}")
