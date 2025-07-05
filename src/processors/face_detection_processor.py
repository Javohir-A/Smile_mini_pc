# src/processors/face_detection_processor.py - OPTIMIZED FOR REAL-TIME STREAMING
import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .emotion_recognizer import DeepFaceEmotionRecognizer
from .face_recognizer import FaceRecognizer
from ucode_sdk import new, Config

from src.config import AppConfig
from src.di.dependencies import DependencyContainer
from src.models.stream import FrameData
from src.models.detection import * #import everything in this modul FaceDtection, DetectionResult
from .unknown_face_processor import UnknownPersonManager

logger = logging.getLogger(__name__)


class FaceDetectionProcessor:
    """Enhanced face detection processor optimized for real-time streaming"""
    
    def __init__(self, config: AppConfig, container: DependencyContainer):
        self.config = config
        self.container = container
        self.net = None
        self.emotion_recognizer = None
        self.face_recognizer = None
        self.face_usecase = None
        self.detection_callbacks: List[callable] = []
        self.face_tracker = {}  # For tracking faces across frames
        self.next_face_id = 1
        self._lock = threading.Lock()
        self.min_face_size: Optional[int] = 80

        self.emotion_cache_clear_every = 30  # Clear every 30 frames (1 second)
        self.ucode_api = new(config=Config(app_id=config.ucode.app_id, base_url=config.ucode.base_url))
        
        self.unknown_person_manager = UnknownPersonManager(config)
        
        # Store detection results for each stream
        self.stream_detections: Dict[str, DetectionResult] = {}
        self.detection_lock = threading.Lock()
        
        # ULTRA OPTIMIZATION: Aggressive caching and processing intervals
        self.face_cache: Dict[str, CachedFaceData] = {}  # Cache all face data
        self.cache_timeout = 0.5  # Keep cache for 3 seconds
        
        # Recognition cache
        self.recognition_cache = {}
        self.recognition_cache_timeout = 0.2  # Cache for 0.2 seconds
        
        # Processing intervals to reduce load
        self.emotion_process_every = 3  # Process emotion every 3 frames
        self.recognition_process_every = 5  # Process face recognition every 5 frames
        self.frame_counter = 0
        
        self.recognition_in_progress = set()  # Track faces being processed for recognition
        self.unknown_processing_lock = threading.Lock()  # Lock for unknown person processing
        self.recognition_lock = threading.Lock()  # Lock for recognition updates
        
        # Performance optimization
        self.background_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face_detection")

        self.database_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="database_save")  # ADD THIS
        self.pending_background_tasks = set()
        
        
        # PERFORMANCE: Reduce logging frequency
        self.log_counter = 0
        self.log_every_n_frames = 100  # Log detailed info only every 30 frames
        self.pose_process_every = 10
        self.pose_cache_ttl = 1.0

        # Initialize all components
        self._initialize_models()
        logger.info("Enhanced FaceDetectionProcessor with recognition initialized")
    
    def _initialize_models(self):
        """Initialize face detection, emotion recognition, and face recognition models"""
        try:
            # Load DNN face detection model
            if not self._load_face_detection_model():
                logger.error("Failed to load face detection model")
                return
            
            # Initialize DeepFace emotion recognizer
            try:
                self.emotion_recognizer = DeepFaceEmotionRecognizer(self.config)
                self.emotion_recognizer.adjust_sensitivity(more_sensitive=False)
                
                self.emotion_recognizer.cache_timeout = 2.0
                
                logger.info("Emotion recognizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize emotion recognizer: {e}")
                self.emotion_recognizer = None
            
            # Initialize face recognizer
            try:
                self.face_recognizer = FaceRecognizer(self.config)
                self.face_recognizer.cache_timeout = 3.0
                
                logger.info("Face recognizer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize face recognizer: {e}")
                self.face_recognizer = None
            
            # Get face usecase for recognition
            self._initialize_face_usecase()
            
            logger.info("All face detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Critical error initializing face detection models: {e}")
    
    class ConservativeRecognitionSettings:
        """Conservative settings to prevent false positives"""
        
        # Primary threshold
        L2_THRESHOLD = 1.8          # Stricter than 2.2
        
        # Validation thresholds  
        MAX_DISTANCE = 1.8          # Must be very close match
        MIN_CONFIDENCE_GAP = 0.3    # Large gap between best and second
        MIN_RECOGNITION_CONF = 0.65 # 65% minimum confidence
        MAX_DISTANCE_RATIO = 0.75   # Best must be 25% better than second
        
        # For different quality faces
        HIGH_QUALITY_THRESHOLD = 1.5    # Frontal, good lighting
        NORMAL_THRESHOLD = 1.8          # Regular conditions  
        LOW_QUALITY_THRESHOLD = 2.0     # Poor lighting, angle
    
    # 2. ADD THESE METHODS AFTER YOUR EXISTING HELPER METHODS:
    
    def get_adaptive_threshold(self, face_id: str) -> float:
        """Get adaptive threshold based on face quality"""
        
        cached_data = self.face_cache.get(face_id)
        if not cached_data or not hasattr(cached_data, 'pose_info'):
            return self.ConservativeRecognitionSettings.NORMAL_THRESHOLD
        
        pose_info = cached_data.pose_info
        pose_quality = pose_info.get('pose_quality', 0.5)
        is_frontal = pose_info.get('is_frontal', False)
        
        if is_frontal and pose_quality > 0.8:
            return self.ConservativeRecognitionSettings.HIGH_QUALITY_THRESHOLD
        elif pose_quality < 0.3:
            return self.ConservativeRecognitionSettings.LOW_QUALITY_THRESHOLD
        else:
            return self.ConservativeRecognitionSettings.NORMAL_THRESHOLD

    def _validate_face_match(self, best_match, all_results, face_id: str) -> bool:
        """FIXED: Multi-layer validation that handles excellent matches properly"""
        
        logger.debug(f"🔍 Validating match for {face_id}: {best_match.name} (L2: {best_match.distance:.3f})")
        
        # VALIDATION 1: Distance threshold - but different thresholds for different quality
        if best_match.distance <= 0.15:
            # EXCELLENT MATCH: Very close, almost certainly correct
            logger.debug(f"✅ EXCELLENT MATCH: Distance {best_match.distance:.3f} - skipping other validations")
            return True
        elif best_match.distance > 1.8:
            # TOO FAR: Definitely reject
            logger.debug(f"❌ Validation failed: Distance {best_match.distance:.3f} > 1.8 (too far)")
            return False
        
        # VALIDATION 2: For good matches (0.15 < distance <= 1.8), check confidence gap
        if len(all_results) > 1:
            second_match = all_results[1]
            confidence_gap = second_match.distance - best_match.distance
            
            # ADAPTIVE GAP REQUIREMENT: Better matches need smaller gaps
            if best_match.distance <= 0.5:
                min_gap = 0.1  # Very good matches only need small gap
            elif best_match.distance <= 1.0:
                min_gap = 0.2  # Good matches need moderate gap
            else:
                min_gap = 0.3  # Average matches need large gap
            
            if confidence_gap < min_gap:
                logger.debug(f"❌ Validation failed: Confidence gap {confidence_gap:.3f} < {min_gap:.3f}")
                logger.debug(f"  Best: {best_match.name} ({best_match.distance:.3f})")
                logger.debug(f"  Second: {second_match.name} ({second_match.distance:.3f})")
                return False
            else:
                logger.debug(f"✅ Confidence gap OK: {confidence_gap:.3f} >= {min_gap:.3f}")
        
        # VALIDATION 3: Minimum recognition confidence
        recognition_conf = self._calculate_recognition_confidence(best_match.distance)
        min_confidence = 0.60  # LOWERED: Was 0.65, now 0.60
        
        if recognition_conf < min_confidence:
            logger.debug(f"❌ Validation failed: Recognition confidence {recognition_conf:.3f} < {min_confidence}")
            return False
        
        # VALIDATION 4: Distance ratio check (only for marginal matches)
        if best_match.distance > 0.5 and len(all_results) > 1:
            second_match = all_results[1]
            distance_ratio = best_match.distance / second_match.distance
            max_ratio = 0.80  # RELAXED: Was 0.75, now 0.80
            
            if distance_ratio > max_ratio:
                logger.debug(f"❌ Validation failed: Distance ratio {distance_ratio:.3f} > {max_ratio}")
                return False
        
        logger.debug(f"✅ All validations passed for {best_match.name}")
        return True

    def _calculate_recognition_confidence(self, l2_distance: float) -> float:
        """Calculate recognition confidence from L2 distance"""
        # L2 distance to confidence mapping
        # 0.0 distance = 100% confidence
        # 1.0 distance = ~75% confidence  
        # 2.0 distance = ~50% confidence
        # 3.0+ distance = <25% confidence
        
        if l2_distance <= 0.5:
            return 1.0  # 100% confidence for very close matches
        elif l2_distance <= 1.0:
            return 0.95 - (l2_distance - 0.5) * 0.4  # 95% to 75%
        elif l2_distance <= 2.0:
            return 0.75 - (l2_distance - 1.0) * 0.25  # 75% to 50%
        else:
            return max(0.0, 0.5 - (l2_distance - 2.0) * 0.2)  # 50% to 0%

    def debug_current_recognition(self, face_roi: np.ndarray, face_id: str) -> dict:
        """Debug current recognition to understand why wrong person is matched"""
        
        if not self.face_recognizer or not self.face_usecase:
            return {"error": "Recognition not available"}
        
        face_embedding = self.face_recognizer.extract_face_embedding(face_roi, face_id)
        if not face_embedding:
            return {"error": "Could not extract embedding"}
        
        # Test with different thresholds
        results = {}
        thresholds = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
        
        for threshold in thresholds:
            try:
                search_results = self.face_usecase.search_similar_faces(
                    face_embedding, limit=5, threshold=threshold
                )
                
                threshold_results = []
                for i, match in enumerate(search_results[:3]):
                    confidence = self._calculate_recognition_confidence(match.distance)
                    is_valid = self._validate_face_match(match, search_results, face_id)
                    
                    threshold_results.append({
                        "rank": i + 1,
                        "name": match.name,
                        "distance": round(match.distance, 3),
                        "confidence": round(confidence, 3),
                        "is_valid": is_valid,
                        "human_guid": match.human_guid
                    })
                
                results[f"threshold_{threshold}"] = threshold_results
                
            except Exception as e:
                results[f"threshold_{threshold}"] = {"error": str(e)}
        
        return results
    
    def _debug_validation_failure(self, best_match, all_results, face_id: str):
        """Debug why validation is failing"""
        
        logger.info(f"🔍 DEBUGGING VALIDATION FAILURE for {face_id}:")
        logger.info(f"  Best match: {best_match.name}")
        logger.info(f"  L2 distance: {best_match.distance:.3f}")
        logger.info(f"  Recognition confidence: {self._calculate_recognition_confidence(best_match.distance):.3f}")
        
        if len(all_results) > 1:
            second_match = all_results[1]
            gap = second_match.distance - best_match.distance
            logger.info(f"  Second match: {second_match.name} (L2: {second_match.distance:.3f})")
            logger.info(f"  Confidence gap: {gap:.3f}")
            
            if best_match.distance <= 0.5:
                required_gap = 0.1
            elif best_match.distance <= 1.0:
                required_gap = 0.2
            else:
                required_gap = 0.3
            
            logger.info(f"  Required gap: {required_gap:.3f}")
            logger.info(f"  Gap sufficient: {gap >= required_gap}")
        
        # Test different thresholds
        logger.info(f"  Would pass with different criteria:")
        logger.info(f"    Distance <= 0.15: {best_match.distance <= 0.15}")
        logger.info(f"    Distance <= 0.20: {best_match.distance <= 0.20}")
        logger.info(f"    Distance <= 0.30: {best_match.distance <= 0.30}")
        logger.info(f"    Distance <= 1.00: {best_match.distance <= 1.00}")

    
    def _initialize_face_usecase(self):
        """Initialize face usecase with proper error handling and validation"""
        try:
            logger.info("Attempting to get face usecase from dependency container...")
            self.face_usecase = self.container.get_face_usecase()
            
            if self.face_usecase is None:
                logger.warning("Face usecase is None from container")
                raise Exception("Face usecase returned None from container")
            
            # Quick test without detailed logging
            test_embedding = [0.0] * 128
            try:
                test_results = self.face_usecase.search_similar_faces(test_embedding, limit=1, threshold=0.9)
                logger.info(f"Face usecase test successful")
            except Exception as test_e:
                logger.warning(f"Face usecase test failed (may be normal if database is empty)")
            
            logger.info("Face usecase initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not get face usecase from container: {e}")

        # Final validation
        if self.face_usecase is None:
            logger.error("CRITICAL: Face usecase could not be initialized - face recognition will be disabled")
        else:
            logger.info("Face usecase is ready for face recognition")
    
    def _load_face_detection_model(self) -> bool:
        """Load the DNN face detection model with optimized settings"""
        try:
            prototxt_path = Path(self.config.detection.model_prototxt)
            model_path = Path(self.config.detection.model_weights)
            
            if not prototxt_path.is_file():
                raise FileNotFoundError(f"Missing prototxt file: {prototxt_path}")
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing model file: {model_path}")
            
            logger.info(f"Loading DNN model from {prototxt_path} and {model_path}")
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
            
            # Optimize DNN backend if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using CUDA backend for face detection")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("Using CPU backend for face detection")
            
            # Lower confidence threshold for better angle detection
            self.min_confidence = getattr(self.config.detection, 'confidence_threshold', 0.4)
            
            logger.info(f"DNN face detection model loaded successfully (confidence threshold: {self.min_confidence})")
            return True
            
        except Exception as e:
            logger.error(f"DNN model loading failed: {e}")
            return False
    
    def add_detection_callback(self, callback: callable):
        """Add a callback function to be called when faces are detected"""
        self.detection_callbacks.append(callback)
        logger.info(f"Added detection callback: {callback.__name__}")


    def _get_cached_face_data(self, face_id: str) -> CachedFaceData:
        """Get cached face data or create new cache entry"""
        current_time = time.time()
        
        if face_id in self.face_cache:
            cached_data = self.face_cache[face_id]
            # Check if cache is still valid
            if current_time - cached_data.last_update < self.cache_timeout:
                # Ensure pose_info exists (for backward compatibility)
                if not hasattr(cached_data, 'pose_info') or cached_data.pose_info is None:
                    cached_data.pose_info = self._get_default_pose_info()
                return cached_data
        
        # Create new cache entry with pose_info
        new_cache_data = CachedFaceData(
            last_update=current_time,
            pose_info=self._get_default_pose_info()
        )
        self.face_cache[face_id] = new_cache_data
        return new_cache_data       

# PRECISE THRESHOLD FIX: Stop false positive matches

    def _update_face_cache_async(self, face_id: str, face_roi_expanded: np.ndarray, 
                        face_roi: np.ndarray, should_process_pose: bool = False):
        """UPDATED: Better validation logic"""
        def background_update():
            try:
                with self.recognition_lock:
                    if face_id in self.recognition_in_progress:
                        return
                    self.recognition_in_progress.add(face_id)
                
                current_time = time.time()
                cached_data = self.face_cache.get(face_id)
                if not cached_data:
                    return

                if not hasattr(cached_data, 'pose_info') or cached_data.pose_info is None:
                    cached_data.pose_info = self._get_default_pose_info()

                # Recognition processing
                if (self.face_recognizer and self.face_recognizer.available and self.face_usecase):
                    should_check_recognition = (
                        not cached_data.is_recognized or
                        (current_time - cached_data.last_update) > 8.0
                    )
                    
                    if should_check_recognition:
                        logger.debug(f"Checking database for face {face_id}")
                        
                        face_embedding = self.face_recognizer.extract_face_embedding(face_roi_expanded, face_id)
                        
                        if face_embedding:
                            search_results = self.face_usecase.search_similar_faces(
                                face_embedding, limit=5, threshold=2.0  # SLIGHTLY INCREASED from 1.8
                            )
                            
                            if search_results and len(search_results) > 0:
                                # Log all matches
                                logger.info(f"🔍 Recognition analysis for {face_id}:")
                                for i, match in enumerate(search_results[:3]):
                                    confidence = self._calculate_recognition_confidence(match.distance)
                                    logger.info(f"  {i+1}. {match.name} - L2: {match.distance:.3f}, conf: {confidence:.3f}")
                                
                                best_match = search_results[0]
                                
                                # USE SIMPLER VALIDATION
                                is_valid_match = self._validate_face_match(best_match, search_results, face_id)                                
                                
                                if is_valid_match:
                                    with self.recognition_lock:
                                        cached_data.human_guid = best_match.human_guid
                                        cached_data.human_name = best_match.name
                                        cached_data.human_type = best_match.human_type
                                        cached_data.recognition_confidence = self._calculate_recognition_confidence(best_match.distance)
                                        cached_data.is_recognized = True
                                    
                                    logger.info(f"✅ RECOGNIZED: Face {face_id} as '{best_match.name}' "
                                            f"(L2: {best_match.distance:.3f}, conf: {cached_data.recognition_confidence:.1%})")
                                    
                                    with self.unknown_processing_lock:
                                        if face_id in self.unknown_person_manager.unknown_faces_buffer:
                                            self.unknown_person_manager.cleanup_face(face_id)
                                        self.unknown_person_manager.saved_faces.add(face_id)
                                else:
                                    logger.info(f"❌ REJECTED MATCH: Face {face_id} -> '{best_match.name}' "
                                            f"(L2: {best_match.distance:.3f}) - failed validation")
                                    
                                    # ADD DEBUGGING FOR FAILURES
                                    self._debug_validation_failure(best_match, search_results, face_id)
                            else:
                                logger.info(f"🔍 NO MATCHES: Face {face_id} - processing as unknown")

                # Emotion processing (unchanged)
                if self.emotion_recognizer and self.emotion_recognizer.available:
                    should_update_emotion = True
                    
                    if should_update_emotion:
                        try:
                            emotion_result = self.emotion_recognizer.predict_emotion(face_roi, None)
                            
                            if emotion_result and len(emotion_result) >= 2:
                                emotion, emotion_conf = emotion_result[0], emotion_result[1]
                                
                                if emotion and emotion_conf > 0.1:
                                    old_emotion = cached_data.emotion
                                    cached_data.emotion = emotion
                                    cached_data.emotion_confidence = emotion_conf
                                    
                                    if old_emotion != emotion:
                                        logger.info(f"🎭 EMOTION CHANGED for {face_id}: {old_emotion} -> {emotion} ({emotion_conf:.2f})")
                            
                        except Exception as e:
                            logger.error(f"🎭 Emotion processing error for {face_id}: {e}")

                cached_data.last_update = current_time
                cached_data.update_count += 1

            except Exception as e:
                logger.error(f"Background face update error for {face_id}: {e}")
            finally:
                with self.recognition_lock:
                    self.recognition_in_progress.discard(face_id)
                self.pending_background_tasks.discard(face_id)

        if face_id not in self.pending_background_tasks:
            self.pending_background_tasks.add(face_id)
            self.background_executor.submit(background_update)


    def _extract_face_regions(self, frame: np.ndarray, detection: np.ndarray, w: int, h: int) -> tuple:
        """Extract face regions from detection - FAST"""
        x1 = max(0, int(detection[3] * w))
        y1 = max(0, int(detection[4] * h))
        x2 = min(w, int(detection[5] * w))
        y2 = min(h, int(detection[6] * h))
        
        width = x2 - x1
        height = y2 - y1
        
        # Skip tiny faces
        if width < 30 or height < 30:
            return None, None, None, None, None, None
        
        # Calculate expanded ROI
        expand_ratio = 0.2
        expand_x = int(width * expand_ratio / 2)
        expand_y = int(height * expand_ratio / 2)
        
        ex1 = max(0, x1 - expand_x)
        ey1 = max(0, y1 - expand_y)
        ex2 = min(w, x2 + expand_x)
        ey2 = min(h, y2 + expand_y)
        
        face_roi_expanded = frame[ey1:ey2, ex1:ex2]
        face_roi = frame[y1:y2, x1:x2]
        
        return x1, y1, width, height, face_roi_expanded, face_roi

    def _should_process_expensive_operations(self) -> tuple:
        """FIXED: Process emotions more frequently for dynamic updates"""
        should_process_recognition = (self.frame_counter % self.recognition_process_every == 0)
        should_process_emotion = (self.frame_counter % 2 == 0)  # FIXED: Every 2 frames instead of 3
        should_process_pose = (self.frame_counter % getattr(self, 'pose_process_every', 10) == 0)
        
        return should_process_recognition, should_process_emotion, should_process_pose

    def _process_single_detection(self, frame: np.ndarray, detection: np.ndarray, 
                                w: int, h: int, should_process_recognition: bool,
                                should_process_emotion: bool, should_process_pose: bool) -> Optional[FaceDetection]:
        """Process a single face detection - OPTIMIZED WITH BETTER UNKNOWN FACE PROCESSING"""
        confidence = detection[2]
        
        if confidence <= self.min_confidence:
            return None
        
        # Extract face regions
        result = self._extract_face_regions(frame, detection, w, h)
        if result[0] is None:  # Skip tiny faces
            return None
        
        x1, y1, width, height, face_roi_expanded, face_roi = result
        
        # Get face ID and cached data
        face_id = self._assign_face_id(x1, y1, width, height)
        cached_data = self._get_cached_face_data(face_id)
        
        # Background processing for expensive operations
        if (should_process_recognition or should_process_emotion or should_process_pose) and face_roi.size > 0:
            self._update_face_cache_async(face_id, face_roi_expanded, face_roi, 
                                        should_process_pose)
        
        # Create face detection object
        face_detection = self._create_face_detection_object(x1, y1, width, height, confidence, face_id, cached_data)
        
        # IMPROVED: Process unknown faces more frequently (every 3 frames instead of every 10)
        should_process_unknown = (self.frame_counter % 3 == 0)
        
        logger.debug(f"🔍 DEBUG: face_id={face_id}, is_recognized={face_detection.is_recognized}, frame_counter={self.frame_counter}, should_process_unknown={should_process_unknown}")
        
        if not face_detection.is_recognized and should_process_unknown:
            logger.info(f"🚀 SUBMITTING unknown face processing for {face_id}")  # ADD THIS
            try:
                self.background_executor.submit(
                    self._process_unknown_face, face_roi, face_detection
                )
            except Exception as e:
                logger.error(f"❌ Error submitting unknown face processing: {e}")  # ADD THIS
        
        return face_detection


    def process_frame(self, frame_data: FrameData) -> Optional[DetectionResult]:
        """FIXED: Filter out duplicate faces and force emotion updates"""
        if not self.net:
            return None
        
        start_time = time.time()
        self.frame_counter += 1
        self.log_counter += 1
        
        if self.frame_counter % self.emotion_cache_clear_every == 0:
            if hasattr(self.emotion_recognizer, 'clear_emotion_cache'):
                self.emotion_recognizer.clear_emotion_cache()
                logger.debug("🎭 Cleared emotion cache for dynamic updates")
        try:
            frame = frame_data.frame
            h, w = frame.shape[:2]
            
            # Face detection
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Determine processing strategy
            should_process_recognition, should_process_emotion, should_process_pose = self._should_process_expensive_operations()
            should_log = (self.log_counter % self.log_every_n_frames == 0)
            
            # CRITICAL FIX: Pre-filter overlapping detections
            filtered_detections = self._filter_overlapping_detections(detections[0, 0, :, :])
            
            # Process filtered detections
            face_detections = []
            for detection in filtered_detections:
                face_detection = self._process_single_detection(
                    frame, detection, w, h, 
                    should_process_recognition, should_process_emotion, should_process_pose
                )
                if face_detection:
                    face_detections.append(face_detection)
            
            # CRITICAL FIX: Remove duplicate faces by name/position
            unique_face_detections = self._remove_duplicate_faces(face_detections)
            
            # Create and store result
            result = self._create_detection_result(frame_data, unique_face_detections, start_time)
            self._store_and_notify(result, frame_data, unique_face_detections, should_log)
            
            return result
            
        except Exception as e:
            if self.log_counter % 200 == 0:
                logger.error(f"Frame processing error: {e}")
            return None

    def _create_detection_result(self, frame_data: FrameData, face_detections: list, start_time: float) -> DetectionResult:
        """Create DetectionResult object"""
        processing_time = time.time() - start_time
        return DetectionResult(
            stream_id=frame_data.stream_id,
            timestamp=frame_data.timestamp,
            frame_number=frame_data.frame_number,
            faces=face_detections,
            processing_time=processing_time
        )

    def _store_and_notify(self, result: DetectionResult, frame_data: FrameData, 
                        face_detections: list, should_log: bool):
        """Store results and notify callbacks"""
        # Store detection results
        with self.detection_lock:
            self.stream_detections[frame_data.stream_id] = result
        
        # Background callbacks
        if face_detections:
            for callback in self.detection_callbacks:
                try:
                    self.background_executor.submit(callback, result, frame_data)
                except Exception:
                    pass
        
        # Minimal logging
        if face_detections and should_log:
            recognized_count = len([f for f in face_detections if f.is_recognized])
            logger.info(f"Stream {frame_data.stream_id}: {len(face_detections)} faces "
                    f"({recognized_count} recognized) in {result.processing_time:.3f}s")    

    def get_stream_detections(self, stream_id: str) -> Optional[DetectionResult]:
        """Get the latest detection results for a stream"""
        with self.detection_lock:
            return self.stream_detections.get(stream_id)

    def _assign_face_id(self, x: int, y: int, width: int, height: int) -> str:
        """FIXED: Better face tracking to prevent multiple boxes for same person"""
        center_x = x + width // 2
        center_y = y + height // 2
        current_time = time.time()
        
        # CRITICAL FIX: Increase distance threshold to merge closer faces
        distance_threshold = 40000  # Increased from 25000 to prevent duplicate boxes
        overlap_threshold = 0.3     # NEW: Check for overlapping faces
        
        min_distance = float('inf')
        closest_id = None
        
        # Enhanced cleanup - more aggressive
        if self.frame_counter % 90 == 0:  # Every 3 seconds
            timeout_unrecognized = 5.0    # Faster cleanup
            timeout_recognized = 12.0     
            
            to_remove = []
            for fid, (_, _, timestamp) in self.face_tracker.items():
                cached_data = self.face_cache.get(fid)
                is_recognized = cached_data and cached_data.is_recognized
                
                timeout = timeout_recognized if is_recognized else timeout_unrecognized
                
                if current_time - timestamp > timeout:
                    to_remove.append(fid)
            
            for fid in to_remove:
                self.face_tracker.pop(fid, None)
                # Clean up all related data
                self.face_cache.pop(fid, None)
                if hasattr(self.face_recognizer, 'embedding_cache'):
                    cache_key = f"{fid}_embedding"
                    self.face_recognizer.embedding_cache.pop(cache_key, None)
                self.unknown_person_manager.cleanup_face(fid)
                logger.debug(f"Cleaned up old face: {fid}")
        
        # CRITICAL FIX: Check for overlapping faces
        for face_id, (prev_x, prev_y, timestamp) in self.face_tracker.items():
            distance = (center_x - prev_x) ** 2 + (center_y - prev_y) ** 2
            
            # Calculate overlap percentage
            overlap = self._calculate_overlap_percentage(x, y, width, height, prev_x, prev_y, width, height)
            
            # Use either distance OR overlap criteria
            if (distance < distance_threshold) or (overlap > overlap_threshold):
                if distance < min_distance:
                    min_distance = distance
                    closest_id = face_id
        
        if closest_id:
            self.face_tracker[closest_id] = (center_x, center_y, current_time)
            logger.debug(f"Updated existing face ID: {closest_id}")
            return closest_id
        else:
            new_id = f"face_{self.next_face_id:03d}"
            self.next_face_id += 1
            self.face_tracker[new_id] = (center_x, center_y, current_time)
            logger.debug(f"Created new face ID: {new_id}")
            return new_id

    def _filter_overlapping_detections(self, detections) -> list:
        """Filter out overlapping face detections before processing"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        filtered = []
        for detection in sorted_detections:
            confidence = detection[2]
            if confidence <= self.min_confidence:
                continue
                
            x1 = detection[3]
            y1 = detection[4]
            x2 = detection[5]
            y2 = detection[6]
            
            # Check if this detection overlaps with any already accepted detection
            overlaps = False
            for accepted in filtered:
                ax1, ay1, ax2, ay2 = accepted[3], accepted[4], accepted[5], accepted[6]
                
                overlap_percentage = self._calculate_detection_overlap(x1, y1, x2, y2, ax1, ay1, ax2, ay2)
                
                if overlap_percentage > 0.5:  # 50% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(detection)
        
        logger.debug(f"Filtered {len(detections)} detections to {len(filtered)} unique faces")
        return filtered

    def _calculate_detection_overlap(self, x1, y1, x2, y2, ax1, ay1, ax2, ay2) -> float:
        """Calculate overlap between two detection boxes"""
        try:
            # Calculate intersection
            left = max(x1, ax1)
            top = max(y1, ay1)
            right = min(x2, ax2)
            bottom = min(y2, ay2)
            
            if left < right and top < bottom:
                intersection_area = (right - left) * (bottom - top)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ax2 - ax1) * (ay2 - ay1)
                union_area = area1 + area2 - intersection_area
                
                return intersection_area / union_area if union_area > 0 else 0
            return 0.0
        except:
            return 0.0

    def _remove_duplicate_faces(self, face_detections: List[FaceDetection]) -> List[FaceDetection]:
        """Remove duplicate faces by name and position"""
        if len(face_detections) <= 1:
            return face_detections
        
        unique_faces = []
        seen_names = set()
        
        # Sort by recognition confidence (highest first)
        sorted_faces = sorted(face_detections, 
                            key=lambda f: f.recognition_confidence if f.recognition_confidence else 0, 
                            reverse=True)
        
        for face in sorted_faces:
            # For recognized faces, check by name
            if face.is_recognized and face.human_name:
                if face.human_name in seen_names:
                    logger.debug(f"Skipping duplicate recognized face: {face.human_name}")
                    continue
                seen_names.add(face.human_name)
            
            # For unknown faces, check by position
            else:
                is_duplicate = False
                for existing in unique_faces:
                    if not existing.is_recognized:  # Compare with other unknown faces
                        distance = ((face.x - existing.x) ** 2 + (face.y - existing.y) ** 2) ** 0.5
                        if distance < 50:  # Within 50 pixels
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    logger.debug(f"Skipping duplicate unknown face at position ({face.x}, {face.y})")
                    continue
            
            unique_faces.append(face)
        
        if len(unique_faces) != len(face_detections):
            logger.info(f"Removed {len(face_detections) - len(unique_faces)} duplicate faces")
        
        return unique_faces

    def _calculate_overlap_percentage(self, x1, y1, w1, h1, x2, y2, w2, h2) -> float:
        """Calculate overlap percentage between two rectangles"""
        try:
            # Calculate intersection
            left = max(x1, x2)
            top = max(y1, y2)
            right = min(x1 + w1, x2 + w2)
            bottom = min(y1 + h1, y2 + h2)
            
            if left < right and top < bottom:
                intersection_area = (right - left) * (bottom - top)
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - intersection_area
                
                overlap_percentage = intersection_area / union_area if union_area > 0 else 0
                return overlap_percentage
            return 0.0
        except:
            return 0.0

    def _notify_callbacks(self, result: DetectionResult, frame_data: FrameData):
        """Notify all registered callbacks about detection results"""
        for callback in self.detection_callbacks:
            try:
                # Run callback in thread pool to avoid blocking
                self.executor.submit(callback, result, frame_data)
            except Exception as e:
                if self.log_counter % 100 == 0:
                    logger.error(f"Error in detection callback: {e}")
    
# Make sure this import is at the top of the file or in the draw_detections method:
    from .emotion_recognizer import normalize_emotion

    # In the draw_detections method, ensure emotion display works:
# Fix the draw_detections method - the logic for showing emotions on recognized faces is broken:

    def draw_detections(self, frame: np.ndarray, detections: List[FaceDetection], 
                    show_emotions: bool = True, show_probabilities: bool = False) -> np.ndarray:
        """Draw detection results on frame with names and enhanced visualization"""
        annotated_frame = frame.copy()

        for detection in detections:
            # Draw bounding box with color based on recognition status
            if detection.is_recognized:
                color = (0, 255, 0)  # Green for recognized faces
                thickness = 3
            else:
                color = self._get_emotion_color(detection.emotion)
                thickness = 2
            
            # Draw main bounding box
            cv2.rectangle(
                annotated_frame,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                color,
                thickness
            )
            
            # Prepare labels
            name_parts = []
            emotion_parts = []
            
            # Build name label
            if detection.is_recognized and detection.human_name:
                name_parts.append(f"{detection.human_name}")
                if detection.recognition_confidence > 0:
                    name_parts.append(f"({detection.recognition_confidence:.1%})")
            else:
                if detection.face_id:
                    name_parts.append(f"Unknown #{detection.face_id.split('_')[-1]}")
            
            # Build emotion label (separate from name)
            if show_emotions and detection.emotion:
                try:
                    # Try to import normalize_emotion, fallback if not available
                    try:
                        from .emotion_recognizer import normalize_emotion
                        emotion_text = f"{normalize_emotion(detection.emotion).title()}"
                    except ImportError:
                        emotion_text = f"{detection.emotion.title()}"
                    
                    if detection.emotion_confidence:
                        emotion_text += f" {detection.emotion_confidence:.0%}"
                    emotion_parts.append(emotion_text)
                except Exception as e:
                    # Fallback emotion display
                    emotion_text = f"{detection.emotion}"
                    if detection.emotion_confidence:
                        emotion_text += f" {detection.emotion_confidence:.0%}"
                    emotion_parts.append(emotion_text)
            
            # Draw labels
            if name_parts or emotion_parts:
                if detection.is_recognized:
                    # For recognized faces: name on top, emotion below
                    if name_parts:
                        name_label = " ".join(name_parts)
                        
                        # Draw name label
                        font_scale = 0.8
                        font_thickness = 2
                        label_size = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        
                        # Draw name background
                        cv2.rectangle(
                            annotated_frame,
                            (detection.x, detection.y - label_size[1] - 20),
                            (detection.x + label_size[0] + 10, detection.y - 5),
                            color,
                            -1
                        )
                        
                        # Draw name text
                        cv2.putText(
                            annotated_frame,
                            name_label,
                            (detection.x + 5, detection.y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            font_thickness
                        )
                    
                    # Draw emotion label below the face (FIXED: Always show emotion if available)
                    if emotion_parts:
                        emotion_label = " ".join(emotion_parts)
                        emotion_size = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        
                        cv2.rectangle(
                            annotated_frame,
                            (detection.x, detection.y + detection.height + 5),
                            (detection.x + emotion_size[0] + 10, detection.y + detection.height + 25),
                            self._get_emotion_color(detection.emotion),
                            -1
                        )
                        
                        cv2.putText(
                            annotated_frame,
                            emotion_label,
                            (detection.x + 5, detection.y + detection.height + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1
                        )
                else:
                    # For unknown faces: show all info in one label
                    all_parts = name_parts + emotion_parts
                    main_label = " | ".join(all_parts)
                    label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (detection.x, detection.y - label_size[1] - 15),
                        (detection.x + label_size[0] + 10, detection.y - 5),
                        color,
                        -1
                    )
                    
                    # Draw main label text
                    cv2.putText(
                        annotated_frame,
                        main_label,
                        (detection.x + 5, detection.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return annotated_frame
    
    def _get_emotion_color(self, emotion: str) -> tuple:
        """Get color for emotion visualization"""
        colors = {
            'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
            'surprise': (0, 255, 255), 'fear': (128, 0, 128),
            'disgust': (0, 128, 0), 'neutral': (128, 128, 128)
        }
        return colors.get(emotion.lower() if emotion else 'neutral', (255, 255, 255))
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        with self._lock:
            recognized_faces = [
                fid for fid in self.face_tracker.keys()
                if fid in self.recognition_cache and self.recognition_cache[fid][0][0] is not None
            ]
            return {
                'active_faces': len(self.face_tracker),
                'recognized_faces': len(recognized_faces),
                'total_faces_detected': self.next_face_id - 1,
                'face_ids': list(self.face_tracker.keys()),
                'deepface_available': self.emotion_recognizer.available if self.emotion_recognizer else False,
                'face_recognition_available': self.face_recognizer.available if self.face_recognizer else False,
                'face_usecase_available': self.face_usecase is not None,
                'models_loaded': {
                    'dnn_model': self.net is not None,
                    'emotion_recognizer': self.emotion_recognizer is not None,
                    'face_recognizer': self.face_recognizer is not None,
                    'face_usecase': self.face_usecase is not None
                }
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.background_executor:
            self.background_executor.shutdown(wait=True)
        self.face_tracker.clear()
        self.recognition_cache.clear()
        with self.detection_lock:
            self.stream_detections.clear()
        logger.info("FaceDetectionProcessor cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

    def _assess_face_quality(self, face_region: np.ndarray, face_detection: FaceDetection) -> dict:
        """Enhanced quality assessment with better frontal face detection"""
        quality_score = 0.0
        quality_reasons = []
        
        # 1. Size check
        min_face_size = 80
        face_area = face_detection.width * face_detection.height
        if face_area >= min_face_size * min_face_size:
            quality_score += 0.2
        else:
            quality_reasons.append("face_too_small")
        
        # 2. Aspect ratio check (more strict for frontal faces)
        aspect_ratio = face_detection.width / face_detection.height
        if 0.8 <= aspect_ratio <= 1.2:  # More strict range for frontal faces
            quality_score += 0.2
        else:
            quality_reasons.append("abnormal_aspect_ratio")
        
        # 3. Blur detection (higher threshold for better quality)
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score > 150:  # Increased from 100 for better quality
                quality_score += 0.2
            else:
                quality_reasons.append("too_blurry")
        except:
            blur_score = 0
            quality_reasons.append("blur_detection_failed")
        
        # 4. Detection confidence
        if face_detection.confidence > 0.7:
            quality_score += 0.2
        else:
            quality_reasons.append("low_detection_confidence")
        
        # 5. ENHANCED: Face pose assessment with eye/mouth detection
        pose_info = self._detect_face_quality_enhanced(face_region)
        if pose_info['is_frontal'] and pose_info['eyes_visible'] and pose_info['mouth_visible']:
            quality_score += 0.2
            pose_bonus = pose_info['overall_quality'] * 0.1  # Up to 0.1 bonus
            quality_score += pose_bonus
        else:
            quality_reasons.extend(pose_info['issues'])
        
        return {
            'quality_score': quality_score,
            'is_good_quality': quality_score >= 0.65,  # Slightly higher threshold
            'reasons': quality_reasons,
            'blur_score': blur_score,
            'face_area': face_area,
            'pose_info': pose_info
        }    

    def _detect_face_quality_enhanced(self, face_region: np.ndarray) -> dict:
        """Enhanced face quality detection focusing on eyes, mouth visibility and frontality"""
        quality_info = {
            'is_frontal': False,
            'eyes_visible': False,
            'mouth_visible': False,
            'overall_quality': 0.0,
            'issues': []
        }
        
        try:
            if face_region.size == 0:
                quality_info['issues'].append("empty_face_region")
                return quality_info
            
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            h, w = gray.shape
            
            # Check for minimum face size
            if h < 120 or w < 120:
                quality_info['issues'].append("face_too_small_for_features")
                return quality_info
            
            # Detect eyes
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(15, 15))
            
            # Detect mouth area
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouth = mouth_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 15))
            
            # Eye analysis
            if len(eyes) >= 2:
                quality_info['eyes_visible'] = True
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye, right_eye = eyes[0], eyes[1]
                
                # Calculate eye symmetry for frontality
                eye_y_diff = abs(left_eye[1] - right_eye[1])
                eye_distance = right_eye[0] - left_eye[0]
                
                if eye_distance > 0:
                    symmetry_ratio = eye_y_diff / eye_distance
                    if symmetry_ratio < 0.15:  # Eyes are fairly level
                        quality_info['is_frontal'] = True
                    else:
                        quality_info['issues'].append("face_not_frontal_tilted")
                
                # Check for obstructions (eyes too close to edges)
                edge_margin = w * 0.1  # 10% margin from edges
                if (left_eye[0] < edge_margin or 
                    right_eye[0] + right_eye[2] > w - edge_margin):
                    quality_info['issues'].append("eyes_near_edge_possible_obstruction")
            else:
                quality_info['issues'].append("insufficient_eyes_detected")
            
            # Mouth analysis
            if len(mouth) > 0:
                quality_info['mouth_visible'] = True
                # Check if mouth is in expected position (lower half of face)
                mouth_y = mouth[0][1]
                if mouth_y > h * 0.5:  # Mouth in lower half
                    pass  # Good position
                else:
                    quality_info['issues'].append("mouth_position_unusual")
            else:
                quality_info['issues'].append("mouth_not_detected")
            
            # Check for hand/obstruction detection (simple brightness variance)
            try:
                # Check upper face region for unusual brightness (possible hand obstruction)
                upper_region = gray[0:h//3, :]
                brightness_var = np.var(upper_region)
                if brightness_var < 200:  # Very low variance might indicate obstruction
                    quality_info['issues'].append("possible_obstruction_detected")
            except:
                pass
            
            # Overall quality calculation
            quality_factors = [
                quality_info['eyes_visible'],
                quality_info['mouth_visible'], 
                quality_info['is_frontal'],
                len(quality_info['issues']) == 0
            ]
            quality_info['overall_quality'] = sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.debug(f"Enhanced face quality detection failed: {e}")
            quality_info['issues'].append("quality_detection_error")
        
        return quality_info

    def _detect_face_pose(self, face_region: np.ndarray, face_detection: FaceDetection) -> dict:
        """
        Detect face pose/angle to determine if face is suitable for recognition
        Returns pose information and quality assessment
        """
        pose_info = {
            'yaw': 0.0,      # Left-right rotation
            'pitch': 0.0,    # Up-down rotation  
            'roll': 0.0,     # Tilt rotation
            'is_frontal': False,
            'pose_quality': 0.0,
            'pose_score': 0.0
        }
        
        try:
            # Method 1: Using facial landmarks (recommended)
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            
            # Detect facial landmarks
            landmarks = self._get_facial_landmarks(gray)
            
            if landmarks is not None:
                pose_info = self._calculate_pose_from_landmarks(landmarks, face_region.shape)
            
            # Method 2: Fallback - use simple geometric analysis
            else:
                pose_info = self._estimate_pose_geometric(face_region)
                
        except Exception as e:
            logger.warning(f"Face pose detection failed: {e}")
        
        return pose_info

    def _get_facial_landmarks(self, gray_face: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks using dlib or MediaPipe"""
        try:
            # Option A: Using dlib (more accurate but requires model download)
            if hasattr(self, 'landmark_predictor'):
                faces = self.face_detector(gray_face)
                if len(faces) > 0:
                    landmarks = self.landmark_predictor(gray_face, faces[0])
                    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            
            # Option B: Using MediaPipe (lighter, built-in)
            elif hasattr(self, 'mp_face_mesh'):
                results = self.mp_face_mesh.process(cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB))
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    h, w = gray_face.shape
                    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
            
        except Exception as e:
            logger.debug(f"Landmark detection failed: {e}")
        
        return None

    def _calculate_pose_from_landmarks(self, landmarks: np.ndarray, face_shape: tuple) -> dict:
        """Calculate face pose angles from facial landmarks"""
        try:
            # Key landmark points for pose estimation
            # Using 68-point dlib model indices
            nose_tip = landmarks[30]           # Nose tip
            left_eye_corner = landmarks[36]    # Left eye outer corner  
            right_eye_corner = landmarks[45]   # Right eye outer corner
            left_mouth = landmarks[48]         # Left mouth corner
            right_mouth = landmarks[54]        # Right mouth corner
            chin = landmarks[8]                # Chin point
            
            # Calculate face center
            face_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
            face_center_y = (left_eye_corner[1] + right_eye_corner[1]) / 2
            
            # Calculate yaw (left-right rotation)
            eye_distance = np.linalg.norm(right_eye_corner - left_eye_corner)
            nose_offset = nose_tip[0] - face_center_x
            yaw_ratio = nose_offset / (eye_distance / 2) if eye_distance > 0 else 0
            yaw_angle = np.arcsin(np.clip(yaw_ratio, -1, 1)) * 180 / np.pi
            
            # Calculate pitch (up-down rotation)
            nose_to_chin = np.linalg.norm(chin - nose_tip)
            expected_nose_chin_ratio = 1.2  # Typical ratio for frontal face
            pitch_ratio = nose_to_chin / (eye_distance * expected_nose_chin_ratio) if eye_distance > 0 else 1
            pitch_angle = (1 - pitch_ratio) * 30  # Approximate pitch in degrees
            
            # Calculate roll (tilt)
            eye_vector = right_eye_corner - left_eye_corner
            roll_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
            
            # Determine if face is frontal enough for recognition
            yaw_threshold = 25   # degrees
            pitch_threshold = 20 # degrees  
            roll_threshold = 15  # degrees
            
            is_frontal = (abs(yaw_angle) < yaw_threshold and 
                        abs(pitch_angle) < pitch_threshold and 
                        abs(roll_angle) < roll_threshold)
            
            # Calculate pose quality score (0-1)
            yaw_score = max(0, 1 - abs(yaw_angle) / yaw_threshold)
            pitch_score = max(0, 1 - abs(pitch_angle) / pitch_threshold)
            roll_score = max(0, 1 - abs(roll_angle) / roll_threshold)
            pose_quality = (yaw_score + pitch_score + roll_score) / 3
            
            return {
                'yaw': yaw_angle,
                'pitch': pitch_angle,
                'roll': roll_angle,
                'is_frontal': is_frontal,
                'pose_quality': pose_quality,
                'pose_score': pose_quality,
                'landmarks_count': len(landmarks)
            }
            
        except Exception as e:
            logger.warning(f"Pose calculation failed: {e}")
            return self._get_default_pose_info()

    def _estimate_pose_geometric(self, face_region: np.ndarray) -> dict:
        """
        Fallback method: estimate pose using simple geometric features
        Less accurate but doesn't require landmark detection
        """
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            h, w = gray.shape
            
            # Detect eyes using Haar cascades (simpler fallback)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1] if len(eyes) > 1 else eyes[0]
                
                # Calculate eye centers
                left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
                
                # Calculate roll angle
                eye_vector = (right_center[0] - left_center[0], right_center[1] - left_center[1])
                roll_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
                
                # Estimate yaw from eye size difference (rough approximation)
                left_eye_area = left_eye[2] * left_eye[3]
                right_eye_area = right_eye[2] * right_eye[3]
                eye_ratio = left_eye_area / right_eye_area if right_eye_area > 0 else 1
                
                # If one eye is much smaller, face might be turned
                yaw_angle = (1 - eye_ratio) * 30 if abs(1 - eye_ratio) > 0.2 else 0
                
                # Simple frontality check
                is_frontal = abs(roll_angle) < 15 and abs(yaw_angle) < 20
                pose_quality = 0.6 if is_frontal else 0.3  # Lower confidence for geometric method
                
                return {
                    'yaw': yaw_angle,
                    'pitch': 0.0,  # Can't estimate pitch reliably with this method
                    'roll': roll_angle,
                    'is_frontal': is_frontal,
                    'pose_quality': pose_quality,
                    'pose_score': pose_quality,
                    'method': 'geometric'
                }
        
        except Exception as e:
            logger.warning(f"Geometric pose estimation failed: {e}")
        
        return self._get_default_pose_info()

    def _get_default_pose_info(self) -> dict:
        """Return default pose info when detection fails"""
        return {
            'yaw': 0.0,
            'pitch': 0.0, 
            'roll': 0.0,
            'is_frontal': False,  # Conservative default
            'pose_quality': 0.0,
            'pose_score': 0.0,
            'method': 'default'
        }
    
    def _detect_face_pose_fast(self, face_roi: np.ndarray) -> dict:
        """Fast pose detection using simple geometric features"""
        try:
            if face_roi.size == 0:
                return self._get_default_pose_info()
            
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            h, w = gray.shape
            
            # Use simple eye detection for pose estimation
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 3, minSize=(10, 10))
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1]
                
                # Calculate eye centers
                left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
                
                # Calculate angles
                eye_distance = np.sqrt((right_center[0] - left_center[0])**2 + 
                                    (right_center[1] - left_center[1])**2)
                
                # Roll angle
                roll_angle = np.arctan2(right_center[1] - left_center[1], 
                                    right_center[0] - left_center[0]) * 180 / np.pi
                
                # Rough yaw estimation from eye sizes
                left_area = left_eye[2] * left_eye[3]
                right_area = right_eye[2] * right_eye[3]
                area_ratio = left_area / right_area if right_area > 0 else 1
                yaw_angle = (1 - area_ratio) * 20 if abs(1 - area_ratio) > 0.3 else 0
                
                # Frontality check
                is_frontal = (abs(roll_angle) < 15 and abs(yaw_angle) < 25)
                pose_quality = 0.8 if is_frontal else 0.4
                
                return {
                    'yaw': yaw_angle,
                    'pitch': 0.0,
                    'roll': roll_angle,
                    'is_frontal': is_frontal,
                    'pose_quality': pose_quality,
                    'pose_score': pose_quality,
                    'method': 'fast_geometric'
                }
        
        except Exception:
            pass
        
        return self._get_default_pose_info()
    
    def _create_empty_cache_entry(self) -> dict:
        """Create empty cache entry with all fields"""
        return {
            'human_guid': None,
            'human_name': None,
            'human_type': None,
            'is_recognized': False,
            'recognition_confidence': 0.0,
            'emotion': None,
            'emotion_confidence': 0.0,
            'pose_info': self._get_default_pose_info(),
            'last_updated': 0,
            'recognition_updated': 0,
            'emotion_updated': 0,
            'pose_updated': 0
        }
        

    def _create_face_detection_object(self, x1: int, y1: int, width: int, height: int, 
                                    confidence: float, face_id: str, cached_data) -> FaceDetection:
        """Create FaceDetection object from processed data - FAST"""
        
        # Debug logging
        if cached_data.is_recognized:
            logger.debug(f"Creating recognized face detection for {face_id}: "
                        f"name={cached_data.human_name}, emotion={cached_data.emotion}, "
                        f"emotion_conf={cached_data.emotion_confidence}")
        
        return FaceDetection(
            x=x1,
            y=y1,
            width=width,
            height=height,
            confidence=float(confidence),
            emotion=cached_data.emotion,
            emotion_confidence=cached_data.emotion_confidence,
            emotion_probabilities=None,  # Skip for performance
            face_id=face_id,
            human_guid=cached_data.human_guid,
            human_name=cached_data.human_name,
            human_type=cached_data.human_type,
            is_recognized=cached_data.is_recognized,
            recognition_confidence=cached_data.recognition_confidence,
            face_embedding=None,  # Skip for performance
        )
        
# Replace the _process_unknown_face method with this corrected version:
    def _process_unknown_face(self, face_roi: np.ndarray, face_detection: FaceDetection) -> bool:
        """Process unknown face with enhanced duplicate prevention"""
        logger.info(f"🔎 ENTERED _process_unknown_face for {face_detection.face_id}")
        
        # ENHANCED: Multiple duplicate checks
        with self.unknown_processing_lock:
            if face_detection.is_recognized:
                logger.debug(f"Skipping {face_detection.face_id} - already recognized as {face_detection.human_name}")
                return False
            
            if face_detection.face_id in self.recognition_in_progress:
                logger.debug(f"Skipping {face_detection.face_id} - recognition in progress")
                return False
            
            cached_data = self.face_cache.get(face_detection.face_id)
            if cached_data and cached_data.is_recognized:
                logger.debug(f"Skipping {face_detection.face_id} - cached as recognized: {cached_data.human_name}")
                return False
            
            if face_detection.face_id in self.unknown_person_manager.saved_faces:
                logger.debug(f"Skipping {face_detection.face_id} - already saved to database")
                return False
                
            # ENHANCED: Check if currently being processed
            if face_detection.face_id in self.unknown_person_manager.processing_faces:
                logger.debug(f"Skipping {face_detection.face_id} - currently being processed")
                return False
        
        # Quality assessment
        quality_assessment = self._assess_face_quality(face_roi, face_detection)
        logger.info(f"📊 QUALITY ASSESSMENT for {face_detection.face_id}: score={quality_assessment['quality_score']:.2f}, "
                    f"good_quality={quality_assessment['is_good_quality']}, frontal={quality_assessment.get('pose_info', {}).get('is_frontal', False)}")

        # Check if frontal
        pose_info = quality_assessment.get('pose_info', {})
        if not pose_info.get('is_frontal', False):
            logger.info(f"❌ REJECTING {face_detection.face_id} - not frontal pose")
            return False
        
        # Check if we should capture this face
        with self.unknown_processing_lock:
            should_capture = self.unknown_person_manager.should_capture_unknown_face(
                face_detection.face_id, quality_assessment, face_detection.is_recognized
            )
            logger.info(f"📋 CAPTURE DECISION for {face_detection.face_id}: should_capture={should_capture}")
            
            if not should_capture:
                logger.info(f"❌ REJECTING {face_detection.face_id} - should not capture")
                return False
        
        # Save face image if good quality
        if quality_assessment['is_good_quality']:
            self.unknown_person_manager.add_face_image(face_detection.face_id, face_roi, quality_assessment)
            logger.info(f"✅ CAPTURED high-quality unknown face {face_detection.face_id}")
        
        # Database save check
        with self.unknown_processing_lock:
            # Re-check recognition status
            cached_data = self.face_cache.get(face_detection.face_id)
            if cached_data and cached_data.is_recognized:
                logger.info(f"STOPPING: {face_detection.face_id} was recognized as {cached_data.human_name} during processing")
                self.unknown_person_manager.cleanup_face(face_detection.face_id)
                return False
            
            # Check if ready for database
            is_ready = self.unknown_person_manager.is_ready_for_database(face_detection.face_id)
            
            if is_ready:
                logger.info(f"🚀 STARTING DATABASE SAVE for {face_detection.face_id}")
                
                # ENHANCED: Mark as being processed
                self.unknown_person_manager.mark_processing_start(face_detection.face_id)
                
                # Save to database
                try:
                    result = self._save_unknown_person_to_database(face_detection.face_id)
                    logger.info(f"📝 DATABASE SAVE RESULT for {face_detection.face_id}: {result}")
                    
                    # Mark processing end
                    self.unknown_person_manager.mark_processing_end(face_detection.face_id, success=result)
                    
                    return result
                except Exception as e:
                    logger.error(f"❌ DATABASE SAVE ERROR for {face_detection.face_id}: {e}")
                    self.unknown_person_manager.mark_processing_end(face_detection.face_id, success=False)
                    return False
            else:
                logger.info(f"⏸️ NOT READY for database yet: {face_detection.face_id}")
        
        return False

    def _save_unknown_person_to_database(self, face_id: str) -> bool:
        """Save unknown person: best image -> UCode -> create human -> save face embedding"""
        import os, json, requests
                
        try:
            if face_id in self.unknown_person_manager.saved_faces:
                return True
            
            # Get best images
            best_images = self.unknown_person_manager.get_best_images(face_id, count=3)
            if not best_images:
                logger.warning(f"No images available for unknown face {face_id}")
                return False

            #should be deleted chechking
            for i, img in enumerate(best_images):
                quality = img['quality']
                logger.info(f"Image {i+1}: quality={quality['quality_score']:.2f}, "
                        f"frontal={quality['pose_info']['is_frontal']}, "
                        f"blur={quality['blur_score']:.0f}")
                
            
            # Get the SINGLE best quality image (first one is highest quality)
            best_image_data = best_images[0]
            face_image = best_image_data['image']
            quality = best_image_data['quality']
            timestamp = best_image_data['timestamp']
            
            company_name = self.config.mini_pc_info.get("company_name", "")
            logger.info(f"Mini PC info in creating human: {self.config.mini_pc_info}")

            if not "company_name" in self.config.mini_pc_info:
                if self.config.mini_pc_info.get("company_id", False):
                    try:
                        api_response, response, error = self.ucode_api.items("company").get_single(self.config.mini_pc_info["company_id"]).exec()
                        # Fix: Unpack the tuple and check for success
                        print(api_response)
                        if error is None and api_response is not None:
                            company_name = api_response.data_container.data.get("response", {}).get("title", "Unknown Company")
                            self.config.mini_pc_info["company_name"] = company_name
                            logger.info(f"Company name successfully extracted: {company_name}")
                        else:
                            logger.error(f"Failed to get company data: {error}")
                            company_name = f"company_{self.config.mini_pc_info['company_id'][:8]}"
                            self.config.mini_pc_info["company_name"] = company_name
                    except Exception as err:
                        logger.error(f"Failed to get company data from ucode: {err}")

                        company_name = f"company_{self.config.mini_pc_info['company_id'][:8]}"
                        self.config.mini_pc_info["company_name"] = company_name
                else:
                    logger.error("mini pc info is not initialized")
                    company_name = "unknown_company"
                    
            logger.info(f"=== STEP 1: SAVING TEMP FILE ===")
            # Create temporary storage for the best image only
            temp_dir = "storage/temp_unknown"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save best image temporarily
            current_timestamp = int(time.time())
            filename = f"{company_name}_client_{current_timestamp}_quality_{quality['quality_score']:.2f}.jpg"
            temp_filepath = os.path.join(temp_dir, filename)
            
            cv2.imwrite(temp_filepath, face_image)
            logger.info(f"Saved best quality image: {filename}")
            
            logger.info(f"=== STEP 2: UPLOADING TO UCODE ===")
            # STEP 1: Upload to UCode
            image_url = "https://cdn.u-code.io/"
            try:
                uploadRes = self.ucode_api.files().upload(temp_filepath).exec()
                
                if uploadRes and len(uploadRes) >= 1:
                    create_response = uploadRes[0]  
                    
                if hasattr(create_response, 'data') and create_response.data:
                        file_link = create_response.data.get('link')
                        if file_link:
                            # construct full URL for download
                            image_url += file_link
                else:
                    return False
                    
                logger.info(f"Successfully uploaded to UCode: {image_url}")
                
            except Exception as e:
                logger.error(f"Exception during UCode upload for {face_id}: {e}")
                return False
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            
            logger.info(f"=== STEP 3: CREATING HUMAN ===")
            # STEP 2: Create human in UCode
            logger.info("COMPANY ID: "+ getattr(self.config.mini_pc_info, "company_id", ""))
            human_guid = ""
            try:
                human_api_data = {
                    "full_name": f"{company_name}_client{current_timestamp}",
                    "company_id": getattr(self.config.mini_pc_info, "company_id", ""),
                    "branch_id": getattr(self.config.mini_pc_info, "branch_id", ""),  
                    "photos": [image_url],
                    "type": ["client"]
                }
                
                resp = self.ucode_api.items("human").create(human_api_data).exec()
                if resp and len(resp) >= 1:
                    human_create_response = resp[0]
                
                human_guid = human_create_response.data_container.data["guid"]
                    
                logger.info(f"Successfully created human in UCode: {human_guid}")
                
            except Exception as e:
                logger.error(f"Exception during UCode human creation for {face_id}: {e}")
                return False
            
            logger.info(f"=== STEP 4: CALLING EXTERNAL API ===")    
            # STEP 3: Save face embedding via external API
            try:

                face_api_data = {
                    "human_guid": human_guid,
                    "name": f"{company_name}_client{current_timestamp}",
                    "human_type": "client",
                    "image_url": image_url,
                    "metadata": self._make_json_serializable({
                        "face_id": face_id,
                        "detection_timestamp": current_timestamp,
                        "original_timestamp": timestamp,
                        "quality_score": quality['quality_score'],
                        "blur_score": quality['blur_score'],
                        "face_area": quality['face_area'],
                        "pose_info": quality['pose_info'],
                        "is_frontal": quality['pose_info']['is_frontal'],
                        "reasons": quality['reasons']
                    })
                }
                # Get the external API URL from config
                external_api_url = getattr(self.config, 'EXTERNAL_FACE_API_URL', 'https://tabassum.mini-tweet.uz/api/v1/faces/from-url')
                
                response = requests.post(external_api_url, json=face_api_data, timeout=30)
                
                if response.status_code == 200 or response.status_code == 201:
                    logger.info(f"Successfully saved face embedding for {face_id} via external API")
                    response_data = response.json()
                    logger.info(f"External API response: {response_data}")
                else:
                    logger.error(f"External API failed for {face_id}: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Exception during external API call for {face_id}: {e}")
                return False
            
            # STEP 4: Save metadata locally for tracking
            try:
                # Create storage directory for successful saves
                success_dir = "storage/unknown_faces_saved"
                os.makedirs(success_dir, exist_ok=True)
                
                success_metadata = {
                    'face_id': face_id,
                    'human_guid': human_guid,
                    'image_url': image_url,
                    'detection_time': current_timestamp,
                    'original_timestamp': timestamp,
                    'quality_score': quality['quality_score'],
                    'blur_score': quality['blur_score'],
                    'pose_info': quality['pose_info'],
                    'external_api_url': external_api_url,
                    'success': True
                }
                
                success_file = os.path.join(success_dir, f"success_{face_id}_{current_timestamp}.json")
                with open(success_file, 'w') as f:
                    json.dump(success_metadata, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Failed to save success metadata for {face_id}: {e}")
                # Don't fail the whole process for this
            
            
            if face_id in self.face_cache:
                cached_data = self.face_cache[face_id]
                # Reset recognition status to force fresh check
                cached_data.is_recognized = False
                cached_data.last_update = 0  # Force immediate update
                logger.info(f"Reset cache for {face_id} to force recognition check") 
            
            if hasattr(self.face_recognizer, 'embedding_cache') and face_id in self.face_recognizer.embedding_cache:
                del self.face_recognizer.embedding_cache[face_id]
                logger.info(f"Cleared embedding cache for {face_id}")
            
            # Mark as saved to prevent duplicate saves
            self.unknown_person_manager.saved_faces.add(face_id)
            logger.info(f"✅ Successfully completed full pipeline for unknown person {face_id}")            
            
            # Clean up after processing
            # self.unknown_person_manager.cleanup_face(face_id)
            return True
            
        except Exception as e:
            logger.error(f"Error in save pipeline for unknown person {face_id}: {e}")
            return False
    

    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format - FIXED"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.int32, np.int64, int)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        else:
            return str(obj)

