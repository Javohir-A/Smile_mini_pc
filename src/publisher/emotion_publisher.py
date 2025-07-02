# src/publishers/emotion_publisher.py
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from .emotion_exchange_manager import EmotionExchangeManager
from .emotion_rabbitmq_publisher import EmotionRabbitMQPublisher
from src.config.settings import AppConfig

logger = logging.getLogger(__name__)

class EmotionPublisher:
    """
    Main emotion publisher that handles both RabbitMQ and REST API fallback
    Drop-in replacement for _send_emotion_to_api() in emotion_processor.py
    """
    
    def __init__(self, config:AppConfig):
        self.config = config
        self.use_rabbitmq = config.use_rabbitmq
        
        # Initialize RabbitMQ components if enabled
        if self.use_rabbitmq:
            self.exchange_manager = EmotionExchangeManager(
                config.redis, 
                config.rabbitmq
            )
            self.rabbitmq_publisher = EmotionRabbitMQPublisher(config.rabbitmq)
        else:
            self.exchange_manager = None
            self.rabbitmq_publisher = None
        
        # REST API settings for fallback
        self.api_url = config.api_base_url
        self.api_timeout = config.api_timeout
        self.api_retry_attempts = config.api_retry_attempts
        
        logger.info(f"🚀 EmotionPublisher initialized: "
                   f"RabbitMQ={'Enabled' if self.use_rabbitmq else 'Disabled'}, "
                   f"Fallback={'Enabled' if self.use_rabbitmq else 'Only Mode'}")
    
    def publish_emotion(self, emotion_data: Dict[str, Any]) -> bool:
        """
        Publish emotion using RabbitMQ or fallback to REST API
        
        Args:
            emotion_data: Emotion data dictionary with keys:
                - human_id, human_name, human_type, emotion_type, confidence
                - camera_id, timestamp, duration_minutes, video_url, etc.
        
        Returns:
            bool: True if successfully published, False otherwise
        """
        human_id = str(emotion_data.get('human_id', ''))
        
        # Try RabbitMQ first if enabled
        if self.use_rabbitmq and self._try_rabbitmq_publish(human_id, emotion_data):
            return True
        
        # Fallback to REST API
        return self._fallback_to_rest_api(emotion_data)
    
    def _try_rabbitmq_publish(self, human_id: str, emotion_data: Dict[str, Any]) -> bool:
        """Try to publish via RabbitMQ"""
        try:
            # Get exchange assignment
            exchange_name = self.exchange_manager.get_exchange_for_human(human_id)
            
            if not exchange_name:
                logger.warning(f"⚠️ No exchange assigned for human {human_id}, using REST fallback")
                return False
            
            # Check RabbitMQ publisher health
            if not self.rabbitmq_publisher.is_healthy():
                logger.warning("⚠️ RabbitMQ publisher unhealthy, using REST fallback")
                return False
            
            # Add exchange info to emotion data
            enriched_data = emotion_data.copy()
            enriched_data['exchange_name'] = exchange_name
            enriched_data['published_via'] = 'rabbitmq'
            enriched_data['published_at'] = datetime.now().isoformat()
            
            # Publish to RabbitMQ
            success = self.rabbitmq_publisher.publish_emotion(exchange_name, enriched_data)
            
            if success:
                logger.info(f"✅ Published via RabbitMQ: {emotion_data.get('human_name', 'Unknown')} "
                           f"→ {exchange_name}")
                return True
            else:
                logger.warning(f"⚠️ RabbitMQ publish failed for {emotion_data.get('human_name', 'Unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ RabbitMQ publish error: {e}")
    
            return False
    def _fallback_to_rest_api(self, emotion_data: Dict[str, Any]) -> bool:
        """Fallback to REST API"""
        try:
            logger.info(f"🔄 Using REST API fallback for {emotion_data.get('human_name', 'Unknown')}")
            
            # Prepare data for REST API (match existing format)
            api_data = {
                "human_id": str(emotion_data.get('human_id', '')),
                "human_type": str(emotion_data.get('human_type', '')),
                "emotion_type": str(emotion_data.get('emotion_type', '')),
                "confidence": float(emotion_data.get('confidence', 0.0)),
                "camera_id": str(emotion_data.get('camera_id', '')),
                "timestamp": emotion_data.get('timestamp', datetime.now().isoformat() + 'Z'),
                "duration_minutes": float(emotion_data.get('duration_minutes', 0.0)),
                "mini_pc_info": emotion_data.get('mini_pc_info', {})
            }
            
            # Add video URL if available
            if emotion_data.get('video_url'):
                api_data["video_url"] = str(emotion_data['video_url'])
            
            # Make REST API call
            response = requests.post(
                f"{self.api_url}/emotions/detect",
                json=api_data,
                timeout=self.api_timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ REST API success: {emotion_data.get('human_name', 'Unknown')}")
                return True
            else:
                logger.error(f"❌ REST API error {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ REST API network error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ REST API error: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get publisher status for monitoring"""
        status = {
            'mode': 'rabbitmq' if self.use_rabbitmq else 'rest_only',
            'api_url': self.api_url
        }
        
        if self.use_rabbitmq:
            status.update({
                'rabbitmq_healthy': self.rabbitmq_publisher.is_healthy() if self.rabbitmq_publisher else False,
                'redis_available': self.exchange_manager.redis_available if self.exchange_manager else False,
                'exchange_stats': self.exchange_manager.get_exchange_stats() if self.exchange_manager else {}
            })
        
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.rabbitmq_publisher:
                self.rabbitmq_publisher.close()
            logger.info("🧹 EmotionPublisher cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")