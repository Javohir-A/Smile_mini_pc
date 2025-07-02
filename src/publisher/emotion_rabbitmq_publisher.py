# src/publishers/emotion_rabbitmq_publisher.py
import logging
import pika
import json
import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import asdict

from src.config.rabbitmq import RabbitMQConfig

logger = logging.getLogger(__name__)

class EmotionRabbitMQPublisher:
    """RabbitMQ publisher for emotion messages"""
    
    def __init__(self, rabbitmq_config:RabbitMQConfig):
        self.config = rabbitmq_config
        self.connection = None
        self.channel = None
        self.is_connected = False
        self.lock = threading.Lock()
        self.last_connection_attempt = None
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize RabbitMQ connection"""
        with self.lock:
            try:
                # Avoid frequent connection attempts
                now = datetime.now()
                if (self.last_connection_attempt and 
                    (now - self.last_connection_attempt).total_seconds() < self.config.retry_delay):
                    return
                
                self.last_connection_attempt = now
                
                # Create connection parameters
                credentials = pika.PlainCredentials(self.config.username, self.config.password)
                parameters = pika.ConnectionParameters(
                    host=self.config.host,
                    port=self.config.port,
                    virtual_host=self.config.virtual_host,
                    credentials=credentials,
                    heartbeat=self.config.heartbeat,
                    blocked_connection_timeout=self.config.blocked_connection_timeout,
                    connection_attempts=self.config.retry_attempts,
                    retry_delay=self.config.retry_delay
                )
                
                # Create connection and channel
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                # Enable delivery confirmations if configured
                if self.config.confirm_delivery:
                    self.channel.confirm_delivery()
                
                self.is_connected = True
                logger.info(f"✅ RabbitMQ publisher connected to {self.config.host}:{self.config.port}")
                
            except Exception as e:
                self.is_connected = False
                self.connection = None
                self.channel = None
                logger.warning(f"⚠️ RabbitMQ connection failed: {e}")
    
    def _ensure_exchange(self, exchange_name: str):
        """Ensure exchange exists"""
        try:
            if self.channel:
                self.channel.exchange_declare(
                    exchange=exchange_name,
                    exchange_type=self.config.exchange_type,
                    durable=self.config.exchange_durable
                )
                logger.debug(f"📡 Ensured exchange exists: {exchange_name}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure exchange {exchange_name}: {e}")
            raise
    
    def publish_emotion(self, exchange_name: str, emotion_data: Dict[str, Any]) -> bool:
        """Publish emotion message to specific exchange"""
        with self.lock:
            # Check connection
            if not self.is_connected or not self.channel:
                self._initialize_connection()
            
            if not self.is_connected:
                return False
            
            try:
                # Ensure exchange exists
                self._ensure_exchange(exchange_name)
                
                # Prepare message
                message_body = json.dumps(emotion_data, default=str)
                
                # Message properties
                properties = pika.BasicProperties(
                    delivery_mode=2 if self.config.message_persistent else 1,
                    content_type='application/json',
                    message_id=emotion_data.get('emotion_id', ''),
                    timestamp=int(datetime.now().timestamp()),
                    app_id='emotion_processor'
                )
                
                # Publish message
                self.channel.basic_publish(
                    exchange=exchange_name,
                    routing_key=self.config.routing_key,
                    body=message_body,
                    properties=properties
                )
                
                logger.info(f"📤 Published emotion to {exchange_name}: "
                           f"{emotion_data.get('human_name', 'Unknown')} - "
                           f"{emotion_data.get('emotion_type', 'Unknown')}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to publish to {exchange_name}: {e}")
                self.is_connected = False
                return False
    
    def is_healthy(self) -> bool:
        """Check if publisher is healthy"""
        return self.is_connected and self.connection and not self.connection.is_closed
    
    def close(self):
        """Close RabbitMQ connection"""
        with self.lock:
            try:
                if self.connection and not self.connection.is_closed:
                    self.connection.close()
                    logger.info("🔌 RabbitMQ publisher connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing RabbitMQ connection: {e}")
            finally:
                self.is_connected = False
                self.connection = None
                self.channel = None