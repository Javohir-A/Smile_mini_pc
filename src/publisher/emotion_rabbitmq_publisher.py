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
        """Ensure exchange AND its queue exist with proper binding"""
        try:
            if not self.channel:
                return
            
            logger.info(f"🏗️ Setting up complete infrastructure for {exchange_name}")
            
            # Step 1: Declare exchange
            self.channel.exchange_declare(
                exchange=exchange_name,
                exchange_type=self.config.exchange_type,  # 'direct'
                durable=self.config.exchange_durable      # True
            )
            logger.debug(f"✅ Exchange declared: {exchange_name}")
            
            # Step 2: Declare corresponding queue
            queue_name = f"{exchange_name}_queue"
            self.channel.queue_declare(
                queue=queue_name,
                durable=self.config.queue_durable  # True - survives restart
            )
            logger.debug(f"✅ Queue declared: {queue_name}")
            
            # Step 3: Bind queue to exchange
            self.channel.queue_bind(
                exchange=exchange_name,
                queue=queue_name,
                routing_key=self.config.routing_key  # 'emotion'
            )
            logger.debug(f"✅ Queue bound: {queue_name} → {exchange_name} (key: {self.config.routing_key})")
            
            logger.info(f"🎉 Complete setup for {exchange_name}:")
            logger.info(f"   Exchange: {exchange_name}")
            logger.info(f"   Queue: {queue_name}")
            logger.info(f"   Binding: {self.config.routing_key}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup infrastructure for {exchange_name}: {e}")
            raise
        
    def publish_emotion(self, exchange_name: str, emotion_data: Dict[str, Any]) -> bool:
        """Publish emotion message with complete infrastructure verification"""
        with self.lock:
            # Check connection
            if not self.is_connected or not self.channel:
                self._initialize_connection()
            
            if not self.is_connected:
                return False
            
            try:
                # Ensure complete infrastructure (exchange + queue + binding)
                self._ensure_exchange(exchange_name)
                
                # Verify setup worked
                if not self.verify_setup(exchange_name):
                    logger.error(f"❌ Infrastructure verification failed for {exchange_name}")
                    return False
                
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
                
                # Publish message to EXCHANGE (not directly to queue!)
                self.channel.basic_publish(
                    exchange=exchange_name,           # Send TO exchange
                    routing_key=self.config.routing_key,  # 'emotion' - routing instruction
                    body=message_body,
                    properties=properties
                )
                
                logger.info(f"📤 Published to EXCHANGE '{exchange_name}' with routing key '{self.config.routing_key}': "
                        f"{emotion_data.get('human_name', 'Unknown')} - "
                        f"{emotion_data.get('emotion_type', 'Unknown')}")
                
                # Log the flow for clarity
                queue_name = f"{exchange_name}_queue"
                logger.debug(f"🔄 Message flow: Publisher → {exchange_name} → {queue_name} → Consumer")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to publish to {exchange_name}: {e}")
                self.is_connected = False
                return False
            
    def verify_setup(self, exchange_name: str) -> bool:
        """Verify that exchange, queue, and binding are properly set up"""
        try:
            if not self.channel:
                return False
            
            queue_name = f"{exchange_name}_queue"
            
            # Test that we can declare (passive=True means check existence only)
            try:
                # Check exchange exists
                self.channel.exchange_declare(
                    exchange=exchange_name,
                    exchange_type=self.config.exchange_type,
                    durable=self.config.exchange_durable,
                    passive=True  # Only check, don't create
                )
                
                # Check queue exists
                method = self.channel.queue_declare(
                    queue=queue_name,
                    durable=self.config.queue_durable,
                    passive=True  # Only check, don't create
                )
                
                message_count = method.method.message_count
                logger.debug(f"📊 {queue_name}: {message_count} messages waiting")
                
                return True
                
            except Exception as verify_error:
                logger.warning(f"⚠️ Setup verification failed for {exchange_name}: {verify_error}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error verifying setup for {exchange_name}: {e}")
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