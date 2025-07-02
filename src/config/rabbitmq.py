# config/rabbitmq.py
from dataclasses import dataclass
import os
from .database import BaseConfig

@dataclass
class RabbitMQConfig(BaseConfig):
    """RabbitMQ configuration for emotion message publishing"""
    # Connection settings
    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    
    # Connection reliability
    heartbeat: int = 600
    blocked_connection_timeout: int = 300
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Exchange settings
    exchange_prefix: str = "emotion_exchange_"
    exchange_type: str = "direct"
    routing_key: str = "emotion"
    exchange_durable: bool = True
    
    # Queue settings  
    queue_durable: bool = True
    message_persistent: bool = True
    
    # Scaling settings
    max_humans_per_exchange: int = 1000
    
    # Publishing settings
    publish_timeout: int = 10
    confirm_delivery: bool = True
    
    @classmethod
    def from_env(cls) -> 'RabbitMQConfig':
        return cls(
            host=os.getenv('RABBITMQ_HOST', 'localhost'),
            port=cls.get_env_int('RABBITMQ_PORT', 5672),
            username=os.getenv('RABBITMQ_USERNAME', 'guest'),
            password=os.getenv('RABBITMQ_PASSWORD', 'guest'),
            virtual_host=os.getenv('RABBITMQ_VIRTUAL_HOST', '/'),
            
            heartbeat=cls.get_env_int('RABBITMQ_HEARTBEAT', 600),
            blocked_connection_timeout=cls.get_env_int('RABBITMQ_BLOCKED_TIMEOUT', 300),
            connection_timeout=cls.get_env_int('RABBITMQ_CONNECTION_TIMEOUT', 30),
            retry_attempts=cls.get_env_int('RABBITMQ_RETRY_ATTEMPTS', 3),
            retry_delay=cls.get_env_float('RABBITMQ_RETRY_DELAY', 2.0),
            
            exchange_prefix=os.getenv('RABBITMQ_EXCHANGE_PREFIX', 'emotion_exchange_'),
            exchange_type=os.getenv('RABBITMQ_EXCHANGE_TYPE', 'direct'),
            routing_key=os.getenv('RABBITMQ_ROUTING_KEY', 'emotion'),
            exchange_durable=cls.get_env_bool('RABBITMQ_EXCHANGE_DURABLE', True),
            
            queue_durable=cls.get_env_bool('RABBITMQ_QUEUE_DURABLE', True),
            message_persistent=cls.get_env_bool('RABBITMQ_MESSAGE_PERSISTENT', True),
            
            max_humans_per_exchange=cls.get_env_int('RABBITMQ_MAX_HUMANS_PER_EXCHANGE', 1000),
            
            publish_timeout=cls.get_env_int('RABBITMQ_PUBLISH_TIMEOUT', 10),
            confirm_delivery=cls.get_env_bool('RABBITMQ_CONFIRM_DELIVERY', True)
        )
    
    def validate(self) -> bool:
        """Validate RabbitMQ configuration"""
        errors = []
        
        if not self.host:
            errors.append("RabbitMQ host is required")
        
        if not (1 <= self.port <= 65535):
            errors.append("RabbitMQ port must be between 1 and 65535")
        
        if not self.username:
            errors.append("RabbitMQ username is required")
        
        if not self.password:
            errors.append("RabbitMQ password is required")
        
        if self.max_humans_per_exchange <= 0:
            errors.append("Max humans per exchange must be positive")
        
        if errors:
            print("RabbitMQ configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True