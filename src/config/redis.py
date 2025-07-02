# config/redis.py
from dataclasses import dataclass
import os
from .database import BaseConfig

@dataclass  
class RedisConfig(BaseConfig):
    """Redis configuration for exchange coordination and caching"""
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    username: str = ""
    
    # Connection reliability
    connection_timeout: int = 30
    socket_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Connection pool settings
    max_connections: int = 50
    health_check_interval: int = 30
    
    # Key naming conventions
    exchange_humans_prefix: str = "exchange_humans:"
    human_exchange_prefix: str = "human_exchange:"
    exchange_stats_prefix: str = "exchange_stats:"
    
    # TTL settings (in seconds)
    human_assignment_ttl: int = 86400  # 24 hours
    exchange_stats_ttl: int = 3600     # 1 hour
    
    # Fallback settings
    enable_fallback: bool = True
    fallback_timeout: float = 2.0  
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        return cls(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=cls.get_env_int('REDIS_PORT', 6379),
            db=cls.get_env_int('REDIS_DB', 0),
            password=os.getenv('REDIS_PASSWORD', ''),
            username=os.getenv('REDIS_USERNAME', ''),
            
            connection_timeout=cls.get_env_int('REDIS_CONNECTION_TIMEOUT', 30),
            socket_timeout=cls.get_env_int('REDIS_SOCKET_TIMEOUT', 30),
            retry_attempts=cls.get_env_int('REDIS_RETRY_ATTEMPTS', 3),
            retry_delay=cls.get_env_float('REDIS_RETRY_DELAY', 1.0),
            
            max_connections=cls.get_env_int('REDIS_MAX_CONNECTIONS', 50),
            health_check_interval=cls.get_env_int('REDIS_HEALTH_CHECK_INTERVAL', 30),
            
            exchange_humans_prefix=os.getenv('REDIS_EXCHANGE_HUMANS_PREFIX', 'exchange_humans:'),
            human_exchange_prefix=os.getenv('REDIS_HUMAN_EXCHANGE_PREFIX', 'human_exchange:'),
            exchange_stats_prefix=os.getenv('REDIS_EXCHANGE_STATS_PREFIX', 'exchange_stats:'),
            
            human_assignment_ttl=cls.get_env_int('REDIS_HUMAN_ASSIGNMENT_TTL', 86400),
            exchange_stats_ttl=cls.get_env_int('REDIS_EXCHANGE_STATS_TTL', 3600),
            
            enable_fallback=cls.get_env_bool('REDIS_ENABLE_FALLBACK', True),
            fallback_timeout=cls.get_env_float('REDIS_FALLBACK_TIMEOUT', 2.0)
        )
    
    def validate(self) -> bool:
        """Validate Redis configuration"""
        errors = []
        
        if not self.host:
            errors.append("Redis host is required")
        
        if not (1 <= self.port <= 65535):
            errors.append("Redis port must be between 1 and 65535")
        
        if not (0 <= self.db <= 15):
            errors.append("Redis DB must be between 0 and 15")
        
        if self.connection_timeout <= 0:
            errors.append("Redis connection timeout must be positive")
        
        if errors:
            print("Redis configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True