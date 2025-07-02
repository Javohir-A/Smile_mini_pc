# src/publishers/emotion_exchange_manager.py
import logging
import redis
import time
from typing import Optional, Set
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

class EmotionExchangeManager:
    """Manages exchange assignment and scaling logic using Redis"""
    
    def __init__(self, redis_config, rabbitmq_config):
        self.redis_config = redis_config
        self.rabbitmq_config = rabbitmq_config
        self.redis_client = None
        self.redis_available = False
        self.last_redis_check = None
        self.lock = Lock()
        
        # Initialize Redis connection
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config.host,
                port=self.redis_config.port,
                db=self.redis_config.db,
                password=self.redis_config.password if self.redis_config.password else None,
                username=self.redis_config.username if self.redis_config.username else None,
                socket_timeout=self.redis_config.socket_timeout,
                socket_connect_timeout=self.redis_config.connection_timeout,
                retry_on_timeout=True,
                health_check_interval=self.redis_config.health_check_interval,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            self.last_redis_check = datetime.now()
            logger.info("✅ Redis connection established for exchange management")
            
        except Exception as e:
            self.redis_available = False
            logger.warning(f"⚠️ Redis connection failed: {e}")
            if self.redis_config.enable_fallback:
                logger.info("🔄 Will use REST API fallback")
    
    def _check_redis_health(self) -> bool:
        """Check Redis health with caching to avoid frequent checks"""
        now = datetime.now()
        
        # Check Redis health every 30 seconds
        if (self.last_redis_check and 
            (now - self.last_redis_check).total_seconds() < 30):
            return self.redis_available
        
        try:
            if self.redis_client:
                self.redis_client.ping()
                self.redis_available = True
                logger.debug("✅ Redis health check passed")
            else:
                self._initialize_redis()
                
        except Exception as e:
            self.redis_available = False
            logger.warning(f"❌ Redis health check failed: {e}")
        
        self.last_redis_check = now
        return self.redis_available
    
    def get_exchange_for_human(self, human_id: str) -> Optional[str]:
        """Get or assign exchange for human. Returns None if Redis unavailable."""
        with self.lock:
            if not self._check_redis_health():
                return None
            
            try:
                # Check if human already has an exchange
                exchange_key = f"{self.redis_config.human_exchange_prefix}{human_id}"
                existing_exchange = self.redis_client.get(exchange_key)
                
                if existing_exchange:
                    logger.debug(f"📋 Human {human_id} already assigned to {existing_exchange}")
                    return existing_exchange
                
                # Find available exchange or create new one
                exchange_name = self._find_or_create_exchange(human_id)
                
                if exchange_name:
                    # Store assignment with TTL
                    self.redis_client.setex(
                        exchange_key, 
                        self.redis_config.human_assignment_ttl, 
                        exchange_name
                    )
                    
                    # Add human to exchange set
                    humans_key = f"{self.redis_config.exchange_humans_prefix}{exchange_name}"
                    self.redis_client.sadd(humans_key, human_id)
                    self.redis_client.expire(humans_key, self.redis_config.human_assignment_ttl)
                    
                    # Update exchange stats
                    self._update_exchange_stats(exchange_name)
                    
                    humans_count = self.redis_client.scard(humans_key)
                    logger.info(f"🔗 Assigned human {human_id} to {exchange_name} "
                               f"({humans_count}/{self.rabbitmq_config.max_humans_per_exchange})")
                    
                return exchange_name
                
            except Exception as e:
                logger.error(f"❌ Error managing exchange for human {human_id}: {e}")
                self.redis_available = False
                return None
    
    def _find_or_create_exchange(self, human_id: str) -> Optional[str]:
        """Find available exchange or create new one"""
        try:
            # Get all existing exchanges
            exchange_pattern = f"{self.redis_config.exchange_humans_prefix}*"
            exchange_keys = list(self.redis_client.scan_iter(match=exchange_pattern))
            
            # Check existing exchanges for available capacity
            for exchange_key in exchange_keys:
                exchange_name = exchange_key.replace(self.redis_config.exchange_humans_prefix, '')
                humans_count = self.redis_client.scard(exchange_key)
                
                if humans_count < self.rabbitmq_config.max_humans_per_exchange:
                    logger.info(f"📍 Found available exchange {exchange_name} "
                               f"({humans_count}/{self.rabbitmq_config.max_humans_per_exchange})")
                    return exchange_name
            
            # No available exchange, create new one
            next_number = len(exchange_keys) + 1
            new_exchange_name = f"{self.rabbitmq_config.exchange_prefix}{next_number}"
            
            logger.info(f"🆕 Creating new exchange: {new_exchange_name}")
            return new_exchange_name
            
        except Exception as e:
            logger.error(f"❌ Error finding/creating exchange: {e}")
            return None
    
    def _update_exchange_stats(self, exchange_name: str):
        """Update exchange statistics in Redis"""
        try:
            stats_key = f"{self.redis_config.exchange_stats_prefix}{exchange_name}"
            stats = {
                'last_assignment': datetime.now().isoformat(),
                'total_assignments': 1
            }
            
            # Get existing stats and increment
            existing_stats = self.redis_client.hgetall(stats_key)
            if existing_stats and 'total_assignments' in existing_stats:
                stats['total_assignments'] = int(existing_stats['total_assignments']) + 1
            
            # Update stats with TTL
            self.redis_client.hmset(stats_key, stats)
            self.redis_client.expire(stats_key, self.redis_config.exchange_stats_ttl)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update exchange stats: {e}")
    
    def get_exchange_stats(self) -> dict:
        """Get statistics about all exchanges"""
        if not self._check_redis_health():
            return {"error": "Redis unavailable"}
        
        try:
            stats = {}
            exchange_pattern = f"{self.redis_config.exchange_humans_prefix}*"
            
            for exchange_key in self.redis_client.scan_iter(match=exchange_pattern):
                exchange_name = exchange_key.replace(self.redis_config.exchange_humans_prefix, '')
                humans_count = self.redis_client.scard(exchange_key)
                
                # Get additional stats
                stats_key = f"{self.redis_config.exchange_stats_prefix}{exchange_name}"
                exchange_stats = self.redis_client.hgetall(stats_key)
                
                stats[exchange_name] = {
                    'humans_count': humans_count,
                    'capacity_used': f"{humans_count}/{self.rabbitmq_config.max_humans_per_exchange}",
                    'last_assignment': exchange_stats.get('last_assignment', 'Unknown'),
                    'total_assignments': int(exchange_stats.get('total_assignments', 0))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting exchange stats: {e}")
            return {"error": str(e)}