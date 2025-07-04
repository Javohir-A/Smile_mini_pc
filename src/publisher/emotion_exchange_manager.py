# src/publisher/emotion_exchange_manager.py - COMPLETE FIXED VERSION
import logging
import redis
import time
import pika
import json
from typing import Optional, Set, List
from datetime import datetime, timedelta
from threading import Lock

from src.config.redis import RedisConfig
from src.config.rabbitmq import RabbitMQConfig

logger = logging.getLogger(__name__)

ATOMIC_ASSIGNMENT_SCRIPT = """
local human_id = ARGV[1]
local max_capacity = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local exchange_prefix = ARGV[4]
local timestamp = ARGV[5]

-- Check if human already assigned
local existing = redis.call('GET', 'human_exchange:' .. human_id)
if existing then
    return existing
end

-- Find available exchange atomically
local pattern = 'exchange_humans:' .. exchange_prefix .. '*'
local exchanges = redis.call('KEYS', pattern)

for i, exchange_key in ipairs(exchanges) do
    local current_count = redis.call('SCARD', exchange_key)
    if current_count < max_capacity then
        local exchange_name = string.gsub(exchange_key, 'exchange_humans:', '')
        
        -- Atomic assignment
        redis.call('SADD', exchange_key, human_id)
        redis.call('SETEX', 'human_exchange:' .. human_id, ttl, exchange_name)
        
        -- Update stats
        local stats_key = 'exchange_stats:' .. exchange_name
        redis.call('HINCRBY', stats_key, 'total_assignments', 1)
        redis.call('HSET', stats_key, 'last_assignment', timestamp)
        
        return exchange_name
    end
end

-- No available exchange found
return 'NEEDS_NEW_EXCHANGE'
"""

class EmotionExchangeManager:
    """Manages exchange assignment and scaling logic using Redis"""
    
    def __init__(self, redis_config: RedisConfig, rabbitmq_config: RabbitMQConfig):
        self.redis_config = redis_config
        self.rabbitmq_config = rabbitmq_config
        self.redis_client = None
        self.redis_available = False
        self.last_redis_check = None
        self.lock = Lock()
        self.assignment_script_sha = None
        
        # Initialize Redis connection FIRST
        self._initialize_redis()
        
        # Then load Lua script
        if self.redis_available:
            self._preload_lua_script()
    
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
            self.redis_client = None
            logger.warning(f"⚠️ Redis connection failed: {e}")
            if self.redis_config.enable_fallback:
                logger.info("🔄 Will use REST API fallback")
    
    def _preload_lua_script(self):
        """Preload Lua script for atomic operations"""
        try:
            if not self.redis_client:
                logger.warning("⚠️ Cannot load Lua script - Redis client not available")
                return
                
            self.assignment_script_sha = self.redis_client.script_load(ATOMIC_ASSIGNMENT_SCRIPT)
            logger.info("✅ Atomic assignment script loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Lua script: {e}")
            self.assignment_script_sha = None
    
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
                
                # Reload script if needed
                if not self.assignment_script_sha:
                    self._preload_lua_script()
            else:
                self._initialize_redis()
                if self.redis_available:
                    self._preload_lua_script()
                
        except Exception as e:
            self.redis_available = False
            logger.warning(f"❌ Redis health check failed: {e}")
        
        self.last_redis_check = now
        return self.redis_available
    
    def get_exchange_for_human(self, human_id: str) -> Optional[str]:
        """Get exchange assignment using atomic Redis operations"""
        with self.lock:
            if not self._check_redis_health():
                return None
            
            try:
                # Ensure script is loaded
                if not self.assignment_script_sha:
                    self._preload_lua_script()
                    if not self.assignment_script_sha:
                        logger.error("❌ Cannot proceed without Lua script")
                        return None
                
                # Use atomic Lua script
                result = self.redis_client.evalsha(
                    self.assignment_script_sha,
                    0,  # Number of keys (we use ARGV only)
                    human_id,
                    str(self.rabbitmq_config.max_humans_per_exchange),
                    str(self.redis_config.human_assignment_ttl),
                    self.rabbitmq_config.exchange_prefix,
                    datetime.now().isoformat()
                )
                
                if result == 'NEEDS_NEW_EXCHANGE':
                    # Handle new exchange creation with distributed lock
                    return self._create_exchange_with_distributed_lock(human_id)
                
                logger.info(f"🔗 Atomic assignment: {human_id} → {result}")
                return result
                
            except redis.exceptions.NoScriptError:
                # Script not loaded, reload and retry
                logger.warning("⚠️ Script not found, reloading...")
                self._preload_lua_script()
                if self.assignment_script_sha:
                    return self.get_exchange_for_human(human_id)
                else:
                    logger.error("❌ Failed to reload script")
                    return None
            except Exception as e:
                logger.error(f"❌ Error in atomic assignment: {e}")
                return None
    
    def _create_exchange_with_distributed_lock(self, human_id: str) -> Optional[str]:
        """Create new exchange with distributed lock to prevent race conditions"""
        lock_key = "exchange_creation_lock"
        lock_timeout = 30  # 30 seconds
        
        try:
            # Acquire distributed lock
            lock_acquired = self.redis_client.set(
                lock_key, 
                human_id, 
                nx=True, 
                ex=lock_timeout
            )
            
            if not lock_acquired:
                # Another process is creating exchange, wait and retry assignment
                logger.info(f"⏳ Waiting for exchange creation to complete...")
                time.sleep(2)
                
                # Retry atomic assignment (new exchange might be available)
                return self.get_exchange_for_human(human_id)
            
            try:
                # We have the lock, proceed with creation
                logger.info(f"🔒 Acquired creation lock for {human_id}")
                
                # Double-check if assignment is still needed
                existing = self.redis_client.get(f"human_exchange:{human_id}")
                if existing:
                    logger.info(f"👤 Human {human_id} was assigned during lock wait: {existing}")
                    return existing
                
                # Double-check if any exchange now has capacity (another process might have created one)
                available_exchange = self._find_available_exchange()
                if available_exchange:
                    logger.info(f"📍 Found available exchange after lock: {available_exchange}")
                    self._assign_human_atomically(human_id, available_exchange)
                    return available_exchange
                
                # Create new exchange
                new_exchange = self._create_new_exchange()
                if new_exchange:
                    logger.info(f"🆕 Created new exchange: {new_exchange}")
                    self._assign_human_atomically(human_id, new_exchange)
                    return new_exchange
                
                logger.error("❌ Failed to create new exchange")
                return None
                
            finally:
                # Release lock
                self.redis_client.delete(lock_key)
                logger.info(f"🔓 Released creation lock")
                
        except Exception as e:
            logger.error(f"❌ Error in distributed lock creation: {e}")
            return None
    
    def _find_available_exchange(self) -> Optional[str]:
        """Find exchange with available capacity"""
        try:
            exchange_pattern = f"{self.redis_config.exchange_humans_prefix}*"
            exchange_keys = list(self.redis_client.scan_iter(match=exchange_pattern))
            
            for exchange_key in exchange_keys:
                exchange_name = exchange_key.replace(self.redis_config.exchange_humans_prefix, '')
                humans_count = self.redis_client.scard(exchange_key)
                
                if humans_count < self.rabbitmq_config.max_humans_per_exchange:
                    logger.debug(f"📍 Found available exchange: {exchange_name} ({humans_count}/1000)")
                    return exchange_name
            
            return None
        except Exception as e:
            logger.error(f"❌ Error finding available exchange: {e}")
            return None
    
    #it doesn't create an actual RabbitMQ exchange creation!
    def _create_new_exchange(self) -> Optional[str]:
        """Create new RabbitMQ exchange and initialize Redis tracking"""
        try:
            # Determine next exchange number
            next_number = self._get_next_exchange_number()
            exchange_name = f"{self.rabbitmq_config.exchange_prefix}{next_number}"
            queue_name = f"{exchange_name}_queue"
            
            logger.info(f"🆕 Creating new exchange: {exchange_name}")
            
            # Create RabbitMQ infrastructure (this requires RabbitMQ connection)
            # For now, we'll just set up Redis tracking - RabbitMQ creation will be handled by publisher
            
            # Initialize Redis tracking for the new exchange
            self._initialize_exchange_in_redis(exchange_name)
            
            return exchange_name
            
        except Exception as e:
            logger.error(f"❌ Failed to create new exchange: {e}")
            return None
    
    def _initialize_exchange_in_redis(self, exchange_name: str):
        """Initialize exchange tracking in Redis"""
        try:
            # Create empty set for humans
            humans_key = f"{self.redis_config.exchange_humans_prefix}{exchange_name}"
            self.redis_client.sadd(humans_key, "")  # Add empty string
            self.redis_client.srem(humans_key, "")  # Remove it (creates empty set)
            
            # Initialize stats
            stats_key = f"{self.redis_config.exchange_stats_prefix}{exchange_name}"
            stats = {
                "created_at": datetime.now().isoformat(),
                "total_assignments": 0,
                "last_assignment": ""
            }
            self.redis_client.hset(stats_key, mapping=stats)
            self.redis_client.expire(stats_key, self.redis_config.exchange_stats_ttl)
            
            logger.info(f"✅ Initialized Redis tracking for {exchange_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange in Redis: {e}")
            raise
    
    def _get_next_exchange_number(self) -> int:
        """Determine the next exchange number to create"""
        try:
            exchange_pattern = f"{self.redis_config.exchange_humans_prefix}*"
            exchange_keys = list(self.redis_client.scan_iter(match=exchange_pattern))
            existing_numbers = []
            
            # Extract numbers from existing exchange names
            for exchange_key in exchange_keys:
                exchange_name = exchange_key.replace(self.redis_config.exchange_humans_prefix, '')
                
                if exchange_name.startswith(self.rabbitmq_config.exchange_prefix):
                    number_part = exchange_name.replace(self.rabbitmq_config.exchange_prefix, '')
                    try:
                        number = int(number_part)
                        existing_numbers.append(number)
                    except ValueError:
                        logger.warning(f"⚠️ Invalid exchange name format: {exchange_name}")
            
            # Return next available number
            if existing_numbers:
                return max(existing_numbers) + 1
            else:
                return 1  # First exchange
                
        except Exception as e:
            logger.error(f"❌ Error determining next exchange number: {e}")
            return 1  # Fallback
    
    def _assign_human_atomically(self, human_id: str, exchange_name: str):
        """Assign human to exchange atomically"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Add human to exchange set
            pipe.sadd(f"exchange_humans:{exchange_name}", human_id)
            
            # Create human → exchange mapping
            pipe.setex(
                f"human_exchange:{human_id}",
                self.redis_config.human_assignment_ttl,
                exchange_name
            )
            
            # Update stats
            stats_key = f"exchange_stats:{exchange_name}"
            pipe.hincrby(stats_key, "total_assignments", 1)
            pipe.hset(stats_key, "last_assignment", datetime.now().isoformat())
            
            # Execute atomically
            pipe.execute()
            
            humans_count = self.redis_client.scard(f"exchange_humans:{exchange_name}")
            logger.info(f"👤 Atomically assigned {human_id} to {exchange_name} ({humans_count}/1000)")
            
        except Exception as e:
            logger.error(f"❌ Error in atomic assignment: {e}")
            raise
    
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