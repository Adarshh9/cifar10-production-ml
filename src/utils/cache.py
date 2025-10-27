"""Redis caching utilities."""
import redis
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis cache wrapper."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: str = None,
        decode_responses: bool = True
    ):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            decode_responses: Whether to decode responses to strings
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        if not self.client:
            return
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache."""
        if self.client:
            try:
                self.client.delete(key)
            except Exception as e:
                logger.error(f"Cache delete error: {e}")
    
    def clear_all(self):
        """Clear all keys (use carefully!)."""
        if self.client:
            try:
                self.client.flushdb()
                logger.info("Cleared all cache")
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
