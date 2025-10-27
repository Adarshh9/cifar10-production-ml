"""Redis caching utilities with Railway support."""
import redis
import json
import logging
from typing import Any, Optional
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis cache wrapper with Railway support."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        password: str = None,
        db: int = 0,
        decode_responses: bool = True
    ):
        # Support Railway's REDIS_URL
        redis_url = os.getenv('REDIS_URL')
        
        if redis_url:
            logger.info("Using Railway REDIS_URL")
            try:
                parsed = urlparse(redis_url)
                self.client = redis.Redis(
                    host=parsed.hostname or host,
                    port=parsed.port or port,
                    password=parsed.password or password,
                    db=db,
                    decode_responses=decode_responses,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self.client.ping()
                logger.info(f"✅ Connected to Redis (Railway)")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.client = None
        else:
            # Local development
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    decode_responses=decode_responses,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
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
