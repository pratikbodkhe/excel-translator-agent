"""
Cache management for Excel translation system.
Provides multi-layer caching with Redis and PostgreSQL support.
"""

import hashlib
import logging
from typing import Optional
from abc import ABC, abstractmethod

import redis
import psycopg2

from src.config.config import config

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get_translation(self, text: str, context: str) -> Optional[str]:
        """Retrieve a translation from cache."""
        pass

    @abstractmethod
    def store_translation(self, text: str, context: str, translation: str):
        """Store a translation in cache."""
        pass


class RedisCache(BaseCache):
    """Redis-based cache implementation for fast lookups."""

    def __init__(self, host: str = None, port: int = None, db: int = None):
        self.host = host or config.REDIS_HOST
        self.port = port or config.REDIS_PORT
        self.db = db or config.REDIS_DB
        self.redis_client = redis.Redis(host=self.host, port=self.port, db=self.db)

    def get_translation(self, text: str, context: str) -> Optional[str]:
        """Get translation from Redis cache."""
        try:
            cache_key = self._generate_key(text, context)
            result = self.redis_client.get(cache_key)
            return result.decode('utf-8') if result else None
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    def store_translation(self, text: str, context: str, translation: str):
        """Store translation in Redis cache with TTL."""
        try:
            cache_key = self._generate_key(text, context)
            self.redis_client.set(cache_key, translation, ex=config.CACHE_EXPIRATION)
        except Exception as e:
            logger.warning(f"Redis store error: {e}")

    def _generate_key(self, text: str, context: str) -> str:
        """Generate cache key from text and context."""
        return f"translate:{hashlib.md5((text + context).encode()).hexdigest()}"

    def clear(self):
        """Clear all translations from Redis cache."""
        try:
            self.redis_client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class PostgreSQLCache(BaseCache):
    """PostgreSQL-based cache for persistent storage."""

    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or config.postgres_uri
        self.conn = psycopg2.connect(self.connection_string)
        self._init_db()

    def _init_db(self):
        """Initialize database table if it doesn't exist."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS translations (
                        cache_key VARCHAR(32) PRIMARY KEY,
                        original_text TEXT NOT NULL,
                        context TEXT NOT NULL,
                        translation TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.conn.commit()
        except Exception as e:
            logger.error(f"PostgreSQL init error: {e}")

    def get_translation(self, text: str, context: str) -> Optional[str]:
        """Get translation from PostgreSQL cache."""
        try:
            cache_key = self._generate_key(text, context)
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT translation FROM translations WHERE cache_key = %s",
                    (cache_key,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.warning(f"PostgreSQL get error: {e}")
            return None

    def store_translation(self, text: str, context: str, translation: str):
        """Store translation in PostgreSQL cache."""
        try:
            cache_key = self._generate_key(text, context)
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO translations (cache_key, original_text, context, translation)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE
                    SET translation = EXCLUDED.translation
                """, (cache_key, text, context, translation))
                self.conn.commit()
        except Exception as e:
            logger.warning(f"PostgreSQL store error: {e}")

    def _generate_key(self, text: str, context: str) -> str:
        """Generate cache key from text and context."""
        return hashlib.md5((text + context).encode()).hexdigest()

    def clear(self):
        """Clear all translations from PostgreSQL cache."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM translations")
                self.conn.commit()
            logger.info("PostgreSQL cache cleared")
        except Exception as e:
            logger.error(f"PostgreSQL clear error: {e}")


class MultiLayerCache:
    """Multi-layer cache with Redis and optional PostgreSQL fallback."""

    def __init__(self, redis_cache: RedisCache, postgres_cache: Optional[PostgreSQLCache] = None):
        self.redis_cache = redis_cache
        self.postgres_cache = postgres_cache

    def get_translation(self, text: str, context: str) -> Optional[str]:
        """Get translation from cache layers."""
        # Check Redis first
        translation = self.redis_cache.get_translation(text, context)
        if translation:
            return translation

        # Fallback to PostgreSQL if configured
        if self.postgres_cache:
            translation = self.postgres_cache.get_translation(text, context)
            if translation:
                # Store back in Redis for faster future access
                self.redis_cache.store_translation(text, context, translation)
                return translation

        return None

    def store_translation(self, text: str, context: str, translation: str):
        """Store translation in all cache layers."""
        # Store in Redis
        self.redis_cache.store_translation(text, context, translation)

        # Store in PostgreSQL if configured
        if self.postgres_cache:
            self.postgres_cache.store_translation(text, context, translation)

    def clear(self):
        """Clear all cache layers."""
        self.redis_cache.clear()
        if self.postgres_cache:
            self.postgres_cache.clear()


class NoCache:
    """Dummy cache that doesn't store anything - for testing or cache-disabled mode."""

    def get_translation(self, text: str, context: str) -> Optional[str]:
        return None

    def store_translation(self, text: str, context: str, translation: str):
        pass

    def clear(self):
        pass
