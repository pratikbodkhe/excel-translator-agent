from datetime import datetime

import psycopg2
import redis

from config.config import config


class MemoryManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        self.postgres_conn = psycopg2.connect(config.postgres_uri) if config.USE_POSTGRES else None

    def sync_to_postgres(self):
        """Synchronize Redis cache to PostgreSQL database"""
        if not self.postgres_conn:
            return

        try:
            # Get all keys from Redis
            keys = self.redis_client.keys('*')
            with self.postgres_conn.cursor() as cursor:
                for key in keys:
                    value = self.redis_client.get(key)
                    # Extract original text and context from key
                    if ':' in key:
                        original, context = key.split(':', 1)
                    else:
                        original, context = key, ""

                    # Upsert into PostgreSQL
                    cursor.execute("""
                        INSERT INTO translations (original, context, translation, last_used)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (original, context)
                        DO UPDATE SET translation = EXCLUDED.translation, last_used = EXCLUDED.last_used
                    """, (original, context, value, datetime.now()))
            self.postgres_conn.commit()
        except Exception as e:
            print(f"Sync error: {e}")

    def close(self):
        """Close connections"""
        self.redis_client.close()
        if self.postgres_conn:
            self.postgres_conn.close()
