import psycopg2
import redis

from agents.excel_reader import ExcelReader  # Use absolute import
from config.config import config


class Translator:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        self.postgres_conn = None
        if config.USE_POSTGRES:
            self.postgres_conn = psycopg2.connect(config.postgres_uri)

    def translate_text(self, text, context=""):
        """Translate text using multi-layered caching strategy"""
        # First try Redis cache
        cache_key = f"{text}:{context}"
        translated = self.redis_client.get(cache_key)
        if translated:
            return translated

        # Then try PostgreSQL if enabled
        if config.USE_POSTGRES and self.postgres_conn:
            try:
                with self.postgres_conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT translation FROM translations WHERE original = %s AND context = %s",
                        (text, context)
                    )
                    result = cursor.fetchone()
                    if result:
                        translated = result[0]
                        # Cache in Redis for future use
                        self.redis_client.setex(cache_key, config.CACHE_EXPIRATION, translated)
                        return translated
            except Exception as e:
                print(f"Database error: {e}")

        # Fallback to LLM translation (placeholder)
        translated = self._translate_with_llm(text, context)

        # Cache the result
        self.redis_client.setex(cache_key, config.CACHE_EXPIRATION, translated)
        if config.USE_POSTGRES and self.postgres_conn:
            self._save_to_postgres(text, context, translated)

        return translated

    def _translate_with_llm(self, text, context):
        """Placeholder for actual LLM translation"""
        # In real implementation, this would call the configured LLM
        print(f"Translating with LLM: {text} (context: {context})")
        return f"Translated: {text}"

    def _save_to_postgres(self, text, context, translation):
        try:
            with self.postgres_conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO translations (original, context, translation) VALUES (%s, %s, %s)",
                    (text, context, translation)
                )
            self.postgres_conn.commit()
        except Exception as e:
            print(f"Error saving to database: {e}")

    def close(self):
        if self.postgres_conn:
            self.postgres_conn.close()
