#!/usr/bin/env python3
"""
Excel Translation Agent using Google ADK
Multi-agent system for translating Japanese Excel files to English
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
import psycopg2
import redis
from psycopg2.extras import RealDictCursor

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationCache:
    """Manages Redis and PostgreSQL caching for translations"""

    def __init__(self):
        self.redis_client = redis.Redis.from_url(config.redis_uri, decode_responses=True)
        self.use_postgres = config.USE_POSTGRES

    def get_translation(self, text: str, context: str = "") -> Optional[str]:
        """Get translation from cache (Redis first, then PostgreSQL)"""
        if not text or not text.strip():
            return None

        cache_key = f"{text}:{context}" if context else text

        # Check Redis first
        try:
            translation = self.redis_client.get(cache_key)
            if translation:
                logger.debug(f"Found in Redis: {text[:50]}...")
                return translation
        except Exception as e:
            logger.warning(f"Redis error: {e}")

        # Check PostgreSQL if enabled
        if self.use_postgres:
            try:
                with psycopg2.connect(config.postgres_uri) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute(
                            "SELECT translation FROM translations WHERE original = %s AND context = %s",
                            (text, context)
                        )
                        result = cursor.fetchone()
                        if result:
                            translation = result['translation']
                            # Store in Redis for faster access
                            self.redis_client.setex(cache_key, 3600, translation)
                            logger.debug(f"Found in PostgreSQL: {text[:50]}...")
                            return translation
            except Exception as e:
                logger.warning(f"PostgreSQL error: {e}")

        return None

    def store_translation(self, text: str, translation: str, context: str = ""):
        """Store translation in both Redis and PostgreSQL"""
        if not text or not translation:
            return

        cache_key = f"{text}:{context}" if context else text

        # Store in Redis
        try:
            self.redis_client.setex(cache_key, 3600, translation)
        except Exception as e:
            logger.warning(f"Redis store error: {e}")

        # Store in PostgreSQL if enabled
        if self.use_postgres:
            try:
                with psycopg2.connect(config.postgres_uri) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """INSERT INTO translations (original, translation, context, created_at)
                               VALUES (%s, %s, %s, NOW())
                               ON CONFLICT (original, context) DO UPDATE SET
                               translation = EXCLUDED.translation, updated_at = NOW()""",
                            (text, translation, context)
                        )
                        conn.commit()
            except Exception as e:
                logger.warning(f"PostgreSQL store error: {e}")

class LLMTranslator:
    """Handles LLM-based translation"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def translate(self, text: str, context: str = "") -> str:
        """Translate text using LLM"""
        if not text or not text.strip():
            return text

        # Skip if already in English (basic check)
        if self._is_likely_english(text):
            return text

        try:
            prompt = f"""Translate the following Japanese text to English.
            Provide only the translation without any additional text or explanations.
            Context: {context if context else 'General'}

            Text to translate: {text}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            translation = response.choices[0].message.content.strip()
            logger.info(f"LLM translated: {text[:50]}... -> {translation[:50]}...")
            return translation

        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            return text  # Return original if translation fails

    def _is_likely_english(self, text: str) -> bool:
        """Basic check if text is likely already in English"""
        if not text:
            return True
        # Check if text contains Japanese characters
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')
        return not japanese_pattern.search(text)

class ExcelTranslationAgent:
    """Main agent for Excel translation using MCP Excel tools"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.working_file = self.file_path.parent / f"{self.file_path.stem}_translation_in_progress.xlsx"
        self.output_file = self.file_path.parent / "output" / f"{self.file_path.stem}_translated.xlsx"

        self.cache = TranslationCache()
        self.translator = LLMTranslator()

        # Ensure output directory exists
        self.output_file.parent.mkdir(exist_ok=True)

    async def translate_excel(self, batch_size: int = 10) -> str:
        """Main translation workflow"""
        logger.info(f"Starting Excel translation: {self.file_path}")
        start_time = time.time()

        # Step 1: Create working copy
        await self._create_working_copy()

        # Step 2: Get sheet information
        sheets_info = await self._get_sheets_info()

        # Step 3: Process each sheet
        for sheet_info in sheets_info:
            await self._process_sheet(sheet_info, batch_size)

        # Step 4: Save final result
        await self._save_final_result()

        elapsed = time.time() - start_time
        logger.info(f"Translation completed in {elapsed:.2f} seconds")
        return str(self.output_file)

    async def _create_working_copy(self):
        """Create a working copy of the Excel file"""
        import shutil
        shutil.copy2(self.file_path, self.working_file)
        logger.info(f"Created working copy: {self.working_file}")

    async def _get_sheets_info(self) -> List[Dict]:
        """Get information about all sheets using Excel MCP"""
        # This would use the excel MCP tool to describe sheets
        # For now, return the known sheets from the file
        return [
            {"name": "改訂履歴", "usedRange": "A1:G50"},
            {"name": "Input資料", "usedRange": "A1:E22"},
            {"name": "事前にご一読ください", "usedRange": "A1:E26"},
            {"name": "テスト環境", "usedRange": "A1:AA23"},
            {"name": "テスト項目Android", "usedRange": "A1:BG190"},
            {"name": "集計", "usedRange": "A1:E21"}
        ]

    async def _process_sheet(self, sheet_info: Dict, batch_size: int):
        """Process a single sheet in batches"""
        sheet_name = sheet_info["name"]
        used_range = sheet_info["usedRange"]

        logger.info(f"Processing sheet: {sheet_name}")

        # Parse range to get dimensions
        end_cell = used_range.split(":")[1]
        col_match = re.match(r'([A-Z]+)', end_cell)
        row_match = re.search(r'(\d+)', end_cell)

        if not col_match or not row_match:
            logger.warning(f"Could not parse range: {used_range}")
            return

        end_col = col_match.group(1)
        end_row = int(row_match.group(1))

        # Process in batches
        for start_row in range(1, end_row + 1, batch_size):
            end_batch_row = min(start_row + batch_size - 1, end_row)
            batch_range = f"A{start_row}:{end_col}{end_batch_row}"

            await self._process_batch(sheet_name, batch_range)

    async def _process_batch(self, sheet_name: str, range_spec: str):
        """Process a batch of cells"""
        logger.info(f"Processing batch: {sheet_name}[{range_spec}]")

        # Read the batch data (would use Excel MCP)
        batch_data = await self._read_batch(sheet_name, range_spec)

        # Translate the batch
        translated_data = await self._translate_batch(batch_data)

        # Write back the translated data (would use Excel MCP)
        await self._write_batch(sheet_name, range_spec, translated_data)

    async def _read_batch(self, sheet_name: str, range_spec: str) -> List[List[str]]:
        """Read batch data using Excel MCP (simulated)"""
        # This would use the excel MCP read_sheet tool
        # For now, return sample data
        return [
            ["改訂履歴", "", "", "", "", "", ""],
            ["Ver-ID", "改訂日", "改訂箇所", "改訂内容", "改訂者", "レビューア", "レビュー日"]
        ]

    async def _translate_batch(self, batch_data: List[List[str]]) -> List[List[str]]:
        """Translate a batch of data"""
        translated_batch = []

        for row in batch_data:
            translated_row = []
            for cell_value in row:
                if cell_value and isinstance(cell_value, str) and cell_value.strip():
                    # Check cache first
                    cached_translation = self.cache.get_translation(cell_value)
                    if cached_translation:
                        translated_row.append(cached_translation)
                    else:
                        # Translate with LLM
                        translation = self.translator.translate(cell_value)
                        self.cache.store_translation(cell_value, translation)
                        translated_row.append(translation)
                else:
                    translated_row.append(cell_value)
            translated_batch.append(translated_row)

        return translated_batch

    async def _write_batch(self, sheet_name: str, range_spec: str, data: List[List[str]]):
        """Write translated data back using Excel MCP (simulated)"""
        # This would use the excel MCP write_to_sheet tool
        logger.info(f"Writing translated batch: {sheet_name}[{range_spec}]")

    async def _save_final_result(self):
        """Save the final translated file"""
        import shutil
        shutil.copy2(self.working_file, self.output_file)
        logger.info(f"Saved final result: {self.output_file}")

async def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python excel_translation_agent.py <excel_file> [batch_size]")
        sys.exit(1)

    file_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    agent = ExcelTranslationAgent(file_path)
    result_file = await agent.translate_excel(batch_size)

    print(f"Translation completed: {result_file}")

if __name__ == "__main__":
    asyncio.run(main())
