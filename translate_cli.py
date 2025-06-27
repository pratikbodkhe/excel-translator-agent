#!/usr/bin/env python3
"""
Simple CLI for Excel Translation Agent
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.cache_manager import MultiLayerCache, RedisCache, PostgreSQLCache, NoCache
from src.llm_providers import BaseLLMProvider, create_llm_provider
from src.excel_translator import ExcelTranslator
from src.config.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_cache():
    """Setup cache based on configuration."""
    try:
        # Try Redis first
        redis_cache = RedisCache(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB
        )

        # Test Redis connection
        redis_cache.redis_client.ping()
        logger.info("Redis cache connected successfully")

        # Setup PostgreSQL if enabled
        postgres_cache = None
        if config.USE_POSTGRES:
            try:
                postgres_cache = PostgreSQLCache(config.postgres_uri)
                logger.info("PostgreSQL cache connected successfully")
            except Exception as e:
                logger.warning(f"PostgreSQL cache failed, using Redis only: {e}")

        return MultiLayerCache(redis_cache, postgres_cache)

    except Exception as e:
        logger.warning(f"Cache setup failed, using no cache: {e}")
        return NoCache()


def setup_llm_provider():
    """Setup LLM provider based on configuration."""
    try:
        provider: BaseLLMProvider = create_llm_provider(
            provider_name=config.LLM_PROVIDER,
            api_key=config.OPENAI_API_KEY if config.LLM_PROVIDER == "openai" else config.ANTHROPIC_API_KEY,
            model=config.LLM_MODEL
        )

        if not provider.is_available():
            logger.error(f"LLM provider {config.LLM_PROVIDER} is not available")
            return None

        logger.info(f"LLM provider {config.LLM_PROVIDER} setup successfully")
        return provider

    except Exception as e:
        logger.error(f"LLM provider setup failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Excel Translation Agent CLI")
    parser.add_argument("input_file", help="Input Excel file path")
    parser.add_argument("-o", "--output", help="Output Excel file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for translation")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM provider for testing")
    parser.add_argument("--context", help="Path to context.md file for additional translation context")
    parser.add_argument("--clean", action="store_true", help="Clear all caches (Redis and PostgreSQL) before translation")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if not input_path.suffix.lower() in ['.xlsx', '.xls']:
        logger.error(f"Input file must be an Excel file (.xlsx or .xls)")
        return 1

    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_translated{input_path.suffix}"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Setup components
    cache = setup_cache()

    # Clear caches if requested
    if args.clean:
        logger.info("Clearing all caches...")
        cache.clear()
        logger.info("Caches cleared successfully")

    # Load additional context if provided
    additional_context = ""
    if args.context:
        context_path = Path(args.context)
        if context_path.exists():
            try:
                additional_context = context_path.read_text(encoding='utf-8')
                logger.info(f"Loaded additional context from {context_path} ({len(additional_context)} characters)")
            except Exception as e:
                logger.warning(f"Failed to read context file {context_path}: {e}")
        else:
            logger.warning(f"Context file not found: {context_path}")

    if args.mock:
        from src.llm_providers import MockProvider
        llm_provider = MockProvider()
        logger.info("Using mock LLM provider")
    else:
        llm_provider = setup_llm_provider()
        if not llm_provider:
            logger.error("Failed to setup LLM provider")
            return 1

    # Create translator
    translator = ExcelTranslator(
        llm_provider=llm_provider,
        cache=cache,
        batch_size=args.batch_size,
        additional_context=additional_context
    )

    # Perform translation
    try:
        logger.info("Starting translation...")
        logger.info(f"Using LLM provider: {type(llm_provider).__name__}")
        logger.info(f"LLM provider available: {llm_provider.is_available()}")
        translator.translate_excel(str(input_path), str(output_path))
        logger.info(f"Translation completed successfully: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
