import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from excel_translator import (
    ExcelTranslator,
    MultiLayerCache,
    RedisCache,
    PostgreSQLCache,
    OpenAILLMProvider,
)
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelTranslationMCPServer:
    def __init__(self):
        self.server = Server("excel-translation-agent")
        self.translator = None
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools for Excel translation."""
            return [
                Tool(
                    name="translate_excel",
                    description="Translate a Japanese Excel file to English while preserving formatting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input_path": {
                                "type": "string",
                                "description": "Path to the input Excel file"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path for the translated output file"
                            },
                            "batch_size": {
                                "type": "integer",
                                "description": "Number of cells to process per batch (default: 50)",
                                "default": 50
                            },
                            "use_cache": {
                                "type": "boolean",
                                "description": "Whether to use translation cache (default: true)",
                                "default": True
                            }
                        },
                        "required": ["input_path", "output_path"]
                    }
                ),
                Tool(
                    name="get_translation_status",
                    description="Get the status of translation cache and system health",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="clear_translation_cache",
                    description="Clear the translation cache",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "cache_type": {
                                "type": "string",
                                "enum": ["redis", "postgres", "all"],
                                "description": "Which cache to clear (default: all)",
                                "default": "all"
                            }
                        },
                        "required": []
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                if name == "translate_excel":
                    return await self._translate_excel(arguments)
                elif name == "get_translation_status":
                    return await self._get_status(arguments)
                elif name == "clear_translation_cache":
                    return await self._clear_cache(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True
                )

    async def _translate_excel(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle Excel translation requests."""
        input_path = arguments["input_path"]
        output_path = arguments["output_path"]
        batch_size = arguments.get("batch_size", 50)
        use_cache = arguments.get("use_cache", True)

        # Validate input file exists
        if not Path(input_path).exists():
            return CallToolResult(
                content=[TextContent(type="text", text=f"Input file not found: {input_path}")],
                isError=True
            )

        try:
            # Initialize translator if not already done
            if self.translator is None:
                self.translator = await self._initialize_translator(batch_size, use_cache)

            # Perform translation
            logger.info(f"Starting translation: {input_path} -> {output_path}")
            self.translator.translate_excel(input_path, output_path)

            result_message = f"Successfully translated {input_path} to {output_path}"
            logger.info(result_message)

            return CallToolResult(
                content=[TextContent(type="text", text=result_message)]
            )

        except Exception as e:
            error_message = f"Translation failed: {str(e)}"
            logger.error(error_message)
            return CallToolResult(
                content=[TextContent(type="text", text=error_message)],
                isError=True
            )

    async def _get_status(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get system status and cache information."""
        try:
            status = {
                "system": "healthy",
                "cache": {
                    "redis": "connected" if self._test_redis_connection() else "disconnected",
                    "postgres": "connected" if config.USE_POSTGRES and self._test_postgres_connection() else "disabled"
                },
                "config": {
                    "llm_provider": config.LLM_PROVIDER,
                    "batch_size": config.DEFAULT_BATCH_SIZE,
                    "use_postgres": config.USE_POSTGRES
                }
            }

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(status, indent=2))]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Status check failed: {str(e)}")],
                isError=True
            )

    async def _clear_cache(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Clear translation cache."""
        cache_type = arguments.get("cache_type", "all")

        try:
            cleared = []

            if cache_type in ["redis", "all"]:
                redis_cache = RedisCache(config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB)
                redis_cache.redis_client.flushdb()
                cleared.append("Redis")

            if cache_type in ["postgres", "all"] and config.USE_POSTGRES:
                postgres_cache = PostgreSQLCache(config.postgres_uri)
                with postgres_cache.conn.cursor() as cursor:
                    cursor.execute("DELETE FROM translations")
                    postgres_cache.conn.commit()
                cleared.append("PostgreSQL")

            message = f"Cleared cache: {', '.join(cleared)}"
            logger.info(message)

            return CallToolResult(
                content=[TextContent(type="text", text=message)]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Cache clear failed: {str(e)}")],
                isError=True
            )

    async def _initialize_translator(self, batch_size: int, use_cache: bool) -> ExcelTranslator:
        """Initialize the translator with proper configuration."""
        # Initialize LLM provider
        if config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")
            llm_provider = OpenAILLMProvider(config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

        # Initialize cache
        if use_cache:
            redis_cache = RedisCache(config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB)
            postgres_cache = None

            if config.USE_POSTGRES:
                postgres_cache = PostgreSQLCache(config.postgres_uri)

            cache = MultiLayerCache(redis_cache, postgres_cache)
        else:
            # Create a dummy cache that doesn't store anything
            class NoCache:
                def get_translation(self, text: str, context: str) -> Optional[str]:
                    return None
                def store_translation(self, text: str, context: str, translation: str):
                    pass
            cache = NoCache()

        return ExcelTranslator(llm_provider, cache, batch_size)

    def _test_redis_connection(self) -> bool:
        """Test Redis connection."""
        try:
            redis_cache = RedisCache(config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB)
            redis_cache.redis_client.ping()
            return True
        except Exception:
            return False

    def _test_postgres_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            postgres_cache = PostgreSQLCache(config.postgres_uri)
            with postgres_cache.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

async def main():
    """Main entry point for the MCP server."""
    server_instance = ExcelTranslationMCPServer()

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="excel-translation-agent",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
