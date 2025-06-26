# Excel Translation Agent

A multi-agent system for translating Japanese Excel files to English while preserving all original formatting, structure, and styles.

## Features

- **Format Preservation**: Maintains all Excel formatting, formulas, and structure
- **Multi-Layer Caching**: Redis + PostgreSQL for efficient translation reuse
- **Batch Processing**: Optimized API calls with configurable batch sizes
- **Multiple LLM Support**: OpenAI, Anthropic, and other providers
- **Context-Aware Translation**: Intelligent handling of headers, data, and formulas

## Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   docker-compose up -d  # Start Redis and PostgreSQL
   ```

2. **Configure API Keys**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. **Translate Excel File**
   ```bash
   python translate_cli.py input.xlsx -o output.xlsx --batch-size 10 -v
   ```

## Architecture

- **Core Engine**: `excel_translator.py` - Main translation logic using openpyxl
- **CLI Interface**: `translate_cli.py` - Command-line interface
- **Caching**: `cache_manager.py` - Multi-layer Redis + PostgreSQL caching
- **LLM Providers**: `llm_providers.py` - Multiple LLM integrations
- **Configuration**: `config/config.py` - Centralized settings

## Usage

```bash
python translate_cli.py [INPUT_FILE] [OPTIONS]

Options:
  -o, --output TEXT       Output file path
  --batch-size INTEGER    Cells per API call (default: 10)
  -v, --verbose          Enable verbose logging
  --provider TEXT        LLM provider (openai, anthropic)
```

## Requirements

- Python 3.8+
- Redis (for caching)
- PostgreSQL (optional, for persistent storage)
- OpenAI API key or other LLM provider

## License

MIT License
