# Project Objective: Excel Translation Agent

This document outlines the requirements, architecture, and operational flow for the Excel Translation Agent, a multi-agent system designed to translate Japanese Excel files into English.

## 1. Core Mission

The primary goal is to create a robust, efficient, and scalable agentic system for translating Excel files in **parallel batches**. The system is designed to preserve all original file formatting, sheet order, and structure while minimizing LLM token consumption through a multi-layered caching strategy.

## 2. Key Features & Requirements

### Batch Translation Workflow

Multi-Agent Architecture for Excel Translation

1. Agent Roles
I propose creating the following specialized agents, be the better judge and see if we can optimize this further.

ExcelReader Agent: Responsible for reading Excel files, extracting content, and understanding the structure of the document.
Translator Agent: Core translation agent that leverages existing translation dictionaries and LLM capabilities to translate Japanese text to English.
MemoryManager Agent: Manages the translation memory database (Redis/DB), storing common translations to minimize LLM usage.
QualityChecker Agent: Validates translations for consistency, terminology, and context-awareness.
ExcelWriter Agent: Formats and writes the translated content back to Excel files, preserving structure and formatting.
2. Implementation Approach
We'll use CrewAI's YAML configuration for defining agents and tasks, and integrate with your existing translation code. The system will:

Use Excel MCP for reading/writing Excel files
Store common translations in Redis for quick retrieval
Only use LLM for new or complex translations
Create an MCP server to expose this functionality
3. Database Integration
For storing translations:

Use Redis for fast in-memory caching of common translations
Implement a simple key-value structure with Japanese text as the key and English translation as the value
Store context information alongside translations for better accuracy

The system uses a parallel batch processing model for high efficiency:

1. **File Analysis:** The pipeline first reads the input Excel file to identify all sheets and determine the total number of rows to be translated.
2. **Batch Creation:** Rows are grouped into batches (default size of 10). The number of parallel batches is configurable to balance performance and resource usage.
3. **Parallel Agent Execution:** For each batch, a dedicated `Batch Translator` agent is dynamically created and executed in a separate process. This allows multiple batches to be translated concurrently.
4. **Multi-Layered Caching:** Each agent checks for existing translations in the following order:
    a. **Redis Cache:** For fast, short-term lookups.
    b. **PostgreSQL Database:** For long-term, persistent storage (optional, can be disabled).
5. **LLM Fallback:** If a translation is not found in the cache, a configured LLM is used.
6. **Consolidated Write-Back:** After all parallel batches are complete, the results are consolidated and written to a single output file. The process ensures that original **formatting, sheet names, and sheet order are perfectly preserved**.

### Technical & Architectural Requirements

- **Excel Interaction:** All Excel file I/O is handled **exclusively by `openpyxl`**. This approach was chosen to ensure that all original formatting, including cell styles, merged cells, sheet order, and chart data, is perfectly preserved during the translation process. `pandas` is explicitly avoided.
- **Agent Framework:** Built on **Google Agent Development Kit (ADK)** to manage the multi-agent architecture and parallel processes.
- **Configurable LLMs:** The LLM provider is configurable via `config/config.py`.
- **Conditional Database:** PostgreSQL usage is controlled by a `USE_POSTGRES` flag, allowing the pipeline to run in a Redis-only mode.
- **Scalability:** The parallel architecture is designed to efficiently handle large files by distributing the workload across multiple CPU cores.

## 3. User Rules & Preferences

This system was developed with the following user-defined rules in mind:

- **Maximize Parallelism:** Always prefer concurrent execution for tasks. The batch processing architecture is a direct implementation of this rule.
- **Smart, Not Repetitive:** Avoid duplicating code and seek intelligent, reusable solutions.
- **Validate After Change:** Always test to ensure new code does not break existing functionality.
- **`openpyxl` Exclusivity:** Use `openpyxl` exclusively for all Excel operations. Do not use `pandas` or any external MCP tools for Excel I/O.

## 4. Technology Stack

- **Programming Language:** Python
- **Agent Framework:** Google Agent Development Kit (ADK)
- **Excel Interaction:** `openpyxl` or better
- **Caching & Database:** Redis, PostgreSQL
- **Containerization:** Docker, Docker Compose
- **LLM Integration:** OpenAI, Anthropic, etc.

Example command:

```bash
python batch_pipeline.py --input-file test_data/simple_test.xlsx
```

**Command-Line Arguments:**

- `--input-file`: (Required) Path to the source Excel file.
- `--output-file`: (Optional) Path for the translated output file. Defaults to `[input_file]_translated.xlsx`.
- `--verbose`: (Optional) Enables detailed logging of the agent and tool execution.
- `--validate`: (Optional) Runs a validation check on the output file after translation.

### Output

- The translated file will be saved in the `output` directory with the format `{original_filename}_translated.xlsx`.

## 6. Nuances & Troubleshooting

- **Redis Container:** The Redis service in `docker-compose.yml` requires a specific `command` with a `shell` entrypoint to start correctly. This was implemented to resolve persistent connection issues.


## 7. Future Development

- **MCP Server:** The final architecture will be exposed as a reusable tool via an MCP (Model Context Protocol) server, built using the MCP Python SDK.
