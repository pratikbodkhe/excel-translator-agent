import openpyxl
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from src.cache_manager import MultiLayerCache, RedisCache, PostgreSQLCache, NoCache
from src.llm_providers import (
    BaseLLMProvider,
    TranslationRequest,
    BatchTranslationRequest,
    BatchTranslationResponse,
    create_llm_provider
)
from src.config.config import config

logger = logging.getLogger(__name__)

@dataclass
class CellData:
    sheet_name: str
    cell_ref: str
    content: str
    context: str = ""  # e.g., "header", "footer", etc.

class LLMBatchProcessor:
    def __init__(self, llm_provider: BaseLLMProvider, max_batch_size: int = 50):
        self.llm_provider = llm_provider
        self.max_batch_size = max_batch_size

    def translate_batch(self, batch: List[CellData]) -> Dict[str, str]:
        # Prepare data for LLM using the new format
        translation_requests = [
            TranslationRequest(
                id=f"{cell.sheet_name}!{cell.cell_ref}",
                text=cell.content,
                context=cell.context
            )
            for cell in batch
        ]

        batch_request = BatchTranslationRequest(
            translations=translation_requests,
            batch_id=hashlib.md5(json.dumps([c.__dict__ for c in batch]).encode()).hexdigest(),
            metadata={
                "file_hash": hashlib.md5(json.dumps([c.__dict__ for c in batch]).encode()).hexdigest(),
                "sheet_name": batch[0].sheet_name if batch else ""
            }
        )

        # Send to LLM and get response
        response = self.llm_provider.translate_batch(batch_request)

        # Convert response to dict mapping cell IDs to translations
        return {
            t.id: t.text
            for t in response.translations
        }

class ExcelTranslator:
    def __init__(self, llm_provider, cache, batch_size: int = 50):
        self.llm_provider = llm_provider
        self.cache = cache
        self.batch_size = batch_size
        self.batch_processor = LLMBatchProcessor(llm_provider, batch_size)

    def _should_translate_cell(self, cell) -> bool:
        # Skip empty cells, numbers, formulas
        if cell is None or cell.value is None:
            return False
        if isinstance(cell.value, (int, float)):
            return False
        if isinstance(cell.value, str) and cell.value.startswith('='):
            return False
        return True

    def _get_cell_context(self, sheet, cell) -> str:
        # Determine if cell is a header, footer, or regular cell
        if cell.row == 1:
            return "header"
        elif cell.row == sheet.max_row:
            return "footer"
        else:
            return "body"

    def _get_batch_for_sheet(self, sheet) -> List[List[CellData]]:
        """Group cells into batches for translation."""
        batch = []
        batches = []

        for row in sheet.iter_rows():
            for cell in row:
                if self._should_translate_cell(cell):
                    context = self._get_cell_context(sheet, cell)
                    cell_data = CellData(
                        sheet_name=sheet.title,
                        cell_ref=cell.coordinate,
                        content=str(cell.value),
                        context=context
                    )
                    batch.append(cell_data)

                    if len(batch) >= self.batch_size:
                        batches.append(batch)
                        batch = []

        if batch:  # Add any remaining cells as the last batch
            batches.append(batch)

        return batches

    def _translate_batch(self, batch: List[CellData]) -> Dict[str, str]:
        """Translate a batch of cells."""
        logger.debug(f"Translating batch of {len(batch)} cells")

        # Check cache first
        translations = {}
        to_translate = []

        for cell in batch:
            cached = self.cache.get_translation(cell.content, cell.context)
            if cached:
                logger.debug(f"Cache hit for '{cell.content}': {cached}")
                translations[f"{cell.sheet_name}!{cell.cell_ref}"] = cached
            else:
                logger.debug(f"Cache miss for '{cell.content}', adding to LLM batch")
                to_translate.append(cell)

        if to_translate:
            logger.debug(f"Sending {len(to_translate)} cells to LLM")
            # Process with LLM
            llm_translations = self.batch_processor.translate_batch(to_translate)
            logger.debug(f"LLM returned {len(llm_translations)} translations")

            for cell_id, translation in llm_translations.items():
                logger.debug(f"LLM translation: {cell_id} -> {translation}")

            translations.update(llm_translations)

            # Store in cache
            for cell in to_translate:
                cell_id = f"{cell.sheet_name}!{cell.cell_ref}"
                if cell_id in llm_translations:
                    self.cache.store_translation(
                        cell.content,
                        cell.context,
                        llm_translations[cell_id]
                    )

        return translations

    def translate_excel(self, input_path: str, output_path: str):
        """Main translation method that handles the entire process."""
        # Load workbook
        wb = openpyxl.load_workbook(input_path)

        # Create a copy of the workbook to preserve formatting
        wb_copy = openpyxl.Workbook()
        wb_copy.remove(wb_copy.active)  # Remove default sheet

        for sheet_name in wb.sheetnames:
            original_sheet = wb[sheet_name]
            new_sheet = wb_copy.create_sheet(sheet_name)

            # Copy styles, formatting, etc.
            for row_idx, row in enumerate(original_sheet.iter_rows(), start=1):
                for col_idx, cell in enumerate(row, start=1):
                    new_cell = new_sheet.cell(row=row_idx, column=col_idx)

                    # Copy formatting
                    if cell.has_style:
                        new_cell.font = cell.font.copy()
                        new_cell.border = cell.border.copy()
                        new_cell.fill = cell.fill.copy()
                        new_cell.number_format = cell.number_format
                        new_cell.protection = cell.protection.copy()
                        new_cell.alignment = cell.alignment.copy()

                    # Copy value if not text (handles formulas, numbers, etc.)
                    if not self._should_translate_cell(cell):
                        new_cell.value = cell.value
                    else:
                        # We'll translate this cell later in batches
                        pass

            # Process sheet in batches
            batches = self._get_batch_for_sheet(original_sheet)
            for batch in batches:
                translations = self._translate_batch(batch)
                for cell in batch:
                    cell_id = f"{cell.sheet_name}!{cell.cell_ref}"
                    if cell_id in translations:
                        # Find the corresponding cell in the new sheet
                        # Cell coordinates are like "A1" - we need to convert to row,col
                        from openpyxl.utils import coordinate_to_tuple
                        row, col = coordinate_to_tuple(cell.cell_ref)
                        new_sheet.cell(row=row, column=col).value = translations[cell_id]

            # Copy sheet properties and dimensions (simplified)
            try:
                new_sheet.sheet_format = original_sheet.sheet_format
                # Copy basic properties without using .copy() method
                if hasattr(original_sheet, 'sheet_properties'):
                    new_sheet.sheet_properties.tabColor = original_sheet.sheet_properties.tabColor
                if hasattr(original_sheet, 'page_setup'):
                    new_sheet.page_setup.orientation = original_sheet.page_setup.orientation
                    new_sheet.page_setup.paperSize = original_sheet.page_setup.paperSize
            except Exception as e:
                logger.warning(f"Could not copy sheet properties: {e}")

        # Save the translated workbook
        wb_copy.save(output_path)
        logger.info(f"Translation complete. Saved to {output_path}")


class MCPAdapter:
    """Adapts the translator to work as an MCP server."""
    def __init__(self, translator: ExcelTranslator):
        self.translator = translator

    def handle_translation_request(self, input_path: str, output_path: str):
        try:
            self.translator.translate_excel(input_path, output_path)
            return {"status": "success", "message": "Translation complete"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
