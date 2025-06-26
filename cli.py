import argparse
import time

from agents.excel_reader import ExcelReader
from agents.excel_writer import ExcelWriter
from agents.memory_manager import MemoryManager
from agents.quality_checker import QualityChecker
from agents.translator import Translator
from config.config import config


def main():
    parser = argparse.ArgumentParser(description='Excel Translation Agent')
    parser.add_argument('input', help='Input Excel file path')
    parser.add_argument('output', help='Output Excel file path')
    parser.add_argument('--context', default="", help='Translation context')
    parser.add_argument('--batch-size', type=int, default=config.DEFAULT_BATCH_SIZE,
                        help='Number of cells to process in a batch')
    parser.add_argument('--preserve-formatting', action='store_true',
                        help='Preserve original cell formatting')
    args = parser.parse_args()

    print("Starting Excel translation...")
    start_time = time.time()

    # Initialize components
    reader = ExcelReader(args.input)
    writer = ExcelWriter(args.input)  # Use input as template
    translator = Translator()
    checker = QualityChecker()
    memory_manager = MemoryManager()

    if not reader.load_workbook():
        print("Error loading workbook")
        return

    print(f"Loaded workbook with sheets: {reader.sheet_names}")

    # Process each sheet
    for sheet_name in reader.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        data = reader.get_sheet_data(sheet_name)

        if not data:
            print(f"Skipping empty sheet: {sheet_name}")
            continue

        translated_data = []
        batch = []

        # Process cells in batches
        for row_idx, row in enumerate(data):
            translated_row = []
            for col_idx, cell in enumerate(row):
                if args.preserve_formatting:
                    cell_copy = cell.copy()
                else:
                    cell_copy = {'value': cell['value']}

                # Add to batch for translation
                batch.append((cell['value'], args.context, cell_copy))

                # Process batch when full
                if len(batch) >= args.batch_size:
                    translated_batch = _process_batch(batch, translator, checker)
                    translated_row.extend(translated_batch)
                    batch = []

            # Process remaining items in batch
            if batch:
                translated_batch = _process_batch(batch, translator, checker)
                translated_row.extend(translated_batch)
                batch = []

            translated_data.append(translated_row)

        # Write translated sheet
        writer.write_translated_sheet(sheet_name, translated_data)
        print(f"Completed sheet: {sheet_name}")

    # Save final workbook
    writer.save(args.output)
    print(f"Saved translated workbook to: {args.output}")

    # Synchronize cache to database
    memory_manager.sync_to_postgres()

    # Cleanup resources
    reader.close()
    translator.close()
    memory_manager.close()

    duration = time.time() - start_time
    print(f"Translation completed in {duration:.2f} seconds")

def _process_batch(batch, translator, checker):
    """Process a batch of cells through translation and validation"""
    results = []
    for original, context, cell in batch:
        # Translate text
        translated = translator.translate_text(original, context)

        # Validate translation
        valid, message = checker.validate_translation(original, translated, context)
        if not valid:
            print(f"Validation warning: {message} for '{original}'")

        # Update cell with translated value
        cell['value'] = translated
        results.append(cell)
    return results

if __name__ == '__main__':
    main()
