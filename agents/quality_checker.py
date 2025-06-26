from config.config import config

from .translator import Translator  # For context-based validation


class QualityChecker:
    def __init__(self):
        self.translator = Translator()

    def validate_translation(self, original, translation, context=""):
        """Validate translation quality with multiple checks"""
        if not translation or translation.strip() == "":
            return False, "Empty translation"

        # Check consistency with cached translations
        cached = self.translator.translate_text(original, context)
        if cached and cached != translation:
            return False, f"Inconsistent with cached: {cached}"

        # TODO: Add more validation rules
        # - Language-specific checks
        # - Format preservation
        # - Length constraints
        # - Custom rules based on context

        return True, "Validation passed"
