"""
LLM provider implementations for Excel translation.
Supports multiple providers with a unified interface.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """Single translation request item."""

    id: str
    text: str
    context: str


@dataclass
class TranslationResponse:
    """Single translation response item."""

    id: str
    text: str


@dataclass
class BatchTranslationRequest:
    """Batch translation request."""

    translations: List[TranslationRequest]
    batch_id: str
    metadata: Dict
    additional_context: str = ""


@dataclass
class BatchTranslationResponse:
    """Batch translation response."""

    translations: List[TranslationResponse]
    batch_id: str
    tokens_used: int = 0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def translate_batch(
        self, request: BatchTranslationRequest
    ) -> BatchTranslationResponse:
        """Translate a batch of text items."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass

    def _create_prompt(self, request: BatchTranslationRequest, intro_text: str = "") -> str:
        """Create translation prompt with common structure."""
        items_text = ""
        for item in request.translations:
            items_text += f'\n{{"id": "{item.id}", "text": "{item.text}", "context": "{item.context}"}}'

        additional_context_section = ""
        if request.additional_context:
            additional_context_section = f"""

Additional Context:
{request.additional_context}

Use this additional context to improve translation accuracy and maintain consistency with domain-specific terminology."""

        # Use default intro if none provided
        if not intro_text:
            intro_text = "Translate the following Japanese text items to English. Preserve all formatting and maintain context awareness."

        prompt = f"""{intro_text}{additional_context_section}

Input items:{items_text}

Return the translations in this exact JSON format:
{{
  "translations": [
    {{"id": "ITEM_ID", "text": "TRANSLATED_TEXT"}},
    ...
  ]
}}

Rules:
- Maintain original formatting (spaces, punctuation, etc.)
- Consider the context for accurate translation:
  * "header": Table headers, column names
  * "body": Regular cell content
  * "footer": Summary or footer content
  * "sheet_name": Excel sheet/tab names (keep concise and descriptive)
- Keep technical terms consistent
- For sheet names, use concise, professional English names
- Use the additional context provided to improve accuracy
- Return only the JSON object, no additional text"""

        return prompt


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider for translation."""

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            client = self._get_client()
            return bool(self.api_key and client)
        except Exception:
            return False

    def translate_batch(
        self, request: BatchTranslationRequest
    ) -> BatchTranslationResponse:
        """Translate batch using OpenAI."""
        try:
            client = self._get_client()

            # Create prompt
            prompt = self._create_prompt(request)

            # Call OpenAI API
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise Japanese to English translator. Preserve all formatting tokens and maintain context awareness. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)

            # Convert to our format
            translations = [
                TranslationResponse(id=t["id"], text=t["text"])
                for t in result.get("translations", [])
            ]

            return BatchTranslationResponse(
                translations=translations,
                batch_id=request.batch_id,
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )

        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return BatchTranslationResponse(
                translations=[], batch_id=request.batch_id, tokens_used=0
            )



class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider for translation."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        try:
            client = self._get_client()
            return bool(self.api_key and client)
        except Exception:
            return False

    def translate_batch(
        self, request: BatchTranslationRequest
    ) -> BatchTranslationResponse:
        """Translate batch using Anthropic Claude."""
        try:
            client = self._get_client()

            # Create prompt
            prompt = self._create_prompt(request)

            # Call Anthropic API
            response = client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            content = response.content[0].text
            result = json.loads(content)

            # Convert to our format
            translations = [
                TranslationResponse(id=t["id"], text=t["text"])
                for t in result.get("translations", [])
            ]

            return BatchTranslationResponse(
                translations=translations,
                batch_id=request.batch_id,
                tokens_used=(
                    response.usage.input_tokens + response.usage.output_tokens
                    if response.usage
                    else 0
                ),
            )

        except Exception as e:
            logger.error(f"Anthropic translation error: {e}")
            return BatchTranslationResponse(
                translations=[], batch_id=request.batch_id, tokens_used=0
            )



class MockProvider(BaseLLMProvider):
    """Mock provider for testing purposes."""

    def __init__(self):
        pass

    def is_available(self) -> bool:
        return True

    def translate_batch(
        self, request: BatchTranslationRequest
    ) -> BatchTranslationResponse:
        """Mock translation - just adds [TRANSLATED] prefix."""
        translations = [
            TranslationResponse(id=item.id, text=f"[TRANSLATED] {item.text}")
            for item in request.translations
        ]

        return BatchTranslationResponse(
            translations=translations,
            batch_id=request.batch_id,
            tokens_used=len(request.translations) * 10,  # Mock token count
        )


def create_llm_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """Factory function to create LLM providers."""
    if provider_name.lower() == "openai":
        api_key = kwargs.get("api_key")
        model = kwargs.get("model", "gpt-4.1")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIProvider(api_key, model)

    elif provider_name.lower() == "anthropic":
        api_key = kwargs.get("api_key")
        model = kwargs.get("model", "claude-3-sonnet-20240229")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        return AnthropicProvider(api_key, model)

    elif provider_name.lower() == "mock":
        return MockProvider()

    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
