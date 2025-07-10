"""
LLM provider implementations for Excel translation.
Supports multiple providers with a unified interface.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

import openai
import anthropic
from src.config.config import config


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

    def _create_prompt(
        self, request: BatchTranslationRequest, intro_text: str = ""
    ) -> str:
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
- STRICTLY Return only the JSON object, no additional text"""

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
            logger.error("OpenAI translation error: %s", e)
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
            except ImportError as exc:
                raise ImportError(
                    "Anthropic package not installed. Run: pip install anthropic"
                ) from exc
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
            logger.error("Anthropic translation error: %s", e)
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


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM translation."""

    def __init__(self, model: str = "gemma3:27b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self._client = None

    def _get_client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                from ollama import Client

                self._client = Client(host=self.host)
            except ImportError as exc:
                raise ImportError(
                    "Ollama package not installed. Run: pip install ollama"
                ) from exc
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            client = self._get_client()
            # Check if the model is available by listing models
            models = client.list()

            # Debug log the structure of the response
            logger.debug("Ollama models response: %s", str(models)[:200] + "...")

            # Based on the debug output, models is a list of Model objects
            # Each Model object has a 'model' attribute with the model name
            if hasattr(models, "__iter__"):
                # Check if our model exists in the list
                for model_info in models:
                    # Handle Model objects
                    if hasattr(model_info, "model") and model_info.model == self.model:
                        logger.info("Found matching Ollama model: %s", self.model)
                        return True
                    # Handle dictionary format
                    elif (
                        isinstance(model_info, dict)
                        and "name" in model_info
                        and model_info["name"] == self.model
                    ):
                        logger.info("Found matching Ollama model: %s", self.model)
                        return True
                    # Handle string format
                    elif isinstance(model_info, str) and model_info == self.model:
                        logger.info("Found matching Ollama model: %s", self.model)
                        return True

                logger.warning(
                    "Model %s not found in available Ollama models", self.model
                )
                # If model doesn't exist, try to pull it
                logger.info(
                    "Attempting to pull model %s. This may take time...", self.model
                )
                try:
                    client.pull(self.model)
                    logger.info("Successfully pulled model %s", self.model)
                    return True
                except Exception as pull_error:
                    logger.error("Failed to pull model %s: %s", self.model, pull_error)
                    return False
            else:
                # Fallback: just check if Ollama is running
                logger.warning(
                    "Could not verify model availability, but Ollama is running"
                )
                return True
        except Exception as e:
            logger.error("Ollama availability check error: %s", e)
            return False

    def translate_batch(
        self, request: BatchTranslationRequest
    ) -> BatchTranslationResponse:
        """Translate batch using Ollama."""
        try:
            client = self._get_client()

            # Create prompt
            prompt = self._create_prompt(request)

            # Call Ollama API
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise Japanese to English translator. Preserve all formatting tokens and maintain context awareness. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,
                },
            )

            # Extract content from response
            content = response["message"]["content"]

            # Try to parse JSON from the response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If the response is not valid JSON, try to extract JSON from the text
                import re

                json_match = re.search(
                    r'\{\s*"translations"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL
                )
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        logger.error(
                            "Failed to extract valid JSON from Ollama response"
                        )
                        result = {"translations": []}
                else:
                    logger.error("No JSON found in Ollama response")
                    result = {"translations": []}

            # Convert to our format
            translations = [
                TranslationResponse(id=t["id"], text=t["text"])
                for t in result.get("translations", [])
            ]

            # Estimate token usage (Ollama doesn't provide this directly)
            # Rough estimate: 1 token ≈ 4 characters for English, 1-2 for Japanese
            prompt_tokens = len(prompt) // 3
            completion_tokens = len(content) // 4
            total_tokens = prompt_tokens + completion_tokens

            return BatchTranslationResponse(
                translations=translations,
                batch_id=request.batch_id,
                tokens_used=total_tokens,
            )

        except Exception as e:
            logger.error("Ollama translation error: %s", e)
            return BatchTranslationResponse(
                translations=[], batch_id=request.batch_id, tokens_used=0
            )


class VertexAIProvider(BaseLLMProvider):
    """Google Vertex AI provider for translation."""

    def __init__(self, project_id: str, location: str, model: str = "gemini-2.5-pro"):
        self.project_id = project_id
        self.location = location
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Vertex AI client."""
        if self._client is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                vertexai.init(project=self.project_id, location=self.location)
                self._client = GenerativeModel(self.model)
            except ImportError as exc:
                raise ImportError(
                    "Google Cloud AI Platform package not installed. Run: pip install google-cloud-aiplatform"
                ) from exc
        return self._client

    def is_available(self) -> bool:
        """Check if Vertex AI is available."""
        if not self.project_id or not self.location:
            logger.warning("Vertex AI project_id or location not configured.")
            return False
        try:
            if config.GOOGLE_APPLICATION_CREDENTIALS:
                logger.info(f"Vertex AI using service account from: {config.GOOGLE_APPLICATION_CREDENTIALS}")
            else:
                logger.info("Vertex AI using application default credentials (ADC)")
            self._get_client()
            return True
        except Exception as e:
            logger.error("Vertex AI availability check error: %s", e)
            return False

    def translate_batch(
        self, request: BatchTranslationRequest
    ) -> BatchTranslationResponse:
        """Translate batch using Vertex AI Gemini."""
        try:
            client = self._get_client()

            # Create prompt
            prompt = self._create_prompt(request)

            # Call Vertex AI API
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            )

            # Parse response
            content = response.text
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Vertex AI response is not valid JSON, attempting to extract.")
                json_match = re.search(
                    r'\{\s*"translations"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL
                )
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        logger.error("Failed to extract valid JSON from Vertex AI response")
                        result = {"translations": []}
                else:
                    logger.error("No JSON found in Vertex AI response")
                    result = {"translations": []}

            # Convert to our format
            translations = [
                TranslationResponse(id=t["id"], text=t["text"])
                for t in result.get("translations", [])
            ]

            tokens_used = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0

            return BatchTranslationResponse(
                translations=translations, batch_id=request.batch_id, tokens_used=tokens_used
            )
        except Exception as e:
            logger.error("Vertex AI translation error: %s", e)
            return BatchTranslationResponse(
                translations=[], batch_id=request.batch_id, tokens_used=0
            )


def create_llm_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """Factory function to create LLM providers."""
    provider_name = provider_name.lower()
    if provider_name == "openai":
        api_key = kwargs.get("api_key")
        model = kwargs.get("model", "gpt-4.1")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIProvider(api_key, model)

    elif provider_name == "anthropic":
        api_key = kwargs.get("api_key")
        model = kwargs.get("model", "claude-3-sonnet-20240229")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        return AnthropicProvider(api_key, model)

    elif provider_name == "ollama":
        # For Ollama, we need to use a model that's available locally
        # Default to gemma3:27b which is shown in the available models list
        model = kwargs.get("model", "gemma3:27b")
        host = kwargs.get("host", "http://localhost:11434")
        return OllamaProvider(model, host)

    elif provider_name in ["google", "vertexai"]:
        project_id = kwargs.get("project_id")
        location = kwargs.get("location")
        model = kwargs.get("model", "gemini-2.5-flash")
        if not project_id:
            raise ValueError("Google Cloud Project ID is required for Vertex AI")
        if not location:
            raise ValueError("Google Cloud Location is required for Vertex AI")
        return VertexAIProvider(project_id, location, model)

    elif provider_name == "mock":
        return MockProvider()

    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
