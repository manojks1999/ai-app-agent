"""
LLM model abstraction layer.

Implements Factory + Strategy patterns: abstract LLMModel protocol,
concrete implementations for OpenAI (GPT-4V/4o), Google Gemini, and Qwen.
ModelFactory.create() instantiates the correct model from config.
"""

from __future__ import annotations

import base64
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

from core.config import ModelConfig
from core.logger import logger


class ModelError(Exception):
    """Raised when an LLM request fails."""


@dataclass
class ModelResponse:
    """Structured response from an LLM."""
    success: bool
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def encode_image_base64(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class LLMModel(ABC):
    """Abstract base for LLM models (Strategy Pattern)."""

    @abstractmethod
    def get_response(self, prompt: str, images: List[str | Path]) -> ModelResponse:
        """
        Send a prompt with optional images to the LLM.

        Args:
            prompt: Text prompt.
            images: List of image file paths to include.

        Returns:
            ModelResponse with the LLM's output.
        """

    def get_response_with_retry(
        self,
        prompt: str,
        images: List[str | Path],
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> ModelResponse:
        """Send a prompt with retry logic and exponential backoff."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.get_response(prompt, images)
            except (ModelError, Exception) as e:
                last_error = e
                wait = backoff_factor ** attempt
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
        raise ModelError(f"All {max_retries} attempts failed. Last error: {last_error}")


class OpenAIModel(LLMModel):
    """OpenAI GPT-4V / GPT-4o model implementation."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            import openai
        except ImportError:
            raise ModelError("openai package not installed. Run: pip install openai")

        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
        )
        self.model = config.openai_api_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def get_response(self, prompt: str, images: List[str | Path]) -> ModelResponse:
        """Send a prompt with images to OpenAI's multimodal API."""
        content: list[dict] = [{"type": "text", "text": prompt}]

        for img_path in images:
            b64 = encode_image_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            # Approximate cost for GPT-4o
            cost = prompt_tokens / 1_000_000 * 2.50 + completion_tokens / 1_000_000 * 10.0

            result = response.choices[0].message.content or ""
            logger.info(
                f"OpenAI response: {prompt_tokens} prompt + {completion_tokens} "
                f"completion tokens (${cost:.4f})"
            )
            return ModelResponse(
                success=True,
                content=result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            )
        except Exception as e:
            raise ModelError(f"OpenAI API error: {e}") from e


class GeminiModel(LLMModel):
    """Google Gemini model implementation."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ModelError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

        genai.configure(api_key=config.gemini_api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(config.gemini_model)
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def get_response(self, prompt: str, images: List[str | Path]) -> ModelResponse:
        """Send a prompt with images to Gemini."""
        from PIL import Image

        parts: list = [prompt]
        for img_path in images:
            img = Image.open(str(img_path))
            parts.append(img)

        try:
            response = self.model.generate_content(
                parts,
                generation_config=self._genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            text = response.text or ""
            logger.info(f"Gemini response received ({len(text)} chars)")
            return ModelResponse(success=True, content=text)
        except Exception as e:
            raise ModelError(f"Gemini API error: {e}") from e


class QwenModel(LLMModel):
    """Alibaba Qwen-VL model implementation."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            import dashscope
        except ImportError:
            raise ModelError("dashscope not installed. Run: pip install dashscope")

        self._dashscope = dashscope
        dashscope.api_key = config.dashscope_api_key
        self.model = config.qwen_model

    def get_response(self, prompt: str, images: List[str | Path]) -> ModelResponse:
        """Send a prompt with images to Qwen-VL."""
        from http import HTTPStatus

        content: list[dict] = [{"text": prompt}]
        for img_path in images:
            content.append({"image": f"file://{img_path}"})

        messages = [{"role": "user", "content": content}]

        try:
            response = self._dashscope.MultiModalConversation.call(
                model=self.model, messages=messages
            )
            if response.status_code == HTTPStatus.OK:
                text = response.output.choices[0].message.content[0]["text"]
                logger.info(f"Qwen response received ({len(text)} chars)")
                return ModelResponse(success=True, content=text)
            else:
                raise ModelError(f"Qwen API error: {response.message}")
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Qwen API error: {e}") from e


class ModelFactory:
    """Factory for creating LLM model instances from config."""

    _registry: dict[str, type[LLMModel]] = {
        "openai": OpenAIModel,
        "gemini": GeminiModel,
        "qwen": QwenModel,
    }

    @classmethod
    def create(cls, config: ModelConfig) -> LLMModel:
        """
        Create an LLM model instance based on the provider in config.

        Args:
            config: Model configuration.

        Returns:
            An LLMModel instance.

        Raises:
            ModelError: If the provider is not supported.
        """
        provider = config.provider.lower()
        model_class = cls._registry.get(provider)
        if not model_class:
            supported = ", ".join(cls._registry.keys())
            raise ModelError(
                f"Unsupported model provider '{config.provider}'. "
                f"Supported: {supported}"
            )
        logger.info(f"Initializing {config.provider} model...")
        return model_class(config)

    @classmethod
    def register(cls, name: str, model_class: type[LLMModel]) -> None:
        """Register a new model provider."""
        cls._registry[name.lower()] = model_class
