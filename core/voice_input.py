"""
Voice input module — the key new feature of VoiceAppAgent.

Implements the Strategy Pattern with two speech-to-text engines:
1. WhisperEngine — OpenAI Whisper API for high-quality cloud STT
2. LocalSpeechEngine — SpeechRecognition + Google free API for local/offline

VoiceInputManager provides a unified interface to get user input via
voice or keyboard fallback.
"""

from __future__ import annotations

import io
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from core.config import VoiceConfig
from core.logger import logger, print_colored


class VoiceError(Exception):
    """Raised when voice input fails."""


class SpeechEngine(ABC):
    """Abstract speech-to-text engine (Strategy Pattern)."""

    @abstractmethod
    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_data: Raw audio data (WAV format).
            language: Language code for transcription.

        Returns:
            Transcribed text string.
        """


class WhisperEngine(SpeechEngine):
    """OpenAI Whisper API-based speech engine (cloud)."""

    def __init__(self, api_key: str) -> None:
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise VoiceError("openai package not installed. Run: pip install openai")

    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """Transcribe audio using OpenAI Whisper API."""
        # Write audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = Path(tmp.name)

        try:
            with open(tmp_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                )
            text = response.text.strip()
            logger.info(f"Whisper transcribed: '{text}'")
            return text
        except Exception as e:
            raise VoiceError(f"Whisper API error: {e}") from e
        finally:
            tmp_path.unlink(missing_ok=True)


class LocalSpeechEngine(SpeechEngine):
    """Local speech engine using SpeechRecognition library (Google free API)."""

    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """Transcribe audio using Google's free Speech Recognition API."""
        try:
            import speech_recognition as sr
        except ImportError:
            raise VoiceError(
                "SpeechRecognition not installed. Run: pip install SpeechRecognition"
            )

        recognizer = sr.Recognizer()
        audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)

        try:
            text = recognizer.recognize_google(audio, language=language)
            logger.info(f"Local STT transcribed: '{text}'")
            return str(text).strip()
        except sr.UnknownValueError:
            raise VoiceError("Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            raise VoiceError(f"Speech recognition service error: {e}") from e


class VoiceInputManager:
    """
    Unified voice/text input manager.

    Provides a single get_input() method that:
    - Records audio from microphone if voice is enabled
    - Transcribes using the configured engine
    - Falls back to keyboard input on failure
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        self._engine: SpeechEngine | None = None

        if self.config.enabled:
            self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the speech engine based on config."""
        if self.config.engine == "whisper":
            # Whisper engine needs OpenAI API key (loaded separately)
            from core.config import load_config
            app_config = load_config()
            self._engine = WhisperEngine(api_key=app_config.model.openai_api_key)
            logger.info("Voice engine initialized: Whisper (cloud)")
        elif self.config.engine == "local":
            self._engine = LocalSpeechEngine()
            logger.info("Voice engine initialized: Local (Google free)")
        else:
            logger.warning(
                f"Unknown voice engine '{self.config.engine}', falling back to keyboard"
            )

    def _record_audio(self) -> bytes:
        """
        Record audio from the microphone.

        Returns:
            Raw audio data as bytes (WAV format).
        """
        try:
            import speech_recognition as sr
        except ImportError:
            raise VoiceError(
                "SpeechRecognition not installed. Run: pip install SpeechRecognition"
            )

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.config.energy_threshold

        try:
            with sr.Microphone(sample_rate=16000) as source:
                print_colored("🎤 Listening... (speak now)", "cyan")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(
                    source, timeout=self.config.timeout, phrase_time_limit=30
                )
                print_colored("✓ Audio captured, transcribing...", "green")
                return audio.get_wav_data()
        except sr.WaitTimeoutError:
            raise VoiceError(
                f"No speech detected within {self.config.timeout}s timeout"
            )
        except Exception as e:
            raise VoiceError(f"Microphone error: {e}") from e

    def listen(self) -> str:
        """
        Record from microphone and transcribe.

        Returns:
            Transcribed text string.

        Raises:
            VoiceError: If recording or transcription fails.
        """
        if not self._engine:
            raise VoiceError("Voice engine not initialized")

        audio_data = self._record_audio()
        return self._engine.transcribe(audio_data, language=self.config.language)

    def get_input(self, prompt: str = "") -> str:
        """
        Get user input via voice or keyboard.

        If voice is enabled, records from mic and transcribes.
        On voice failure, falls back to keyboard input.
        If voice is disabled, uses keyboard directly.

        Args:
            prompt: Text prompt to display to the user.

        Returns:
            User's input text.
        """
        if prompt:
            print_colored(prompt, "blue")

        if self.config.enabled and self._engine:
            print_colored(
                "  [Press ENTER to type instead, or speak your response]", "yellow"
            )
            try:
                text = self.listen()
                print_colored(f"  Heard: \"{text}\"", "green")

                # Confirm with user
                print_colored("  Accept? (y/n or press ENTER for yes): ", "yellow")
                confirm = input().strip().lower()
                if confirm in ("", "y", "yes"):
                    return text
                else:
                    print_colored("  Type your response instead:", "blue")
                    return input().strip()
            except VoiceError as e:
                logger.warning(f"Voice input failed: {e}")
                print_colored(f"  Voice failed: {e}", "yellow")
                print_colored("  Type your response instead:", "blue")
                return input().strip()
        else:
            return input().strip()

    def get_choice(self, prompt: str, valid_choices: list[str]) -> str:
        """
        Get a validated choice from voice or keyboard input.

        Args:
            prompt: Text prompt explaining the choices.
            valid_choices: List of valid responses.

        Returns:
            One of the valid choices.
        """
        while True:
            response = self.get_input(prompt).strip().lower()
            if response in [c.lower() for c in valid_choices]:
                return response
            print_colored(
                f"  Invalid choice '{response}'. Valid: {valid_choices}", "red"
            )
