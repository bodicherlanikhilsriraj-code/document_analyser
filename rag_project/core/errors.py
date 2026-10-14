"""User-safe domain errors for StudyPilot AI."""


class StudyPilotError(Exception):
    """Base class for errors that can be shown as a friendly UI message."""


class ConfigurationError(StudyPilotError):
    """Invalid local application configuration."""


class DocumentError(StudyPilotError):
    """An uploaded document cannot be read or indexed."""


class EmbeddingError(StudyPilotError):
    """The local embedding model could not be loaded or used."""


class RetrievalError(StudyPilotError):
    """Vector storage or document retrieval failed."""


class OllamaUnavailableError(StudyPilotError):
    """The configured Ollama service could not be reached."""


class OllamaModelMissingError(StudyPilotError):
    """The requested model is not installed in Ollama."""


class OllamaTimeoutError(StudyPilotError):
    """Ollama did not finish generation before the configured timeout."""


class GenerationError(StudyPilotError):
    """The model returned an invalid or unsuccessful response."""


class WebSearchError(StudyPilotError):
    """External web search failed."""
