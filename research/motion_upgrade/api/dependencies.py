"""
FastAPI application-level dependencies (dependency injection).

Provides a shared InferenceService instance managed by the application
lifespan, so every endpoint receives the same pre-loaded model without
re-reading from disk.
"""

from api.inference import InferenceService

# Module-level singleton — populated during app lifespan startup
_inference_service: InferenceService | None = None


def set_inference_service(service: InferenceService) -> None:
    """Called once during application startup to inject the loaded service."""
    global _inference_service
    _inference_service = service


def get_inference_service() -> InferenceService:
    """FastAPI Depends() resolver. Returns the loaded InferenceService."""
    if _inference_service is None:
        raise RuntimeError(
            "InferenceService not initialised. "
            "The application lifespan did not complete startup."
        )
    return _inference_service
