"""
API Configuration — environment-driven settings for the inference service.

All values can be overridden via environment variables. Defaults point at
the motion-upgrade ST-GCN checkpoint so the service works out-of-the-box.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """Immutable configuration for the FastAPI inference service."""

    # Model — path to the ST-GCN .pt checkpoint
    model_path: str = os.environ.get(
        "MODEL_PATH", "models/motion_stgcn.pt"
    )
    # Device for PyTorch inference
    device: str = os.environ.get("DEVICE", "cpu")

    # Server
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "info")

    # Limits
    max_sequence_length: int = int(os.environ.get("MAX_SEQUENCE_LENGTH", "500"))
    min_sequence_length: int = int(os.environ.get("MIN_SEQUENCE_LENGTH", "1"))


def load_config() -> APIConfig:
    """Constructs an APIConfig from current environment variables."""
    return APIConfig()
