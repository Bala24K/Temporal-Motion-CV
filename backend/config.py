import os
import secrets
from dataclasses import dataclass, field


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./fitness.db"))
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY") or secrets.token_urlsafe(32))
    token_ttl_seconds: int = int(os.getenv("TOKEN_TTL_SECONDS", 60 * 60 * 24 * 7))
    admin_emails: frozenset = frozenset(e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip())
    cookie_secure: bool = os.getenv("COOKIE_SECURE", "0") == "1"
    min_calibration_frames: int = 10
    max_frames_per_request: int = 3000


settings = Settings()
