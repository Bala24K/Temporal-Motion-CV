"""Stdlib-only password hashing (PBKDF2-SHA256) and HS256 JWTs."""
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional


def hash_password(password: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, it, salt, dk = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(calc.hex(), dk)
    except Exception:
        return False


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_token(sub: int, secret: str, ttl: int) -> str:
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({"sub": sub, "exp": int(time.time()) + ttl}).encode())
    sig = _b64(hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


def decode_token(token: str, secret: str) -> Optional[int]:
    try:
        head, body, sig = token.split(".")
        good = _b64(hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        payload = json.loads(_unb64(body))
        return int(payload["sub"]) if payload["exp"] > time.time() else None
    except Exception:
        return None


import threading
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window limiter, in-process. Swap for Redis when running >1 replica."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, now: float = None) -> bool:
        now = time.time() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True

    def reset(self):
        with self.lock:
            self.hits.clear()


login_limiter = RateLimiter(limit=8, window=60.0)       # per (ip, email)
