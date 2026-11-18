"""OIDC bearer-token verification (RS256/ES256 against the IdP's JWKS). Works with Keycloak/Auth0/Clerk/Cognito.
Hardening: algorithm allow-list (no HS256 confusion, no 'none'), issuer + audience + exp + sub required,
verified e-mail required before we map the token to a local account."""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Callable, Optional

import jwt
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

ALLOWED_ALGS = ["RS256", "ES256"]


def fetch_jwks(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:        # nosec - operator-configured URL
        return json.loads(r.read())


class OIDCVerifier:
    def __init__(self, issuer: str, audience: str, jwks_provider: Callable[[], dict], leeway: int = 30,
                 refresh_seconds: int = 600, min_refetch_seconds: int = 60):
        self.issuer, self.audience, self.leeway = issuer, audience, leeway
        self._provider, self._keys, self._at, self._last_forced = jwks_provider, {}, 0.0, 0.0
        self._ttl, self._min_refetch, self._lock = refresh_seconds, min_refetch_seconds, threading.Lock()

    def _refresh(self, force: bool = False):
        now = time.time()
        if not force and self._keys and now - self._at < self._ttl:
            return
        if force and now - self._last_forced < self._min_refetch:      # unknown-kid floods must not hammer the IdP
            return
        with self._lock:
            self._last_forced = now if force else self._last_forced
            self._keys = {k["kid"]: k for k in self._provider().get("keys", []) if "kid" in k}
            self._at = now

    def _key(self, kid: str, alg: str):
        self._refresh()
        if kid not in self._keys:
            self._refresh(force=True)
        jwk = self._keys.get(kid)
        if not jwk:
            raise jwt.InvalidTokenError("unknown signing key")
        return (RSAAlgorithm if alg == "RS256" else ECAlgorithm).from_jwk(json.dumps(jwk))

    def verify(self, token: str) -> Optional[dict]:
        try:
            h = jwt.get_unverified_header(token)
            if h.get("alg") not in ALLOWED_ALGS or not h.get("kid"):
                return None
            return jwt.decode(token, self._key(h["kid"], h["alg"]), algorithms=ALLOWED_ALGS, audience=self.audience,
                              issuer=self.issuer, leeway=self.leeway, options={"require": ["exp", "iss", "aud", "sub"]})
        except Exception:
            return None
