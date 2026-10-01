"""Demo-grade role-based access and per-client rate limiting.

Two roles: ``customer`` (the UPI app: gets a band + reasons, never a score) and ``analyst`` (scores,
evidence, dispositions). Tokens come from the environment; the defaults are for the offline demo only.
Rate limiting is a per-token sliding window so a fraudster cannot map the decision boundary by
hammering the scoring endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import deque

DEFAULT_TOKENS = {"demo-customer-token": "customer", "demo-analyst-token": "analyst"}


def tokens() -> dict[str, str]:
    out = {}
    c, a = os.environ.get("CITADEL_CUSTOMER_TOKEN"), os.environ.get("CITADEL_ANALYST_TOKEN")
    out[c or "demo-customer-token"] = "customer"
    out[a or "demo-analyst-token"] = "analyst"
    return out


def role_of(authorization: str | None) -> tuple[str | None, str | None]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None, None
    tok = authorization.split(" ", 1)[1].strip()
    for known, role in tokens().items():
        if hmac.compare_digest(tok, known):
            return role, tok
    return None, None


class RateLimiter:
    def __init__(self, per_minute: dict[str, int]):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, key: str, role: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        limit = self.per_minute.get(role, 30)
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


def hmac_id(value: str, salt: str | None = None) -> str:
    """Salted HMAC of an identifier. The salt lives in the environment, never in the repo or the store."""
    key = (salt or os.environ.get("CITADEL_HMAC_SALT", "citadel-demo-salt-not-for-production")).encode()
    return hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()[:16]
