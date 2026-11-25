"""
Distributed Token Bucket / Sliding Window Rate Limiter.

Protects against Denial-of-Wallet (DoW) and API abuse.
Uses Redis when available, with in-memory fallback.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import HTTPException, Request, status

from src.config import settings

logger = logging.getLogger("hermes.security.rate_limiter")

_IN_MEMORY_BUCKETS: dict[str, list[float]] = {}


class RateLimiter:
    def __init__(self):
        self._redis = None

    @property
    def redis(self):
        if self._redis is None:
            try:
                import redis
                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            except Exception:
                self._redis = None
        return self._redis

    def is_rate_limited(self, identifier: str, max_requests: int = 60, window_seconds: int = 60) -> tuple[bool, int]:
        """Check if identifier exceeds rate limit in the current sliding window.
        Returns: (is_limited: bool, current_usage: int)
        """
        now = time.time()
        r = self.redis

        if r:
            try:
                key = f"hermes:ratelimit:{identifier}"
                pipe = r.pipeline()
                # Remove timestamps older than window
                pipe.zremrangebyscore(key, 0, now - window_seconds)
                # Count current valid requests
                pipe.zcard(key)
                # Add current request
                pipe.zadd(key, {f"{now}_{time.perf_counter()}": now})
                # Set TTL
                pipe.expire(key, window_seconds + 5)
                _, count, _, _ = pipe.execute()

                if count >= max_requests:
                    return True, count + 1
                return False, count + 1
            except Exception as exc:
                logger.warning(f"Redis rate limit check failed: {exc}. Falling back to in-memory limiter.")

        # In-memory sliding window fallback
        bucket = _IN_MEMORY_BUCKETS.setdefault(identifier, [])
        cutoff = now - window_seconds
        # Evict old timestamps
        _IN_MEMORY_BUCKETS[identifier] = [ts for ts in bucket if ts > cutoff]
        _IN_MEMORY_BUCKETS[identifier].append(now)

        current_count = len(_IN_MEMORY_BUCKETS[identifier])
        if current_count > max_requests:
            return True, current_count
        return False, current_count


rate_limiter = RateLimiter()


def rate_limit_dependency(max_rpm: int | None = None) -> Callable:
    """FastAPI Dependency for route-level rate limiting."""

    async def _check_rate_limit(request: Request):
        if not settings.rate_limit_enabled:
            return

        rpm = max_rpm or settings.rate_limit_rpm

        # Identify client via user_id header, API Key, or Client IP
        client_id = (
            request.headers.get("X-API-Key")
            or request.headers.get("X-User-Id")
            or (request.client.host if request.client else "unknown_client")
        )

        limited, usage = rate_limiter.is_rate_limited(client_id, max_requests=rpm, window_seconds=60)
        if limited:
            logger.warning(
                "rate_limit_exceeded",
                extra={"client": client_id, "usage": usage, "limit": rpm},
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {rpm} requests per minute allowed.",
                headers={"Retry-After": "60"},
            )

    return _check_rate_limit
