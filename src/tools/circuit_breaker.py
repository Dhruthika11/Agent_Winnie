"""
Distributed Circuit Breaker for Hermes Tool Registry.

Implements the standard 3-state Circuit Breaker pattern (CLOSED, OPEN, HALF-OPEN)
with optional Redis synchronization for distributed agent deployments.

Guarantees:
1. Fast-Fail: If an upstream tool service/subprocess fails repeatedly, subsequent calls
   fail immediately (<1ms) instead of waiting for timeouts.
2. Self-Healing: Allows probe executions in HALF-OPEN state once cooldown expires.
3. Multi-Node Sync: Distributes circuit states across agent replicas via Redis (with
   instant in-memory fallback if Redis is unavailable).
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Any

from src.config import settings

logger = logging.getLogger("hermes.tools.circuit_breaker")


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"      # Normal operation: traffic flows through
    OPEN = "OPEN"          # Tripped: fail-fast immediately, route to fallback
    HALF_OPEN = "HALF_OPEN"  # Testing recovery: probe request allowed through


class ServerCircuit:
    """In-memory circuit breaker state representation for a single tool server."""

    def __init__(
        self,
        server_name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        recovery_threshold: int = 1,
    ):
        self.server_name = server_name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.recovery_threshold = recovery_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_successes = 0
        self.tripped_at: float | None = None
        self.last_failure_at: float | None = None
        self.last_failure_reason: str | None = None
        self.last_success_at: float | None = None
        self.total_calls = 0
        self.total_failures = 0
        self.total_trips = 0

    def cooldown_remaining(self) -> float:
        if self.state != CircuitState.OPEN or self.tripped_at is None:
            return 0.0
        elapsed = time.time() - self.tripped_at
        return max(0.0, self.cooldown_seconds - elapsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "cooldown_remaining_seconds": round(self.cooldown_remaining(), 2),
            "tripped_at": self.tripped_at,
            "last_failure_reason": self.last_failure_reason,
            "last_failure_at": self.last_failure_at,
            "last_success_at": self.last_success_at,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_trips": self.total_trips,
        }


class ToolCircuitBreaker:
    """Distributed Tool Circuit Breaker with dual-layer storage (Redis + Local Memory)."""

    def __init__(
        self,
        default_failure_threshold: int = 3,
        default_cooldown_seconds: float = 30.0,
        redis_url: str | None = None,
    ):
        self.default_failure_threshold = default_failure_threshold
        self.default_cooldown_seconds = default_cooldown_seconds
        self.redis_url = redis_url or getattr(settings, "redis_url", "")
        self._lock = threading.Lock()
        self._circuits: dict[str, ServerCircuit] = {}
        self._redis_client = None
        self._init_redis()

    def _init_redis(self) -> None:
        if not self.redis_url:
            return
        try:
            import redis
            client = redis.Redis.from_url(self.redis_url, socket_timeout=1.0, decode_responses=True)
            client.ping()
            self._redis_client = client
            logger.info("ToolCircuitBreaker: Distributed Redis synchronization active.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"ToolCircuitBreaker: Redis unreachable ({exc.__class__.__name__}: {exc}). "
                "Operating in resilient standalone in-memory mode."
            )
            self._redis_client = None

    def _get_or_create(self, server_name: str) -> ServerCircuit:
        with self._lock:
            if server_name not in self._circuits:
                self._circuits[server_name] = ServerCircuit(
                    server_name=server_name,
                    failure_threshold=self.default_failure_threshold,
                    cooldown_seconds=self.default_cooldown_seconds,
                )
            circuit = self._circuits[server_name]

        # Sync from Redis if available
        if self._redis_client:
            try:
                pipe = self._redis_client.pipeline()
                pipe.get(f"hermes:circuit:{server_name}:state")
                pipe.get(f"hermes:circuit:{server_name}:tripped_at")
                pipe.get(f"hermes:circuit:{server_name}:reason")
                r_state, r_tripped, r_reason = pipe.execute()

                if r_state:
                    circuit.state = CircuitState(r_state)
                if r_tripped:
                    circuit.tripped_at = float(r_tripped)
                if r_reason:
                    circuit.last_failure_reason = r_reason
            except Exception:
                pass

        return circuit

    def can_execute(self, server_name: str) -> tuple[bool, str]:
        """Check if a tool call to the specified server should proceed or fail fast.

        Returns:
            (allow_execution, reason_string)
        """
        circuit = self._get_or_create(server_name)
        with self._lock:
            circuit.total_calls += 1

            if circuit.state == CircuitState.CLOSED:
                return True, "Circuit CLOSED: Healthy"

            if circuit.state == CircuitState.OPEN:
                remaining = circuit.cooldown_remaining()
                if remaining > 0:
                    return False, f"Circuit OPEN: Cooling down ({remaining:.1f}s remaining, Reason: {circuit.last_failure_reason})"

                # Cooldown expired -> Transition to HALF_OPEN for probe
                circuit.state = CircuitState.HALF_OPEN
                logger.info(
                    f"[CIRCUIT BREAKER] Cooldown expired for '{server_name}'. "
                    "Transitioning to HALF_OPEN. Permitting probe execution."
                )
                self._sync_to_redis(circuit)
                return True, "Circuit HALF_OPEN: Probe permitted"

            if circuit.state == CircuitState.HALF_OPEN:
                # In half-open, allow the probe
                return True, "Circuit HALF_OPEN: Probe in-flight"

        return True, "Default allow"

    def record_success(self, server_name: str) -> None:
        """Record successful execution of a tool on the server."""
        circuit = self._get_or_create(server_name)
        with self._lock:
            circuit.last_success_at = time.time()
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.consecutive_successes += 1
                if circuit.consecutive_successes >= circuit.recovery_threshold:
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    circuit.consecutive_successes = 0
                    circuit.tripped_at = None
                    circuit.last_failure_reason = None
                    logger.info(
                        f"[CIRCUIT BREAKER] Probe succeeded. Server '{server_name}' "
                        "has recovered. Circuit reset to CLOSED."
                    )
                    self._sync_to_redis(circuit)
            else:
                # Normal closed operation success
                circuit.failure_count = 0
                circuit.consecutive_successes += 1

    def record_failure(self, server_name: str, error_message: str) -> None:
        """Record a failure (timeout, network drop, process crash) on the server."""
        circuit = self._get_or_create(server_name)
        with self._lock:
            now = time.time()
            circuit.failure_count += 1
            circuit.total_failures += 1
            circuit.last_failure_at = now
            circuit.last_failure_reason = error_message[:200]
            circuit.consecutive_successes = 0

            # If failed during HALF_OPEN probe, trip immediately back to OPEN
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.OPEN
                circuit.tripped_at = now
                circuit.total_trips += 1
                logger.warning(
                    f"[CIRCUIT BREAKER] Probe failed for '{server_name}' ({error_message}). "
                    f"Tripping back to OPEN for {circuit.cooldown_seconds}s."
                )
                self._sync_to_redis(circuit)
                return

            # If failure threshold exceeded in CLOSED, trip to OPEN
            if circuit.state == CircuitState.CLOSED and circuit.failure_count >= circuit.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.tripped_at = now
                circuit.total_trips += 1
                logger.warning(
                    f"[CIRCUIT BREAKER] Failure threshold reached ({circuit.failure_count}/{circuit.failure_threshold}) "
                    f"for '{server_name}'. Circuit TRIPPED to OPEN. Entering {circuit.cooldown_seconds}s cooldown."
                )
                self._sync_to_redis(circuit)

    def reset(self, server_name: str) -> bool:
        """Manually reset a circuit to CLOSED."""
        circuit = self._get_or_create(server_name)
        with self._lock:
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.consecutive_successes = 0
            circuit.tripped_at = None
            circuit.last_failure_reason = None
            logger.info(f"[CIRCUIT BREAKER] Manually reset circuit for '{server_name}' to CLOSED.")
            self._sync_to_redis(circuit)
        return True

    def trip(self, server_name: str, reason: str = "Manual operator override") -> bool:
        """Manually trip a circuit to OPEN for testing or maintenance."""
        circuit = self._get_or_create(server_name)
        with self._lock:
            circuit.state = CircuitState.OPEN
            circuit.tripped_at = time.time()
            circuit.last_failure_reason = reason
            circuit.total_trips += 1
            logger.warning(f"[CIRCUIT BREAKER] Manually tripped circuit for '{server_name}' to OPEN ({reason}).")
            self._sync_to_redis(circuit)
        return True

    def get_status(self, server_name: str | None = None) -> dict[str, Any]:
        """Return diagnostic status of all or a specific circuit."""
        with self._lock:
            if server_name:
                circuit = self._circuits.get(server_name)
                return circuit.to_dict() if circuit else {"error": f"Server '{server_name}' not tracked"}

            return {
                "redis_synced": self._redis_client is not None,
                "circuits": {name: c.to_dict() for name, c in self._circuits.items()},
            }

    def _sync_to_redis(self, circuit: ServerCircuit) -> None:
        if not self._redis_client:
            return
        try:
            name = circuit.server_name
            pipe = self._redis_client.pipeline()
            pipe.set(f"hermes:circuit:{name}:state", circuit.state.value)
            pipe.set(f"hermes:circuit:{name}:failures", str(circuit.failure_count))
            if circuit.tripped_at:
                pipe.set(f"hermes:circuit:{name}:tripped_at", str(circuit.tripped_at))
            else:
                pipe.delete(f"hermes:circuit:{name}:tripped_at")
            if circuit.last_failure_reason:
                pipe.set(f"hermes:circuit:{name}:reason", circuit.last_failure_reason)
            else:
                pipe.delete(f"hermes:circuit:{name}:reason")
            pipe.execute()
        except Exception as exc:
            logger.debug(f"ToolCircuitBreaker: Redis sync error: {exc}")

    def get(self, server_name: str) -> ServerCircuitProxy:
        return ServerCircuitProxy(self, server_name)


class ServerCircuitProxy:
    """Convenience adapter offering an object-oriented view of a specific server circuit."""

    def __init__(self, breaker: ToolCircuitBreaker, server_name: str):
        self._breaker = breaker
        self.server_name = server_name

    def allow_request(self) -> bool:
        allowed, _ = self._breaker.can_execute(self.server_name)
        return allowed

    def record_success(self) -> None:
        self._breaker.record_success(self.server_name)

    def record_failure(self, error: str = "") -> None:
        self._breaker.record_failure(self.server_name, error)

    @property
    def is_open(self) -> bool:
        return not self.allow_request()


# Global Singleton Circuit Breaker
tool_circuit_breaker = ToolCircuitBreaker()
circuit_breaker_registry = tool_circuit_breaker
