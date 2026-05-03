# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prometheus counters for the policy hot-swap path.

Lazy-initialized so importing this module costs nothing if `prometheus_client`
is missing or the feature is disabled. All counters are no-ops when the
Prometheus client is not importable; this matches the rest of vLLM's
metrics surface.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


_PROM: dict[str, Any] | None = None


def _ensure_prom() -> dict[str, Any] | None:
    global _PROM
    if _PROM is not None:
        return _PROM
    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:
        # No prometheus_client available — record-no-op shim.
        logger.debug(
            "prometheus_client not importable; policy hot-swap metrics disabled"
        )
        _PROM = {}
        return _PROM

    _PROM = {
        "swap_count": Counter(
            "vllm_policy_swap_count_total",
            "Number of policy hot-swap attempts.",
            ["result"],
        ),
        "swap_latency_ms": Histogram(
            "vllm_policy_swap_latency_ms",
            "Policy hot-swap latency in milliseconds.",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
        ),
        "policy_errors": Counter(
            "vllm_policy_errors_total",
            "Cumulative supervised CachePolicy errors recorded by the registry.",
        ),
        "policy_rollbacks": Counter(
            "vllm_policy_rollbacks_total",
            "Number of times a policy was rolled back to the built-in baseline.",
        ),
        "active_generation": Gauge(
            "vllm_policy_active_generation",
            "Generation number of the currently active policy per engine.",
            ["engine_id"],
        ),
    }
    return _PROM


def record_swap_result(result: str, latency_ms: float) -> None:
    prom = _ensure_prom()
    if not prom:
        return
    prom["swap_count"].labels(result=result).inc()
    prom["swap_latency_ms"].observe(latency_ms)


def record_policy_error(n: int = 1) -> None:
    prom = _ensure_prom()
    if not prom:
        return
    prom["policy_errors"].inc(n)


def record_rollback() -> None:
    prom = _ensure_prom()
    if not prom:
        return
    prom["policy_rollbacks"].inc()


def set_active_generation(engine_id: str, generation: int) -> None:
    prom = _ensure_prom()
    if not prom:
        return
    prom["active_generation"].labels(engine_id=engine_id).set(generation)


__all__ = [
    "record_policy_error",
    "record_rollback",
    "record_swap_result",
    "set_active_generation",
]
