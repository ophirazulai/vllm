# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SupervisedCachePolicy — buggy-policy protection wrapper.

Wraps an evolved `CachePolicy` so that exceptions on read-side / suggestion
ops degrade to safe defaults rather than crash the engine. State-mutating
ops (`insert`, `remove`) re-raise so that manager-level hardening
(`CPUOffloadingManager` §14.3) can roll back partial writes coherently.

When the error budget is exhausted, the supervisor calls `on_trip()` once.
The registry uses that hook to schedule cold-rollback to a built-in LRU.

See `design/evolved_cpu_offloading.md` §14.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

logger = init_logger(__name__)


class SupervisedCachePolicy(CachePolicy):
    """Wraps an `inner` `CachePolicy` and counts/suppresses exceptions.

    Constructed inline by the registry (§6 step 8) — never via
    `PolicyLoader`. The widened-ABC signature `(cache_capacity, **kwargs)`
    is intentionally not honored: the supervisor takes its `inner` as a
    constructor argument because Python only requires the abstract method
    to be overridden, not signature-compatible with the base ABC.
    """

    POLICY_NAME = "supervisor"
    POLICY_VERSION = "builtin"

    def __init__(
        self,
        inner: CachePolicy,
        error_budget: int = 8,
        on_trip: Callable[[], None] | None = None,
    ) -> None:  # type: ignore[override]
        # Note: deliberately does NOT call super().__init__ — the wrapped
        # `inner` already owns its data structures, and the supervisor has
        # no `cache_capacity` of its own.
        self._inner = inner
        self._errors = 0
        self._budget = error_budget
        self._on_trip = on_trip
        self._tripped = False

    # ------------------------------------------------------------------
    # Diagnostic accessors used by the registry / stats path.
    # ------------------------------------------------------------------

    @property
    def inner(self) -> CachePolicy:
        return self._inner

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def budget(self) -> int:
        return self._budget

    # ------------------------------------------------------------------
    # Internal — record an error, trip on budget exhaustion.
    # ------------------------------------------------------------------

    def _bump(self) -> None:
        """Increment error count and trip on budget exhaustion. Internal."""
        self._errors += 1
        if not self._tripped and self._errors >= self._budget:
            self._tripped = True
            cb = self._on_trip
            if cb is not None:
                try:
                    cb()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Supervisor on_trip callback raised: %s", e)

    def _record(self, where: str, err: BaseException) -> None:
        logger.warning(
            "Supervised CachePolicy.%s raised (%d/%d): %s",
            where,
            self._errors + 1,
            self._budget,
            err,
        )
        self._bump()

    # Manager-side write-error notification — design §14.3.
    def record_write_error(self) -> None:
        self._bump()

    # ------------------------------------------------------------------
    # Read-side / suggestion ops: degrade to safe defaults on exception.
    #
    # The manager already has graceful-degradation contracts that absorb
    # these defaults:
    #   - `lookup` returns False when `_policy.get(...) is None`
    #     → degraded gets become misses
    #   - `prepare_store` returns None when `_policy.evict(...)` returns
    #     None → "couldn't free a slot this step", retried next step
    #   - `touch` is advisory; suppressing it only loses recency info
    # ------------------------------------------------------------------

    def get(self, key: OffloadKey) -> BlockStatus | None:
        try:
            block = self._inner.get(key)
        except Exception as e:  # noqa: BLE001
            self._record("get", e)
            return None
        if block is not None and not isinstance(block, BlockStatus):
            self._record(
                "get",
                TypeError(f"expected BlockStatus | None, got {type(block).__name__}"),
            )
            return None
        return block

    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
        req_context: ReqContext | None = None,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        try:
            evicted = self._inner.evict(n, protected, req_context)
        except Exception as e:  # noqa: BLE001
            self._record("evict", e)
            return None
        if evicted is None:
            return None
        error = self._validate_eviction_result(n, protected, evicted)
        if error is not None:
            self._record("evict", error)
            return None
        return evicted

    def touch(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext | None = None,
    ) -> None:
        try:
            self._inner.touch(keys, req_context)
        except Exception as e:  # noqa: BLE001
            self._record("touch", e)

    # ------------------------------------------------------------------
    # State-mutating ops: propagate exceptions so the manager can undo
    # partial writes (design §14.3). The manager calls
    # `record_write_error()` after an undo.
    # ------------------------------------------------------------------

    def insert(
        self,
        key: OffloadKey,
        block: BlockStatus,
        req_context: ReqContext | None = None,
    ) -> None:
        self._inner.insert(key, block, req_context)

    def remove(self, key: OffloadKey) -> None:
        self._inner.remove(key)

    # ------------------------------------------------------------------
    # State-transfer hooks: forward to the inner so the *next* swap
    # (which sees `manager._policy` as a `SupervisedCachePolicy`) can
    # still snapshot resident blocks during §6 step 1.
    # ------------------------------------------------------------------

    def export_state(self) -> Iterable[tuple[OffloadKey, BlockStatus]]:
        return self._inner.export_state()

    def import_state(self, items: Iterable[tuple[OffloadKey, BlockStatus]]) -> None:
        self._inner.import_state(items)

    @staticmethod
    def _validate_eviction_result(
        n: int,
        protected: set[OffloadKey],
        evicted: object,
    ) -> TypeError | None:
        if not isinstance(evicted, list):
            return TypeError(
                "expected evict() to return list[tuple[OffloadKey, BlockStatus]] "
                f"| None, got {type(evicted).__name__}"
            )
        if len(evicted) != n:
            return TypeError(
                f"expected evict() to return exactly {n} entries, got {len(evicted)}"
            )
        for item in evicted:
            if not isinstance(item, tuple) or len(item) != 2:
                return TypeError(
                    "expected each evict() entry to be (OffloadKey, BlockStatus)"
                )
            key, block = item
            if not isinstance(key, bytes):
                return TypeError(
                    "expected evict() key to be OffloadKey/bytes, "
                    f"got {type(key).__name__}"
                )
            if key in protected:
                return TypeError("evict() returned a protected key")
            if not isinstance(block, BlockStatus):
                return TypeError(
                    "expected evict() block to be BlockStatus, "
                    f"got {type(block).__name__}"
                )
            if block.ref_cnt != 0:
                return TypeError(
                    f"expected evict() block to have ref_cnt == 0, got {block.ref_cnt}"
                )
        return None


__all__ = ["SupervisedCachePolicy"]
