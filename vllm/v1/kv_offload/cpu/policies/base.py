# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    block_id - index of the physical CPU buffer slot.
    """

    _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

    def __init__(self, block_id: int):
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1
        self.block_id = block_id

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the block is ready to be read.
        """
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """
    Encapsulates both block organization (data structures) and replacement
    decisions (which block to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.

    Subclasses *must* override ``__init__``; the constructor signature is
    ``(self, cache_capacity: int, **kwargs: Any)`` so that the hot-swap
    request body can pass an opaque ``policy_kwargs`` dict to evolved
    policies without ABC changes (see design/evolved_cpu_offloading.md §4).

    Subclasses *should* set ``POLICY_NAME`` and ``POLICY_VERSION`` class
    attributes for telemetry / reproducibility. They default to empty
    strings so canary policies and tests can subclass without boilerplate.
    """

    # Class-level metadata exposed by `GET /v1/offload_policy`. Populated
    # by built-ins; evolved policies should set these on their own subclass.
    POLICY_NAME: str = ""
    POLICY_VERSION: str = ""

    @abstractmethod
    def __init__(self, cache_capacity: int, **kwargs: Any) -> None: ...

    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None:
        """Find block in data structures. Returns None if not present."""

    @abstractmethod
    def insert(
        self,
        key: OffloadKey,
        block: BlockStatus,
        req_context: ReqContext | None = None,
    ) -> None:
        """Add a newly allocated block. For ARC: also removes from ghost lists.

        ``req_context`` carries per-request hints (priority, session id, etc.)
        sourced from ``sampling_params.extra_args["policy_hints"]``. Built-in
        policies ignore it; evolved policies opt in by reading it.
        """

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a block (used to clean up after a failed store).

        Intentionally context-free — `remove` is a manager-internal cleanup
        with no caller-meaningful request scope.
        """

    @abstractmethod
    def touch(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext | None = None,
    ) -> None:
        """Mark blocks as recently used."""

    @abstractmethod
    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
        req_context: ReqContext | None = None,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (key, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """

    # ------------------------------------------------------------------
    # State-transfer hooks for hot-swap (see design §4).
    # ------------------------------------------------------------------

    @abstractmethod
    def export_state(self) -> Iterable[tuple[OffloadKey, BlockStatus]]:
        """Iterate over all currently resident `(key, BlockStatus)` pairs.

        Called once on the *outgoing* policy during a hot-swap. Recency /
        ghost-list metadata is intentionally not preserved; only resident
        blocks transfer. Implementations must yield each resident block
        exactly once and must not mutate the policy.
        """

    def import_state(
        self, items: Iterable[tuple[OffloadKey, BlockStatus]]
    ) -> None:
        """Bulk-load resident blocks into a freshly constructed policy.

        Default implementation just calls `insert` per item without a
        request context. Subclasses may override for efficiency.
        """
        for key, block in items:
            self.insert(key, block)

    # ------------------------------------------------------------------
    # Write-error notification hook (see design §14.1 / §14.3).
    # ------------------------------------------------------------------

    def record_write_error(self) -> None:
        """Called by the manager when `_policy.insert` (or `remove` during
        a failed store) raises. The base implementation is a no-op so the
        manager can call this unconditionally without an `isinstance`
        check; `SupervisedCachePolicy` overrides it to advance its error
        budget.
        """
        return
