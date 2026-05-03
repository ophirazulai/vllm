# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PolicySwapRegistry — process-singleton orchestrator for the CPU-offloading
policy hot-swap path.

Lives in the scheduler process (same process as
`OffloadingConnectorScheduler`). One registry per process; entries are keyed
by `VllmConfig.instance_id` so multiple in-process engines and unit tests
do not share a swap target by accident.

Thread safety: `CPUOffloadingManager` runs only on the scheduler thread,
so the registry's `RLock` only serializes concurrent *external* swap
requests. See design §13.
"""

from __future__ import annotations

import dataclasses
import secrets
import sys
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies import metrics
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.loader import (
    LoadedPolicy,
    PolicyLoadError,
    discard_module,
)
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.policies.supervisor import SupervisedCachePolicy

if TYPE_CHECKING:
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Plain dataclasses (msgpack-safe). The HTTP / LLM layer turns them into
# JSON; cross-process IPC encodes them via msgspec without insecure pickle.
# ---------------------------------------------------------------------------


@dataclass
class ActivePolicy:
    engine_id: str
    generation: int
    policy_name: str
    policy_version: str
    source_hash: str
    source_origin: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SwapResult:
    ok: bool
    generation: int  # current generation after the call
    policy_name: str | None = None
    policy_version: str | None = None
    source_hash: str | None = None
    previous_generation: int = 0
    dry_run: bool = False
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class OffloadPolicyStats:
    lookups: int = 0
    hits: int = 0
    stores: int = 0
    evictions: int = 0
    loads: int = 0
    hit_rate: float = 0.0
    window_start_generation: int = 0
    active_generation: int = 0
    hint_keys: tuple[str, ...] = ()
    policy_errors: int = 0
    policy_rolled_back: bool = False
    rollback_generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Per-engine registry entry.
# ---------------------------------------------------------------------------


@dataclass
class PolicyRegistryEntry:
    manager_ref: weakref.ref | None = None  # type: ignore[type-arg]
    generation: int = 0
    active: ActivePolicy | None = None
    active_module_name: str | None = None
    supervisor: SupervisedCachePolicy | None = None
    enqueue_idle_callback: Callable[[Callable[[], None]], None] | None = None
    # Sticky-within-window state cleared only by `take_policy_stats(reset=True)`.
    cumulative_policy_errors: int = 0
    policy_rolled_back: bool = False
    rollback_generation: int = 0
    # Default-built-in metadata so `current()` returns useful data before
    # the first hot-swap. Populated by `attach()`.
    _builtin_name: str = "lru"
    _builtin_version: str = "builtin"


# ---------------------------------------------------------------------------
# Canary helpers — exercise the policy's full read+write surface before we
# install it. Runs on a separate empty instance so the canary's sentinel
# blocks never enter the real policy or the manager's block pool. See §6 step 5.
# ---------------------------------------------------------------------------


def _make_sentinel_key(byte: int) -> OffloadKey:
    # Different length than real keys (which are block_hash + 4-byte group_idx,
    # block_hash itself is typically 32 bytes), so collisions are impossible.
    return OffloadKey(secrets.token_bytes(48) + bytes([byte]))


def _run_canary(cls: type[CachePolicy], cache_capacity: int, kwargs: dict) -> None:
    """Execute the §6 step 5 canary on a freshly constructed instance.

    Raises on any failure. The caller is responsible for discarding the
    canary instance regardless of outcome.
    """
    canary = cls(cache_capacity=cache_capacity, **kwargs)

    # 1. empty-input edge cases
    canary.touch([])
    if canary.evict(0, set()) != []:
        raise RuntimeError("canary: evict(0, set()) must return []")

    # 2. empty cache: cannot satisfy
    if canary.evict(1, set()) is not None:
        raise RuntimeError("canary: evict(1, set()) on empty cache must return None")

    # 3. insert + get round-trip
    canary_key = _make_sentinel_key(0xC1)
    fake_block = BlockStatus(block_id=-1)
    canary.insert(canary_key, fake_block)
    if canary.get(canary_key) is not fake_block:
        raise RuntimeError("canary: get(canary_key) did not return inserted block")

    # 4. touch then re-get; mark block free (ref_cnt=0) so it's evictable later
    fake_block.ref_cnt = 0
    canary.touch([canary_key])
    if canary.get(canary_key) is not fake_block:
        raise RuntimeError("canary: get after touch did not return inserted block")

    # 5. protected-set short-circuit
    if canary.evict(1, {canary_key}) is not None:
        raise RuntimeError(
            "canary: evict(1, {canary_key}) must return None when only "
            "entry is protected"
        )

    # 6. successful eviction with a second sentinel
    evict_key = _make_sentinel_key(0xC2)
    evict_block = BlockStatus(block_id=-2)
    evict_block.ref_cnt = 0  # default ctor sets -1; explicitly mark evictable
    canary.insert(evict_key, evict_block)
    evicted = canary.evict(1, set())
    if evicted is None or len(evicted) != 1:
        raise RuntimeError(
            "canary: evict(1, set()) on 2-block cache must return 1 entry"
        )
    evicted_key = evicted[0][0]
    if evicted_key not in (canary_key, evict_key):
        raise RuntimeError("canary: evicted unrecognized key")
    if canary.get(evicted_key) is not None:
        raise RuntimeError("canary: evicted key still present")

    # 7. remove the surviving sentinel; both should be absent
    surviving_key = evict_key if evicted_key == canary_key else canary_key
    canary.remove(surviving_key)
    if canary.get(canary_key) is not None or canary.get(evict_key) is not None:
        raise RuntimeError("canary: removed key still present")

    # 8. with-context probe
    ctx = ReqContext(policy_hints={"_canary": True})
    ctx_key = _make_sentinel_key(0xC3)
    ctx_block = BlockStatus(block_id=-3)
    ctx_block.ref_cnt = 0
    canary.insert(ctx_key, ctx_block, req_context=ctx)
    canary.touch([ctx_key], req_context=ctx)
    if canary.get(ctx_key) is not ctx_block:
        raise RuntimeError("canary: get after with-context insert failed")
    ctx_evicted = canary.evict(1, set(), req_context=ctx)
    if ctx_evicted is None or len(ctx_evicted) != 1:
        raise RuntimeError("canary: with-context evict failed")
    if canary.get(ctx_key) is not None:
        raise RuntimeError("canary: with-context evicted key still present")


# ---------------------------------------------------------------------------
# Process-wide singleton.
# ---------------------------------------------------------------------------


class PolicySwapRegistry:
    _instance: PolicySwapRegistry | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def singleton(cls) -> PolicySwapRegistry:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, PolicyRegistryEntry] = {}

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def attach(
        self,
        engine_id: str,
        manager: CPUOffloadingManager,
        builtin_name: str = "lru",
        builtin_version: str = "builtin",
    ) -> None:
        with self._lock:
            entry = self._entries.get(engine_id)
            if entry is None:
                entry = PolicyRegistryEntry()
                self._entries[engine_id] = entry
            elif entry.manager_ref is not None and entry.manager_ref() is not None:
                logger.warning(
                    "PolicySwapRegistry.attach called twice for engine_id=%s; "
                    "replacing prior manager and dropping prior synthetic module",
                    engine_id,
                )
                discard_module(entry.active_module_name)
                entry.active_module_name = None
                entry.supervisor = None

            entry.manager_ref = weakref.ref(manager)
            entry._builtin_name = builtin_name
            entry._builtin_version = builtin_version
            if entry.active is None:
                entry.active = ActivePolicy(
                    engine_id=engine_id,
                    generation=entry.generation,
                    policy_name=builtin_name,
                    policy_version=builtin_version,
                    source_hash="builtin",
                    source_origin="builtin",
                )
            metrics.set_active_generation(engine_id, entry.generation)

    def attach_idle_callback(
        self,
        engine_id: str,
        enqueue_idle_callback: Callable[[Callable[[], None]], None],
    ) -> None:
        with self._lock:
            entry = self._entries.get(engine_id)
            if entry is None:
                # `attach()` may not have run yet — order-tolerant per §6.
                entry = PolicyRegistryEntry()
                self._entries[engine_id] = entry
            entry.enqueue_idle_callback = enqueue_idle_callback

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def current(self, engine_id: str) -> ActivePolicy | None:
        with self._lock:
            entry = self._entries.get(engine_id)
            return entry.active if entry is not None else None

    def stats(
        self, engine_id: str, reset: bool = False
    ) -> OffloadPolicyStats:
        with self._lock:
            entry = self._entries.get(engine_id)
            if entry is None or entry.manager_ref is None:
                return OffloadPolicyStats()
            manager = entry.manager_ref()
            if manager is None:
                return OffloadPolicyStats(
                    window_start_generation=entry.generation,
                    active_generation=entry.generation,
                    policy_errors=entry.cumulative_policy_errors,
                    policy_rolled_back=entry.policy_rolled_back,
                    rollback_generation=entry.rollback_generation,
                )

            snap = manager.take_policy_stats(
                reset=reset, active_generation=entry.generation
            )
            sup = entry.supervisor
            sup_errors = sup.errors if sup is not None else 0
            policy_errors = entry.cumulative_policy_errors + sup_errors

            stats = OffloadPolicyStats(
                lookups=int(snap["lookups"]),
                hits=int(snap["hits"]),
                stores=int(snap["stores"]),
                evictions=int(snap["evictions"]),
                loads=int(snap["loads"]),
                hit_rate=float(snap["hit_rate"]),
                window_start_generation=int(snap["window_start_generation"]),
                active_generation=int(snap["active_generation"]),
                hint_keys=tuple(snap["hint_keys"]),  # type: ignore[arg-type]
                policy_errors=policy_errors,
                policy_rolled_back=entry.policy_rolled_back,
                rollback_generation=entry.rollback_generation,
            )

            if reset:
                # Sticky-within-window fields are cleared only here.
                entry.cumulative_policy_errors = 0
                entry.policy_rolled_back = False
                entry.rollback_generation = 0
                if sup is not None:
                    sup._errors = 0  # noqa: SLF001 — co-owned with registry

            return stats

    # ------------------------------------------------------------------
    # The hot-swap algorithm — design §6.
    # ------------------------------------------------------------------

    def swap(
        self,
        engine_id: str,
        loaded: LoadedPolicy,
        policy_kwargs: dict[str, Any] | None = None,
        policy_name: str | None = None,
        policy_version: str | None = None,
        dry_run: bool = False,
    ) -> SwapResult:
        kwargs = policy_kwargs or {}
        t0 = time.perf_counter()

        with self._lock:
            entry = self._entries.get(engine_id)
            if entry is None or entry.manager_ref is None:
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("missing_engine", 0.0)
                return SwapResult(
                    ok=False,
                    generation=0,
                    previous_generation=0,
                    error="CPU offloading manager is not attached",
                )

            manager = entry.manager_ref()
            previous_generation = entry.generation
            if manager is None:
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("manager_gc", 0.0)
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    error="CPU offloading manager has been garbage collected",
                )

            # Step 0a: pre-rollback if a previous candidate has tripped.
            if entry.supervisor is not None and entry.supervisor.tripped:
                logger.info(
                    "Pre-rollback: previous policy generation %d tripped its "
                    "error budget; rolling back to built-in LRU before next swap.",
                    entry.generation,
                )
                self._cold_recover(engine_id, entry, manager)
                previous_generation = entry.generation

            # Step 1: snapshot resident blocks from the outgoing policy.
            old_policy = manager._policy  # noqa: SLF001
            try:
                state = list(old_policy.export_state())
            except Exception as e:  # noqa: BLE001
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("export_failed", _ms(t0))
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    latency_ms=_ms(t0),
                    error=f"outgoing policy export_state failed: {e}",
                )

            # Step 2: construct the candidate.
            try:
                new_policy = loaded.cls(
                    cache_capacity=manager._num_blocks, **kwargs  # noqa: SLF001
                )
            except Exception as e:  # noqa: BLE001
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("construct_failed", _ms(t0))
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    latency_ms=_ms(t0),
                    error=f"candidate __init__ raised: {e}",
                )

            # Step 3: migrate state into the candidate.
            try:
                new_policy.import_state(state)
            except Exception as e:  # noqa: BLE001
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("import_failed", _ms(t0))
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    latency_ms=_ms(t0),
                    error=f"candidate import_state raised: {e}",
                )

            # Step 5: canary (skip step 4: drain barrier — see design §6).
            try:
                _run_canary(loaded.cls, manager._num_blocks, kwargs)  # noqa: SLF001
            except Exception as e:  # noqa: BLE001
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("canary_failed", _ms(t0))
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    latency_ms=_ms(t0),
                    error=f"canary failed: {e}",
                )

            # Step 6: resolve metadata.
            resolved_name = (
                policy_name
                if policy_name is not None and policy_name != ""
                else getattr(loaded.cls, "POLICY_NAME", "") or ""
            )
            resolved_version = (
                policy_version
                if policy_version is not None and policy_version != ""
                else getattr(loaded.cls, "POLICY_VERSION", "") or ""
            )
            if not resolved_name or not resolved_version:
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("missing_metadata", _ms(t0))
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    latency_ms=_ms(t0),
                    error=(
                        "policy metadata missing: POLICY_NAME / POLICY_VERSION "
                        "must be non-empty (or override via name/version request "
                        "fields)"
                    ),
                )

            # Step 7: dry-run early return.
            if dry_run:
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("dry_run", _ms(t0))
                return SwapResult(
                    ok=True,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    policy_name=resolved_name,
                    policy_version=resolved_version,
                    source_hash=loaded.source_hash,
                    dry_run=True,
                    latency_ms=_ms(t0),
                )

            # Step 8: pointer flip.
            on_trip = self._make_on_trip(engine_id)
            supervisor = SupervisedCachePolicy(new_policy, on_trip=on_trip)
            try:
                manager._policy = supervisor  # noqa: SLF001
            except Exception as e:  # noqa: BLE001 — defensive
                discard_module(loaded.module.__name__)
                metrics.record_swap_result("flip_failed", _ms(t0))
                manager._policy = old_policy  # noqa: SLF001
                return SwapResult(
                    ok=False,
                    generation=previous_generation,
                    previous_generation=previous_generation,
                    latency_ms=_ms(t0),
                    error=f"pointer flip raised: {e}",
                )

            # Active state update — past the point of no rollback.
            prior_module = entry.active_module_name
            entry.generation += 1
            entry.supervisor = supervisor
            entry.active_module_name = loaded.module.__name__
            entry.active = ActivePolicy(
                engine_id=engine_id,
                generation=entry.generation,
                policy_name=resolved_name,
                policy_version=resolved_version,
                source_hash=loaded.source_hash,
                source_origin=loaded.source_origin,
            )

            # Synthetic-module hygiene: drop the previous active module so
            # long-running evolutionary loops don't leak `sys.modules`.
            discard_module(prior_module)

            metrics.record_swap_result("ok", _ms(t0))
            metrics.set_active_generation(engine_id, entry.generation)

            return SwapResult(
                ok=True,
                generation=entry.generation,
                previous_generation=previous_generation,
                policy_name=resolved_name,
                policy_version=resolved_version,
                source_hash=loaded.source_hash,
                latency_ms=_ms(t0),
            )

    # ------------------------------------------------------------------
    # Auto-rollback
    # ------------------------------------------------------------------

    def _make_on_trip(self, engine_id: str) -> Callable[[], None]:
        """Return a closure invoked exactly once when the supervisor trips.

        Reads `entry.enqueue_idle_callback` at trip time (not at supervisor
        construction) so a late-arriving `attach_idle_callback` still wires
        rollback. If still unset at trip time, the trip is recorded; the
        next external swap will perform recovery before installing a new
        candidate.
        """
        registry_ref = weakref.ref(self)

        def on_trip() -> None:
            self_ = registry_ref()
            if self_ is None:
                return
            with self_._lock:
                entry = self_._entries.get(engine_id)
                if entry is None or entry.enqueue_idle_callback is None:
                    return
                enqueue = entry.enqueue_idle_callback

            def idle_recover() -> None:
                self_inner = registry_ref()
                if self_inner is None:
                    return
                with self_inner._lock:
                    entry_inner = self_inner._entries.get(engine_id)
                    if entry_inner is None or entry_inner.manager_ref is None:
                        return
                    if entry_inner.supervisor is None:
                        # Already recovered (e.g. via the next-swap path).
                        return
                    if not entry_inner.supervisor.tripped:
                        return
                    manager = entry_inner.manager_ref()
                    if manager is None:
                        return
                    self_inner._cold_recover(engine_id, entry_inner, manager)

            try:
                enqueue(idle_recover)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to enqueue idle rollback callback: %s", e)

        return on_trip

    def _cold_recover(
        self,
        engine_id: str,
        entry: PolicyRegistryEntry,
        manager: CPUOffloadingManager,
    ) -> None:
        """Phase-1 cold rollback — install a fresh built-in LRU populated
        from the buggy candidate's exported residents. See design §14.2.

        Called under `self._lock`. Idempotent: no-ops if `entry.supervisor`
        has already been cleared by a concurrent recovery path.
        """
        if entry.supervisor is None:
            return

        rolled_back_module = entry.active_module_name
        rollback_generation = entry.generation
        supervisor = entry.supervisor

        residents: list[tuple[OffloadKey, BlockStatus]] = []
        export_failed = False
        try:
            for key, block in supervisor.inner.export_state():
                residents.append((key, block))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Cold-recovery: inner.export_state() raised: %s; "
                "installing empty LRU and rebuilding free list.",
                e,
            )
            export_failed = True

        # Validate residents against the manager's allocated block range.
        recovered_block_ids: set[int] = set()
        if not export_failed:
            seen: set[int] = set()
            valid_residents: list[tuple[OffloadKey, BlockStatus]] = []
            for key, block in residents:
                bid = block.block_id
                if bid < 0 or bid >= manager._num_allocated_blocks:  # noqa: SLF001
                    logger.warning(
                        "Cold-recovery: dropping resident with out-of-range "
                        "block_id=%d (allocated=%d).",
                        bid,
                        manager._num_allocated_blocks,  # noqa: SLF001
                    )
                    continue
                if bid in seen:
                    logger.warning(
                        "Cold-recovery: dropping duplicate resident block_id=%d.",
                        bid,
                    )
                    continue
                seen.add(bid)
                recovered_block_ids.add(bid)
                valid_residents.append((key, block))
            residents = valid_residents

        if export_failed:
            recovered = LRUCachePolicy(cache_capacity=manager._num_blocks)  # noqa: SLF001
            manager._free_list = list(  # noqa: SLF001
                range(manager._num_allocated_blocks)  # noqa: SLF001
            )
        else:
            recovered = LRUCachePolicy(cache_capacity=manager._num_blocks)  # noqa: SLF001
            try:
                recovered.import_state(residents)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Cold-recovery: LRUCachePolicy.import_state raised: %s; "
                    "installing empty LRU.",
                    e,
                )
                recovered = LRUCachePolicy(  # noqa: SLF001
                    cache_capacity=manager._num_blocks  # noqa: SLF001
                )
                recovered_block_ids = set()
            # Reconcile the block pool: any allocated slot not in the
            # recovered set is returned to the free list. See §14.2 step 5
            # in-flight caveat.
            manager._free_list = sorted(  # noqa: SLF001
                set(range(manager._num_allocated_blocks))  # noqa: SLF001
                - recovered_block_ids
            )

        manager._policy = recovered  # noqa: SLF001

        discard_module(rolled_back_module)
        entry.supervisor = None
        entry.active_module_name = None
        entry.generation += 1
        entry.active = ActivePolicy(
            engine_id=engine_id,
            generation=entry.generation,
            policy_name=LRUCachePolicy.POLICY_NAME,
            policy_version=LRUCachePolicy.POLICY_VERSION,
            source_hash="builtin",
            source_origin=f"rollback:{rollback_generation}",
        )

        entry.policy_rolled_back = True
        entry.rollback_generation = rollback_generation
        entry.cumulative_policy_errors += supervisor.errors

        metrics.record_rollback()
        metrics.record_policy_error(supervisor.errors)
        metrics.set_active_generation(engine_id, entry.generation)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


__all__ = [
    "ActivePolicy",
    "OffloadPolicyStats",
    "PolicyLoadError",
    "PolicyRegistryEntry",
    "PolicySwapRegistry",
    "SwapResult",
]


# Silence unused-import warnings for `field` / `sys` (kept for future hooks).
_ = field
_ = sys
