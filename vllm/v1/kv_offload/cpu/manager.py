# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection, Iterable
from contextlib import suppress
from typing import Literal

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

logger = init_logger(__name__)

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
    ):
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)

        # --- counters consumed by `take_policy_stats` (see design §8.3) ---
        self._stat_lookups: int = 0
        self._stat_hits: int = 0
        self._stat_stores: int = 0
        self._stat_evictions: int = 0
        self._stat_loads: int = 0
        # Window in which stats are accumulated. Updated by the registry on
        # each forward swap and by `take_policy_stats(reset=True)`.
        self._stat_window_start_generation: int = 0
        # Top-level keys observed in `req_context.policy_hints` since last
        # reset. Used as a smoke signal that a CORAL client shim actually
        # populated hints; values are not exposed.
        self._stat_hint_keys: set[str] = set()

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    def _record_hint_keys(self, req_context: ReqContext | None) -> None:
        if req_context is None or req_context.policy_hints is None:
            return
        # `policy_hints` is shallow-copied at request init; iterating its
        # keys is safe and bounded by what the user supplied. Keep only
        # string keys so telemetry collection cannot fail on mixed key types.
        for key in req_context.policy_hints:
            if isinstance(key, str):
                self._stat_hint_keys.add(key)

    # --- OffloadingManager interface ---

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        self._stat_lookups += 1
        self._record_hint_keys(req_context)
        block = self._policy.get(key)
        is_hit = block is not None and block.is_ready
        if is_hit:
            self._stat_hits += 1
        return is_hit

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        self._record_hint_keys(req_context)
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            block.ref_cnt += 1
            blocks.append(block)
        self._stat_loads += len(blocks)
        return self._get_load_store_spec(keys, blocks)

    def touch(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        self._record_hint_keys(req_context)
        self._policy.touch(keys, req_context)

    def complete_load(self, keys: Collection[OffloadKey]) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        self._record_hint_keys(req_context)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected, req_context)
            if evicted is None:
                return None

            malformed_reason: str | None = None
            seen_evicted_keys: set[OffloadKey] = set()
            seen_evicted_block_ids: set[int] = set()
            if not isinstance(evicted, list):
                malformed_reason = (
                    "evict returned invalid type "
                    f"{type(evicted).__name__} (expected list)"
                )
            elif len(evicted) != num_blocks_to_evict:
                malformed_reason = (
                    f"evict returned {len(evicted)} items for n={num_blocks_to_evict}"
                )
            else:
                for item in evicted:
                    if not isinstance(item, tuple) or len(item) != 2:
                        malformed_reason = (
                            "evict returned an entry that is not a (key, block) tuple"
                        )
                        break
                    evict_key, evict_block = item
                    if not isinstance(evict_key, bytes):
                        malformed_reason = (
                            "evict returned key with invalid type "
                            f"{type(evict_key).__name__}"
                        )
                        break
                    if evict_key in protected:
                        malformed_reason = "evict returned a protected key"
                        break
                    if not isinstance(evict_block, BlockStatus):
                        malformed_reason = (
                            "evict returned block with invalid type "
                            f"{type(evict_block).__name__}"
                        )
                        break
                    if evict_block.ref_cnt != 0:
                        malformed_reason = (
                            "evict returned block with ref_cnt "
                            f"{evict_block.ref_cnt} (expected 0)"
                        )
                        break
                    block_id = evict_block.block_id
                    if block_id < 0 or block_id >= self._num_allocated_blocks:
                        malformed_reason = (
                            "evict returned out-of-range block_id "
                            f"{block_id} (allocated={self._num_allocated_blocks})"
                        )
                        break
                    if evict_key in seen_evicted_keys:
                        malformed_reason = "evict returned duplicate key entries"
                        break
                    if block_id in seen_evicted_block_ids:
                        malformed_reason = "evict returned duplicate block IDs"
                        break
                    try:
                        still_present = self._policy.get(evict_key) is not None
                    except Exception as e:  # noqa: BLE001
                        malformed_reason = (
                            f"evict returned a key whose post-evict lookup raised: {e}"
                        )
                        break
                    if still_present:
                        malformed_reason = (
                            "evict returned a key that is still present in the policy"
                        )
                        break
                    seen_evicted_keys.add(evict_key)
                    seen_evicted_block_ids.add(block_id)
            if malformed_reason is not None:
                logger.warning(
                    "CachePolicy.evict returned malformed output: %s. "
                    "Skipping store attempt.",
                    malformed_reason,
                )
                with suppress(Exception):
                    self._policy.record_write_error()
                return None

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)
            self._stat_evictions += len(evicted)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        # Hardened insert loop (design §14.3): if `_policy.insert` raises
        # mid-loop, undo successful inserts, free every allocated block,
        # notify the policy via `record_write_error`, and report a
        # recoverable `prepare_store -> None` rather than crashing the
        # engine. Built-in policies inherit a no-op `record_write_error`;
        # `SupervisedCachePolicy` uses it to advance its error budget.
        inserted = 0
        try:
            for key, block in zip(keys_to_store, blocks):
                self._policy.insert(key, block, req_context)
                inserted += 1
        except Exception as e:  # noqa: BLE001 - propagated to grader via stats
            logger.warning(
                "CachePolicy.insert raised mid-prepare_store after %d/%d "
                "successful inserts: %s. Rolling back.",
                inserted,
                len(keys_to_store),
                e,
            )
            # Include the key whose `insert` raised: a buggy policy can
            # mutate its table and then throw, and leaving that not-ready
            # entry behind would point at a block we are about to free.
            failed_idx = min(inserted + 1, len(keys_to_store))
            for undo_key, undo_block in zip(
                keys_to_store[:failed_idx], blocks[:failed_idx]
            ):
                with suppress(Exception):
                    self._policy.remove(undo_key)
                self._free_block(undo_block)
            for pending_block in blocks[failed_idx:]:
                self._free_block(pending_block)
            with suppress(Exception):
                self._policy.record_write_error()
            return None

        self._stat_stores += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def complete_store(
        self, keys: Collection[OffloadKey], success: bool = True
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    # Guard `_policy.remove`: a buggy evolved policy must
                    # not crash the engine here. Always free the block,
                    # best-effort remove from the policy, surface the
                    # error to the supervisor, and continue.
                    try:
                        self._policy.remove(key)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "CachePolicy.remove raised in failed-store cleanup: %s",
                            e,
                        )
                        with suppress(Exception):
                            self._policy.record_write_error()
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    # ------------------------------------------------------------------
    # Hot-swap stats hook (see design §8.3 / §14.5).
    # ------------------------------------------------------------------

    def take_policy_stats(
        self, reset: bool, active_generation: int
    ) -> dict[str, object]:
        """Return a msgpack-safe snapshot of policy/manager counters.

        Returns a plain dict (not a typed struct) so the registry can
        compose it with supervisor / sticky-rollback fields without a
        circular import. The registry is responsible for the final
        `OffloadPolicyStats` shape.
        """
        snapshot: dict[str, object] = {
            "lookups": self._stat_lookups,
            "hits": self._stat_hits,
            "stores": self._stat_stores,
            "evictions": self._stat_evictions,
            "loads": self._stat_loads,
            "hit_rate": self._stat_hits / max(self._stat_lookups, 1),
            "window_start_generation": self._stat_window_start_generation,
            "active_generation": active_generation,
            "hint_keys": tuple(sorted(self._stat_hint_keys)),
        }
        if reset:
            self._stat_lookups = 0
            self._stat_hits = 0
            self._stat_stores = 0
            self._stat_evictions = 0
            self._stat_loads = 0
            self._stat_window_start_generation = active_generation
            self._stat_hint_keys.clear()
        return snapshot
