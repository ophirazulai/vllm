# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class LRUCachePolicy(CachePolicy):
    """LRU cache policy backed by a single OrderedDict."""

    POLICY_NAME = "lru"
    POLICY_VERSION = "builtin"

    def __init__(self, cache_capacity: int, **kwargs: Any):
        # cache_capacity unused by LRU but accepted for a uniform constructor.
        # **kwargs is accepted-and-ignored so the hot-swap loader can pass
        # `policy_kwargs` through without needing a per-policy switch.
        del cache_capacity, kwargs
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(
        self,
        key: OffloadKey,
        block: BlockStatus,
        req_context: ReqContext | None = None,
    ) -> None:
        del req_context  # built-in LRU ignores per-request hints
        self.blocks[key] = block

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]

    def touch(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext | None = None,
    ) -> None:
        del req_context
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
        req_context: ReqContext | None = None,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        del req_context
        if n == 0:
            return []
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
                if len(candidates) == n:
                    break
        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.blocks[key]
        return candidates

    def export_state(self) -> Iterable[tuple[OffloadKey, BlockStatus]]:
        # Snapshot to a list so callers can mutate the policy without
        # invalidating the iterator.
        return list(self.blocks.items())
