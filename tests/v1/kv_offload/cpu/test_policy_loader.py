# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.cpu.policies.loader import PolicyLoader, discard_module

pytestmark = pytest.mark.skip_global_cleanup


def test_loader_deduplicates_aliases_of_the_same_policy_class():
    loaded = PolicyLoader().load_from_source(
        """
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class AliasPolicy(LRUCachePolicy):
    POLICY_NAME = "alias"
    POLICY_VERSION = "test"


AliasAgain = AliasPolicy
"""
    )
    try:
        assert loaded.cls.__name__ == "AliasPolicy"
    finally:
        discard_module(loaded.module.__name__)
