# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.policy_hotswap import api_router

pytestmark = pytest.mark.skip_global_cleanup


class _FakeEngineClient:
    async def swap_offload_policy(self, **kwargs):
        raise AssertionError("invalid source payload should fail before engine call")

    async def get_offload_policy(self):
        return None

    async def get_offload_policy_stats(self, reset: bool = False):
        return {}


def test_swap_offload_policy_source_validation_returns_400(monkeypatch):
    monkeypatch.setattr(api_router.envs, "VLLM_POLICY_HOTSWAP_ALLOW_REMOTE", True)

    app = FastAPI()
    app.state.args = SimpleNamespace(enable_policy_hotswap=True)
    app.state.engine_client = _FakeEngineClient()
    api_router.attach_router(app)

    response = TestClient(app).post(
        "/v1/swap_offload_policy",
        json={"source_path": "/tmp/policy.py", "source": "class Policy: pass"},
    )

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]
