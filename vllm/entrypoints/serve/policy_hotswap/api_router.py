# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HTTP routes for the CPU-offloading `CachePolicy` hot-swap path.

Default-off and localhost-only-by-default. The endpoints accept arbitrary
Python source / module names and `exec()` them in the engine process; see
`design/evolved_cpu_offloading.md` §5 for the full threat model.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.policy_hotswap.protocol import SwapOffloadPolicyRequest
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


_LOCALHOST_HOSTS = frozenset(("127.0.0.1", "::1", "localhost"))


def _enabled(app: FastAPI) -> bool:
    args = getattr(app.state, "args", None)
    cli_flag = bool(getattr(args, "enable_policy_hotswap", False))
    return cli_flag or envs.VLLM_ENABLE_POLICY_HOTSWAP


def _ensure_localhost(raw_request: Request) -> None:
    if envs.VLLM_POLICY_HOTSWAP_ALLOW_REMOTE:
        return
    client = raw_request.client
    host = client.host if client is not None else None
    if host not in _LOCALHOST_HOSTS:
        raise HTTPException(
            status_code=403,
            detail=(
                "policy hot-swap is restricted to localhost callers; set "
                "VLLM_POLICY_HOTSWAP_ALLOW_REMOTE=1 to override (NOT "
                "recommended for production)"
            ),
        )


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def attach_router(app: FastAPI) -> None:
    if not _enabled(app):
        # Default deployment: routes are not registered with FastAPI at all.
        return
    logger.warning(
        "Policy hot-swap endpoints are enabled. This loads arbitrary Python "
        "into the engine process; do NOT enable in production."
    )

    @router.post(
        "/v1/swap_offload_policy",
        dependencies=[Depends(validate_json_request)],
    )
    async def swap_offload_policy(
        body: SwapOffloadPolicyRequest, raw_request: Request
    ):
        _ensure_localhost(raw_request)
        ec = engine_client(raw_request)
        try:
            result = await ec.swap_offload_policy(
                source_path=body.source_path,
                module=body.module,
                source=body.source,
                name=body.name,
                version=body.version,
                policy_kwargs=dict(body.policy_kwargs),
                dry_run=body.dry_run,
            )
        except NotImplementedError as e:
            raise HTTPException(
                status_code=501,
                detail=f"swap_offload_policy not implemented on this engine: {e}",
            )
        # Candidate-caused failures get HTTP 400 (bad input → bad code).
        # Internal failures bubble up as 500 via FastAPI's default handler.
        if not result.get("ok", False):
            return JSONResponse(content=result, status_code=400)
        return JSONResponse(content=result, status_code=200)

    @router.get("/v1/offload_policy")
    async def offload_policy(raw_request: Request):
        _ensure_localhost(raw_request)
        ec = engine_client(raw_request)
        active = await ec.get_offload_policy()
        return JSONResponse(content=active or {})

    @router.get("/v1/offload_policy_stats")
    async def offload_policy_stats(raw_request: Request):
        _ensure_localhost(raw_request)
        reset = raw_request.query_params.get("reset", "false").lower() in (
            "1",
            "true",
        )
        ec = engine_client(raw_request)
        stats = await ec.get_offload_policy_stats(reset=reset)
        return JSONResponse(content=stats)

    app.include_router(router)
