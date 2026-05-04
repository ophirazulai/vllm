# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from pydantic import BaseModel, Field


class SwapOffloadPolicyRequest(BaseModel):
    """Body for `POST /v1/swap_offload_policy`. See design §8.1."""

    source_path: str | None = Field(default=None)
    module: str | None = Field(default=None)
    source: str | None = Field(default=None)
    name: str | None = Field(default=None)
    version: str | None = Field(default=None)
    policy_kwargs: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
