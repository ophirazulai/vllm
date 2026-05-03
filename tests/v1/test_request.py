# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_policy_hints_injects_request_id():
    request = Request(
        request_id="req-123",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"policy_hints": {"tenant": "acme"}}
        ),
        pooling_params=None,
    )

    assert request.policy_hints == {"tenant": "acme", "_request_id": "req-123"}


def test_request_policy_hints_rejects_non_string_keys():
    with pytest.raises(TypeError, match="must use string keys"):
        Request(
            request_id="req-123",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(
                max_tokens=1, extra_args={"policy_hints": {1: "bad"}}
            ),
            pooling_params=None,
        )


def test_request_policy_hints_rejects_reserved_request_id_key():
    with pytest.raises(ValueError, match="reserved key '_request_id'"):
        Request(
            request_id="req-123",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"policy_hints": {"_request_id": "spoofed"}},
            ),
            pooling_params=None,
        )
