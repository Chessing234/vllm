# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError


def test_responses_tools_must_be_an_array():
    with pytest.raises(VLLMValidationError, match="must be an array"):
        ResponsesRequest.model_validate(
            {
                "model": "facebook/opt-125m",
                "input": "hello",
                "tools": {"type": "function", "name": "lookup"},
            }
        )
