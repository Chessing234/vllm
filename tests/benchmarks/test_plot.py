# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from datetime import datetime

import pytest

from vllm.benchmarks.plot import construct_timeline_data


@pytest.mark.parametrize(
    "duration,expected",
    [(59.9996, "00:01:00.000"), (3599.9996, "01:00:00.000")],
)
def test_timeline_timestamps_carry_rounded_milliseconds(duration, expected):
    segments = construct_timeline_data(
        [{"start_time": 100.0, "ttft": duration, "latency": duration}],
        [0.025, 0.050],
        ["fast", "medium", "slow"],
    )
    assert segments[0]["end"] == expected
    assert segments[0]["req_finish_time"] == expected
    datetime.strptime(segments[0]["end"], "%H:%M:%S.%f")


def test_timeline_itl_segments_carry_rounded_milliseconds():
    segments = construct_timeline_data(
        [
            {
                "start_time": 100.0,
                "ttft": 59.99,
                "itl": [0.0096, 0.02],
                "latency": 60.0196,
            }
        ],
        [0.025, 0.050],
        ["fast", "medium", "slow"],
    )
    assert [segment["end"] for segment in segments] == [
        "00:00:59.990",
        "00:01:00.000",
        "00:01:00.020",
    ]
    assert segments[2]["start"] == segments[1]["end"]
