#!/usr/bin/env python3
"""Steam frame-rate figures for a few common games. The number is the Steam mean."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from canrun_steam_fps import resolve_fps


def test_realistic_predictions():
    print("Testing Steam FPS figures for common GPUs")
    print("=" * 60)

    cases = [
        {
            "game": "Cyberpunk 2077",
            "gpu": "RTX 4090",
            "cpu": "AMD Ryzen 7 7800X3D",
            "mean": 72,
            "stddev": 6,
            "expected_fps": 72,
            "fps_min": 66,
            "fps_max": 78,
        },
        {
            "game": "Diablo 4",
            "gpu": "RTX 4090",
            "cpu": "AMD Ryzen 7 7800X3D",
            "mean": 120,
            "stddev": 8,
            "expected_fps": 120,
            "fps_min": 110,
            "fps_max": 130,
        },
        {
            "game": "Counter-Strike 2",
            "gpu": "RTX 4090",
            "cpu": "AMD Ryzen 7 7800X3D",
            "mean": 240,
            "stddev": 12,
            "expected_fps": 240,
            "fps_min": 221,
            "fps_max": 259,
        },
    ]

    for case in cases:
        estimate = resolve_fps(
            case["gpu"],
            case["cpu"],
            {"response": {"frame_rates": [{
                "gpu_name": case["gpu"],
                "cpu_name": case["cpu"],
                "report_count": 80,
                "mean_frame_rate": case["mean"],
                "mean_frame_rate_stddev": case["stddev"],
            }]}},
            cluster_map={"clusters": {}, "reference_gpu": "RTX 5070"},
        )
        print(f"{case['game']}: {estimate.expected_fps} FPS ({estimate.fps_source})")
        assert estimate.expected_fps == case["expected_fps"]
        assert estimate.fps_min == case["fps_min"]
        assert estimate.fps_max == case["fps_max"]
        assert estimate.fps_source == "steam"


if __name__ == "__main__":
    test_realistic_predictions()
