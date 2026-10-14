"""Held-out accuracy of the similar-PC curve against measured benchmark FPS."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from canrun_steam_fps import leave_one_out_fps, resolve_fps

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_benchmarks.json")
RESOLUTIONS = ("1080p", "1440p", "4K")


def _measured_rows():
    with open(BENCHMARK_PATH, encoding="utf-8") as handle:
        games = json.load(handle)["games"]
    by_game = []
    for game in sorted(games):
        rows = []
        gpu_rows = games[game].get("gpus", {})
        for resolution in RESOLUTIONS:
            group = []
            for gpu in sorted(gpu_rows):
                spec = gpu_rows[gpu]
                if resolution in spec and spec.get("passmark") and spec[resolution]:
                    group.append((game, resolution, gpu, int(spec[resolution])))
            if len(group) >= 3:
                rows.extend(group)
        by_game.append(rows)

    picked = []
    index = 0
    while len(picked) < 100:
        progressed = False
        for rows in by_game:
            if index < len(rows):
                picked.append(rows[index])
                progressed = True
                if len(picked) == 100:
                    break
        if not progressed:
            break
        index += 1
    return picked


ACCURACY_CASES = _measured_rows()


def _r_squared(actuals, predictions):
    mean_actual = sum(actuals) / len(actuals)
    total = sum((actual - mean_actual) ** 2 for actual in actuals)
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(actuals, predictions))
    return 1 - residual / total


def test_at_least_100_held_out_rows():
    assert len(ACCURACY_CASES) >= 100


def test_similar_pc_curve_explains_at_least_95_percent():
    actuals = []
    predictions = []
    for game, resolution, gpu, actual in ACCURACY_CASES:
        predicted = leave_one_out_fps(game, resolution, gpu)
        assert predicted is not None, f"{game} {gpu} {resolution} has no leave-one-out prediction"
        actuals.append(actual)
        predictions.append(predicted)
    assert _r_squared(actuals, predictions) >= 0.95


def test_steam_exact_sample_beats_the_curve():
    estimate = resolve_fps(
        "RTX 4090",
        "AMD Ryzen 7 7800X3D",
        {"response": {"frame_rates": [{
            "gpu_name": "RTX 4090",
            "cpu_name": "AMD Ryzen 7 7800X3D",
            "report_count": 40,
            "mean_frame_rate": 50,
            "mean_frame_rate_stddev": 4,
        }]}},
        game_name="Cyberpunk 2077",
        resolution="1080p",
        cluster_map={"clusters": {}, "reference_gpu": "RTX 5070"},
    )
    assert estimate.fps_source == "steam"
    assert estimate.expected_fps == 50
    assert estimate.fps_min == 46
    assert estimate.fps_max == 54


def test_previously_missing_games_use_the_measured_curve():
    for game_name in ("elden ring", "hogwarts legacy", "black myth wukong", "diablo 4"):
        estimate = resolve_fps(
            "RTX 4090",
            "AMD Ryzen 7 7800X3D",
            None,
            api_available=False,
            game_name=game_name,
            resolution="1440p",
        )
        assert estimate.fps_source == "steam_similar", game_name
        assert estimate.expected_fps > 0, game_name
        assert estimate.fps_min < estimate.expected_fps < estimate.fps_max
    elden = resolve_fps(
        "RTX 4090",
        "AMD Ryzen 7 7800X3D",
        None,
        api_available=False,
        game_name="elden ring",
        resolution="1440p",
    )
    assert elden.expected_fps == 60
