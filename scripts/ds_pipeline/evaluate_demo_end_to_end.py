"""
End-to-end check of the FPS the demo and plugin actually show, against measured benchmarks.

Runs CanRunEngine.check_game_compatibility() for a typed game name, with the hardware
replaced per card: title matching, live Steam Store requirements, measured-table curve
or store-spec model, the 240 FPS cap and the shown range. The Steam Web API key is
cleared, as in the shipped keyless config.json.

Source: TechPowerUp, "NVIDIA GeForce RTX 5070 Founders Edition Review" (Mar 2025),
Ryzen 7 9800X3D, highest settings, ray tracing off.
https://www.techpowerup.com/review/nvidia-geforce-rtx-5070-founders-edition/
Elden Ring is left out: the review unlocks the game's 60 FPS cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from canrun_engine import CanRunEngine  # noqa: E402
from canrun_hardware_detector import HardwareSpecs  # noqa: E402
from scripts.ds_pipeline.evaluate_external_benchmarks import FPS_1080P, GAMES, VRAM_GB  # noqa: E402

TEST_CPU = "AMD Ryzen 7 9800X3D"
CARDS = [
    "RTX 5080", "RTX 5070 Ti", "RTX 5070", "RTX 4080", "RTX 4070 Ti", "RTX 4070", "RTX 3090 Ti",
    "RTX 3090", "RTX 3080", "RTX 3070 Ti", "RTX 3070", "RX 7900 XTX", "RX 7800 XT",
]
DISPLAY = {"1080p": (1920, 1080), "1440p": (2560, 1440)}
# Steam documents no limit for the store endpoints. About 200 requests per
# 5 minutes per IP is where they start answering 429 and then 403.
STEAM_MIN_INTERVAL_S = 2.0
STEAM_REQUESTS: Counter[str] = Counter()
STEAM_BLOCK: dict[str, str] = {}


class SteamBlocked(RuntimeError):
    pass


def pace_steam_requests() -> None:
    original = aiohttp.ClientSession._request
    lock = asyncio.Lock()
    last = {"t": 0.0}

    async def paced(self, method, url, **kwargs):
        parts = urlsplit(str(url))
        if not parts.hostname or not parts.hostname.endswith(("steampowered.com", "steamcommunity.com")):
            return await original(self, method, url, **kwargs)
        if STEAM_BLOCK:
            raise SteamBlocked(STEAM_BLOCK["reason"])
        async with lock:
            wait = last["t"] + STEAM_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            last["t"] = time.monotonic()
        STEAM_REQUESTS[f"{parts.hostname}{parts.path}"] += 1
        response = await original(self, method, url, **kwargs)
        if response.status in (403, 429):
            STEAM_BLOCK["reason"] = f"{response.status} from {parts.hostname}{parts.path}"
            raise SteamBlocked(STEAM_BLOCK["reason"])
        return response

    aiohttp.ClientSession._request = paced

# 1440p values in CARDS order, for games that are also in data/training_benchmarks.json
# (or, for No Rest for the Wicked and The Last of Us, reached only through store specs).
FPS_1440P_MORE: dict[str, list[float]] = {
    "Alan Wake 2": [104.5, 93.0, 72.6, 96.6, 79.9, 63.8, 82.7, 74.2, 66.3, 54.1, 51.1, 102.4, 68.4],
    "Baldur's Gate 3": [186.8, 168.8, 141.5, 184.9, 152.7, 121.1, 150.6, 135.0, 121.9, 99.4, 93.2, 180.2, 120.2],
    "Black Myth Wukong": [59.6, 53.3, 42.7, 54.8, 45.5, 35.1, 44.7, 40.1, 35.5, 29.3, 27.3, 49.6, 33.1],
    "Counter-Strike 2": [409.2, 360.3, 294.5, 349.4, 281.3, 227.0, 313.7, 289.1, 242.6, 216.6, 204.0, 327.5, 204.1],
    "Cyberpunk 2077": [142.9, 125.2, 104.4, 123.0, 100.3, 72.0, 106.9, 95.3, 84.8, 69.3, 65.4, 129.2, 87.1],
    "DOOM Eternal": [268.2, 237.6, 191.9, 259.3, 222.9, 180.0, 207.4, 191.3, 172.6, 141.2, 135.2, 268.5, 180.7],
    "Ghost of Tsushima": [134.8, 119.4, 92.3, 125.1, 100.7, 83.9, 104.9, 95.4, 85.0, 72.4, 66.1, 135.6, 97.0],
    "God of War Ragnarok": [148.4, 133.1, 107.2, 134.7, 108.6, 86.0, 112.5, 101.9, 89.6, 73.2, 69.3, 120.2, 79.4],
    "Hogwarts Legacy": [107.0, 94.8, 79.5, 98.0, 77.6, 61.9, 85.6, 78.1, 66.8, 55.5, 51.6, 108.3, 69.3],
    "Horizon Forbidden West": [130.5, 116.2, 98.0, 121.5, 100.0, 83.0, 96.7, 87.1, 77.8, 69.3, 64.9, 118.3, 78.4],
    "Resident Evil 4": [156.5, 140.3, 107.7, 133.6, 108.4, 84.8, 109.5, 98.1, 86.2, 67.2, 62.5, 165.5, 107.7],
    "Space Marine 2": [141.4, 124.1, 96.2, 122.0, 97.2, 76.5, 97.9, 88.1, 77.0, 62.2, 58.3, 121.3, 79.4],
    "Stalker 2": [91.0, 84.1, 71.3, 82.0, 71.4, 57.7, 72.0, 63.4, 57.0, 50.1, 45.0, 76.0, 52.5],
    "Starfield": [110.1, 101.2, 81.0, 104.5, 87.8, 71.0, 84.6, 75.9, 67.6, 55.9, 52.7, 111.1, 73.5],
    "The Witcher 3": [318.4, 269.4, 223.5, 283.8, 225.6, 172.5, 226.2, 201.9, 177.3, 149.2, 138.5, 279.7, 169.4],
    "No Rest for the Wicked": [162.2, 143.1, 114.7, 141.1, 115.0, 84.7, 121.7, 108.7, 97.2, 79.1, 75.3, 144.8, 91.5],
    "The Last of Us Part I": [113.6, 100.5, 78.9, 100.2, 81.9, 67.4, 83.3, 76.8, 68.6, 46.2, 43.7, 105.9, 74.0],
}

# What a user would type for the games defined in evaluate_external_benchmarks.py.
TYPED = {
    "Assassin's Creed Mirage": "Assassin's Creed Mirage",
    "Dragon Age: The Veilguard": "Dragon Age The Veilguard",
    "F1 24": "F1 24",
    "Like a Dragon: Infinite Wealth": "Like a Dragon Infinite Wealth",
    "Ratchet & Clank: Rift Apart": "Ratchet and Clank Rift Apart",
    "SILENT HILL 2": "Silent Hill 2",
    "Star Wars Outlaws": "Star Wars Outlaws",
}


def build_cases() -> list[tuple[str, str, str, float]]:
    cases = []
    for game, info in GAMES.items():
        for res, table in (("1440p", info["fps"]), ("1080p", FPS_1080P[game])):
            for card in CARDS:
                if card in table:
                    cases.append((TYPED[game], card, res, table[card]))
    for game, values in FPS_1440P_MORE.items():
        for card, fps in zip(CARDS, values):
            cases.append((game, card, "1440p", fps))
    return cases


def hardware_for(card: str) -> HardwareSpecs:
    nvidia = card.startswith("RTX")
    return HardwareSpecs(
        gpu_model=card, gpu_vram_gb=VRAM_GB[card], cpu_cores=8, cpu_threads=16, ram_total_gb=32,
        ram_speed_mhz=6000, storage_type="NVMe SSD", primary_monitor_refresh_hz=144,
        primary_monitor_resolution="2560x1440", os_version="Windows 11", directx_version="DirectX 12",
        gpu_vendor="NVIDIA" if nvidia else "AMD", cpu_model=TEST_CPU, is_nvidia_gpu=nvidia,
        supports_rtx=nvidia, supports_dlss=nvidia,
    )


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["shown"] > 0]
    ape = [abs(r["shown"] - r["true"]) / r["true"] * 100 for r in scored]
    in_range = [r["lo"] <= r["true"] <= r["hi"] for r in scored]
    return {
        "n": len(rows),
        "unknown": len(rows) - len(scored),
        "MdAPE_pct": round(statistics.median(ape), 1) if ape else None,
        "MAPE_pct": round(statistics.fmean(ape), 1) if ape else None,
        "within_10pct": round(sum(a <= 10 for a in ape) / len(ape) * 100, 1) if ape else None,
        "true_inside_shown_range_pct": round(sum(in_range) / len(in_range) * 100, 1) if in_range else None,
        "bias_pct": round(statistics.fmean((r["shown"] - r["true"]) / r["true"] * 100 for r in scored), 1)
        if scored else None,
    }


async def run() -> list[dict]:
    logging.disable(logging.WARNING)
    engine = CanRunEngine(cache_dir=str(ROOT_DIR / "cache" / "e2e_eval"))
    engine.steam_fps.api_key = ""
    requirement_cache: dict[str, object] = {}
    original_fetch = engine._fetch_game_requirements

    async def cached_fetch(name: str):
        if name not in requirement_cache:
            requirement_cache[name] = await original_fetch(name)
        return requirement_cache[name]

    engine._fetch_game_requirements = cached_fetch
    current = {"res": "1440p"}
    engine.hardware_detector.get_display_resolution = lambda: dict(
        zip(("width", "height"), DISPLAY[current["res"]])
    )

    rows = []
    for typed, card, res, true_fps in build_cases():
        engine._hardware_cache = hardware_for(card)
        current["res"] = res
        try:
            result = await engine.check_game_compatibility(typed, use_cache=False)
        except SteamBlocked:
            raise
        except (ValueError, AssertionError) as e:
            rows.append({
                "typed": typed, "matched": "-", "card": card, "res": res, "true": true_fps,
                "shown": 0.0, "lo": 0.0, "hi": 0.0, "source": f"error: {e}"[:60],
            })
            continue
        perf = result.performance_prediction
        rows.append({
            "typed": typed, "matched": result.game_requirements.game_name, "card": card, "res": res,
            "true": true_fps, "shown": float(perf.expected_fps or 0), "lo": float(perf.fps_min or 0),
            "hi": float(perf.fps_max or 0), "source": perf.fps_source,
        })
    return rows


def main() -> int:
    pace_steam_requests()
    started = time.monotonic()
    try:
        rows = asyncio.run(run())
    except SteamBlocked as e:
        STEAM_BLOCK.setdefault("reason", str(e))
    if STEAM_BLOCK:
        print(f"Stopped: Steam answered {STEAM_BLOCK['reason']} after {sum(STEAM_REQUESTS.values())} requests")
        return 1
    print(f"Steam requests: {sum(STEAM_REQUESTS.values())} in {time.monotonic() - started:.0f} s "
          f"{dict(STEAM_REQUESTS)}")
    (ROOT_DIR / "cache" / "e2e_eval").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "cache" / "e2e_eval" / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"All {len(rows)} shown predictions: {summarize(rows)}")
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for source, group in sorted(by_source.items()):
        print(f"  source={source:26s} {summarize(group)}")

    print("\nPer typed game (matched title, source, median error, bias):")
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["typed"]].append(r)
    for typed, group in by_game.items():
        s = summarize(group)
        print(
            f"  {typed:30s} -> {group[0]['matched'][:34]:34s} {group[0]['source']:22s} "
            f"MdAPE {s['MdAPE_pct']}%  bias {s['bias_pct']}%  in-range {s['true_inside_shown_range_pct']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
