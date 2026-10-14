"""FPS estimates for the same recent Steam games the title check uses.

This calls the live store. It is not part of pytest.
"""

import asyncio
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from canrun_game_fetcher import CanRunGameFetcher
from canrun_steam_fps import estimate_from_requirements, first_known_gpu, requirement_passmarks
from check_recent_titles import HEADERS, collect_games

USER_GPU = "RTX 4090"
USER_CPU = "AMD Ryzen 7 7800X3D"
RESOLUTION = "1440p"


async def _app_details(session, app_id: str) -> dict:
    for _ in range(3):
        try:
            async with session.get(
                "https://store.steampowered.com/api/appdetails",
                params={"appids": app_id, "cc": "US", "l": "english"},
                headers=HEADERS,
            ) as response:
                if response.status == 429:
                    await asyncio.sleep(1.5)
                    continue
                if response.status != 200:
                    return {}
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(0.4)
            continue
        entry = payload.get(str(app_id), {}) if isinstance(payload, dict) else {}
        data = entry.get("data") if isinstance(entry, dict) else None
        return data if isinstance(data, dict) else {}
    return {}


async def main() -> None:
    fetcher = CanRunGameFetcher()
    passmarks = requirement_passmarks()
    timeout = aiohttp.ClientTimeout(total=30)
    semaphore = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        games = await collect_games(session)

        async def one(app_id: str, title: str, released: str) -> dict:
            async with semaphore:
                info = await _app_details(session, app_id)
            minimum = fetcher._parse_requirement_text((info.get("pc_requirements") or {}).get("minimum", ""))
            recommended = fetcher._parse_requirement_text((info.get("pc_requirements") or {}).get("recommended", ""))
            rec_gpu = recommended.get("graphics", "")
            min_gpu = minimum.get("graphics", "")
            estimate = estimate_from_requirements(USER_GPU, USER_CPU, rec_gpu, min_gpu, RESOLUTION, passmarks)
            rec_card = first_known_gpu(rec_gpu, passmarks)
            min_card = first_known_gpu(min_gpu, passmarks)
            return {
                "title": title,
                "released": released,
                "rec": rec_gpu,
                "minimum": min_gpu,
                "rec_card": rec_card[0] if rec_card else "",
                "min_card": min_card[0] if min_card else "",
                "fps": estimate.expected_fps,
                "low": estimate.fps_min,
                "high": estimate.fps_max,
                "source": estimate.fps_source,
            }

        rows = await asyncio.gather(*(one(app_id, title, released) for app_id, title, released in games))

    scored = [
        row for row in rows
        if row["source"] != "unknown" and row["fps"] > 0 and row["low"] <= row["fps"] <= row["high"]
    ]
    unknown = [row for row in rows if row["source"] == "unknown" or row["fps"] <= 0]
    unordered = [
        row for row in rows
        if row["source"] != "unknown" and not (row["low"] <= row["fps"] <= row["high"])
    ]
    by_card: dict[str, int] = {}
    for row in scored:
        card = row["rec_card"] or row["min_card"] or "unspecified"
        by_card.setdefault(card, row["fps"])
    print(f"\nTITLES {len(rows)} SCORED {len(scored)} UNKNOWN {len(unknown)} UNORDERED {len(unordered)}")
    print("CARDS " + ", ".join(f"{name}={fps}" for name, fps in sorted(by_card.items(), key=lambda item: item[1])))
    print("\nUNKNOWN")
    for row in unknown:
        print(f"  {row['title']} | rec={row['rec'][:100]!r} | min={row['minimum'][:100]!r}")
    print("\nSAMPLE")
    for row in scored[:8]:
        print(f"  {row['fps']} ({row['low']}-{row['high']}) {row['source']} via {row['rec_card'] or row['min_card']} | {row['title']}")
    if len(scored) < 100:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
