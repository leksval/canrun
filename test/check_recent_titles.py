"""Resolve 100 Steam games from the last three months, including mistyped titles.

This talks to the live Steam store search API. It is not part of the default
pytest run.
"""

import asyncio
from html import unescape
import json
import re
import sys
from datetime import datetime

import aiohttp

sys.path.insert(0, ".")

from canrun_game_fetcher import CanRunGameFetcher

CUTOFF = datetime(2026, 7, 14)
ROW = re.compile(
    r'data-ds-appid="(\d+)"[\s\S]{0,1800}?class="title"[^>]*>([^<]+)</span>'
    r'[\s\S]{0,500}?search_released[^>]*>([^<]*)',
)
LONG_WORD = re.compile(r"[A-Za-z]{6,}")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/html"}


def _released_on(text: str) -> datetime | None:
    released = " ".join((text or "").split())
    for fmt in ("%d %b, %Y", "%b %d, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(released, fmt)
        except ValueError:
            continue
    return None


def mistyped(title: str) -> dict[str, str]:
    """A few realistic slips against one published title."""
    queries = {"lower": title.lower()}
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", title)
    long_words = [word for word in words if len(word) >= 6 and not word.isdigit()]
    if long_words:
        word = max(long_words, key=len)
        middle = len(word) // 2
        missing = word[:middle] + word[middle + 1:]
        chars = list(word)
        chars[middle], chars[middle - 1] = chars[middle - 1], chars[middle]
        queries["missing_letter"] = title.replace(word, missing, 1)
        queries["swapped_letters"] = title.replace(word, "".join(chars), 1)
        queries["double_letter"] = title.replace(word, word[:middle] + word[middle] + word[middle:], 1)
    if title.lower().startswith("the "):
        queries["dropped_the"] = title[4:]
    else:
        queries["added_the"] = "the " + title
    return queries


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict | str:
    for _ in range(3):
        try:
            async with session.get(url, params=params, headers=HEADERS) as response:
                text = await response.text()
                if response.status == 429:
                    await asyncio.sleep(1.5)
                    continue
                if response.status != 200:
                    return ""
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(0.5)
    return ""


async def collect_games(session: aiohttp.ClientSession) -> list[tuple[str, str, str]]:
    games: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for start in range(0, 12000, 250):
        payload = await _get_json(
            session,
            "https://store.steampowered.com/search/results/",
            {
                "query": "",
                "start": start,
                "count": 50,
                "sort_by": "Released_DESC",
                "infinite": 1,
                "category1": 998,
                "os": "win",
                "ignore_preferences": 1,
            },
        )
        html = payload.get("results_html", "") if isinstance(payload, dict) else ""
        rows = ROW.findall(html)
        if not rows:
            break
        older = 0
        for appid, title, released in rows:
            when = _released_on(released)
            title = " ".join(unescape(title).split())
            if when is None:
                continue
            if when < CUTOFF:
                older += 1
                continue
            if appid in seen or LONG_WORD.search(title) is None:
                continue
            seen.add(appid)
            games.append((appid, title, when.strftime("%Y-%m-%d")))
        print(f"page {start}: kept {len(games)} older {older}", flush=True)
        if older > 30 or len(games) >= 110:
            break
    return games[:110]


async def main() -> None:
    timeout = aiohttp.ClientTimeout(total=30)
    fetcher = CanRunGameFetcher()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        games = await collect_games(session)
        if len(games) < 100:
            print(f"Only found {len(games)} recent titles")
            return
        counts: dict[str, dict[str, int]] = {}
        misses: dict[str, list[str]] = {}
        semaphore = asyncio.Semaphore(3)

        async def check(appid: str, title: str, released: str, kind: str, query: str) -> None:
            async with semaphore:
                chosen = await fetcher._search_steam_store_api(query) or ""
            bucket = counts.setdefault(kind, {"chosen": 0, "n": 0})
            bucket["n"] += 1
            bucket["chosen"] += int(chosen == appid)
            if chosen != appid and len(misses.setdefault(kind, [])) < 8:
                misses[kind].append(f"{released} | {title} | query={query} | chose={chosen or 'none'}")

        tasks = []
        for appid, title, released in games:
            for kind, query in mistyped(title).items():
                if query.strip().lower() == title.strip().lower() and kind != "lower":
                    continue
                tasks.append(check(appid, title, released, kind, query))
        await asyncio.gather(*tasks)

    print(f"\nTITLES {len(games)}  FROM {games[-1][2]} TO {games[0][2]}")
    for kind, bucket in counts.items():
        print(f"{kind}: chose {bucket['chosen']}/{bucket['n']}")
    for kind, rows in misses.items():
        print(f"\n{kind} misses")
        for row in rows:
            print(" ", row)


if __name__ == "__main__":
    asyncio.run(main())
