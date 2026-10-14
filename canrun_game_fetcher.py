"""
CanRun Game Fetcher - Standalone module for NVIDIA G-Assist Plugin
Fetches game requirements from Steam API with comprehensive parsing.
Flattened structure with all dependencies bundled for PyInstaller compatibility.
"""

import json
import logging
import asyncio
import threading
import aiohttp
import re
import time
import sys
import os
from datetime import datetime, timedelta
from html import unescape
from typing import Dict, List, Union, Optional, Any
from urllib.parse import quote_plus
from pathlib import Path
from dataclasses import dataclass

from canrun_steam_fps import redact_secrets, steam_client_roots, valid_steam_app_id


def steam_store_search_url(game_name: str) -> str:
    """Quote the search term so it cannot add extra query parameters."""
    return "https://store.steampowered.com/search/?term=" + quote_plus(game_name or "")


_ROMAN_NUMERALS = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}
STEAM_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/json"}


_OPTIONAL_QUERY_WORDS = {"the", "a", "an"}
_FILLER_WORDS = {"the", "a", "an", "of", "and"}


def steam_title_tokens(text: str) -> list[str]:
    """Lowercase words, with sequel numerals written as digits."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return [_ROMAN_NUMERALS.get(token, token) for token in cleaned.split() if token]


def _change_kind(left: str, right: str) -> str | None:
    """One insertion, deletion, adjacent swap, or substitution. Otherwise none."""
    if left == right:
        return "same"
    if abs(len(left) - len(right)) > 1:
        return None
    if len(left) == len(right):
        differ = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(differ) == 1:
            return "substitution"
        if len(differ) == 2:
            index = differ[0]
            swapped = (
                index + 1 == differ[1]
                and left[index] == right[index + 1]
                and left[index + 1] == right[index]
                and left[index + 2:] == right[index + 2:]
            )
            if swapped:
                return "transposition"
        return None
    if len(left) > len(right):
        left, right = right, left
    index = 0
    while index < len(left) and left[index] == right[index]:
        index += 1
    if left[index:] == right[index + 1:]:
        return "insertion"
    return None


def _token_edit_cost(query_token: str, title_token: str) -> int | None:
    """0 for the same word. 1 for a single slip on a long word. Digits stay exact."""
    if query_token == title_token:
        return 0
    if query_token.isdigit() or title_token.isdigit():
        return None
    kind = _change_kind(query_token, title_token)
    if kind not in {"insertion", "transposition", "substitution"}:
        return None
    shorter = min(len(query_token), len(title_token))
    longer = max(len(query_token), len(title_token))
    if shorter < 4 or longer < 5:
        return None
    if kind == "substitution" and longer < 8:
        return None
    return 1


def _best_token_cost(token: str, title_tokens: list[str], used: set[int]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for index, title_token in enumerate(title_tokens):
        if index in used:
            continue
        cost = _token_edit_cost(token, title_token)
        if cost is None:
            continue
        if best is None or cost < best[0]:
            best = (cost, index)
            if cost == 0:
                break
    return best


def _compound_match_cost(query_tokens: list[str], title_tokens: list[str]) -> int | None:
    """Match 'Dawn of the Walkers' to a title token such as dawnwalker.

    Filler words are removed and the remaining words are joined. The join has
    to be one slip from one long title word, and the query cannot contain a digit.
    """
    if any(token.isdigit() for token in query_tokens):
        return None
    content = [token for token in query_tokens if token not in _FILLER_WORDS]
    if len(content) < 2 or len(content) == len(query_tokens):
        return None
    compound = "".join(content)
    if len(compound) < 8 or any(character.isdigit() for character in compound):
        return None
    best: int | None = None
    for token in title_tokens:
        if len(token) < 8 or token.isdigit():
            continue
        cost = _token_edit_cost(compound, token)
        if cost is None:
            continue
        if best is None or cost < best:
            best = cost
            if cost == 0:
                break
    return best


def steam_title_match_cost(query: str, title: str) -> int | None:
    """How many slipped words were needed, or None when the title is a different game."""
    query_tokens = steam_title_tokens(query)
    title_tokens = steam_title_tokens(title)
    if not query_tokens or not title_tokens:
        return None
    title_token_set = set(title_tokens)
    if any(token in _OPTIONAL_QUERY_WORDS and token not in title_token_set for token in query_tokens):
        kept = [
            token for token in query_tokens
            if token not in _OPTIONAL_QUERY_WORDS or token in title_token_set
        ]
        if kept:
            query_tokens = kept
    used: set[int] = set()
    index = 0
    edits = 0
    while index < len(query_tokens):
        found = _best_token_cost(query_tokens[index], title_tokens, used)
        if found is not None:
            edits += found[0]
            used.add(found[1])
            index += 1
            continue
        if index + 1 < len(query_tokens):
            found = _best_token_cost(query_tokens[index] + query_tokens[index + 1], title_tokens, used)
            if found is not None:
                edits += found[0]
                used.add(found[1])
                index += 2
                continue
        edits = None
        break
    if edits is not None:
        # A short word may open a longer title only when it is that title's first word.
        # "sekiro" matches "Sekiro: Shadows Die Twice"; "within" does not match "Knights Within".
        if len(query_tokens) == 1 and len(query_tokens[0]) < 8 and len(title_tokens) != 1:
            token = query_tokens[0]
            if title_tokens[0] != token or len(token) < 4:
                return None
        return edits
    return _compound_match_cost(steam_title_tokens(query), title_tokens)


def steam_title_matches(query: str, title: str) -> bool:
    """True when the query is this Steam title, allowing one slip in a long word."""
    return steam_title_match_cost(query, title) is not None


def steam_title_order_penalty(query: str, title: str) -> int:
    """0 when the query words appear in the same order as the Steam title."""
    query_tokens = steam_title_tokens(query)
    index = 0
    for token in steam_title_tokens(title):
        if index < len(query_tokens) and _token_edit_cost(query_tokens[index], token) == 0:
            index += 1
    return 0 if query_tokens and index == len(query_tokens) else 1


def title_search_repairs(query: str) -> list[str]:
    """Spellings to try when Steam returns nothing for the typed title.

    Drops a leading "the", collapses a doubled letter, swaps two adjacent
    letters, and searches a short prefix of each long word. The original
    query is not included.
    """
    tokens = steam_title_tokens(query)
    if not tokens:
        return []
    repairs: list[str] = []

    def add(parts: list[str]) -> None:
        text = " ".join(part for part in parts if part)
        if text and text not in repairs and text != " ".join(tokens):
            repairs.append(text)

    if tokens[0] in _OPTIONAL_QUERY_WORDS and len(tokens) > 1:
        add(tokens[1:])
    for word_index, word in enumerate(tokens):
        if len(word) < 5 or word.isdigit():
            continue
        for index in range(1, len(word)):
            if word[index] == word[index - 1]:
                add(tokens[:word_index] + [word[:index] + word[index + 1:]] + tokens[word_index + 1:])
        center = max(len(word) // 2 - 1, 0)
        for index in (center, center - 1, center + 1):
            if not 0 <= index < len(word) - 1:
                continue
            chars = list(word)
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            add(tokens[:word_index] + ["".join(chars)] + tokens[word_index + 1:])
        add(tokens[:word_index] + [word[:4]] + tokens[word_index + 1:])
        add(tokens[:word_index] + [word[:3]] + tokens[word_index + 1:])
        if len(repairs) >= 12:
            break
    for repair in _compound_repairs(tokens):
        add([repair])
        if len(repairs) >= 12:
            break
    return repairs[:12]


def _compound_repairs(tokens: list[str]) -> list[str]:
    """Joined form of a split title, such as 'Dawn Walker', then one-letter cuts."""
    if any(token.isdigit() for token in tokens):
        return []
    content = [token for token in tokens if token not in _FILLER_WORDS]
    if len(content) < 2:
        return []
    compound = "".join(content)
    if len(compound) < 8:
        return []
    repairs = [compound]
    if compound.endswith("s") and not compound.endswith("ss") and len(compound) > 8:
        repairs.append(compound[:-1])
    for index in range(len(compound) - 1, -1, -1):
        shortened = compound[:index] + compound[index + 1:]
        if len(shortened) >= 8 and shortened not in repairs:
            repairs.append(shortened)
    return repairs


def _edit_distance(left: str, right: str) -> int:
    """Character edits between two short titles."""
    if left == right:
        return 0
    if not left or not right:
        return max(len(left), len(right))
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _catalog_pair(query: str, title: str) -> tuple[list[str], list[str]]:
    """Query and title words with one leading article removed when the other side lacks it."""
    query_tokens = steam_title_tokens(query)
    title_tokens = steam_title_tokens(title)
    if (
        len(query_tokens) > 1
        and query_tokens[0] in _OPTIONAL_QUERY_WORDS
        and (not title_tokens or title_tokens[0] != query_tokens[0])
    ):
        query_tokens = query_tokens[1:]
    if (
        title_tokens
        and title_tokens[0] in _OPTIONAL_QUERY_WORDS
        and (not query_tokens or query_tokens[0] != title_tokens[0])
    ):
        title_tokens = title_tokens[1:]
    return query_tokens, title_tokens


def _locks_long_title_word(query: str, title: str) -> bool:
    """True when a split query collapses onto one long word in the title.

    "Dawn Walker" locks onto dawnwalker inside "The Blood of Dawnwalker".
    A single short word such as "within" does not lock onto a longer title.
    """
    query_tokens = [token for token in steam_title_tokens(query) if token not in _FILLER_WORDS]
    if len(query_tokens) < 2 or any(token.isdigit() for token in query_tokens):
        return False
    compound = "".join(query_tokens)
    if len(compound) < 8 or any(character.isdigit() for character in compound):
        return False
    for title_token in steam_title_tokens(title):
        if len(title_token) < 8 or title_token.isdigit():
            continue
        if _token_edit_cost(compound, title_token) is not None:
            return True
    return False


def choose_recent_title(query: str, games: list[tuple[str, str]]) -> str:
    """Newest recent release that is the same title, or a split of one of its words.

    A one-letter slip still requires the same words. A split phrase such as
    "Dawn Walker" may match a longer recent title when those words join into
    one of its long words. A short alias such as "dawnwalker" still goes to
    live Steam search. The list is newest first.
    """
    best: tuple | None = None
    for index, (app_id, name) in enumerate(games):
        steam_id = valid_steam_app_id(str(app_id))
        title = " ".join(str(name or "").split())
        if not steam_id or not title:
            continue
        query_tokens, title_tokens = _catalog_pair(query, title)
        if not query_tokens or not title_tokens:
            continue
        adjusted_query = " ".join(query_tokens)
        adjusted_title = " ".join(title_tokens)
        edits = steam_title_match_cost(adjusted_query, adjusted_title)
        if edits is None or edits > 1:
            continue
        locked = _locks_long_title_word(query, title)
        extra = len(title_tokens) - len(query_tokens)
        if not locked and (extra > 0 or extra < -1):
            continue
        if locked and extra < -1:
            continue
        distance = _edit_distance(adjusted_query, adjusted_title)
        if not locked and edits > 0 and distance > 3:
            continue
        order = steam_title_order_penalty(adjusted_query, adjusted_title)
        rank = (edits, abs(extra), order, distance, index)
        if best is None or rank < best[0]:
            best = (rank, steam_id)
    return best[1] if best else ""


_RECENT_SEARCH_URL = "https://store.steampowered.com/search/results/"
_RECENT_ROW = re.compile(
    r'data-ds-appid="(\d+)"[\s\S]{0,1800}?class="title"[^>]*>([^<]+)</span>'
    r'[\s\S]{0,500}?search_released[^>]*>([^<]*)',
)
_RECENT_TTL_SECONDS = 6 * 60 * 60
_RECENT_PAGE_SIZE = 50
_RECENT_PAGE_CAP = 40
_RECENT_DAYS = 30
_BESTSELLER_COUNT = 100
_RECENT_CATALOG = "recent_steam_titles.json"


def _parse_release_date(text: str) -> datetime | None:
    released = " ".join((text or "").split())
    for fmt in ("%d %b, %Y", "%b %d, %Y", "%d %B, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(released, fmt)
        except ValueError:
            continue
    return None


def recent_titles_from_search_html(html_text: str, cutoff: datetime) -> list[tuple[str, str]]:
    """App id and title rows from one Released_DESC search page."""
    games: list[tuple[str, str]] = []
    for app_id, title, released in _RECENT_ROW.findall(html_text or ""):
        steam_id = valid_steam_app_id(app_id)
        name = " ".join(unescape(title).split())
        when = _parse_release_date(released)
        if not steam_id or not name:
            continue
        if when is not None and when < cutoff:
            continue
        games.append((steam_id, name))
    return games


def merge_catalog_titles(
    recent: list[tuple[str, str]],
    bestsellers: list[tuple[str, str]],
    bestseller_limit: int = _BESTSELLER_COUNT,
) -> list[tuple[str, str]]:
    """Recent releases first, then the current bestsellers that are not already listed."""
    seen: set[str] = set()
    merged: list[tuple[str, str]] = []
    for app_id, name in recent:
        if not app_id or app_id in seen:
            continue
        seen.add(app_id)
        merged.append((app_id, name))
    for app_id, name in bestsellers[:bestseller_limit]:
        if not app_id or app_id in seen:
            continue
        seen.add(app_id)
        merged.append((app_id, name))
    return merged


_VDF_LIMIT = 8_000_000
_ACF_LIMIT = 1_000_000
_MANIFEST_NAME = re.compile(r"^appmanifest_(\d+)\.acf$", re.IGNORECASE)


@dataclass(frozen=True)
class LibraryGame:
    """One app named in a local Steam library manifest."""

    app_id: str
    name: str
    installed: bool


def _tokenize_vdf(text: str) -> list[str]:
    """Split a Valve text file into strings and braces."""
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            index += 2
            while index < length and text[index] not in "\r\n":
                index += 1
            continue
        if char in "{}":
            tokens.append(char)
            index += 1
            continue
        if char == '"':
            index += 1
            chars: list[str] = []
            while index < length:
                if text[index] == "\\" and index + 1 < length:
                    chars.append(text[index + 1])
                    index += 2
                    continue
                if text[index] == '"':
                    index += 1
                    break
                chars.append(text[index])
                index += 1
            tokens.append("".join(chars))
            continue
        start = index
        while index < length and not text[index].isspace() and text[index] not in "{}\"":
            index += 1
        if start != index:
            tokens.append(text[start:index])
    return tokens


def _parse_vdf_block(tokens: list[str], index: int) -> tuple[dict, int]:
    node: dict = {}
    if index >= len(tokens) or tokens[index] != "{":
        return node, index
    index += 1
    while index < len(tokens) and tokens[index] != "}":
        key = tokens[index]
        index += 1
        if index >= len(tokens) or key in "{}":
            break
        if tokens[index] == "{":
            child, index = _parse_vdf_block(tokens, index)
            node[key] = child
        elif tokens[index] == "}":
            break
        else:
            node[key] = tokens[index]
            index += 1
    if index < len(tokens) and tokens[index] == "}":
        index += 1
    return node, index


def parse_vdf(text: str) -> dict:
    """Parse a small Valve key-value file. Nested keys become dicts."""
    tokens = _tokenize_vdf(text)
    node: dict = {}
    index = 0
    while index < len(tokens):
        if tokens[index] in "{}":
            index += 1
            continue
        key = tokens[index]
        index += 1
        if index >= len(tokens):
            break
        if tokens[index] == "{":
            child, index = _parse_vdf_block(tokens, index)
            node[key] = child
        else:
            node[key] = tokens[index]
            index += 1
    return node


def _read_text_limited(path: Path, limit: int) -> str:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _library_paths(node: Any) -> list[str]:
    found: list[str] = []
    if not isinstance(node, dict):
        return found
    for key, value in node.items():
        if str(key).lower() == "path" and isinstance(value, str) and value.strip():
            found.append(value)
        elif isinstance(value, dict):
            found.extend(_library_paths(value))
    return found


def _remember_path(path: Path, found: list[Path], seen: set[str]) -> None:
    try:
        key = os.path.normcase(os.path.abspath(path))
    except OSError:
        return
    if key in seen:
        return
    seen.add(key)
    found.append(Path(key))


def steam_library_directories(client_roots: Optional[list[Path]] = None) -> list[Path]:
    """Steam install roots plus every library path listed in libraryfolders.vdf."""
    roots = list(client_roots) if client_roots is not None else steam_client_roots()
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        _remember_path(root, found, seen)
        text = _read_text_limited(root / "steamapps" / "libraryfolders.vdf", _VDF_LIMIT)
        if not text:
            continue
        for raw in _library_paths(parse_vdf(text)):
            _remember_path(Path(raw), found, seen)
    return found


def _manifest_game(path: Path) -> Optional[LibraryGame]:
    match = _MANIFEST_NAME.match(path.name)
    if match is None:
        return None
    text = _read_text_limited(path, _ACF_LIMIT)
    if not text:
        return None
    parsed = parse_vdf(text)
    state = None
    for key, value in parsed.items():
        if str(key).lower() == "appstate" and isinstance(value, dict):
            state = value
            break
    if not isinstance(state, dict):
        return None
    fields = {str(key).lower(): value for key, value in state.items() if isinstance(value, str)}
    file_id = valid_steam_app_id(match.group(1))
    app_id = valid_steam_app_id(str(fields.get("appid") or file_id))
    if not app_id or (file_id and app_id != file_id):
        return None
    name = str(fields.get("name") or "").strip()
    if not name:
        return None
    try:
        installed = (int(str(fields.get("stateflags") or "0")) & 4) != 0
    except ValueError:
        installed = False
    return LibraryGame(app_id, name, installed)


def steam_library_games(client_roots: Optional[list[Path]] = None) -> list[LibraryGame]:
    """Names and app ids from appmanifest files in the local Steam libraries."""
    games: dict[str, LibraryGame] = {}
    for library in steam_library_directories(client_roots):
        steamapps = library / "steamapps"
        try:
            if not steamapps.is_dir():
                continue
            manifests = list(steamapps.glob("appmanifest_*.acf"))
        except OSError:
            continue
        for path in manifests:
            game = _manifest_game(path)
            if game is None:
                continue
            current = games.get(game.app_id)
            if current is None or (game.installed and not current.installed):
                games[game.app_id] = game
    return list(games.values())


def choose_library_app(query: str, games: list[LibraryGame]) -> str:
    """Pick one library app. Installed titles win, then the closest name."""
    query_tokens = steam_title_tokens(query)
    if not query_tokens:
        return ""
    ranked: list[tuple] = []
    for game in games:
        if not steam_title_matches(query, game.name):
            continue
        extra = len(steam_title_tokens(game.name)) - len(query_tokens)
        edits = steam_title_match_cost(query, game.name) or 0
        order = steam_title_order_penalty(query, game.name)
        tool = 1 if _title_is_tool(game.name) else 0
        ranked.append((0 if game.installed else 1, tool, edits, extra, order, game.name.lower(), game.app_id))
    if not ranked:
        return ""
    ranked.sort()
    return ranked[0][6]


_STORE_SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
_NON_GAME_TYPES = {"dlc", "demo", "music", "bundle", "advertising", "mod", "video", "episode"}
_TOOL_TITLE_WORDS = {"redkit", "soundtrack", "sdk", "benchmark"}


def _title_is_tool(title: str) -> bool:
    """True for a kit, soundtrack, or benchmark that shares a game's name."""
    return any(token in _TOOL_TITLE_WORDS for token in steam_title_tokens(title))


def official_steam_title(app_info: Dict) -> str:
    """Steam's own app name, with the spacing collapsed."""
    return " ".join(str((app_info or {}).get("name") or "").split())


def _search_items(data: Any) -> list:
    """A Steam search body is either a list or an object with an items list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []


def _app_id_from_search_body(game_name: str, text: str) -> str:
    """Title-matched app id from a Steam search body, or empty when none fits."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ""
    return matching_steam_result(game_name, _search_items(data))


def matching_steam_result(game_name: str, items: list) -> str:
    """Closest titled Steam hit. A base game beats a demo, DLC, or soundtrack."""
    query_tokens = steam_title_tokens(game_name)
    if not query_tokens or not isinstance(items, list):
        return ""
    ranked: list[tuple] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("name") or item.get("title") or "").split())
        app_id = valid_steam_app_id(str(item.get("id") or item.get("appid") or ""))
        if not app_id or not title or not steam_title_matches(game_name, title):
            continue
        kind = str(item.get("type") or "app").lower()
        kind_rank = 1 if kind in _NON_GAME_TYPES or _title_is_tool(title) else 0
        extra = len(steam_title_tokens(title)) - len(query_tokens)
        edits = steam_title_match_cost(game_name, title) or 0
        order = steam_title_order_penalty(game_name, title)
        ranked.append((kind_rank, edits, extra, order, len(title), title.lower(), app_id))
    if not ranked:
        return ""
    ranked.sort()
    return ranked[0][6]


def best_title_app_id(query: str, rows: list[tuple[str, str]]) -> str:
    """Closest app id from (app id, title) rows."""
    items = [{"id": app_id, "name": title, "type": "app"} for app_id, title in rows]
    return matching_steam_result(query, items)


def _suggestion_titles(query: str, titles: list[str], limit: int = 3) -> list[str]:
    """Steam titles that start with the same word as the query."""
    query_tokens = steam_title_tokens(query)
    if not query_tokens:
        return []
    first = query_tokens[0]
    suggestions = []
    for title in titles:
        clean = " ".join(title.split())
        tokens = steam_title_tokens(clean)
        if tokens and tokens[0] == first and clean not in suggestions:
            suggestions.append(clean)
        if len(suggestions) >= limit:
            break
    return suggestions


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable - use temporary extraction path
        base_path = sys._MEIPASS
        data_path = os.path.join(base_path, relative_path)
        return data_path
    else:
        # Running as normal Python script - use relative path from project root
        base_path = Path(__file__).parent
        return os.path.join(base_path, relative_path)


@dataclass
class GameRequirements:
    """
    Data class for storing game requirements fetched from Steam API.
    Supports the S-A-B-C-D-F tier system for performance prediction.
    """
    game_name: str
    minimum_cpu: str
    minimum_gpu: str
    minimum_ram_gb: int
    minimum_vram_gb: int
    minimum_storage_gb: int
    minimum_directx: str = "DirectX 11"
    minimum_os: str = "Windows 10"
    recommended_cpu: str = "Unknown"
    recommended_gpu: str = "Unknown"
    recommended_ram_gb: int = 0
    recommended_vram_gb: int = 0
    recommended_storage_gb: int = 0
    recommended_directx: str = "DirectX 12"
    recommended_os: str = "Windows 11"
    source: str = "Unknown"
    last_updated: str = ""
    steam_api_name: str = ""  # Actual name returned by Steam API for reference
    steam_app_id: str = ""


class CanRunGameFetcher:
    """
    Main game requirements fetcher using Steam API exclusively.
    No caching - always fetches fresh data for accurate performance calculations.
    Data loaded once in constructor for efficiency.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load static data once in constructor (following user's efficiency pattern)
        self._load_static_data()
        self.recent_suggestions = []
        self._recent_lock = asyncio.Lock()
        self._recent_memory: tuple[float, list[tuple[str, str]]] | None = None
        self._catalog_refresh_started = False
        
        self.logger.info("CanRun game fetcher initialized")
    
    def _load_static_data(self):
        """Load all static parsing data once (following user's pattern)."""
        
        # Steam API endpoints for comprehensive game search (loaded once)
        self.base_url = "https://store.steampowered.com/api"
        self.search_url = "https://steamcommunity.com/actions/SearchApps"
        self.store_search_url = "https://store.steampowered.com/search/suggest"
        
        # Regex patterns to extract specific requirement fields (loaded once)
        self.requirement_patterns = {
            'os': r'OS:\s*([^<>\n]*?)(?=\s*(?:Processor|Memory|Graphics|DirectX|Storage|Sound|Additional|$))',
            'processor': r'Processor:\s*([^<>\n]*?)(?=\s*(?:Memory|Graphics|DirectX|Storage|Sound|Additional|$))',
            'memory': r'Memory:\s*([^<>\n]*?)(?=\s*(?:Graphics|DirectX|Storage|Sound|Additional|$))',
            'graphics': r'Graphics:\s*([^<>\n]*?)(?=\s*(?:DirectX|Storage|Sound|Additional|$))',
            'directx': r'DirectX:\s*([^<>\n]*?)(?=\s*(?:Storage|Sound|Additional|$))',
            'storage': r'Storage:\s*([^<>\n]*?)(?=\s*(?:Sound|Additional|$))',
            'sound': r'Sound Card:\s*([^<>\n]*?)(?=\s*(?:Additional|$))'
        }
        
        # App ID extraction patterns for various Steam responses (loaded once)
        self.app_id_patterns = [
            r'data-ds-appid="(\d+)"',
            r'"appid":\s*(\d+)',
            r'app/(\d+)/',
            r'appid=(\d+)'
        ]
        
        # Basic VRAM fallbacks (detailed VRAM estimation handled by hardware detector)
        self.vram_fallbacks = {
            'rtx': 8, 'gtx': 4, 'amd': 6, 'radeon': 6
        }

    async def fetch_requirements(self, game_name: str) -> Optional[GameRequirements]:
        """
        Fetch game requirements from Steam API only.
        Always returns fresh data - no caching involved.
        """
        try:
            # Direct Steam API call with timeout for G-Assist compatibility
            steam_requirements = await asyncio.wait_for(
                self._fetch_from_steam(game_name),
                timeout=15.0
            )
            
            if steam_requirements:
                # The query stays the cache key. The sentence uses Steam's own title.
                official = (steam_requirements.steam_api_name or "").strip()
                steam_requirements.game_name = game_name
                steam_requirements.steam_api_name = official or game_name
                
                self.logger.info(
                    "Fetched requirements for '%s' as Steam title '%s'",
                    game_name,
                    steam_requirements.steam_api_name,
                )
                return steam_requirements
            
            self.logger.warning(f"No requirements found for '{game_name}' in Steam API")
            return None

        except Exception as e:
            self.logger.error("Steam API fetch failed for '%s': %s", game_name, redact_secrets(e))
            return None

    async def _fetch_from_steam(self, game_name: str) -> Optional[GameRequirements]:
        """
        Fetch game requirements from Steam API using multiple search strategies.
        Returns fresh data every time - no caching involved.
        """
        try:
            tried: list[str] = []

            async def consider(app_id: Optional[str]) -> Optional[GameRequirements]:
                steam_id = valid_steam_app_id(str(app_id or ""))
                if not steam_id or steam_id in tried:
                    return None
                tried.append(steam_id)
                return await self._requirements_for_known_app(steam_id, game_name)

            found = await consider(choose_library_app(game_name, steam_library_games()))
            if found:
                return found

            for method in (
                self._search_steam_store_api,
                self._search_steam_store_suggest,
                self._search_steam_community,
                self._search_steam_store_direct,
            ):
                try:
                    found = await consider(await method(game_name))
                except Exception as e:
                    self.logger.debug("Search method %s failed: %s", method.__name__, e)
                    continue
                if found:
                    return found

            self.logger.warning(f"No Steam ID found for '{game_name}'")
            return None
            
        except Exception as e:
            self.logger.error("Steam API fetch failed for %s: %s", game_name, redact_secrets(e))
            return None

    async def _requirements_for_known_app(self, steam_id: str, game_name: str) -> Optional[GameRequirements]:
        """Accept an app id only when Steam's own name still matches the query."""
        app_info = await self._get_app_info(steam_id)
        if not app_info:
            self.logger.warning(f"No app info found for Steam ID {steam_id}")
            return None
        official = official_steam_title(app_info)
        if official and not steam_title_matches(game_name, official):
            self.logger.info(
                "Steam app %s is named %s, which does not match %s",
                steam_id,
                official,
                game_name,
            )
            return None
        requirements = self._parse_requirements(app_info, game_name)
        if not requirements:
            return None
        requirements.steam_app_id = steam_id
        requirements.steam_api_name = official
        requirements.game_name = game_name
        return requirements

    async def _search_game(self, game_name: str) -> Optional[str]:
        """
        Search for a game using multiple Steam API endpoints for maximum success rate.
        Returns the Steam App ID if found, None otherwise.
        """
        self.logger.debug(f"Searching Steam for game: {game_name}")
        self.recent_suggestions = []

        library_id = choose_library_app(game_name, steam_library_games())
        if library_id:
            self.logger.info("Found Steam library app %s for '%s'", library_id, game_name)
            return library_id
        
        # Multiple search strategies ordered by reliability and speed
        search_methods = [
            self._search_steam_store_api,       # Stable JSON. Ranked by title, not page order
            self._search_steam_store_suggest,   # Store suggest JSON
            self._search_steam_community,       # Community search JSON
            self._search_steam_store_direct     # HTML fallback when the JSON APIs have no titled hit
        ]
        
        # Try each search method until one succeeds
        for method in search_methods:
            try:
                steam_id = await method(game_name)
                if steam_id:
                    self.logger.info(f"Found Steam ID {steam_id} for '{game_name}' using {method.__name__}")
                    return steam_id
            except Exception as e:
                self.logger.debug(f"Search method {method.__name__} failed: {e}")
                continue
        
        self.logger.warning(f"All Steam search methods failed for '{game_name}'")
        return None

    async def _storesearch_items(self, session: aiohttp.ClientSession, term: str) -> list:
        """Items from the stable store-search JSON endpoint."""
        try:
            params = {"term": term, "l": "english", "cc": "US"}
            async with session.get(_STORE_SEARCH_URL, params=params, headers=STEAM_HEADERS) as response:
                if response.status != 200:
                    return []
                return _search_items(json.loads(await response.text()))
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            return []

    def _recent_catalog_path(self) -> Path:
        return Path(get_resource_path("cache")) / _RECENT_CATALOG

    def _read_recent_disk(self) -> tuple[float, list[tuple[str, str]]] | None:
        path = self._recent_catalog_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        fetched_at = payload.get("fetched_at")
        rows = payload.get("games")
        if payload.get("days") != _RECENT_DAYS or payload.get("bestsellers") != _BESTSELLER_COUNT:
            return None
        if not isinstance(fetched_at, (int, float)) or not isinstance(rows, list):
            return None
        games = [
            (valid_steam_app_id(str(app_id)), " ".join(str(name).split()))
            for app_id, name in rows
            if isinstance(app_id, str) and isinstance(name, str)
        ]
        games = [(app_id, name) for app_id, name in games if app_id and name]
        if len(games) < 50:
            return None
        return float(fetched_at), games

    def _store_recent(self, games: list[tuple[str, str]]) -> None:
        self._recent_memory = (time.time(), games)
        path = self._recent_catalog_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "fetched_at": self._recent_memory[0],
                    "days": _RECENT_DAYS,
                    "bestsellers": _BESTSELLER_COUNT,
                    "games": games,
                }),
                encoding="utf-8",
            )
        except OSError as error:
            self.logger.debug("Could not store the recent Steam catalog: %s", error)

    async def _download_recent_titles(self) -> list[tuple[str, str]]:
        """Windows games released in the last 30 days, newest first."""
        cutoff = datetime.now() - timedelta(days=_RECENT_DAYS)
        timeout = aiohttp.ClientTimeout(total=40, connect=5)
        semaphore = asyncio.Semaphore(4)

        async def page(session: aiohttp.ClientSession, start: int) -> tuple[list[tuple[str, str]], bool]:
            params = {
                "query": "",
                "start": start,
                "count": _RECENT_PAGE_SIZE,
                "sort_by": "Released_DESC",
                "infinite": 1,
                "category1": 998,
                "os": "win",
                "ignore_preferences": 1,
            }
            async with semaphore:
                for attempt in range(3):
                    try:
                        async with session.get(
                            _RECENT_SEARCH_URL, params=params, headers=STEAM_HEADERS
                        ) as response:
                            if response.status == 429:
                                await asyncio.sleep(1.0 + attempt)
                                continue
                            if response.status != 200:
                                return [], False
                            payload = json.loads(await response.text())
                    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
                        await asyncio.sleep(0.4)
                        continue
                    html_text = payload.get("results_html", "") if isinstance(payload, dict) else ""
                    older = any(
                        (when := _parse_release_date(released)) is not None and when < cutoff
                        for _app_id, _title, released in _RECENT_ROW.findall(html_text or "")
                    )
                    return recent_titles_from_search_html(html_text, cutoff), older or not html_text
            return [], False

        games: list[tuple[str, str]] = []
        seen: set[str] = set()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for start in range(0, _RECENT_PAGE_CAP * _RECENT_PAGE_SIZE, _RECENT_PAGE_SIZE * 4):
                    pages = await asyncio.gather(
                        *(
                            page(session, offset)
                            for offset in range(start, start + _RECENT_PAGE_SIZE * 4, _RECENT_PAGE_SIZE)
                            if offset < _RECENT_PAGE_CAP * _RECENT_PAGE_SIZE
                        )
                    )
                    passed_cutoff = False
                    for rows, older in pages:
                        for app_id, name in rows:
                            if app_id in seen:
                                continue
                            seen.add(app_id)
                            games.append((app_id, name))
                        passed_cutoff = passed_cutoff or older
                    if passed_cutoff:
                        break
                bestsellers = await self._download_bestsellers(session, semaphore)
        except Exception as error:
            self.logger.debug("Recent Steam catalog fetch failed: %s", error)
            return games
        games = merge_catalog_titles(games, bestsellers)
        self.logger.info(
            "Loaded %s Steam titles from the last %s days plus %s bestsellers",
            len(games),
            _RECENT_DAYS,
            _BESTSELLER_COUNT,
        )
        return games

    async def _download_bestsellers(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> list[tuple[str, str]]:
        """Current Windows topsellers. Release date is ignored."""
        games: list[tuple[str, str]] = []
        seen: set[str] = set()
        for start in range(0, _BESTSELLER_COUNT, _RECENT_PAGE_SIZE):
            params = {
                "query": "",
                "start": start,
                "count": _RECENT_PAGE_SIZE,
                "filter": "topsellers",
                "infinite": 1,
                "category1": 998,
                "os": "win",
                "ignore_preferences": 1,
            }
            html_text = ""
            async with semaphore:
                for attempt in range(3):
                    try:
                        async with session.get(
                            _RECENT_SEARCH_URL, params=params, headers=STEAM_HEADERS
                        ) as response:
                            if response.status == 429:
                                await asyncio.sleep(1.0 + attempt)
                                continue
                            if response.status != 200:
                                break
                            payload = json.loads(await response.text())
                    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
                        await asyncio.sleep(0.4)
                        continue
                    html_text = payload.get("results_html", "") if isinstance(payload, dict) else ""
                    break
            for app_id, name in recent_titles_from_search_html(html_text, datetime.min):
                if app_id in seen:
                    continue
                seen.add(app_id)
                games.append((app_id, name))
                if len(games) >= _BESTSELLER_COUNT:
                    return games
        return games

    def _refresh_catalog_later(self) -> None:
        """Fill the 30-day list off the request so a cold start still answers."""
        if self._catalog_refresh_started:
            return
        self._catalog_refresh_started = True

        def run() -> None:
            try:
                fresh = asyncio.run(self._download_recent_titles())
            except Exception as error:
                self.logger.debug("Recent Steam catalog refresh failed: %s", error)
                return
            if len(fresh) >= 50:
                self._store_recent(fresh)

        threading.Thread(target=run, name="canrun-recent-catalog", daemon=True).start()

    async def _recent_steam_titles(self) -> list[tuple[str, str]]:
        """Saved recent releases. A missing list does not block the Steam search."""
        async with self._recent_lock:
            now = time.time()
            if self._recent_memory and now - self._recent_memory[0] < _RECENT_TTL_SECONDS:
                return self._recent_memory[1]
            disk = self._read_recent_disk()
            if disk and now - disk[0] < _RECENT_TTL_SECONDS:
                self._recent_memory = disk
                return disk[1]
            stale = disk[1] if disk else []
        self._refresh_catalog_later()
        return stale

    async def _search_steam_store_api(self, game_name: str) -> Optional[str]:
        """
        Store search JSON. The response shape is {items: [{id, name, type}]}.
        Recent releases are matched locally first, so a one-letter slip still
        lands on the published title. Otherwise a doubled letter, a swap, or a
        short prefix is retried. Results stay ranked against the typed words.
        """
        try:
            catalog_id = choose_recent_title(game_name, await self._recent_steam_titles())
            if catalog_id:
                return catalog_id
            timeout = aiohttp.ClientTimeout(total=8, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                items = await self._storesearch_items(session, game_name)
                app_id = matching_steam_result(game_name, items)
                if app_id:
                    return app_id
                repairs = title_search_repairs(game_name)
                if not repairs:
                    return None
                batches = await asyncio.gather(
                    *(self._storesearch_items(session, repair) for repair in repairs)
                )
                combined: list = []
                for batch in batches:
                    combined.extend(batch)
                app_id = matching_steam_result(game_name, combined)
                return app_id or None
        except Exception as e:
            self.logger.debug(f"Steam storesearch failed for '{game_name}': {e}")
        return None

    async def _search_steam_store_suggest(self, game_name: str) -> Optional[str]:
        """
        Search using Steam Store suggest API - fastest and most accurate method.
        This endpoint is used by Steam's own search suggestions.
        """
        try:
            # Short timeout for G-Assist compatibility
            timeout = aiohttp.ClientTimeout(total=5, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                params = {
                    'term': game_name,
                    'f': 'games',        # Filter to games only
                    'cc': 'US',          # Country code for consistent results
                    'l': 'english'       # Language for parsing
                }
                
                async with session.get(self.store_search_url, params=params, headers=STEAM_HEADERS) as response:
                    if response.status == 200:
                        app_id = _app_id_from_search_body(game_name, await response.text())
                        if app_id:
                            return app_id
                        
        except Exception as e:
            self.logger.debug(f"Steam store suggest search failed for '{game_name}': {e}")
        
        return None

    async def _search_steam_community(self, game_name: str) -> Optional[str]:
        """
        Search using Steam Community API - good for partial matches and fuzzy search.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                params = {
                    'text': game_name,
                    'max_results': 10
                }
                
                async with session.get(self.search_url, params=params, headers=STEAM_HEADERS) as response:
                    if response.status == 200:
                        app_id = _app_id_from_search_body(game_name, await response.text())
                        if app_id:
                            return app_id
                                    
        except Exception as e:
            self.logger.debug(f"Steam community search failed for '{game_name}': {e}")
        
        return None

    async def _search_steam_store_direct(self, game_name: str) -> Optional[str]:
        """
        Direct search on Steam store page - most comprehensive but slowest method.
        Used as final fallback when other methods fail.
        """
        try:
            # Construct direct search URL
            search_url = steam_store_search_url(game_name)
            
            timeout = aiohttp.ClientTimeout(total=8, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(search_url, headers=STEAM_HEADERS) as response:
                    if response.status == 200:
                        text = await response.text()
                        rows = re.findall(
                            r'data-ds-appid="(\d+)"[\s\S]{0,2500}?class="title"[^>]*>([^<]+)',
                            text,
                        )
                        self.recent_suggestions = _suggestion_titles(game_name, [title for _, title in rows])
                        app_id = best_title_app_id(game_name, rows)
                        if app_id:
                            self.logger.info("Steam search matched %s for %s", app_id, game_name)
                            return app_id
                                
        except Exception as e:
            self.logger.debug(f"Steam store direct search failed for '{game_name}': {e}")
        
        return None

    async def _get_app_info(self, steam_id: str) -> Optional[Dict]:
        """
        Get detailed app information from Steam Store API.
        Includes retry logic for reliability and rate limiting handling.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url}/appdetails"
                params = {
                    'appids': steam_id,
                    'cc': 'US',          # Country code for consistent pricing/content
                    'l': 'english'       # Language for requirements text
                }
                
                # Retry logic for handling temporary failures and rate limits
                for attempt in range(3):
                    try:
                        self.logger.debug(f"Fetching Steam app info for ID {steam_id}, attempt {attempt + 1}")
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if steam_id in data and data[steam_id].get('success'):
                                    self.logger.debug(f"Successfully fetched app info for {steam_id}")
                                    return data[steam_id]['data']
                                else:
                                    self.logger.warning(f"Steam API returned unsuccessful response for {steam_id}")
                                    return None
                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt  # Exponential backoff
                                self.logger.warning(f"Rate limited by Steam API, waiting {wait_time}s")
                                if attempt < 2:
                                    await asyncio.sleep(wait_time)
                                continue
                            else:
                                self.logger.warning(f"Steam API returned status {response.status}")
                                return None
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Steam API timeout, attempt {attempt + 1}/3")
                        if attempt < 2:
                            await asyncio.sleep(1)
                        continue
                    except Exception as e:
                        self.logger.warning(f"Steam API error: {e}, attempt {attempt + 1}/3")
                        if attempt < 2:
                            await asyncio.sleep(1)
                        continue
                    
                    break  # Success or final failure
                        
        except Exception as e:
            self.logger.error("Steam app info fetch failed: %s", redact_secrets(e))
        
        return None

    def _parse_requirements(self, app_info: Dict, game_name: str) -> Optional[GameRequirements]:
        """
        Parse requirements from Steam app info into structured GameRequirements object.
        Handles both minimum and recommended specifications for the S-A-B-C-D-F tier system.
        """
        try:
            pc_requirements = app_info.get('pc_requirements', {})
            if not pc_requirements:
                self.logger.debug(f"No PC requirements found for {game_name}")
                return None
            
            # Parse minimum and recommended requirements separately
            minimum = self._parse_requirement_text(pc_requirements.get('minimum', ''))
            recommended = self._parse_requirement_text(pc_requirements.get('recommended', ''))
            
            official = official_steam_title(app_info)
            if official and not steam_title_matches(game_name, official):
                self.logger.info(
                    "Steam app name %s does not match query %s",
                    official,
                    game_name,
                )
                return None
            
            return GameRequirements(
                game_name=game_name,
                steam_api_name=official,
                **self._dict_to_dataclass_fields(minimum, recommended),
                source='Steam API',
                last_updated=str(int(time.time()))
            )
        except Exception as e:
            self.logger.debug(f"Steam requirements parsing failed: {e}")
            return None

    def _parse_requirement_text(self, text: str) -> Dict[str, str]:
        """
        Parse requirement text from Steam into structured format.
        Handles HTML tags, special characters, and various formatting styles.
        """
        requirements = {}
        
        # Clean HTML tags and normalize whitespace
        clean_text = re.sub(r'<[^>]+>', '\n', text)
        clean_text = re.sub(r'&nbsp;', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Apply each pattern using pre-loaded patterns to extract requirements
        for key, pattern in self.requirement_patterns.items():
            match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up trailing punctuation
                value = re.sub(r'[.,:;]+$', '', value).strip()
                if value:
                    requirements[key] = value
        
        # Fallback parsing if structured patterns fail
        if not requirements:
            self._fallback_requirement_parsing(clean_text, requirements)
        
        return requirements

    def _fallback_requirement_parsing(self, clean_text: str, requirements: Dict[str, str]):
        """
        Fallback method for parsing requirements when structured patterns fail.
        Uses line-by-line analysis to extract key-value pairs.
        """
        lines = re.split(r'[<>]|(?:\s*(?:Processor|Memory|Graphics|DirectX|Storage|Sound)\s*:)', clean_text)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line contains a requirement key
            if re.match(r'^(OS|Processor|Memory|Graphics|DirectX|Storage|Sound)', line, re.IGNORECASE):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    if key in ['os', 'processor', 'memory', 'graphics', 'directx', 'storage', 'sound']:
                        requirements[key] = value

    def _dict_to_dataclass_fields(self, minimum: Dict[str, str], recommended: Dict[str, str]) -> Dict[str, any]:
        """
        Convert parsed requirement dictionaries to GameRequirements dataclass fields.
        Implements intelligent parsing for the S-A-B-C-D-F tier system calculations.
        """
        def parse_storage(value: str) -> int:
            """Parse storage value like '25 GB' to integer GB."""
            if not value:
                return 0
            # Extract number from strings like "25 GB", "2.5 GB", etc.
            match = re.search(r'(\d+\.?\d*)', str(value))
            return int(float(match.group(1))) if match else 0
        
        def parse_ram(value: str) -> int:
            """
            Parse RAM value with proper MB/GB unit handling.
            Critical for accurate performance tier calculations.
            """
            if not value:
                return 0
                
            value_upper = str(value).upper()
            
            # Handle MB specifications (convert to GB)
            if 'MB' in value_upper:
                mb_match = re.search(r'(\d+\.?\d*)\s*MB', value_upper)
                if mb_match:
                    mb_value = float(mb_match.group(1))
                    if mb_value < 512:
                        return 1  # Minimum 1GB for very small values
                    else:
                        return max(1, int(mb_value / 1024))  # Convert MB to GB
            
            # Handle GB specifications
            gb_match = re.search(r'(\d+\.?\d*)\s*G?B?', value_upper)
            if gb_match:
                return int(float(gb_match.group(1)))
                
            return 0

        def estimate_vram_from_gpu(gpu_str: str) -> int:
            """
            Estimate VRAM from GPU model string using pre-loaded database.
            Based on known GPU specifications and market data.
            """
            if not gpu_str:
                return 2  # Conservative default
            
            gpu_lower = gpu_str.lower()
            
            # Look for explicit VRAM mention first
            vram_match = re.search(r'(\d+)\s*gb', gpu_lower)
            if vram_match:
                return int(vram_match.group(1))
            
            # Use pre-loaded VRAM fallbacks for common GPU types
            for gpu_type, vram in self.vram_fallbacks.items():
                if gpu_type in gpu_lower:
                    return vram
            
            return 2  # Conservative fallback
        
        # Calculate VRAM estimates for both minimum and recommended using pre-loaded data
        min_vram = estimate_vram_from_gpu(minimum.get('graphics', ''))
        rec_vram = estimate_vram_from_gpu(recommended.get('graphics', ''))
        
        # Ensure recommended VRAM is at least as high as minimum
        if rec_vram < min_vram:
            rec_vram = min_vram
        
        return {
            'minimum_cpu': minimum.get('processor', 'Unknown'),
            'minimum_gpu': minimum.get('graphics', 'Unknown'),
            'minimum_ram_gb': parse_ram(minimum.get('memory', '0')),
            'minimum_vram_gb': min_vram,
            'minimum_storage_gb': parse_storage(minimum.get('storage', '0')),
            'minimum_directx': minimum.get('directx', 'DirectX 11'),
            'minimum_os': minimum.get('os', 'Windows 10'),
            'recommended_cpu': recommended.get('processor', 'Unknown'),
            'recommended_gpu': recommended.get('graphics', 'Unknown'),
            'recommended_ram_gb': parse_ram(recommended.get('memory', '0')),
            'recommended_vram_gb': rec_vram,
            'recommended_storage_gb': parse_storage(recommended.get('storage', '0')),
            'recommended_directx': recommended.get('directx', 'DirectX 12'),
            'recommended_os': recommended.get('os', 'Windows 11')
        }

    async def batch_fetch(self, game_names: List[str]) -> Dict[str, Optional[GameRequirements]]:
        """
        Fetch requirements for multiple games concurrently.
        Each game gets fresh Steam API data - no caching.
        """
        tasks = []
        for game_name in game_names:
            task = asyncio.create_task(self.fetch_requirements(game_name))
            tasks.append((game_name, task))
        
        results = {}
        for game_name, task in tasks:
            try:
                results[game_name] = await task
            except Exception as e:
                self.logger.error(f"Batch fetch failed for {game_name}: {e}")
                results[game_name] = None
        
        return results

    def quick_requirements_summary(self, requirements: GameRequirements) -> Dict[str, Any]:
        """Quick summary of requirements for G-Assist responses."""
        if not requirements:
            return {}
            
        return {
            'game_name': requirements.game_name,
            'min_ram': f"{requirements.minimum_ram_gb}GB",
            'min_vram': f"{requirements.minimum_vram_gb}GB", 
            'min_gpu': requirements.minimum_gpu,
            'min_cpu': requirements.minimum_cpu,
            'rec_ram': f"{requirements.recommended_ram_gb}GB" if requirements.recommended_ram_gb > 0 else "N/A",
            'rec_vram': f"{requirements.recommended_vram_gb}GB" if requirements.recommended_vram_gb > 0 else "N/A",
            'source': requirements.source
        }


async def test_game_fetcher():
    """Test the game requirements fetcher with Steam API only."""
    fetcher = CanRunGameFetcher()
    
    print("Testing Steam API-only game fetch...")
    requirements = await fetcher.fetch_requirements("Cyberpunk 2077")
    if requirements:
        summary = fetcher.quick_requirements_summary(requirements)
        print(f"Game: {summary['game_name']}")
        print(f"Source: {summary['source']}")
        print(f"Min GPU: {summary['min_gpu']}")
        print(f"Min RAM: {summary['min_ram']}")
        print(f"Min VRAM: {summary['min_vram']}")
    else:
        print("No requirements found")


if __name__ == "__main__":
    asyncio.run(test_game_fetcher())