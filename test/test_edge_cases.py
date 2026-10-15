"""Edge cases, cybersecurity checks, and optimization checks for CanRun."""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "libs"))

from canrun_engine import sanitize_game_name, safe_cache_path
from canrun_game_fetcher import (
    LibraryGame,
    _app_id_from_search_body,
    choose_library_app,
    matching_steam_result,
    steam_library_games,
    steam_store_search_url,
    steam_title_matches,
)
from canrun_game_matcher import CanRunGameMatcher
from canrun_steam_fps import (
    SteamFpsClient,
    load_gpu_passmarks,
    lookup_cpu_passmark,
    measured_range,
    parse_frame_rate_payload,
    redact_secrets,
    resolve_fps,
    scale_from_reference,
    valid_steam_app_id,
)
from gassist_sdk.types import JsonRpcRequest
from plugin import (
    CanRunPlugin,
    extract_game_name,
    protocol_probe_requested,
)


def test_security_cache_path_cannot_escape_directory():
    matcher = CanRunGameMatcher()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.realpath(tmp)
        for raw in ("../../etc/passwd", "..\\..\\Windows\\System32", "game/name", "!!!", ""):
            normalized = matcher.normalize_game_name(raw)
            path = safe_cache_path(tmp, normalized)
            assert os.path.dirname(path) == root
            assert ".." not in os.path.basename(path)
            assert os.path.commonpath([root, path]) == root


def test_security_search_url_has_no_header_injection():
    url = steam_store_search_url("a\r\nHost: evil.example")
    assert "\n" not in url and "\r" not in url
    assert url.startswith("https://store.steampowered.com/search/?term=")


def test_security_cache_rejects_oversize_and_symlink():
    with tempfile.TemporaryDirectory() as tmp:
        path = safe_cache_path(tmp, "elden ring")
        Path(path).write_bytes(b"x" * (256 * 1024 + 1))
        from canrun_engine import cache_file_is_trusted
        assert cache_file_is_trusted(tmp, path) is False

        outside = Path(tmp) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = Path(tmp) / "link.json"
        try:
            os.symlink(outside, link)
        except OSError:
            pytest.skip("creating a symlink is not permitted here")
        assert cache_file_is_trusted(tmp, str(link)) is False


def test_security_cache_record_drops_extra_fields_and_bad_app_id():
    from canrun_engine import requirements_from_cache
    requirements = requirements_from_cache({
        "game_name": "Hades",
        "minimum_cpu": "i5",
        "minimum_gpu": "GTX 1060",
        "minimum_ram_gb": 8,
        "minimum_vram_gb": 3,
        "minimum_storage_gb": 20,
        "steam_app_id": "12;drop",
        "__import__": "os",
    })
    assert requirements.steam_app_id == ""
    assert not hasattr(requirements, "__import__")
    assert requirements.game_name == "Hades"


def test_security_long_cpu_pattern_is_not_compiled():
    assert lookup_cpu_passmark("Ryzen 7 7800X3D", [("a" * 200, 1), ("7800X3D", 38000)]) == 38000


def test_security_ui_stays_on_localhost_and_hides_errors():
    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text(encoding="utf-8")
    assert 'LOCAL_ONLY_HOST = "127.0.0.1"' in source
    assert "0.0.0.0" not in source
    assert "redact_secrets" in source
    assert "An error occurred during analysis: {e}" not in source
    assert "An error occurred during hardware detection: {e}" not in source


def test_security_steam_search_url_does_not_add_parameters():
    url = steam_store_search_url("half life & cmd=1")
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "store.steampowered.com"
    assert parsed.path == "/search/"
    assert parse_qs(parsed.query) == {"term": ["half life & cmd=1"]}


def test_steam_search_rejects_a_different_diablo_game():
    assert steam_title_matches("diablo 3", "Diablo III")
    assert steam_title_matches("diablo 4", "Diablo® IV")
    assert not steam_title_matches("diablo 3", "Diablo II: Resurrected – Infernal Edition")
    assert not steam_title_matches("diablo 3", "Diablo® IV")


def test_same_words_keep_the_published_order():
    items = [
        {"name": "Reincarnation Tower", "id": "1"},
        {"name": "Tower Reincarnation", "id": "2"},
    ]
    assert matching_steam_result("tower reincarnation", items) == "2"


def test_title_repairs_cover_a_doubled_letter_and_a_swap():
    from canrun_game_fetcher import title_search_repairs

    assert "doomsday landlord" in title_search_repairs("doomssday landlord")
    assert "doomsday landlord" in title_search_repairs("doosmday landlord")
    assert "doomsday landlord" in title_search_repairs("the doomsday landlord")
    assert "doom landlord" in title_search_repairs("doomday landlord")
    assert "dawnwalker" in title_search_repairs("Dawn of the Walkers")
    assert "dawnwalker" in title_search_repairs("Dawn Walker")


def test_recent_catalog_keeps_a_one_letter_slip_on_the_new_title():
    from canrun_game_fetcher import choose_recent_title, recent_titles_from_search_html
    from datetime import datetime

    games = [
        ("1", "Knights Within"),
        ("2", "Within"),
        ("11", "Reincarnation Tower"),
        ("12", "Tower Reincarnation"),
        ("10", "Doomsday Landlord"),
        ("3", "The Blood of Dawnwalker"),
        ("9", "Legend of the Sword"),
    ]
    assert choose_recent_title("doomday landlord", games) == "10"
    assert choose_recent_title("the doomsday landlord", games) == "10"
    assert choose_recent_title("tower reincarnation", games) == "12"
    assert choose_recent_title("within", games) == "2"
    assert choose_recent_title("the legend of the sword", games) == "9"
    assert choose_recent_title("dawnwalker", games) == ""
    assert choose_recent_title("Dawn Walker", games) == "3"
    assert choose_recent_title("Dawn of the Walkers", games) == "3"
    assert choose_recent_title("legend", games) == ""
    assert choose_recent_title("diablo 3", [("4", "Diablo IV")]) == ""
    both = games + [("4417550", "The Blood of Dawnwalker - Eclipse Edition Content")]
    assert choose_recent_title("Dawn Walker", both) == "3"
    from canrun_game_fetcher import merge_catalog_titles

    merged = merge_catalog_titles(
        [("10", "Doomsday Landlord")],
        [
            ("10", "Doomsday Landlord"),
            ("3751260", "The Blood of Dawnwalker"),
            ("730", "Counter-Strike 2"),
        ],
        bestseller_limit=2,
    )
    assert merged == [("10", "Doomsday Landlord"), ("3751260", "The Blood of Dawnwalker")]
    assert choose_recent_title("Dawn Walker", merged) == "3751260"
    old_bestseller = (
        '<a data-ds-appid="3751260"><span class="title">The Blood of Dawnwalker</span>'
        '<div class="search_released">2 Sep, 2026</div></a>'
    )
    assert recent_titles_from_search_html(old_bestseller, datetime.min) == [
        ("3751260", "The Blood of Dawnwalker")
    ]
    html_text = (
        '<a data-ds-appid="10"><span class="title">Doomsday &amp; Landlord</span>'
        '<div class="search_released">14 Oct, 2026</div></a>'
    )
    assert recent_titles_from_search_html(html_text, datetime(2026, 7, 14)) == [
        ("10", "Doomsday & Landlord")
    ]


@pytest.mark.asyncio
async def test_missing_recent_catalog_does_not_download_during_the_check(monkeypatch):
    from canrun_game_fetcher import CanRunGameFetcher

    fetcher = CanRunGameFetcher()

    async def explode():
        raise AssertionError("catalog download")

    monkeypatch.setattr(fetcher, "_download_recent_titles", explode)
    monkeypatch.setattr(fetcher, "_read_recent_disk", lambda: None)
    monkeypatch.setattr(fetcher, "_refresh_catalog_later", lambda: None)
    assert await fetcher._recent_steam_titles() == []


@pytest.mark.asyncio
async def test_store_search_uses_the_recent_catalog_before_steam(monkeypatch):
    from canrun_game_fetcher import CanRunGameFetcher

    fetcher = CanRunGameFetcher()

    async def titles():
        return [("10", "Doomsday Landlord")]

    async def explode(session, term):
        raise AssertionError(term)

    monkeypatch.setattr(fetcher, "_recent_steam_titles", titles)
    monkeypatch.setattr(fetcher, "_storesearch_items", explode)
    assert await fetcher._search_steam_store_api("doomday landlord") == "10"
    assert steam_title_matches("doomday landlord", "Doomsday Landlord")
    assert steam_title_matches("the doomsday landlord", "Doomsday Landlord")
    assert not steam_title_matches("within", "Knights Within")


def test_title_slips_still_match_the_same_game():
    assert steam_title_matches("dawnwlaker", "The Blood of Dawnwalker")
    assert steam_title_matches("dawnwalkerr", "The Blood of Dawnwalker")
    assert steam_title_matches("dawn walker", "The Blood of Dawnwalker")
    assert steam_title_matches("the elden ring", "Elden Ring")
    assert not steam_title_matches("dawnwalker", "Dawn Walker")
    assert not steam_title_matches("diablo 3", "Diablo® IV")
    assert not steam_title_matches("storm", "Store")
    assert steam_title_matches("sekiro", "Sekiro™: Shadows Die Twice - GOTY Edition")
    assert matching_steam_result(
        "The Witcher 3",
        [
            {"id": "2684660", "name": "The Witcher 3 REDkit", "type": "app"},
            {"id": "1229320", "name": "The Witcher 3: Wild Hunt — Remastered Soundtrack", "type": "app"},
            {"id": "292030", "name": "The Witcher 3: Wild Hunt — Remastered", "type": "app"},
        ],
    ) == "292030"


def test_missing_ram_requirement_does_not_divide_by_zero():
    from canrun_ml_predictor import CanRunMLPredictor

    predictor = CanRunMLPredictor()
    score = predictor._get_ram_score(
        {"total": 32, "available": 24},
        {"recommended_ram_gb": 0, "minimum_ram_gb": 0},
    )
    assert score == 100


def test_dawnwalker_library_prefers_the_installed_title():
    assert steam_title_matches("dawnwalker", "The Blood of Dawnwalker")
    assert not steam_title_matches("dawnwalker", "Dawn Walker")
    assert steam_title_matches("Dawn of the Walkers", "The Blood of Dawnwalker")
    assert not steam_title_matches("Dawn of the Walkers", "Dawn Walker")
    assert not steam_title_matches("Dawn of the Walkers", "Dawn of the Monsters")
    assert matching_steam_result(
        "Dawn of the Walkers",
        [
            {"id": "4417550", "name": "The Blood of Dawnwalker - Eclipse Edition Content", "type": "app"},
            {"id": "3751260", "name": "The Blood of Dawnwalker", "type": "app"},
        ],
    ) == "3751260"
    assert matching_steam_result(
        "Dawn Walker",
        [
            {"id": "4417550", "name": "The Blood of Dawnwalker - Eclipse Edition Content", "type": "app"},
            {"id": "3751260", "name": "The Blood of Dawnwalker", "type": "app"},
        ],
    ) == "3751260"
    games = [
        LibraryGame("111", "Dawn Walker", True),
        LibraryGame("222", "Dawnwalker", False),
        LibraryGame("3751260", "The Blood of Dawnwalker", True),
    ]
    assert choose_library_app("Dawnwalker", games) == "3751260"

    both_installed = [
        LibraryGame("9", "Dawnwalker and the Long Night Saga", True),
        LibraryGame("3751260", "The Blood of Dawnwalker", True),
    ]
    assert choose_library_app("Dawnwalker", both_installed) == "3751260"


def test_dawnwalker_resolves_from_libraryfolders():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        steam = root / "Steam"
        library = root / "SteamLibrary"
        (steam / "steamapps").mkdir(parents=True)
        (library / "steamapps").mkdir(parents=True)
        library_path = str(library).replace("\\", "\\\\")
        (steam / "steamapps" / "libraryfolders.vdf").write_text(
            '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"' + library_path + '"\n\t}\n}\n',
            encoding="utf-8",
        )
        (steam / "steamapps" / "appmanifest_222.acf").write_text(
            '"AppState"\n{\n\t"appid"\t\t"222"\n\t"name"\t\t"Dawnwalker"\n\t"StateFlags"\t\t"2"\n}\n',
            encoding="utf-8",
        )
        (library / "steamapps" / "appmanifest_111.acf").write_text(
            '"AppState"\n{\n\t"appid"\t\t"111"\n\t"name"\t\t"Dawn Walker"\n\t"StateFlags"\t\t"4"\n}\n',
            encoding="utf-8",
        )
        (library / "steamapps" / "appmanifest_3751260.acf").write_text(
            '"AppState"\n{\n\t"appid"\t\t"3751260"\n\t"name"\t\t"The Blood of Dawnwalker"\n\t"StateFlags"\t\t"4"\n}\n',
            encoding="utf-8",
        )
        assert choose_library_app("Dawnwalker", steam_library_games([steam])) == "3751260"


def test_suggest_skips_a_wrong_first_hit():
    items = [
        {"name": "Dawn Walker", "id": "111"},
        {"name": "The Blood of Dawnwalker", "id": "3751260"},
    ]
    assert matching_steam_result("Dawnwalker", items) == "3751260"
    body = (
        '[{"name":"Something Else","appid":"1"},'
        '{"name":"The Blood of Dawnwalker","id":3751260}]'
    )
    assert _app_id_from_search_body("dawnwalker", body) == "3751260"


def test_bare_cyberpunk_selects_2077():
    items = [
        {"name": "Cyberpunk", "id": "1", "type": "app"},
        {"name": "Cyberpunk 2077: Phantom Liberty", "id": "2138330", "type": "dlc"},
        {"name": "Cyberpunk 2077", "id": "1091500", "type": "app"},
    ]
    assert matching_steam_result("cyberpunk", items) == "1091500"
    assert matching_steam_result("Cyberpunk 2077", items) == "1091500"


def test_suggest_prefers_the_closer_base_game():
    items = [
        {"name": "Dawnwalker", "id": "9", "type": "demo"},
        {"name": "The Blood of Dawnwalker Soundtrack", "id": "8", "type": "music"},
        {"name": "The Blood of Dawnwalker", "id": "3751260", "type": "app"},
    ]
    assert matching_steam_result("dawnwalker", items) == "3751260"
    wrapped = (
        '{"items":['
        '{"name":"The Blood of Dawnwalker Soundtrack","id":8,"type":"music"},'
        '{"name":"The Blood of Dawnwalker","id":3751260,"type":"app"}'
        ']}'
    )
    assert _app_id_from_search_body("dawnwalker", wrapped) == "3751260"


def test_official_steam_title_survives_a_short_query():
    from canrun_game_fetcher import CanRunGameFetcher, GameRequirements

    fetcher = CanRunGameFetcher()
    app_info = {
        "name": "The Blood of Dawnwalker",
        "pc_requirements": {
            "minimum": "OS: Windows 10 Processor: Intel Core i5 Memory: 16 GB RAM Graphics: NVIDIA GTX 1060 DirectX: Version 12 Storage: 60 GB",
            "recommended": "OS: Windows 10 Processor: Intel Core i7 Memory: 16 GB RAM Graphics: NVIDIA RTX 4060 DirectX: Version 12 Storage: 60 GB",
        },
    }
    requirements = fetcher._parse_requirements(app_info, "dawnwalker")
    assert requirements is not None
    assert requirements.game_name == "dawnwalker"
    assert requirements.steam_api_name == "The Blood of Dawnwalker"
    assert fetcher._parse_requirements({**app_info, "name": "Dawn Walker"}, "dawnwalker") is None

    kept = GameRequirements(
        game_name="dawnwalker",
        steam_api_name="The Blood of Dawnwalker",
        minimum_cpu="Intel Core i5",
        minimum_gpu="NVIDIA GTX 1060",
        minimum_ram_gb=16,
        minimum_vram_gb=6,
        minimum_storage_gb=60,
    )

    async def fake_fetch(game_name):
        assert game_name == "dawnwalker"
        return kept

    fetcher._fetch_from_steam = fake_fetch
    import asyncio
    result = asyncio.run(fetcher.fetch_requirements("dawnwalker"))
    assert result.steam_api_name == "The Blood of Dawnwalker"
    assert result.game_name == "dawnwalker"


@pytest.mark.asyncio
async def test_search_checks_the_library_before_the_store(monkeypatch):
    from canrun_game_fetcher import CanRunGameFetcher

    monkeypatch.setattr(
        "canrun_game_fetcher.steam_library_games",
        lambda client_roots=None: [LibraryGame("3751260", "The Blood of Dawnwalker", True)],
    )
    fetcher = CanRunGameFetcher()

    async def fail(game_name):
        raise AssertionError(game_name)

    fetcher._search_steam_store_suggest = fail
    fetcher._search_steam_community = fail
    fetcher._search_steam_store_direct = fail
    assert await fetcher._search_game("Dawnwalker") == "3751260"


def test_missing_steam_game_is_a_user_message():
    from plugin import missing_steam_game_message

    message = missing_steam_game_message(
        "diablo 3",
        ["Diablo II: Resurrected – Infernal Edition", "El Taco Diablo"],
    )
    assert "could not find" in message
    assert "Battle.net" in message
    assert "Diablo IV" in message
    assert "Diablo II: Resurrected" in message
    assert "El Taco Diablo" not in message
    assert "Error" not in message


def test_security_app_id_must_be_numeric():
    assert valid_steam_app_id("1091500") == "1091500"
    assert valid_steam_app_id(" 42 ") == "42"
    assert valid_steam_app_id("12abc") == ""
    assert valid_steam_app_id("../10") == ""
    assert valid_steam_app_id("1" * 13) == ""
    assert valid_steam_app_id("") == ""


@pytest.mark.asyncio
async def test_security_non_numeric_app_id_is_not_fetched():
    client = SteamFpsClient(api_key="not-sent")
    called = {"n": 0}

    async def fail_if_called(appid):
        called["n"] += 1
        raise AssertionError(appid)

    client._fetch = fail_if_called
    estimate = await client.estimate("12;key=stolen", "RTX 4070", "Ryzen 7 7800X3D")
    assert called["n"] == 0
    assert estimate.fps_source == "unknown"


def test_security_api_key_is_redacted():
    secret = "super-secret-key"
    logged = redact_secrets(
        f"ClientError: GET https://api.steampowered.com/?key={secret}&appid=10",
        secret,
    )
    assert secret not in logged
    assert "key=[redacted]" in logged


def test_security_game_name_drops_controls_and_caps_length():
    assert "\n" not in sanitize_game_name("Elden\nRing")
    assert "\x00" not in sanitize_game_name("Elden\x00Ring")
    assert sanitize_game_name("   ") == ""
    assert len(sanitize_game_name("A" * 500)) == 120


def test_security_malformed_steam_payload_does_not_raise():
    payload = {
        "response": {
            "frame_rates": [
                "nope",
                {"mean_frame_rate": "not-a-number", "report_count": 40},
                {"mean_frame_rate": None},
                {"gpu_name": "RTX 4070", "report_count": "lots", "mean_frame_rate": 60, "mean_frame_rate_stddev": "bad"},
            ]
        }
    }
    buckets = parse_frame_rate_payload(payload)
    assert len(buckets) == 1
    assert buckets[0].mean_fps == 60
    assert buckets[0].report_count == 0
    assert buckets[0].stddev == 0.0
    estimate = resolve_fps("RTX 4070", "cpu", payload, api_available=True, cluster_map={"clusters": {}})
    assert estimate.fps_source in {"steam", "steam_scaled", "unknown"}


def test_security_protocol_probe_requires_explicit_env(monkeypatch):
    monkeypatch.delenv("CANRUN_PROTOCOL_PROBE", raising=False)
    assert protocol_probe_requested("protocol-probe") is False
    monkeypatch.setenv("CANRUN_PROTOCOL_PROBE", "1")
    assert protocol_probe_requested("protocol-probe") is True
    assert protocol_probe_requested("cyberpunk") is False


def test_edge_extract_game_name():
    assert extract_game_name("canrun?") == ""
    assert extract_game_name("CANRUN   Hades") == "hades"
    assert extract_game_name("Elden Ring?") == "Elden Ring"
    assert extract_game_name("Can I run Diablo IV") == "Diablo IV"
    assert extract_game_name("Can I run Diablo IV?") == "Diablo IV"
    assert extract_game_name("can my pc run Hogwarts Legacy") == "Hogwarts Legacy"
    assert extract_game_name("") == ""
    assert extract_game_name("can I run cyberpunk") == "Cyberpunk 2077"
    assert extract_game_name("cyberpunk?") == "Cyberpunk 2077"
    assert extract_game_name("Cyberpunk 2077") == "Cyberpunk 2077"
    assert extract_game_name("cyberpunk phantom liberty") == "cyberpunk phantom liberty"


def test_edge_fps_ranges_are_clamped_and_ordered():
    high = measured_range(5000, 50)
    assert high.expected_fps == 1000
    assert high.fps_min <= high.expected_fps <= high.fps_max
    floor = measured_range(0, 0)
    assert floor.expected_fps == 1
    wide = measured_range(30, -80)
    assert wide.fps_min <= wide.expected_fps <= wide.fps_max
    assert scale_from_reference(60, 5, 10000, 0, 1.0).fps_source == "unknown"


def test_new_game_fps_scales_from_the_recommended_gpu():
    from canrun_steam_fps import estimate_from_requirements

    estimate = estimate_from_requirements(
        "RTX 4090",
        "AMD Ryzen 7 7800X3D",
        "NVIDIA RTX 4060 / Radeon RX 7600 XT / Intel Arc B580",
        "GTX 1060",
        "1440p",
    )
    assert estimate.fps_source == "requirements"
    assert estimate.fps_min < estimate.expected_fps < estimate.fps_max
    assert estimate.expected_fps > 60

    on_the_spec = estimate_from_requirements(
        "RTX 4060",
        "AMD Ryzen 7 7800X3D",
        "NVIDIA RTX 4060",
        "",
        "1080p",
    )
    assert 50 <= on_the_spec.expected_fps <= 70

    weaker = estimate_from_requirements(
        "GTX 1650",
        "AMD Ryzen 7 7800X3D",
        "NVIDIA RTX 4060",
        "",
        "1080p",
    )
    assert weaker.expected_fps < on_the_spec.expected_fps
    assert estimate_from_requirements("RTX 4090", "AMD Ryzen 7 7800X3D", "not a gpu", "", "1080p").fps_source == "requirements_unspecified"
    typed = estimate_from_requirements("RTX 4090", "AMD Ryzen 7 7800X3D", "GTX 3060", "", "1440p")
    real = estimate_from_requirements("RTX 4090", "AMD Ryzen 7 7800X3D", "RTX 3060", "", "1440p")
    assert typed.fps_source == "requirements"
    assert typed.expected_fps == real.expected_fps
    bare = estimate_from_requirements("RTX 4090", "AMD Ryzen 7 7800X3D", "GTX 1060+", "", "1440p")
    full = estimate_from_requirements("RTX 4090", "AMD Ryzen 7 7800X3D", "GTX 1060 6GB", "", "1440p")
    assert bare.expected_fps == full.expected_fps
    amd = estimate_from_requirements("RTX 4090", "AMD Ryzen 7 7800X3D", "AMD Radeon RX 580", "", "1440p")
    assert amd.fps_source == "requirements"
    assert amd.fps_min <= amd.expected_fps <= amd.fps_max
    tight = measured_range(120, 8)
    assert tight.expected_fps == 120
    assert tight.fps_min == 110
    assert tight.fps_max == 130
    kept = measured_range(100, 20)
    assert kept.fps_min == 80
    assert kept.fps_max == 120
    similar = scale_from_reference(100, 1, 10000, 10000, 1.0)
    assert similar.expected_fps == 100
    assert similar.fps_min == 70
    assert similar.fps_max == 130


def test_edge_invalid_cpu_pattern_is_skipped():
    assert lookup_cpu_passmark("Ryzen 7 7800X3D", [("[", 100), ("7800x3d", 42000)]) == 42000
    assert lookup_cpu_passmark("x" * 500, [(".", 1)]) is None


def test_edge_unknown_command_returns_jsonrpc_error():
    plugin, sent = _plugin_with_capture()
    plugin._handle_execute(JsonRpcRequest(method="execute", id=7, params={"function": "missing", "arguments": {}}))
    assert sent[0]["id"] == 7
    assert sent[0]["error"]["code"] == -32601


def test_optimization_second_execute_is_rejected_while_busy():
    plugin, sent = _plugin_with_capture()
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        assert release.wait(2)
        return "done"

    plugin.command("slow")(slow)
    plugin._handle_execute(JsonRpcRequest(method="execute", id=1, params={"function": "slow", "arguments": {}}))
    assert started.wait(1)
    plugin._handle_execute(JsonRpcRequest(method="execute", id=2, params={"function": "slow", "arguments": {}}))
    release.set()
    deadline = time.time() + 2
    while plugin._execute_lock.locked() and time.time() < deadline:
        time.sleep(0.01)
    errors = [item for item in sent if item.get("method") == "error"]
    completes = [item for item in sent if item.get("method") == "complete"]
    assert errors and errors[0]["params"]["message"] == "Another command is already running"
    assert completes and completes[0]["params"]["data"] == "done"


def test_optimization_reference_tables_are_loaded_once(monkeypatch):
    load_gpu_passmarks.cache_clear()
    reads = {"n": 0}
    original = Path.read_text

    def counting(self, *args, **kwargs):
        if self.name == "gpu_hierarchy.json":
            reads["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    first = load_gpu_passmarks()
    second = load_gpu_passmarks()
    assert first is second
    assert reads["n"] == 1
    assert len(first) > 0


def test_optimization_cached_fps_resolution_stays_fast():
    payload = {
        "response": {
            "frame_rates": [
                {
                    "gpu_name": "RTX 4070",
                    "cpu_name": "AMD Ryzen 7 7800X3D",
                    "report_count": 80,
                    "mean_frame_rate": 90,
                    "mean_frame_rate_stddev": 6,
                }
            ]
        }
    }
    resolve_fps("RTX 4070", "AMD Ryzen 7 7800X3D", payload, cluster_map={"clusters": {}})
    started = time.perf_counter()
    for _ in range(40):
        estimate = resolve_fps("RTX 4070", "AMD Ryzen 7 7800X3D", payload, cluster_map={"clusters": {}})
    elapsed = time.perf_counter() - started
    assert estimate.fps_source == "steam"
    assert elapsed < 0.5


def _plugin_with_capture():
    plugin = CanRunPlugin("canrun", version="12.0.0")
    sent = []

    class Protocol:
        def send_response(self, response):
            sent.append(response.to_dict())
            return True

        def send_notification(self, notification):
            sent.append(notification.to_dict())
            return True

    plugin._protocol = Protocol()
    return plugin, sent
