"""
Published review measurements, used to train and to test CanRun on real FPS.

The numbers are transcribed from review charts into data/real_benchmarks/ (gitignored,
the reviews own them). Steam store details for those games are cached next to them.

    py -3 scripts/ds_pipeline/real_benchmarks.py fetch-store     # cache Steam store details
    py -3 scripts/ds_pipeline/real_benchmarks.py evaluate        # score the app on every point
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

REAL_DIR = ROOT_DIR / "data" / "real_benchmarks"
STORE_CACHE = REAL_DIR / "store.json"

# Review slug -> Steam app id. None: not sold on Steam, so the app has no store specs for it.
STEAM_APP_IDS = {
    "alan-wake-2": None,
    "assassins-creed-shadows": "3159330",
    "baldurs-gate-3": "1086940",
    "black-myth-wukong": "2358720",
    "borderlands-4": "1285190",
    "expedition-33": "1903340",
    "counter-strike-2": "730",
    "cyberpunk-2077": "1091500",
    "death-stranding-2": "3280350",
    "f1-25": "3059520",
    "ghost-of-tsushima": "2215430",
    "god-of-war-ragnarok": "2322010",
    "hogwarts-legacy": "990080",
    "kingdom-come-2": "1771300",
    "monster-hunter-wilds": "2246340",
    "resident-evil-requiem": "3764200",
    "spider-man-2": "2651280",
    "stalker-2": "1643320",
    "starfield": "1716740",
    "star-wars-outlaws": "2842040",
    "the-last-of-us-part-2": "2531310",
    "the-witcher-3": "292030",
    "battlefield-6": "2807960",
    "crimson-desert": "3321460",
    "forza-horizon-6": "2483190",
    "doom-the-dark-ages": "3017860",
    "silent-hill-f": "2947440",
    "mafia-the-old-country": "1941540",
    "dying-light-the-beast": "3008130",
    "the-outer-worlds-2": "1449110",
    "007-first-light": "3768760",
    "pragmata": "3357650",
}

_STRIP_TAGS = re.compile(r"<[^>]+>")


def _flatten(html: str) -> str:
    return re.sub(r"\s+", " ", _STRIP_TAGS.sub(" ", str(html or ""))).strip()


def fetch_store(app_ids: list[str]) -> dict:
    cache = json.loads(STORE_CACHE.read_text(encoding="utf-8")) if STORE_CACHE.exists() else {}
    for app_id in app_ids:
        if app_id in cache:
            continue
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=english&cc=us"
        req = urllib.request.Request(url, headers={"User-Agent": "CanRunEval/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                entry = json.loads(resp.read().decode("utf-8"))[app_id]
        except Exception as exc:  # noqa: BLE001
            print(f"{app_id}: {type(exc).__name__} {exc}")
            continue
        if not entry.get("success"):
            print(f"{app_id}: no store data")
            continue
        data = entry["data"]
        pc = data.get("pc_requirements") if isinstance(data.get("pc_requirements"), dict) else {}
        cache[app_id] = {
            "name": data.get("name", ""),
            "release_date": (data.get("release_date") or {}).get("date", ""),
            "minimum_raw": _flatten(pc.get("minimum")),
            "recommended_raw": _flatten(pc.get("recommended")),
            "genres": ", ".join(g.get("description", "") for g in data.get("genres") or []),
        }
        print(f"{app_id}: {cache[app_id]['name']}")
        time.sleep(1.0)
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    STORE_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def card_name(label: str) -> tuple[str, float | None]:
    """'RTX 5060 Ti 8 GB' -> ('RTX 5060 Ti', 8.0). Partner names such as 'Acer RX 9070 GRE 12 Nitro GB' keep the model."""
    vram = re.search(r"(\d+)\s*(?:Nitro\s*)?GB", label)
    model = re.search(r"((?:RTX|GTX|RX|Arc)\s+[A-Z]?\d{3,4}(?:\s+(?:Ti|Super|XTX|XT|GRE))*)", label)
    return (model.group(1) if model else label.strip()), (float(vram.group(1)) if vram else None)


def load_points() -> list[dict]:
    """Every (game, resolution, card, measured FPS) from the transcribed reviews."""
    points = []
    for path in sorted(REAL_DIR.glob("tpu_*.json")):
        review = json.loads(path.read_text(encoding="utf-8"))
        for slug, game in review["games"].items():
            for res, cards in game["fps"].items():
                for label, fps in cards.items():
                    card, vram = card_name(label)
                    points.append({
                        "review": path.stem, "slug": slug, "title": game["title"], "released": game.get("released"),
                        "res": res, "card": card, "vram": vram, "fps": float(fps), "cpu": game.get("cpu", review["cpu"]),
                    })
    return points


def metrics(pairs: list[tuple[float, float]]) -> dict:
    ape = [abs(p - t) / t * 100 for t, p in pairs]
    return {
        "n": len(pairs),
        "MdAPE": round(statistics.median(ape), 1),
        "MAPE": round(statistics.fmean(ape), 1),
        "within10": round(sum(a <= 10 for a in ape) / len(ape) * 100, 1),
        "within25": round(sum(a <= 25 for a in ape) / len(ape) * 100, 1),
        "bias": round(statistics.fmean((p - t) / t * 100 for t, p in pairs), 1),
    }


def store_inputs(entry: dict) -> dict:
    from scripts.ds_pipeline.collect_and_audit import _extract_field_from_raw_req, _extract_ram_gb

    year = re.search(r"\b(19|20)\d\d\b", entry.get("release_date", ""))
    return {
        "name": entry["name"],
        "rec_gpu": _extract_field_from_raw_req(entry["recommended_raw"], "(?:Graphics|Video Card)") or "",
        "min_gpu": _extract_field_from_raw_req(entry["minimum_raw"], "(?:Graphics|Video Card)") or "",
        "rec_cpu": _extract_field_from_raw_req(entry["recommended_raw"], "Processor") or "",
        "rec_ram": float(_extract_ram_gb(entry["recommended_raw"], 16)),
        "year": int(year.group(0)) if year else 0,
        "genre": entry.get("genres", ""),
    }


def evaluate() -> int:
    from canrun_steam_fps import (
        estimate_from_requirements, load_cpu_patterns, load_gpu_passmarks, lookup_gpu_passmark,
        requirement_passmarks, resolve_fps, first_known_gpu,
    )

    store = json.loads(STORE_CACHE.read_text(encoding="utf-8"))
    gpu_pms = load_gpu_passmarks()
    cards = requirement_passmarks(gpu_pms)
    cpu_patterns = load_cpu_patterns()
    rows = {"app": [], "store_spec": [], "rule": []}
    per_game: dict[str, dict[str, list]] = {}
    skipped_cards: set[str] = set()
    for p in load_points():
        app_id = STEAM_APP_IDS.get(p["slug"])
        if not app_id or app_id not in store or p["res"] not in ("1080p", "1440p", "4K"):
            continue
        s = store_inputs(store[app_id])
        if not lookup_gpu_passmark(p["card"], cards):
            skipped_cards.add(p["card"])
            continue
        spec = estimate_from_requirements(
            p["card"], p["cpu"], s["rec_gpu"], s["min_gpu"], p["res"], gpu_passmarks=gpu_pms,
            cpu_patterns=cpu_patterns, game_name=s["name"], user_vram_gb=p["vram"] or 0.0,
            recommended_cpu=s["rec_cpu"], recommended_ram_gb=s["rec_ram"], release_year=s["year"], genre=s["genre"],
        )
        shown = resolve_fps(p["card"], p["cpu"], None, api_available=False, gpu_passmarks=gpu_pms,
                            cpu_patterns=cpu_patterns, game_name=s["name"], resolution=p["res"])
        if shown.fps_source == "unknown" or shown.expected_fps <= 0:
            shown = spec
        rec = first_known_gpu(s["rec_gpu"], cards) or first_known_gpu(s["min_gpu"], cards)
        gpu_pm = lookup_gpu_passmark(p["card"], cards)
        rule = 60.0 * (gpu_pm / rec[1]) ** 0.85 * {"1080p": 1.0, "1440p": 0.75, "4K": 0.42}[p["res"]] if rec else None
        if spec.expected_fps <= 0:
            continue
        rows["app"].append((p["fps"], float(shown.expected_fps)))
        rows["store_spec"].append((p["fps"], float(spec.expected_fps)))
        if rule:
            rows["rule"].append((p["fps"], rule))
        g = per_game.setdefault(p["title"], {"app": [], "store_spec": [], "src": set()})
        g["app"].append((p["fps"], float(shown.expected_fps)))
        g["store_spec"].append((p["fps"], float(spec.expected_fps)))
        g["src"].add(shown.fps_source)
    print("Cards without a PassMark score (skipped):", sorted(skipped_cards))
    for key, pairs in rows.items():
        print(f"  {key:10s} {metrics(pairs)}")
    print("Per game, median abs % error (app | store-spec model), app source:")
    for title, g in sorted(per_game.items()):
        print(f"  {title:32s} {metrics(g['app'])['MdAPE']:6.1f} | {metrics(g['store_spec'])['MdAPE']:6.1f}  bias "
              f"{metrics(g['store_spec'])['bias']:+6.1f}  {sorted(g['src'])}")
    return 0


def spec_rows(captured: list | None = None) -> list[dict]:
    """Store-spec model prediction for every review point, as the app computes it for an unknown game.

    When `captured` is a list the model hook appends to, each row gets the feature vector as "x".
    """
    from canrun_steam_fps import (
        estimate_from_requirements, first_known_gpu, load_cpu_patterns, load_gpu_passmarks, lookup_gpu_passmark,
        requirement_passmarks,
    )

    store = json.loads(STORE_CACHE.read_text(encoding="utf-8"))
    gpu_pms = load_gpu_passmarks()
    cards = requirement_passmarks(gpu_pms)
    cpu_patterns = load_cpu_patterns()
    rows = []
    for p in load_points():
        app_id = STEAM_APP_IDS.get(p["slug"])
        if not app_id or app_id not in store:
            continue
        s = store_inputs(store[app_id])
        gpu_pm = lookup_gpu_passmark(p["card"], cards)
        rec = first_known_gpu(s["rec_gpu"], cards) or first_known_gpu(s["min_gpu"], cards)
        if not gpu_pm or not rec:
            continue
        if captured is not None:
            captured.clear()
        spec = estimate_from_requirements(
            p["card"], p["cpu"], s["rec_gpu"], s["min_gpu"], p["res"], gpu_passmarks=gpu_pms,
            cpu_patterns=cpu_patterns, game_name=s["name"], user_vram_gb=p["vram"] or 0.0,
            recommended_cpu=s["rec_cpu"], recommended_ram_gb=s["rec_ram"], release_year=s["year"], genre=s["genre"],
        )
        if spec.expected_fps <= 0:
            continue
        rows.append({**p, "pred": float(spec.expected_fps), "gpu_pm": float(gpu_pm), "rec_pm": float(rec[1]),
                     "year": s["year"], "vendor": _vendor(p["card"]),
                     "x": captured[-1] if captured else None})
    return rows


def _least_squares(xs: list[list[float]], ys: list[float], ridge: float) -> list[float]:
    """Ridge-regularized normal equations, solved by Gaussian elimination."""
    k = len(xs[0])
    a = [[sum(x[i] * x[j] for x in xs) + (ridge if i == j and i > 0 else 0.0) for j in range(k)] for i in range(k)]
    b = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(a[r][col]))
        a[col], a[pivot], b[col], b[pivot] = a[pivot], a[col], b[pivot], b[col]
        for r in range(k):
            if r != col and a[col][col]:
                f = a[r][col] / a[col][col]
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
                b[r] -= f * b[col]
    return [b[i] / a[i][i] if a[i][i] else 0.0 for i in range(k)]


CALIBRATION_FEATURES = {
    "none": None,
    "offset": lambda r: [1.0],
    "offset_res": lambda r: [1.0, r["res"] == "1440p", r["res"] == "4K"],
    "headroom": lambda r: [1.0, r["res"] == "1440p", r["res"] == "4K", math.log(r["gpu_pm"] / r["rec_pm"])],
    "headroom_year": lambda r: [1.0, r["res"] == "1440p", r["res"] == "4K", math.log(r["gpu_pm"] / r["rec_pm"]),
                                (r["year"] - 2023) / 3.0],
}


def calibrate(vendor: str = "nvidia") -> int:
    """Leave-one-game-out: fit log(measured / model) on the other games, apply it to the held-out game."""
    rows = [r for r in spec_rows() if vendor == "all" or r["vendor"] == vendor]
    games = sorted({r["slug"] for r in rows})
    print(f"{len(rows)} points, {len(games)} games, vendor {vendor}")
    for name, feats in CALIBRATION_FEATURES.items():
        pairs = []
        for held in games:
            test = [r for r in rows if r["slug"] == held]
            if feats is None:
                pairs += [(r["fps"], r["pred"]) for r in test]
                continue
            train = [r for r in rows if r["slug"] != held]
            coef = _least_squares([[float(v) for v in feats(r)] for r in train],
                                  [math.log(r["fps"] / r["pred"]) for r in train], ridge=1.0)
            pairs += [(r["fps"], r["pred"] * math.exp(sum(c * float(v) for c, v in zip(coef, feats(r))))) for r in test]
        print(f"  {name:14s} {metrics(pairs)}")
    return 0


WIKIDATA_ENGINES = REAL_DIR / "wikidata_engines.json"
GAME_ENGINES = ROOT_DIR / "data" / "game_engines.json"
_WIKIDATA_QUERY = (
    "SELECT ?appid ?engine ?engineLabel WHERE { ?game wdt:P1733 ?appid; wdt:P408 ?engine. "
    'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }'
)
# Wikidata label (or item id when it has no English label) -> engine family.
_ENGINE_FAMILIES = {
    "Q94277753": "Unreal Engine 5", "Unreal Engine 5": "Unreal Engine 5",
    "Unreal Engine 4": "Unreal Engine 4", "Unreal Engine 3": "Unreal Engine 3",
    "Q63966": "Unity", "Unity": "Unity",
    "Q30892050": "RE Engine", "RE Engine": "RE Engine",
    "REDengine": "REDengine", "Decima": "Decima", "Ubisoft Anvil": "Anvil", "Snowdrop": "Snowdrop",
    "Source": "Source", "Source 2": "Source 2", "Frostbite": "Frostbite", "Frostbite 3": "Frostbite",
    "Q870033": "CryEngine", "CryEngine": "CryEngine", "CryEngine 3": "CryEngine",
    "Q107215616": "Creation Engine", "Creation Engine": "Creation Engine",
    "IW": "IW", "EGO": "EGO", "Godot": "Godot", "GameMaker": "GameMaker", "MT Framework": "MT Framework",
    "Clausewitz Engine": "Clausewitz", "Glacier": "Glacier", "Glacier 2": "Glacier",
}
# Engines of measured games that Wikidata does not list (public facts, not measurements).
MEASURED_GAME_ENGINES = {
    "1285190": "Unreal Engine 5", "1941540": "Unreal Engine 5", "1449110": "Unreal Engine 5",
    "2246340": "RE Engine", "3764200": "RE Engine", "3357650": "RE Engine",
    "2807960": "Frostbite", "3017860": "id Tech", "3008130": "C-Engine", "3768760": "Glacier",
    "3059520": "EGO", "1086940": "Divinity", "3321460": "BlackSpace", "2483190": "ForzaTech",
}
ENGINE_SHRINK = 2.0
ENGINE_MIN_GAMES = 3


def fetch_engines() -> int:
    import urllib.parse

    url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(_WIKIDATA_QUERY)
    req = urllib.request.Request(url, headers={"User-Agent": "CanRunEval/1.0 (game engine lookup)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        rows = json.loads(resp.read().decode("utf-8"))["results"]["bindings"]
    pairs = [{"appid": r["appid"]["value"], "engine": r["engineLabel"]["value"]} for r in rows]
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    WIKIDATA_ENGINES.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pairs)} Wikidata (Steam app id, engine) pairs")
    return 0


def engine_map() -> dict[str, str]:
    """Steam app id -> engine family, from the Wikidata export plus MEASURED_GAME_ENGINES."""
    apps: dict[str, str] = {}
    for row in json.loads(WIKIDATA_ENGINES.read_text(encoding="utf-8")):
        family = _ENGINE_FAMILIES.get(row["engine"])
        if family and row["appid"].isdigit():
            apps.setdefault(row["appid"], family)
    apps.update(MEASURED_GAME_ENGINES)
    return apps


def _game_offsets(vendor: str = "nvidia") -> tuple[dict[str, float], dict[str, list]]:
    """Mean log(measured / store-spec model) per game, one review per game."""
    rows = [r for r in spec_rows() if r["vendor"] == vendor]
    sizes: dict[tuple[str, str], int] = {}
    for r in rows:
        sizes[(r["slug"], r["review"])] = sizes.get((r["slug"], r["review"]), 0) + 1
    chosen: dict[str, str] = {}
    for (slug, review), size in sorted(sizes.items()):
        if size > sizes.get((slug, chosen.get(slug, "")), 0):
            chosen[slug] = review
    by_game: dict[str, list] = {}
    for r in rows:
        if r["review"] == chosen[r["slug"]]:
            by_game.setdefault(r["slug"], []).append(r)
    offsets = {s: statistics.fmean(math.log(r["fps"] / r["pred"]) for r in rs) for s, rs in by_game.items()}
    return offsets, by_game


def _engine_factors(offsets: dict[str, float], engines: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list] = {}
    for slug, off in offsets.items():
        family = engines.get(STEAM_APP_IDS.get(slug) or "")
        if family:
            grouped.setdefault(family, []).append(off)
    return {f: round(math.exp(sum(v) / (len(v) + ENGINE_SHRINK)), 3)
            for f, v in sorted(grouped.items()) if len(v) >= ENGINE_MIN_GAMES}


def engine_factors() -> int:
    """Leave-one-game-out test of the engine correction, then write data/game_engines.json."""
    engines = engine_map()
    offsets, by_game = _game_offsets()
    plain, corrected = [], []
    for slug, rows in by_game.items():
        factors = _engine_factors({s: o for s, o in offsets.items() if s != slug}, engines)
        factor = factors.get(engines.get(STEAM_APP_IDS.get(slug) or "", ""), 1.0)
        plain += [(r["fps"], r["pred"]) for r in rows]
        corrected += [(r["fps"], r["pred"] * factor) for r in rows]
    print(f"{len(by_game)} games, NVIDIA, store-spec model, each game held out")
    print(f"  no correction      {metrics(plain)}")
    print(f"  engine correction  {metrics(corrected)}")
    factors = _engine_factors(offsets, engines)
    print("  factors:", factors)
    GAME_ENGINES.write_text(json.dumps({
        "description": "Steam app id -> game engine family. Engines from Wikidata (CC0, P408 software engine), "
                       "factors from the leave-one-game-out fit in scripts/ds_pipeline/real_benchmarks.py.",
        "factors": factors,
        "apps": dict(sorted(engines.items(), key=lambda kv: int(kv[0]))),
    }, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"{GAME_ENGINES.name}: {len(engines)} apps")
    return 0


STEAMFPS_CACHE = REAL_DIR / "steamfps_search.json"
COMPARE_CARD = "RTX 4060 8GB"


def _steamfps_game(app_id: str, title: str, cache: dict) -> dict:
    if app_id not in cache:
        import urllib.parse

        url = "https://steamfps.com/api/v1/search?" + urllib.parse.urlencode({"q": title, "limit": 10})
        req = urllib.request.Request(url, headers={"User-Agent": "CanRunEval/1.0", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            games = json.loads(resp.read().decode("utf-8")).get("games", [])
        cache[app_id] = next((g for g in games if str(g.get("appid")) == app_id), {})
        time.sleep(0.5)
    return cache[app_id]


def steamfps_compare() -> int:
    """SteamFPS Steam Machine "high" estimates and CanRun against TechPowerUp's RTX 4060 at 1080p."""
    from canrun_steam_fps import (
        estimate_from_requirements, fit_log_fps_curve, load_measured_table, same_vendor_points,
    )

    store = json.loads(STORE_CACHE.read_text(encoding="utf-8"))
    titles = {p["slug"]: p["title"] for p in load_points()}
    table = load_measured_table()
    cache = json.loads(STEAMFPS_CACHE.read_text(encoding="utf-8")) if STEAMFPS_CACHE.exists() else {}
    rows = []
    for slug, app_id in STEAM_APP_IDS.items():
        if app_id not in store or slug not in titles:
            continue
        points = table.get((_table_name(slug, titles[slug], store), "1080p"), [])
        truth = next(((pm, fps) for gpu, pm, fps in points if gpu == COMPARE_CARD), None)
        if truth is None:
            continue
        rest = [(pm, f) for _, pm, f in same_vendor_points([p for p in points if p[0] != COMPARE_CARD], COMPARE_CARD)]
        fitted = fit_log_fps_curve(rest)
        curve = math.exp(fitted[0] + fitted[1] * math.log(truth[0])) if fitted else None
        s = store_inputs(store[app_id])
        spec = estimate_from_requirements(
            "RTX 4060", "AMD Ryzen 7 9800X3D", s["rec_gpu"], s["min_gpu"], "1080p", game_name=s["name"],
            user_vram_gb=8.0, recommended_cpu=s["rec_cpu"], recommended_ram_gb=s["rec_ram"],
            release_year=s["year"], genre=s["genre"],
        ).expected_fps
        high = ((_steamfps_game(app_id, s["name"], cache).get("steamMachine") or {}).get("tiers") or {}).get("high") or {}
        sfps = high.get("fps") if high.get("upscaler") in (None, "off") and high.get("resolution") == "1080p" else None
        rows.append((slug, truth[1], sfps, curve, spec or None))
    STEAMFPS_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"{'game':28s} {'TPU':>6s} {'SteamFPS':>9s} {'curve':>7s} {'spec':>7s}")
    for slug, t, sf, cv, sp in rows:
        cells = [f"{v:7.0f}" if v else "      -" for v in (sf, cv, sp)]
        print(f"{slug:28s} {t:6.0f}  {cells[0]}  {cells[1]} {cells[2]}")
    both = [r for r in rows if r[2]]
    print(f"\n{len(both)} games where SteamFPS gives a native 1080p 'high' figure:")
    for label, i in (("SteamFPS (Steam Machine, high)", 2), ("CanRun curve, RTX 4060 left out", 3),
                     ("CanRun store-spec model only", 4)):
        pairs = [(r[1], r[i]) for r in both if r[i]]
        if pairs:
            print(f"  {label:34s} {metrics(pairs)}")
    print(f"All {len(rows)} games:")
    for label, i in (("CanRun curve, RTX 4060 left out", 3), ("CanRun store-spec model only", 4)):
        print(f"  {label:34s} {metrics([(r[1], r[i]) for r in rows if r[i]])}")
    return 0


def real_feature_rows() -> list[dict]:
    """spec_rows() plus the exact feature vector the app feeds the model for each point."""
    from canrun_model_loader import LightweightMLPredictor

    original = LightweightMLPredictor.predict_two_stage_from_features
    captured: list[list[float]] = []

    def capture(self, x_vec):
        captured.append(list(x_vec))
        return original(self, x_vec)

    LightweightMLPredictor.predict_two_stage_from_features = capture
    try:
        rows = spec_rows(captured)
    finally:
        LightweightMLPredictor.predict_two_stage_from_features = original
    return [r for r in rows if r["x"] is not None]


def _fit_two_stage(X, y_log, weights):
    """TwoStageLogResidualFPSModel.fit with per-row weights."""
    from scripts.ds_pipeline.train_and_evaluate import TwoStageLogResidualFPSModel

    model = TwoStageLogResidualFPSModel(use_monotonic=True)
    X_scaled = model.scaler.fit_transform(X)
    model.ridge.fit(X_scaled, y_log, sample_weight=weights)
    model.ridge.coef_ = model.ridge.coef_ * 0.05
    model.gbdt.fit(X, y_log - model.ridge.predict(X_scaled), sample_weight=weights)
    return model


def retrain(real_weight: float, synthetic_rows: int, vendor: str = "all") -> int:
    """Leave-one-game-out: train on synthetic rows plus the other games' review rows, score the held-out game."""
    import numpy as np
    from canrun_steam_fps import _REQUIREMENTS_FPS_CAP
    from scripts.ds_pipeline.features import records_to_matrix

    real = [r for r in real_feature_rows() if vendor == "all" or r["vendor"] == vendor]
    dataset = json.loads((ROOT_DIR / "data" / "clean_training_dataset.json").read_text(encoding="utf-8"))
    records = dataset["records"]
    rng = np.random.default_rng(42)
    if synthetic_rows and synthetic_rows < len(records):
        records = [records[i] for i in rng.choice(len(records), synthetic_rows, replace=False)]
    X_syn, y_syn, _ = records_to_matrix(records)
    X_real = np.array([r["x"] for r in real])
    y_real = np.log(np.array([r["fps"] for r in real]))
    slugs = np.array([r["slug"] for r in real])

    def predict(model, X):
        fps = np.exp(np.clip(model.predict_log_ml_only(X), math.log(5.0), math.log(500.0)))
        return np.minimum(fps, _REQUIREMENTS_FPS_CAP)

    base = _fit_two_stage(X_syn, y_syn, np.ones(len(y_syn)))
    base_pred = predict(base, X_real)
    shipped, retrained = [], []
    per_game = {}
    for held in sorted(set(slugs)):
        test, train = slugs == held, slugs != held
        X = np.vstack([X_syn, X_real[train]])
        y = np.concatenate([y_syn, y_real[train]])
        w = np.concatenate([np.ones(len(y_syn)), np.full(int(train.sum()), real_weight)])
        pred = predict(_fit_two_stage(X, y, w), X_real[test])
        truth = np.exp(y_real[test])
        pairs_new = list(zip(truth, pred))
        shipped += [(t, r["pred"]) for t, r in zip(truth, (r for r, m in zip(real, test) if m))]
        retrained += pairs_new
        per_game[held] = (metrics([(t, r["pred"]) for t, r in zip(truth, (r for r, m in zip(real, test) if m))])["MdAPE"],
                          metrics(pairs_new)["MdAPE"])
    print(f"{len(real)} review points, {len(per_game)} games, vendor {vendor}, "
          f"{len(y_syn)} synthetic rows, real weight {real_weight}")
    print(f"  shipped model            {metrics(shipped)}")
    print(f"  refit, synthetic only    {metrics(list(zip(np.exp(y_real), base_pred)))}")
    print(f"  refit, + other games     {metrics(retrained)}")
    for slug, (old, new) in sorted(per_game.items(), key=lambda kv: kv[1][1] - kv[1][0]):
        print(f"    {slug:26s} {old:6.1f} -> {new:6.1f}")
    return 0


NOTEBOOKCHECK_DIR = REAL_DIR / "notebookcheck"
_NBC_SETTINGS = {"ultra 1920x1080": "1080p", "QHD 2560x1440": "1440p", "4K 3840x2160": "4K"}
_DESKTOP_NAMESAKE = re.compile(r"\s+Laptop GPU$")
# Notebookcheck's "ultra" in these games is 2-7x below review numbers for similar
# cards with ray tracing off, so it includes ray or path tracing. Scored separately.
NBC_RAY_TRACED_ULTRA = frozenset({"F1 25", "Forza Horizon 6", "Resident Evil Requiem"})


def load_notebookcheck() -> list[dict]:
    """Per-laptop game results from saved Notebookcheck GPU pages (<GPU name>.txt).

    Only the ultra 1080p, QHD and 4K rows are kept: Notebookcheck runs those at
    the highest preset, like the review curves. Each laptop's own CPU is kept.
    """
    points = []
    for path in sorted(NOTEBOOKCHECK_DIR.glob("*.txt")):
        gpu = path.stem
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        try:
            start = lines.index("## Game Benchmarks")
        except ValueError:
            continue
        title, res, i = None, None, start
        while i < len(lines):
            line = lines[i]
            if line.startswith("## ") and i > start:
                break
            if line.startswith("### "):
                title, res = line[4:].strip(), None
            elif re.search(r"\s\d{3,4}x\d{3,4}$", line):
                res = _NBC_SETTINGS.get(line)
            elif res and title and re.fullmatch(r"\d+(?:\.\d+)?", line) and i + 3 < len(lines) and lines[i + 2].startswith("[X]"):
                points.append({"gpu": gpu, "title": title, "res": res, "fps": float(line),
                               "laptop": lines[i + 2][3:].strip(), "cpu": lines[i + 3]})
            i += 1
    return points


def evaluate_laptops() -> int:
    """Notebookcheck laptop results against the app, with the laptop score and with the desktop namesake's."""
    from canrun_steam_fps import load_gpu_passmarks, resolve_fps

    gpu_pms = load_gpu_passmarks()
    rows = {"laptop score": [], "desktop namesake (12.1)": []}
    by_source: dict[str, list] = {}
    per_gpu: dict[str, list] = {}
    ray_traced: list = []
    games, skipped = set(), set()
    for p in load_notebookcheck():
        if p["fps"] <= 0:
            continue
        if p["title"] in NBC_RAY_TRACED_ULTRA:
            shown = resolve_fps(p["gpu"], p["cpu"], None, api_available=False, gpu_passmarks=gpu_pms,
                                game_name=p["title"], resolution=p["res"])
            if shown.expected_fps > 0:
                ray_traced.append((p["fps"], float(shown.expected_fps)))
            continue
        shown = resolve_fps(p["gpu"], p["cpu"], None, api_available=False, gpu_passmarks=gpu_pms,
                            game_name=p["title"], resolution=p["res"])
        if shown.fps_source == "unknown" or shown.expected_fps <= 0:
            skipped.add(p["title"])
            continue
        desktop = resolve_fps(_DESKTOP_NAMESAKE.sub("", p["gpu"]), p["cpu"], None, api_available=False,
                              gpu_passmarks=gpu_pms, game_name=p["title"], resolution=p["res"])
        rows["laptop score"].append((p["fps"], float(shown.expected_fps)))
        rows["desktop namesake (12.1)"].append((p["fps"], float(desktop.expected_fps)))
        source = "measured review curve" if _benchmark_source(p["title"], p["res"]) else "older benchmark table"
        by_source.setdefault(source, []).append((p["fps"], float(shown.expected_fps)))
        per_gpu.setdefault(p["gpu"], []).append((p["fps"], float(shown.expected_fps)))
        games.add(p["title"])
    print(f"{len(rows['laptop score'])} laptop results, {len(games)} games with a curve; "
          f"{len(skipped)} games without a curve skipped")
    for key, pairs in rows.items():
        print(f"  {key:26s} {metrics(pairs)}")
    for key, pairs in sorted(by_source.items()):
        print(f"  {key:26s} {metrics(pairs)}")
    for gpu, pairs in sorted(per_gpu.items()):
        print(f"    {gpu:24s} {metrics(pairs)}")
    if ray_traced:
        print(f"  ray-traced ultra, not counted ({', '.join(sorted(NBC_RAY_TRACED_ULTRA))}): {metrics(ray_traced)}")
    return 0


def _benchmark_source(title: str, res: str) -> bool:
    from canrun_steam_fps import _benchmark_points, load_measured_table

    return bool(_benchmark_points(title, res, load_measured_table()))


MEASURED_TABLE = ROOT_DIR / "data" / "measured_benchmarks.json"


def _table_name(slug: str, title: str, store: dict) -> str:
    app_id = STEAM_APP_IDS.get(slug)
    return store[app_id]["name"] if app_id and app_id in store else title


def build_table(exclude: set[str], vendors: set[str]) -> dict:
    """Review points in the training_benchmarks.json layout, one curve per game and resolution."""
    from canrun_steam_fps import load_gpu_passmarks, lookup_gpu_passmark

    store = json.loads(STORE_CACHE.read_text(encoding="utf-8")) if STORE_CACHE.exists() else {}
    gpu_pms = load_gpu_passmarks()
    points = load_points()
    # One review per game: curves from two test scenes would mix their offsets.
    sizes: dict[tuple[str, str], int] = {}
    for p in points:
        sizes[(p["slug"], p["review"])] = sizes.get((p["slug"], p["review"]), 0) + 1
    chosen = {}
    for (slug, review), size in sorted(sizes.items()):
        if size > sizes.get((slug, chosen.get(slug, "")), 0):
            chosen[slug] = review
    games: dict[str, dict] = {}
    for p in points:
        if p["review"] != chosen[p["slug"]] or p["slug"] in exclude or _vendor(p["card"]) not in vendors:
            continue
        passmark = lookup_gpu_passmark(p["card"], gpu_pms)
        if not passmark:
            continue
        label = f"{p['card']} {int(p['vram'])}GB" if p["vram"] else p["card"]
        for name in {_table_name(p["slug"], p["title"], store), p["title"]}:
            game = games.setdefault(name, {"source": p["review"], "gpus": {}})
            game["gpus"].setdefault(label, {"passmark": passmark})[p["res"]] = p["fps"]
    table = {"description": "Published review measurements (see data/real_benchmarks/).", "games": games}
    MEASURED_TABLE.write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{MEASURED_TABLE.name}: {len(games)} games, vendors {sorted(vendors)}, excluded {sorted(exclude)}")
    return table


LOW_END_PASSMARK = 21000


def low_end_check(review: str) -> int:
    """NVIDIA cards below an RTX 3070 in one review, predicted by the app's current tables."""
    from canrun_steam_fps import load_gpu_passmarks, lookup_gpu_passmark, resolve_fps

    store = json.loads(STORE_CACHE.read_text(encoding="utf-8")) if STORE_CACHE.exists() else {}
    gpu_pms = load_gpu_passmarks()
    by_source: dict[str, list] = {}
    per_card: dict[str, list] = {}
    missing = set()
    for p in load_points():
        passmark = lookup_gpu_passmark(p["card"], gpu_pms)
        if p["review"] != review or _vendor(p["card"]) != "nvidia" or not passmark or passmark > LOW_END_PASSMARK:
            continue
        name = _table_name(p["slug"], p["title"], store)
        shown = resolve_fps(p["card"], p["cpu"], None, api_available=False, gpu_passmarks=gpu_pms,
                            game_name=name, resolution=p["res"])
        if shown.expected_fps <= 0:
            missing.add(p["title"])
            continue
        source = "measured review curve" if _benchmark_source(name, p["res"]) else "older benchmark table"
        by_source.setdefault(source, []).append((p["fps"], float(shown.expected_fps)))
        label = f"{p['card']} {int(p['vram'])}GB" if p["vram"] else p["card"]
        per_card.setdefault(label, []).append((p["fps"], float(shown.expected_fps)))
    for source, pairs in sorted(by_source.items()):
        print(f"  {source:24s} {metrics(pairs)}")
    print(f"  all                      {metrics([x for v in by_source.values() for x in v])}")
    for card, pairs in sorted(per_card.items()):
        print(f"    {card:18s} {metrics(pairs)}")
    if missing:
        print("  no curve:", sorted(missing))
    return 0


def _vendor(card: str) -> str:
    return "nvidia" if card.startswith(("RTX", "GTX")) else ("intel" if card.startswith("Arc") else "amd")


def leave_one_card_out() -> int:
    """Each review point predicted from the same game's curve with that card left out."""
    from canrun_steam_fps import fit_log_fps_curve, load_measured_table, same_vendor_points

    raw = json.loads(MEASURED_TABLE.read_text(encoding="utf-8"))["games"]
    first_name = {}
    for name, info in raw.items():
        first_name.setdefault(json.dumps(info, sort_keys=True), name)
    keep = set(first_name.values())
    table = {key: rows for key, rows in load_measured_table().items() if key[0] in keep}
    by_vendor: dict[str, list] = {}
    per_game: dict[str, list] = {}
    low_end: list = []
    for (game, res), points in table.items():
        for gpu, passmark, fps in points:
            rest = [(pm, f) for g, pm, f in same_vendor_points([r for r in points if r[0] != gpu], gpu)]
            fitted = fit_log_fps_curve(rest)
            if not fitted:
                continue
            pred = math.exp(fitted[0] + fitted[1] * math.log(passmark))
            by_vendor.setdefault(_vendor(gpu), []).append((fps, pred))
            per_game.setdefault(game, []).append((fps, pred))
            if _vendor(gpu) == "nvidia" and passmark <= LOW_END_PASSMARK:
                low_end.append((fps, pred))
    for vendor, pairs in sorted(by_vendor.items()):
        print(f"  {vendor:7s} {metrics(pairs)}")
    if low_end:
        print(f"  NVIDIA below RTX 3070 {metrics(low_end)}")
    print(f"  all     {metrics([p for v in by_vendor.values() for p in v])}")
    worst = sorted(per_game.items(), key=lambda kv: -metrics(kv[1])["MdAPE"])[:6]
    print("  worst games:", [(g, metrics(p)["MdAPE"]) for g, p in worst])
    return 0


def extrapolate_down(k: int) -> int:
    """Predict the k slowest NVIDIA cards of each curve from the faster NVIDIA cards only."""
    from canrun_steam_fps import fit_log_fps_curve

    raw = json.loads(MEASURED_TABLE.read_text(encoding="utf-8"))["games"]
    seen, pairs, by_res = set(), [], {}
    for name, info in raw.items():
        key = json.dumps(info, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        for res in ("1080p", "1440p", "4K"):
            pts = sorted(((g, r["passmark"], r[res]) for g, r in info["gpus"].items()
                          if res in r and _vendor(g) == "nvidia"), key=lambda t: t[1])
            if len(pts) < k + 4:
                continue
            fitted = fit_log_fps_curve([(pm, f) for _, pm, f in pts[k:]])
            if not fitted:
                continue
            for _, pm, f in pts[:k]:
                pred = math.exp(fitted[0] + fitted[1] * math.log(pm))
                pairs.append((f, pred))
                by_res.setdefault(res, []).append((f, pred))
    print(f"  slowest {k} NVIDIA cards held out: {metrics(pairs)}")
    for res, p in by_res.items():
        print(f"    {res}: {metrics(p)}")
    return 0


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    args = sys.argv[2:]
    if command == "fetch-store":
        fetch_store([a for a in STEAM_APP_IDS.values() if a])
        return 0
    if command == "build-table":
        exclude = {a.split("=", 1)[1] for a in args if a.startswith("--exclude=")}
        exclude = {s for e in exclude for s in e.split(",") if s}
        vendors = next((set(a.split("=", 1)[1].split(",")) for a in args if a.startswith("--vendors=")),
                       {"nvidia", "amd", "intel"})
        build_table(exclude, vendors)
        return 0
    if command == "loo":
        return leave_one_card_out()
    if command == "extrapolate":
        return extrapolate_down(int(args[0]) if args else 3)
    if command == "calibrate":
        return calibrate(args[0] if args else "nvidia")
    if command == "fetch-engines":
        return fetch_engines()
    if command == "engine-factors":
        return engine_factors()
    if command == "steamfps-compare":
        return steamfps_compare()
    if command == "lowend":
        return low_end_check(args[0] if args else "tpu_games_2026")
    if command == "laptops":
        return evaluate_laptops()
    if command == "retrain":
        return retrain(float(args[0]) if args else 50.0, int(args[1]) if len(args) > 1 else 150000,
                       args[2] if len(args) > 2 else "all")
    return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
