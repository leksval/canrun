"""
Hold-out evaluation on FPS figures stated in TechSpot (Hardware Unboxed) GPU benchmarks.

Only numbers written out in the article text are used, never values derived from
"X% faster" statements. Test system: Ryzen 7 9800X3D, DDR5-6000, native resolution.
  Forza Horizon 6, May 2026      https://www.techspot.com/article/3128-forza-horizon-6-benchmark/
  Crimson Desert, Mar 2026       https://www.techspot.com/article/3104-crimson-desert-benchmark/
  Battlefield 6, Oct 2025        https://www.techspot.com/review/3042-battlefield-6-benchmark/
  Doom: The Dark Ages, May 2025  https://www.techspot.com/review/2984-doom-dark-ages-benchmark/

Steam requirements come from the local store harvest, so the run makes no Steam requests.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from canrun_steam_fps import (  # noqa: E402
    estimate_from_requirements,
    first_known_gpu,
    gpu_vendor,
    load_benchmark_table,
    load_cpu_patterns,
    load_gpu_passmarks,
    lookup_cpu_passmark,
    lookup_gpu_passmark,
    match_benchmark_game,
    requirement_passmarks,
    resolve_fps,
)
from scripts.ds_pipeline.collect_and_audit import _extract_field_from_raw_req, _extract_ram_gb  # noqa: E402
from scripts.ds_pipeline.evaluate_external_benchmarks import heuristic_m0, metrics  # noqa: E402

TEST_CPU = "AMD Ryzen 7 9800X3D"
FPS_CAP = 240

STEAM_APPS = {
    "Forza Horizon 6": "2483190",
    "Crimson Desert": "3321460",
    "Battlefield 6": "2807960",
    "DOOM: The Dark Ages": "3017860",
}

VRAM_GB = {
    "RTX 5090": 32, "RTX 4090": 24, "RTX 5080": 16, "RTX 5070 Ti": 16, "RTX 5070": 12, "RTX 4080 Super": 16,
    "RTX 4070 Ti Super": 16, "RTX 4070": 12, "RTX 5060 Ti 16GB": 16, "RTX 5060 Ti 8GB": 8,
    "RTX 4060 Ti 16GB": 16, "RTX 4060 Ti 8GB": 8, "RTX 4060 Ti": 8, "RTX 4060": 8, "RTX 5050": 8, "RTX 3090": 24,
    "RTX 3080": 10, "RTX 3070": 8, "RTX 3060 Ti": 8, "RTX 3060": 12, "RTX 3050": 8, "RTX 2060": 6,
    "RX 9070 XT": 16, "RX 9070": 16, "RX 9060 XT 16GB": 16, "RX 9060 XT 8GB": 8, "RX 7900 XTX": 24,
    "RX 7900 XT": 20, "RX 7900 GRE": 16, "RX 7800 XT": 16, "RX 7700 XT": 12, "RX 7600 XT": 16, "RX 7600": 8,
    "RX 6750 XT": 12, "RX 6600": 8, "RX 5700 XT": 8, "Arc B580": 12,
}

# (game, preset, preset class, resolution) -> {card: average fps}
MEASURED: dict[tuple[str, str, str, str], dict[str, float]] = {
    ("Forza Horizon 6", "Extreme", "max", "1080p"): {
        "RTX 5090": 204, "RTX 5080": 172, "RTX 5070 Ti": 163, "RX 9070 XT": 136, "RX 9070": 132,
        "RTX 5060 Ti 16GB": 108, "RX 9060 XT 16GB": 95, "RTX 3060 Ti": 70, "RX 7600 XT": 64, "RTX 3060": 62,
        "RTX 5050": 50, "Arc B580": 83, "RTX 5060 Ti 8GB": 83, "RTX 3050": 39, "RTX 2060": 39, "RX 5700 XT": 46},
    ("Forza Horizon 6", "Extreme", "max", "1440p"): {
        "RTX 5090": 187, "RTX 5080": 142, "RTX 5070 Ti": 132, "RX 9070 XT": 117, "RTX 4080 Super": 118,
        "RX 9070": 112, "RTX 5070": 112, "RX 7900 XTX": 103, "RX 7900 XT": 94, "RTX 5060 Ti 16GB": 85,
        "RTX 5060 Ti 8GB": 63, "RX 9060 XT 16GB": 77, "RTX 3080": 79, "Arc B580": 67, "RTX 4060 Ti 16GB": 68,
        "RTX 3060 Ti": 55, "RX 7600 XT": 51, "RTX 3060": 48, "RTX 5050": 39},
    ("Forza Horizon 6", "High", "high", "1080p"): {
        "RTX 5080": 235, "RTX 5070 Ti": 228, "RX 9070 XT": 191, "RX 9070": 184, "RX 7900 XT": 148,
        "RTX 5060 Ti 16GB": 167, "RX 9060 XT 16GB": 143, "RX 9060 XT 8GB": 143, "RTX 3060 Ti": 104,
        "RTX 3060": 95, "RX 7600 XT": 94, "RTX 5050": 87, "Arc B580": 121},
    ("Forza Horizon 6", "High", "high", "1440p"): {
        "RTX 5080": 205, "RX 9070 XT": 166, "RX 9070": 159, "RX 7900 XTX": 141, "RTX 4070 Ti Super": 141,
        "RX 7900 XT": 129},
    ("Crimson Desert", "Cinematic", "max", "1080p"): {"RTX 5090": 175, "RX 7900 XTX": 121},
    ("Crimson Desert", "Cinematic", "max", "1440p"): {
        "RTX 4090": 123, "RX 7900 XTX": 93, "RX 9070 XT": 89, "RX 7900 GRE": 68, "RTX 4070": 66},
    ("Crimson Desert", "High", "high", "1080p"): {
        "RTX 5090": 188, "RTX 4090": 166, "RX 9070 XT": 127, "RX 9070": 117, "RTX 4070": 97, "RX 7800 XT": 95,
        "RTX 5060 Ti 16GB": 92, "RTX 3080": 92, "RTX 3070": 72, "RTX 4060 Ti": 68, "RTX 3060 Ti": 68,
        "RTX 4060": 61, "RX 7600": 59, "RX 7600 XT": 59},
    ("Crimson Desert", "High", "high", "1440p"): {
        "RX 7900 XTX": 110, "RTX 5070 Ti": 116, "RX 9070 XT": 103, "RX 9070": 93, "RTX 3060 Ti": 53,
        "RTX 4060 Ti 8GB": 52, "RTX 4060 Ti 16GB": 52, "RX 6750 XT": 51, "RTX 4060": 45},
    ("Battlefield 6", "High", "high", "1440p"): {"RTX 5060 Ti 16GB": 94, "RX 9060 XT 16GB": 84},
    ("Battlefield 6", "Overkill", "max", "1440p"): {"RTX 5060 Ti 16GB": 70},
    ("Battlefield 6", "Overkill", "max", "1080p"): {"Arc B580": 66},
    ("DOOM: The Dark Ages", "Ultra Nightmare", "max", "1080p"): {"RTX 5090": 151, "RX 7700 XT": 68},
    ("DOOM: The Dark Ages", "Ultra Nightmare", "max", "1440p"): {"RTX 5090": 125, "RTX 3080": 65},
    ("DOOM: The Dark Ages", "Medium", "medium", "1080p"): {
        "RX 7600": 52, "RX 7600 XT": 52, "RTX 4060": 64, "RX 6600": 41},
    ("DOOM: The Dark Ages", "Medium", "medium", "1440p"): {"RTX 5090": 135, "RTX 4090": 130},
}


def main() -> int:
    store = json.loads((ROOT_DIR / "data" / "steam_api_harvest.json").read_text(encoding="utf-8"))["store_appdetails"]
    gpu_pms = load_gpu_passmarks()
    cards = requirement_passmarks(gpu_pms)
    cpu_patterns = load_cpu_patterns()
    cpu_pm = float(lookup_cpu_passmark(TEST_CPU, cpu_patterns) or 40000)
    table_games = sorted({key[0] for key in load_benchmark_table()})

    spec = importlib.util.spec_from_file_location("hf_inf", ROOT_DIR / "dist" / "hf_model" / "inference.py")
    hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf)
    model = hf.CanRunFPSModel.from_pretrained(ROOT_DIR / "dist" / "hf_model" / "model.json")

    info = {}
    for game, app_id in STEAM_APPS.items():
        entry = store[app_id]
        rec_text = _extract_field_from_raw_req(entry.get("recommended_raw", ""), "(?:Graphics|Video Card)") or ""
        min_text = _extract_field_from_raw_req(entry.get("minimum_raw", ""), "(?:Graphics|Video Card)") or ""
        rec, mn = first_known_gpu(rec_text, cards), first_known_gpu(min_text, cards)
        min_pm = float(mn[1]) if mn else 6385.0
        rec_pm = float(rec[1]) if rec else min_pm * 1.65
        year = re.search(r"(19|20)\d\d", str(entry.get("release_date", "")))
        info[game] = {
            "name": entry["name"], "rec_text": rec_text, "min_text": min_text, "rec_pm": rec_pm, "min_pm": min_pm,
            "rec_cpu_pm": float(lookup_cpu_passmark(
                _extract_field_from_raw_req(entry.get("recommended_raw", ""), "Processor"), cpu_patterns) or 16000),
            "rec_ram": _extract_ram_gb(entry.get("recommended_raw", ""), 16),
            "year": int(year.group(0)) if year else 2025,
            "in_table": match_benchmark_game(entry["name"], table_games) is not None,
        }
        print(f"{game:22s} rec={rec[0] if rec else None} min={mn[0] if mn else None} "
              f"path={'measured-table curve' if info[game]['in_table'] else 'store-spec estimate'}")

    rows, skipped = [], set()
    for (game, preset, preset_class, res), measured in MEASURED.items():
        g = info[game]
        for card, true_fps in measured.items():
            gpu_pm = lookup_gpu_passmark(card, gpu_pms)
            if not gpu_pm or true_fps > FPS_CAP:
                skipped.add(card) if not gpu_pm else None
                continue
            min_vram = 4 if g["min_pm"] < 10000 else (6 if g["min_pm"] < 17000 else 8)
            model_fps = model.predict_fps(
                gpu_passmark=float(gpu_pm), cpu_passmark=cpu_pm, gpu_vram_gb=float(VRAM_GB[card]), gpu_name=card,
                resolution=res, rec_gpu_passmark=g["rec_pm"], min_gpu_passmark=g["min_pm"],
                rec_cpu_passmark=g["rec_cpu_pm"], rec_vram_gb=float(max(min_vram, 6 if g["rec_pm"] < 14500 else 8)),
                rec_ram_gb=float(g["rec_ram"]), release_year=g["year"], genre="", game_name=g["name"],
            )
            shown = resolve_fps(card, TEST_CPU, None, api_available=False, gpu_passmarks=gpu_pms,
                                cpu_patterns=cpu_patterns, game_name=g["name"], resolution=res)
            if shown.fps_source == "unknown" or shown.expected_fps <= 0:
                shown = estimate_from_requirements(card, TEST_CPU, g["rec_text"], g["min_text"], res,
                                                   gpu_passmarks=gpu_pms, cpu_patterns=cpu_patterns,
                                                   game_name=g["name"])
            rows.append({
                "game": game, "preset": preset, "preset_class": preset_class, "res": res, "card": card,
                "true": float(true_fps), "model": min(model_fps, FPS_CAP), "plugin": float(shown.expected_fps),
                "lo": float(shown.fps_min), "hi": float(shown.fps_max), "source": shown.fps_source,
                "heuristic": min(heuristic_m0(float(gpu_pm), g["rec_pm"], res), FPS_CAP), "in_table": g["in_table"],
            })

    if skipped:
        print(f"Skipped cards with no PassMark entry: {sorted(skipped)}")

    def report(title: str, subset: list[dict]) -> None:
        if len(subset) < 5:
            return
        print(f"\n{title} (n={len(subset)}):")
        for key in ("model", "plugin", "heuristic"):
            print(f"  {key:10s} {metrics([(r['true'], r[key]) for r in subset])}")
        inside = sum(r["lo"] <= r["true"] <= r["hi"] for r in subset) / len(subset) * 100
        print(f"  measured FPS inside the plugin's shown range: {inside:.1f}%")

    report("All stated TechSpot results", rows)
    report("Highest preset only (Extreme, Cinematic, Overkill, Ultra Nightmare)", [r for r in rows if r["preset_class"] == "max"])
    report("High preset only", [r for r in rows if r["preset_class"] == "high"])
    report("Games with a measured-table curve", [r for r in rows if r["in_table"]])
    report("Games on the store-spec path (no table curve)", [r for r in rows if not r["in_table"]])
    report("NVIDIA cards", [r for r in rows if gpu_vendor(r["card"]) == "nvidia"])
    report("AMD and Intel cards", [r for r in rows if gpu_vendor(r["card"]) != "nvidia"])

    print("\nPer game and preset (median abs % error, bias):")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["game"], r["preset"])].append(r)
    for (game, preset), group in groups.items():
        if len(group) < 3:
            continue
        m = {k: metrics([(r["true"], r[k]) for r in group]) for k in ("model", "plugin", "heuristic")}
        print(f"  {game:20s} {preset:16s} n={len(group):3d} model {m['model']['MdAPE_pct']:5.1f}% ({m['model']['bias_pct']:+6.1f})"
              f" | plugin {m['plugin']['MdAPE_pct']:5.1f}% ({m['plugin']['bias_pct']:+6.1f})"
              f" | heuristic {m['heuristic']['MdAPE_pct']:5.1f}% ({m['heuristic']['bias_pct']:+6.1f})  {group[0]['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
