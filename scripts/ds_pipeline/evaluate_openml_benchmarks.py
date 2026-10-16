"""
Hold-out evaluation on OpenML dataset 42737 ("fps-in-video-games", CC BY).

Peeters, Melnikov and Huellermeier, "Performance Prediction for Hardware-Software
Configurations: A Case Study for Video Games", IDA 2021. https://www.openml.org/d/42737

Rows carry a CPU, a GPU, a game, a resolution, a quality setting and a measured FPS.
"userbenchmark" rows are FPS reported by users on their own PCs; "fpsbenchmark" rows
come from fps-benchmark.com. Steam requirements are read from the local store harvest
(data/steam_api_harvest.json), so the run makes no Steam requests. The plugin column
follows the keyless shipped path: measured-table curve when the game has one, otherwise
the store-spec estimate.

Usage: py -3 scripts/ds_pipeline/evaluate_openml_benchmarks.py [path/to/fps-in-video-games.arff]
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from canrun_steam_fps import (  # noqa: E402
    estimate_from_requirements,
    first_known_gpu,
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

ARFF_URL = "https://openml.org/data/v1/download/22044773/fps-in-video-games.arff"
DEFAULT_ARFF = Path(os.environ.get("TEMP", "/tmp")) / "openml_fps" / "fps.arff"

# OpenML game key -> Steam app id in the local harvest. The two marked games are a
# different release today (CS:GO became Counter-Strike 2, Overwatch became Overwatch 2),
# so their current Steam requirements do not describe the measured game.
STEAM_APPS = {
    "counterStrikeGlobalOffensive": ("730", True),
    "overwatch": ("2357570", True),
    "playerUnknownsBattlegrounds": ("578080", False),
    "grandTheftAuto5": ("271590", False),
    "apexLegends": ("1172470", False),
    "dota2": ("570", False),
    "worldOfTanks": ("1407200", False),
    "rust": ("252490", False),
    "arkSurvivalEvolved": ("346110", False),
    "rocketLeague": ("252950", False),
    "callOfDutyWW2": ("476600", False),
    "pathOfExile": ("238960", False),
    "destiny2": ("1085660", False),
    "totalWar3Kingdoms": ("779340", False),
    "warframe": ("230410", False),
    "rainbowSixSiege": ("359550", False),
}
RESOLUTIONS = {"1080": "1080p", "1440": "1440p"}
SETTINGS = {"high", "max"}
FPS_CAP = 240


def read_arff(path: Path) -> tuple[list[str], list[list[str]]]:
    attrs: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text.lower().startswith("@attribute"):
                attrs.append(text.split()[1].strip("'"))
            elif text.lower().startswith("@data"):
                break
        rows = [row for row in csv.reader(handle, quotechar="'", skipinitialspace=True) if row]
    return attrs, rows


def plugin_gpu_name(name: str) -> str:
    """The detector reports names like 'NVIDIA GeForce GTX 1060 6GB'; drop OpenML's memory-clock suffixes."""
    name = re.sub(r"\b\d+\s*Gbps\b|\bGDDR5X?\b|\bOEM\b|\bRev\.\s*\d+\b", "", name)
    name = re.sub(r"(\d+)\s+GB\b", r"\1GB", name)
    return re.sub(r"\s+", " ", name).strip()


def load_store_requirements() -> dict[str, dict]:
    harvest = json.loads((ROOT_DIR / "data" / "steam_api_harvest.json").read_text(encoding="utf-8"))
    return harvest["store_appdetails"]


def main() -> int:
    arff = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARFF
    if not arff.exists():
        arff.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ARFF_URL, arff)
    attrs, raw_rows = read_arff(arff)
    col = {name: i for i, name in enumerate(attrs)}

    groups: dict[tuple, list[float]] = defaultdict(list)
    vram_gb: dict[str, float] = {}
    for row in raw_rows:
        game = row[col["GameName"]]
        res = RESOLUTIONS.get(row[col["GameResolution"]])
        setting = row[col["GameSetting"]]
        if game not in STEAM_APPS or res is None or setting not in SETTINGS:
            continue
        try:
            fps = float(row[col["FPS"]])
        except ValueError:
            continue
        if fps > 0:
            key = (row[col["Dataset"]], game, row[col["GpuName"]], row[col["CpuName"]], res, setting)
            groups[key].append(fps)
            try:
                vram_gb[row[col["GpuName"]]] = float(row[col["GpuMemorySize"]]) / 1024
            except ValueError:
                pass

    store = load_store_requirements()
    gpu_pms = load_gpu_passmarks()
    cards = requirement_passmarks(gpu_pms)
    cpu_patterns = load_cpu_patterns()
    table_games = sorted({key[0] for key in load_benchmark_table()})

    spec = importlib.util.spec_from_file_location("hf_inf", ROOT_DIR / "dist" / "hf_model" / "inference.py")
    hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf)
    model = hf.CanRunFPSModel.from_pretrained(ROOT_DIR / "dist" / "hf_model" / "model.json")

    game_info = {}
    for game, (app_id, mismatch) in STEAM_APPS.items():
        entry = store[app_id]
        rec_text = _extract_field_from_raw_req(entry.get("recommended_raw", ""), "(?:Graphics|Video Card)") or ""
        min_text = _extract_field_from_raw_req(entry.get("minimum_raw", ""), "(?:Graphics|Video Card)") or ""
        rec, mn = first_known_gpu(rec_text, cards), first_known_gpu(min_text, cards)
        min_pm = float(mn[1]) if mn else 6385.0
        rec_pm = float(rec[1]) if rec else min_pm * 1.65
        year = re.search(r"(19|20)\d\d", str(entry.get("release_date", "")))
        game_info[game] = {
            "name": entry["name"], "mismatch": mismatch, "rec_text": rec_text, "min_text": min_text,
            "rec_pm": rec_pm, "min_pm": min_pm,
            "rec_cpu_pm": float(lookup_cpu_passmark(
                _extract_field_from_raw_req(entry.get("recommended_raw", ""), "Processor"), cpu_patterns) or 16000),
            "rec_ram": _extract_ram_gb(entry.get("recommended_raw", ""), 16),
            "year": int(year.group(0)) if year else 2018,
            "in_table": match_benchmark_game(entry["name"], table_games) is not None,
        }
        print(f"{game:30s} -> {entry['name'][:30]:30s} rec={rec[0] if rec else None} min={mn[0] if mn else None}"
              f" table={'yes' if game_info[game]['in_table'] else 'no'}{' (version mismatch)' if mismatch else ''}")

    results = []
    unknown_gpus, unknown_cpus = set(), set()
    curve_cache: dict[tuple, object] = {}
    for (part, game, gpu_raw, cpu, res, setting), values in groups.items():
        gpu = plugin_gpu_name(gpu_raw)
        gpu_pm = lookup_gpu_passmark(gpu, gpu_pms)
        cpu_pm = lookup_cpu_passmark(cpu, cpu_patterns)
        if not gpu_pm:
            unknown_gpus.add(gpu)
            continue
        if not cpu_pm:
            unknown_cpus.add(cpu)
            continue
        info = game_info[game]
        true_fps = statistics.median(values)
        min_vram = 4 if info["min_pm"] < 10000 else (6 if info["min_pm"] < 17000 else 8)
        model_fps = model.predict_fps(
            gpu_passmark=float(gpu_pm), cpu_passmark=float(cpu_pm), gpu_vram_gb=vram_gb.get(gpu_raw, 4.0), gpu_name=gpu,
            resolution=res, rec_gpu_passmark=info["rec_pm"], min_gpu_passmark=info["min_pm"],
            rec_cpu_passmark=info["rec_cpu_pm"], rec_vram_gb=float(max(min_vram, 6 if info["rec_pm"] < 14500 else 8)),
            rec_ram_gb=float(info["rec_ram"]), release_year=info["year"], genre="", game_name=info["name"],
        )
        curve_key = (info["name"], res, gpu)
        if curve_key not in curve_cache:
            curve_cache[curve_key] = resolve_fps(gpu, cpu, None, api_available=False, gpu_passmarks=gpu_pms,
                                                 cpu_patterns=cpu_patterns, game_name=info["name"], resolution=res)
        shown = curve_cache[curve_key]
        if shown.fps_source == "unknown" or shown.expected_fps <= 0:
            shown = estimate_from_requirements(gpu, cpu, info["rec_text"], info["min_text"], res,
                                               gpu_passmarks=gpu_pms, cpu_patterns=cpu_patterns,
                                               game_name=info["name"])
        results.append({
            "part": part, "game": game, "gpu": gpu, "cpu": cpu, "cpu_pm": cpu_pm, "res": res, "setting": setting,
            "n": len(values), "true": true_fps, "model": min(model_fps, FPS_CAP), "plugin": float(shown.expected_fps),
            "plugin_lo": float(shown.fps_min), "plugin_hi": float(shown.fps_max), "source": shown.fps_source,
            "heuristic": min(heuristic_m0(float(gpu_pm), info["rec_pm"], res), FPS_CAP),
            "mismatch": info["mismatch"], "in_table": info["in_table"],
        })

    print(f"\n{len(groups)} (game, GPU, CPU, resolution, setting) groups; {len(results)} scored.")
    print(f"GPU names without PassMark: {len(unknown_gpus)} (e.g. {sorted(unknown_gpus)[:6]})")
    print(f"CPU names without PassMark: {len(unknown_cpus)} (e.g. {sorted(unknown_cpus)[:6]})")

    def report(title: str, subset: list[dict]) -> None:
        subset = [r for r in subset if r["true"] <= FPS_CAP and r["plugin"] > 0]
        if len(subset) < 20:
            return
        print(f"\n{title} (n={len(subset)}):")
        for key in ("model", "plugin", "heuristic"):
            print(f"  {key:10s} {metrics([(r['true'], r[key]) for r in subset])}")
        inside = sum(r["plugin_lo"] <= r["true"] <= r["plugin_hi"] for r in subset) / len(subset) * 100
        print(f"  measured FPS inside the plugin's shown range: {inside:.1f}%")

    clean = [r for r in results if not r["mismatch"]]
    over_cap = sum(r["true"] > FPS_CAP for r in clean)
    print(f"Excluded from metrics: {over_cap} groups measured above the plugin's {FPS_CAP} FPS display cap.")
    report("All games with matching Steam release, both data parts", clean)
    for part in ("userbenchmark", "fpsbenchmark"):
        report(f"Data part: {part}", [r for r in clean if r["part"] == part])
    report("Users, group has 3+ reports", [r for r in clean if r["part"] == "userbenchmark" and r["n"] >= 3])
    report("Games with a measured-table curve (seen in training)", [r for r in clean if r["in_table"]])
    report("Games with no table curve (store-spec path, unseen)", [r for r in clean if not r["in_table"]])
    for lo, hi, label in ((0, 6000, "weak CPU (<6k PassMark)"), (6000, 10000, "mid CPU (6-10k)"), (10000, 10**9, "strong CPU (10k+)")):
        report(f"CPU tier: {label}", [r for r in clean if lo <= r["cpu_pm"] < hi])
    for setting in sorted(SETTINGS):
        report(f"Setting: {setting}", [r for r in clean if r["setting"] == setting])
    report("Version-mismatched games (CS:GO vs CS2 specs, Overwatch vs Overwatch 2 specs)",
           [r for r in results if r["mismatch"]])

    print("\nPer game, both parts, matching releases (median abs % error and bias):")
    by_game = defaultdict(list)
    for r in clean:
        if r["true"] <= FPS_CAP:
            by_game[r["game"]].append(r)
    for game, rows in sorted(by_game.items(), key=lambda kv: -len(kv[1])):
        m = {k: metrics([(r["true"], r[k]) for r in rows]) for k in ("model", "plugin", "heuristic")}
        sources = sorted({r["source"] for r in rows})
        print(f"  {game:28s} n={len(rows):5d} model {m['model']['MdAPE_pct']:5.1f}% ({m['model']['bias_pct']:+6.1f})"
              f" | plugin {m['plugin']['MdAPE_pct']:5.1f}% ({m['plugin']['bias_pct']:+6.1f})"
              f" | heuristic {m['heuristic']['MdAPE_pct']:5.1f}% ({m['heuristic']['bias_pct']:+6.1f})  {sources}")

    out = ROOT_DIR / "cache" / "openml_eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
