"""
External hold-out evaluation on measured benchmarks the model never saw.

Source: TechPowerUp, "NVIDIA GeForce RTX 5070 Founders Edition Review" (Mar 2025),
1920x1080 and 2560x1440 per-game charts, Ryzen 7 9800X3D test system, highest settings, RT off.
https://www.techpowerup.com/review/nvidia-geforce-rtx-5070-founders-edition/

All 7 games are absent from data/clean_training_dataset.json. Steam requirements
are fetched live from the public Store API, the same input the plugin uses.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from canrun_steam_fps import (  # noqa: E402
    estimate_from_requirements,
    first_known_gpu,
    load_cpu_patterns,
    load_gpu_passmarks,
    lookup_cpu_passmark,
    lookup_gpu_passmark,
    requirement_passmarks,
)
from scripts.ds_pipeline.collect_and_audit import _extract_field_from_raw_req, _extract_ram_gb  # noqa: E402

TEST_CPU = "AMD Ryzen 7 9800X3D"

GAMES: dict[str, dict] = {
    "Assassin's Creed Mirage": {"app_id": "3035570", "fps": {
        "RX 7900 XTX": 156.6, "RTX 5080": 145.6, "RTX 4080 Super": 145.4, "RTX 4080": 142.7, "RX 7900 XT": 138.3,
        "RTX 5070 Ti": 136.4, "RTX 4070 Ti Super": 129.5, "RTX 4070 Ti": 125.9, "RX 7900 GRE": 119.7,
        "RTX 4070 Super": 117.4, "RTX 5070": 116.3, "RX 7800 XT": 115.1, "RX 6900 XT": 113.8, "RTX 3090 Ti": 111.0,
        "RTX 4070": 105.9, "RTX 3090": 104.9, "RTX 3080": 97.2, "RTX 3070 Ti": 84.8, "RTX 3070": 80.6}},
    "Dragon Age: The Veilguard": {"app_id": "1845910", "fps": {
        "RTX 5080": 114.7, "RTX 4080 Super": 103.2, "RTX 5070 Ti": 102.6, "RTX 4080": 101.3, "RX 7900 XTX": 93.4,
        "RTX 4070 Ti Super": 87.4, "RTX 5070": 84.0, "RTX 4070 Ti": 83.6, "RX 7900 XT": 83.1, "RTX 3090 Ti": 83.0,
        "RTX 4070 Super": 75.0, "RTX 3090": 73.0, "RX 7900 GRE": 70.1, "RTX 4070": 65.4, "RTX 3080": 65.1,
        "RX 6900 XT": 64.8, "RX 7800 XT": 62.6, "RTX 3070 Ti": 54.7, "RTX 3070": 50.3}},
    "F1 24": {"app_id": "2488620", "fps": {
        "RTX 5080": 236.5, "RX 7900 XTX": 223.5, "RTX 4080 Super": 213.6, "RTX 4080": 208.1, "RTX 5070 Ti": 195.1,
        "RX 7900 XT": 192.1, "RTX 3090 Ti": 189.5, "RTX 4070 Ti Super": 182.7, "RX 7900 GRE": 178.4,
        "RTX 4070 Ti": 171.6, "RTX 3090": 166.0, "RTX 5070": 163.7, "RTX 4070 Super": 162.1, "RX 6900 XT": 155.7,
        "RX 7800 XT": 154.9, "RTX 3080": 152.1, "RTX 4070": 144.4, "RTX 3070 Ti": 122.7, "RTX 3070": 116.6}},
    "Like a Dragon: Infinite Wealth": {"app_id": "2072450", "fps": {
        "RTX 5080": 163.8, "RTX 5070 Ti": 144.1, "RTX 4080 Super": 141.5, "RTX 4080": 137.5, "RX 7900 XTX": 132.2,
        "RTX 4070 Ti Super": 121.5, "RTX 5070": 115.6, "RX 7900 XT": 115.0, "RTX 4070 Ti": 109.6,
        "RTX 4070 Super": 101.2, "RTX 3090 Ti": 100.5, "RX 7900 GRE": 100.4, "RX 7800 XT": 93.2, "RTX 3090": 92.9,
        "RX 6900 XT": 90.7, "RTX 4070": 86.9, "RTX 3080": 78.9, "RTX 3070 Ti": 63.5, "RTX 3070": 59.4}},
    "Ratchet & Clank: Rift Apart": {"app_id": "1895880", "fps": {
        "RTX 5080": 168.0, "RTX 4080 Super": 153.7, "RTX 4080": 151.6, "RTX 5070 Ti": 146.7, "RX 7900 XTX": 140.2,
        "RTX 4070 Ti Super": 133.1, "RTX 4070 Ti": 126.3, "RTX 3090 Ti": 122.2, "RX 7900 XT": 120.0,
        "RTX 4070 Super": 115.4, "RTX 5070": 115.2, "RTX 3090": 110.2, "RX 7900 GRE": 102.4, "RTX 4070": 101.0,
        "RTX 3080": 99.6, "RX 6900 XT": 97.1, "RX 7800 XT": 92.6, "RTX 3070 Ti": 69.2, "RTX 3070": 66.3}},
    "SILENT HILL 2": {"app_id": "2124490", "fps": {
        "RTX 5080": 79.7, "RTX 4080 Super": 73.6, "RTX 4080": 72.4, "RTX 5070 Ti": 71.6, "RTX 4070 Ti Super": 63.8,
        "RTX 4070 Ti": 60.1, "RTX 3090 Ti": 59.0, "RX 7900 XTX": 56.3, "RTX 5070": 56.3, "RTX 4070 Super": 54.7,
        "RTX 3090": 52.9, "RTX 4070": 47.8, "RX 7900 XT": 47.7, "RTX 3080": 47.0, "RX 7900 GRE": 42.6,
        "RTX 3070 Ti": 38.7, "RX 6900 XT": 38.1, "RX 7800 XT": 38.0, "RTX 3070": 36.4}},
    "Star Wars Outlaws": {"app_id": "2842040", "fps": {
        "RTX 5080": 87.1, "RTX 4080 Super": 81.8, "RTX 4080": 80.6, "RTX 5070 Ti": 74.5, "RTX 4070 Ti Super": 69.0,
        "RX 7900 XTX": 68.5, "RTX 3090 Ti": 66.1, "RTX 4070 Ti": 65.6, "RX 7900 XT": 59.2, "RTX 4070 Super": 59.0,
        "RTX 5070": 57.2, "RTX 3090": 57.0, "RTX 3080": 51.0, "RTX 4070": 50.5, "RX 7900 GRE": 49.5,
        "RX 6900 XT": 44.2, "RX 7800 XT": 43.4, "RTX 3070 Ti": 40.5, "RTX 3070": 38.4}},
}

FPS_1080P: dict[str, dict[str, float]] = {
    "Assassin's Creed Mirage": {
        "RX 7900 XTX": 191.2, "RTX 5080": 178.4, "RTX 4080 Super": 176.4, "RTX 4080": 175.9, "RX 7900 XT": 173.3,
        "RTX 5070 Ti": 165.3, "RTX 4070 Ti Super": 161.0, "RTX 4070 Ti": 158.6, "RX 7900 GRE": 150.2,
        "RTX 5070": 148.7, "RX 6900 XT": 147.8, "RTX 4070 Super": 147.5, "RX 7800 XT": 142.3, "RTX 4070": 136.5,
        "RTX 3090 Ti": 131.3, "RTX 3090": 123.8, "RTX 3080": 116.9, "RTX 3070 Ti": 105.9, "RTX 3070": 101.4},
    "Dragon Age: The Veilguard": {
        "RTX 5080": 139.4, "RTX 4080 Super": 129.7, "RTX 4080": 129.0, "RTX 5070 Ti": 125.4, "RX 7900 XTX": 112.8,
        "RTX 4070 Ti Super": 111.4, "RTX 4070 Ti": 107.5, "RTX 3090 Ti": 104.0, "RTX 5070": 103.8,
        "RX 7900 XT": 101.2, "RTX 4070 Super": 97.1, "RTX 3090": 91.8, "RTX 4070": 86.9, "RX 7900 GRE": 86.1,
        "RTX 3080": 82.4, "RX 6900 XT": 79.1, "RX 7800 XT": 75.9, "RTX 3070 Ti": 70.9, "RTX 3070": 64.7},
    "F1 24": {
        "RTX 5080": 270.4, "RX 7900 XTX": 265.8, "RTX 4080 Super": 246.1, "RTX 4080": 245.2, "RX 7900 XT": 233.0,
        "RTX 5070 Ti": 230.3, "RTX 3090 Ti": 223.4, "RTX 4070 Ti Super": 213.2, "RX 7900 GRE": 206.5,
        "RTX 4070 Ti": 204.1, "RTX 3090": 202.1, "RTX 4070 Super": 194.1, "RTX 5070": 193.3, "RX 6900 XT": 187.1,
        "RX 7800 XT": 184.6, "RTX 3080": 182.2, "RTX 4070": 173.3, "RTX 3070 Ti": 147.1, "RTX 3070": 138.6},
    "Like a Dragon: Infinite Wealth": {
        "RTX 5080": 246.7, "RTX 5070 Ti": 224.6, "RTX 4080 Super": 219.6, "RTX 4080": 218.4, "RX 7900 XTX": 195.6,
        "RTX 4070 Ti Super": 190.1, "RTX 5070": 187.4, "RTX 4070 Ti": 178.2, "RX 7900 XT": 171.1,
        "RTX 4070 Super": 168.2, "RTX 3090 Ti": 157.8, "RX 7900 GRE": 151.9, "RTX 3090": 148.2, "RX 7800 XT": 144.1,
        "RTX 4070": 143.8, "RX 6900 XT": 139.3, "RTX 3080": 130.0, "RTX 3070 Ti": 109.5, "RTX 3070": 101.4},
    "Ratchet & Clank: Rift Apart": {
        "RTX 5080": 214.0, "RTX 4080 Super": 198.9, "RTX 4080": 197.8, "RTX 5070 Ti": 193.5, "RX 7900 XTX": 187.3,
        "RTX 4070 Ti Super": 176.9, "RTX 4070 Ti": 168.8, "RX 7900 XT": 164.3, "RTX 3090 Ti": 160.2,
        "RTX 5070": 158.5, "RTX 4070 Super": 156.2, "RTX 3090": 146.1, "RX 7900 GRE": 142.5, "RTX 4070": 138.3,
        "RTX 3080": 136.8, "RX 6900 XT": 132.4, "RX 7800 XT": 127.5, "RTX 3070 Ti": 90.7, "RTX 3070": 87.4},
    "SILENT HILL 2": {
        "RTX 5080": 105.3, "RTX 4080 Super": 99.7, "RTX 4080": 98.2, "RTX 5070 Ti": 97.4, "RTX 4070 Ti Super": 87.6,
        "RTX 4070 Ti": 84.6, "RTX 3090 Ti": 78.8, "RTX 5070": 78.6, "RTX 4070 Super": 77.6, "RX 7900 XTX": 77.3,
        "RTX 3090": 71.5, "RTX 4070": 68.5, "RX 7900 XT": 67.0, "RTX 3080": 64.6, "RX 7900 GRE": 58.9,
        "RX 6900 XT": 53.9, "RTX 3070 Ti": 53.9, "RX 7800 XT": 53.8, "RTX 3070": 51.1},
    "Star Wars Outlaws": {
        "RTX 5080": 123.9, "RTX 4080 Super": 118.4, "RTX 4080": 117.9, "RTX 5070 Ti": 109.5, "RTX 4070 Ti Super": 102.2,
        "RX 7900 XTX": 98.9, "RTX 4070 Ti": 98.2, "RTX 3090 Ti": 93.8, "RTX 4070 Super": 88.2, "RX 7900 XT": 87.2,
        "RTX 5070": 85.2, "RTX 3090": 83.8, "RTX 4070": 76.6, "RTX 3080": 74.3, "RX 7900 GRE": 73.5,
        "RX 6900 XT": 65.6, "RX 7800 XT": 65.0, "RTX 3070 Ti": 60.5, "RTX 3070": 57.4},
}

VRAM_GB = {
    "RTX 5080": 16, "RTX 5070 Ti": 16, "RTX 5070": 12, "RTX 4080 Super": 16, "RTX 4080": 16,
    "RTX 4070 Ti Super": 16, "RTX 4070 Ti": 12, "RTX 4070 Super": 12, "RTX 4070": 12, "RTX 3090 Ti": 24,
    "RTX 3090": 24, "RTX 3080": 10, "RTX 3070 Ti": 8, "RTX 3070": 8, "RX 7900 XTX": 24, "RX 7900 XT": 20,
    "RX 7900 GRE": 16, "RX 7800 XT": 16, "RX 6900 XT": 16,
}


def fetch_store(app_id: str) -> dict:
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=english&cc=us"
    req = urllib.request.Request(url, headers={"User-Agent": "CanRunEval/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        entry = json.loads(resp.read().decode("utf-8"))[app_id]
    data = entry["data"]
    pc = data.get("pc_requirements") or {}
    strip = lambda s: re.sub(r"<[^>]+>", " ", str(s or ""))  # noqa: E731
    year = re.search(r"(19|20)\d\d", str((data.get("release_date") or {}).get("date", "")))
    return {
        "name": data.get("name", ""),
        "min_raw": strip(pc.get("minimum")) if isinstance(pc, dict) else "",
        "rec_raw": strip(pc.get("recommended")) if isinstance(pc, dict) else "",
        "genre": ", ".join(g["description"] for g in data.get("genres") or []),
        "release_year": int(year.group(0)) if year else 2023,
    }


def metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    ape = [abs(p - t) / t * 100 for t, p in pairs]
    truth = [t for t, _ in pairs]
    mean_t = statistics.fmean(truth)
    ss_res = sum((t - p) ** 2 for t, p in pairs)
    ss_tot = sum((t - mean_t) ** 2 for t in truth)
    return {
        "n": len(pairs),
        "MdAPE_pct": round(statistics.median(ape), 1),
        "MAPE_pct": round(statistics.fmean(ape), 1),
        "MAE_fps": round(statistics.fmean(abs(p - t) for t, p in pairs), 1),
        "R2": round(1 - ss_res / ss_tot, 3),
        "within_10pct": round(sum(a <= 10 for a in ape) / len(ape) * 100, 1),
        "within_22pct": round(sum(a <= 22 for a in ape) / len(ape) * 100, 1),
        "bias_pct": round(statistics.fmean((p - t) / t * 100 for t, p in pairs), 1),
    }


def heuristic_m0(gpu_pm: float, rec_pm: float, resolution: str) -> float:
    return 60.0 * (gpu_pm / rec_pm) ** 0.85 * {"1080p": 1.0, "1440p": 0.75}[resolution]


def main() -> int:
    spec = importlib.util.spec_from_file_location("hf_inf", ROOT_DIR / "dist" / "hf_model" / "inference.py")
    hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf)
    model_path = ROOT_DIR / "dist" / "hf_model" / "model.json"
    model = hf.CanRunFPSModel.from_pretrained(model_path)
    bands = json.loads(model_path.read_text(encoding="utf-8"))["conformal_quantiles"]["unseen_game_spec"]

    gpu_pms = load_gpu_passmarks()
    cards = requirement_passmarks(gpu_pms)
    cpu_patterns = load_cpu_patterns()
    cpu_pm = lookup_cpu_passmark(TEST_CPU, cpu_patterns) or 40000

    rows = {"model": [], "plugin": [], "heuristic": []}
    by_res: dict[str, dict[str, list]] = {r: {"model": [], "plugin": [], "heuristic": []} for r in ("1080p", "1440p")}
    per_game: dict[str, dict] = {}
    model_in_band = 0
    skipped: set[str] = set()

    for game, info in GAMES.items():
        store = fetch_store(info["app_id"])
        rec_gpu_text = _extract_field_from_raw_req(store["rec_raw"], "(?:Graphics|Video Card)") or store["rec_raw"]
        min_gpu_text = _extract_field_from_raw_req(store["min_raw"], "(?:Graphics|Video Card)") or store["min_raw"]
        rec = first_known_gpu(rec_gpu_text, cards)
        mn = first_known_gpu(min_gpu_text, cards)
        min_pm = float(mn[1]) if mn else 6385.0
        rec_pm = float(rec[1]) if rec else min_pm * 1.65
        rec_cpu_text = _extract_field_from_raw_req(store["rec_raw"], "Processor")
        rec_cpu_pm = float(lookup_cpu_passmark(rec_cpu_text, cpu_patterns) or 16000)
        rec_ram = _extract_ram_gb(store["rec_raw"], 16)
        min_vram = 4 if min_pm < 10000 else (6 if min_pm < 17000 else 8)
        rec_vram = max(min_vram, 6 if rec_pm < 14500 else 8)
        print(f"{game}: rec={rec[0] if rec else None} ({rec_pm:.0f}) min={mn[0] if mn else None} ({min_pm:.0f})")

        game_rows = {"model": [], "plugin": [], "heuristic": []}
        for resolution, measured in (("1440p", info["fps"]), ("1080p", FPS_1080P[game])):
            for gpu, true_fps in measured.items():
                gpu_pm = lookup_gpu_passmark(gpu, cards)
                if not gpu_pm:
                    skipped.add(gpu)
                    continue
                pred = model.predict_fps(
                    gpu_passmark=float(gpu_pm), cpu_passmark=float(cpu_pm), gpu_vram_gb=float(VRAM_GB[gpu]),
                    gpu_name=gpu, resolution=resolution, rec_gpu_passmark=rec_pm, min_gpu_passmark=min_pm,
                    rec_cpu_passmark=rec_cpu_pm, rec_vram_gb=float(rec_vram), rec_ram_gb=float(rec_ram),
                    release_year=store["release_year"], genre=store["genre"], game_name=store["name"],
                )
                plugin = estimate_from_requirements(
                    gpu, TEST_CPU, recommended_gpu=rec_gpu_text, minimum_gpu=min_gpu_text,
                    resolution=resolution, gpu_passmarks=gpu_pms, cpu_patterns=cpu_patterns,
                    game_name=store["name"], user_vram_gb=float(VRAM_GB[gpu]), recommended_cpu=rec_cpu_text,
                    recommended_ram_gb=float(rec_ram), release_year=store["release_year"], genre=store["genre"],
                ).expected_fps
                heur = heuristic_m0(float(gpu_pm), rec_pm, resolution)
                if abs(math.log(pred) - math.log(true_fps)) <= bands[resolution]:
                    model_in_band += 1
                for key, value in (("model", pred), ("plugin", float(plugin)), ("heuristic", heur)):
                    rows[key].append((true_fps, value))
                    game_rows[key].append((true_fps, value))
                    by_res[resolution][key].append((true_fps, value))
        per_game[game] = {k: metrics(v) for k, v in game_rows.items()}

    if skipped:
        print(f"Skipped GPUs with no PassMark entry: {sorted(skipped)}")
    print(f"\nOverall on {len(rows['model'])} unseen measured results (1080p + 1440p):")
    for key, pairs in rows.items():
        print(f"  {key:10s} {metrics(pairs)}")
    print(f"  model inside its published 80% band: {model_in_band / len(rows['model']) * 100:.1f}%")
    for resolution, res_rows in by_res.items():
        print(f"\n{resolution}:")
        for key, pairs in res_rows.items():
            print(f"  {key:10s} {metrics(pairs)}")
    print("\nPer game (median abs % error, bias %):")
    for game, res in per_game.items():
        print(
            f"  {game:32s} model {res['model']['MdAPE_pct']:5.1f}% ({res['model']['bias_pct']:+6.1f})"
            f" | plugin {res['plugin']['MdAPE_pct']:5.1f}% ({res['plugin']['bias_pct']:+6.1f})"
            f" | heuristic {res['heuristic']['MdAPE_pct']:5.1f}% ({res['heuristic']['bias_pct']:+6.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
