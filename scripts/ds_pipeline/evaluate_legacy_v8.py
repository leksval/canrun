"""
Legacy CanRun v8.0 model, rebuilt for comparison on the independent benchmarks.

v8.0 was a scikit-learn StackingRegressor (XGBoost, LightGBM, CatBoost, random forest,
gradient boosting, extra trees and ridge, combined by Bayesian ridge, then an isotonic
calibrator and a 0.88 correction). Its pickles stay in data/. The training-time feature
code is reproduced here, and the v8-era benchmark, GPU and model files are read from
git revision 5a61ba3^. This rebuild matches 635 of the 675 FPS values v8 saved, within
0.1 FPS.

    py -3 scripts/ds_pipeline/evaluate_legacy_v8.py
"""

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA = ROOT_DIR / "data"
V8_COMMIT = "5a61ba3^"


def _git_json(path: str) -> dict:
    """A data file as it was when v8.0 was trained."""
    blob = subprocess.check_output(["git", "show", f"{V8_COMMIT}:{path}"], cwd=ROOT_DIR)
    return json.loads(blob.decode("utf-8"))

PIXELS = {"720p": 921600, "1080p": 2073600, "1440p": 3686400, "4K": 8294400}
RES_FACTOR = {"720p": 0.56, "1080p": 1.0, "1440p": 1.78, "4K": 4.0}
VRAM_DIV = {"720p": 2, "1080p": 4, "1440p": 8, "4K": 16}
TIER_TO_GEN = {"RTX 5": "RTX_50", "RTX 4": "RTX_40", "RTX 3": "RTX_30", "RTX 2": "RTX_20",
               "GTX 16": "GTX_16", "GTX 10": "GTX_10", "GTX 9": "GTX_9"}
ARCH = {"RTX_50": "Blackwell", "RTX_40": "Ada_Lovelace", "RTX_30": "Ampere", "RTX_20": "Turing",
        "GTX_16": "Turing", "GTX_10": "Pascal", "GTX_9": "Maxwell"}
TIER = {"Ultra": "Flagship", "High": "High-End", "Medium": "Mid-Range", "Low": "Entry", "Entry": "Entry"}


def game_info_from_description(name: str, description: str) -> dict:
    """The keyword rules from v8's load_training_data()."""
    d = description.lower()
    complexity = 0.6
    if any(k in d for k in ["demanding", "very demanding"]):
        complexity = 0.95
    elif any(k in d for k in ["aaa", "open world"]):
        complexity = 0.85
    elif any(k in d for k in ["rt", "ray tracing", "rtx"]):
        complexity = 0.90
    elif any(k in d for k in ["simulation", "strategy"]):
        complexity = 0.80
    elif any(k in d for k in ["racing", "horror"]):
        complexity = 0.75
    elif any(k in d for k in ["battle royale", "fps"]):
        complexity = 0.70
    elif any(k in d for k in ["competitive", "esports"]):
        complexity = 0.35
    elif any(k in d for k in ["indie", "2d"]):
        complexity = 0.25
    rt = 0.5
    if any(k in d for k in ["rt", "ray tracing", "rtx"]):
        rt = 0.9
    elif any(k in d for k in ["competitive", "esports"]):
        rt = 0.0
    elif any(k in d for k in ["aaa", "demanding"]):
        rt = 0.7
    elif any(k in d for k in ["indie", "2d"]):
        rt = 0.0
    n = name.lower()
    engine = "Unknown"
    if "unreal" in d or any(x in n for x in ["fortnite", "valorant"]):
        engine = "Unreal Engine"
    elif "source" in d or any(x in n for x in ["counter-strike", "apex"]):
        engine = "Source Engine"
    elif "cyberpunk" in n:
        engine = "RED Engine"
    elif "unity" in d:
        engine = "Unity"
    elif any(k in d for k in ["id tech", "doom"]):
        engine = "id Tech"
    # v8 looked up an int vram_usage in a string-keyed dict, so it always encoded 2.
    return {"complexity": complexity, "rt_intensity": rt, "vram_usage_encoded": 2, "engine": engine}


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace(":", "").replace("'", "").replace("-", "_").lower()


class V8Model:
    def __init__(self, gpu_hierarchy_path: Path | None = None):
        self.model = joblib.load(DATA / "unified_stacking_model.pkl")
        self.scaler = joblib.load(DATA / "unified_scaler.pkl")
        self.calibrator = joblib.load(DATA / "unified_calibrator.pkl")
        meta = _git_json("data/ml_fps_model_unified_2025.json")
        self.domain_correction = float(meta["performance"]["domain_correction_factor"])
        self.features = list(self.scaler.feature_names_in_)
        tb = _git_json("data/training_benchmarks.json")
        self.games = {g: game_info_from_description(g, v.get("description", "")) for g, v in tb["games"].items()}
        self.game_keys = {_safe(g): g for g in self.games}
        gh = json.loads(Path(gpu_hierarchy_path).read_text(encoding="utf-8")) if gpu_hierarchy_path else _git_json("data/gpu_hierarchy.json")
        self.gpus = {}
        for name, info in gh.get("nvidia", {}).items():
            gen = next((g for p, g in TIER_TO_GEN.items() if name.startswith(p)), "Unknown")
            self.gpus[name.lower()] = {"memory": info["memory"], "gen": gen, "arch": ARCH.get(gen, "Unknown"),
                                       "tier": TIER.get(info["tier"], "Mid-Range"), "year": info["launch_year"]}

    def _gpu(self, gpu_name: str, vram: float) -> dict:
        key = gpu_name.lower().replace("geforce ", "").replace("nvidia ", "").strip()
        spec = self.gpus.get(key)
        if spec:
            return spec
        return {"memory": vram or 8, "gen": "Unknown", "arch": "Unknown", "tier": "Mid-Range", "year": 2020}

    def _game(self, game_name: str, genre: str) -> tuple[str | None, dict]:
        """A v8 training game keeps its own inputs. Any other game gets its Steam genres as the description."""
        for g in self.games:
            if g.lower() == (game_name or "").lower():
                return g, self.games[g]
        return None, game_info_from_description(game_name or "", genre or "")

    def features_for(self, gpu_name, gpu_passmark, gpu_vram_gb, resolution, game_name="", genre="",
                     cores=3000, base_clock=1500):
        g = self._gpu(gpu_name, gpu_vram_gb)
        known, gi = self._game(game_name, genre)
        pm = float(gpu_passmark)
        mem = float(g["memory"])
        px = PIXELS[resolution]
        f = {
            "passmark": pm, "passmark_normalized": pm / 40000, "memory_gb": mem, "cores": cores, "base_clock": base_clock,
            "generation_age": 2025 - g["year"], "memory_per_passmark": mem / (pm / 1000), "cores_per_memory": cores / mem,
            "performance_per_watt_estimate": pm / (200 + g["year"] * 10), "pixel_count": px,
            "resolution_factor": RES_FACTOR[resolution], "pixels_per_memory_gb": px / mem, "launch_year": g["year"],
            "complexity_score": gi["complexity"], "rt_intensity": gi["rt_intensity"],
            "vram_usage_encoded": gi["vram_usage_encoded"], "quality_encoded": 4,
            "gpu_game_balance": pm * (1 - gi["complexity"]), "vram_resolution_ratio": mem / VRAM_DIV[resolution],
            "performance_density": pm / px * 1e6,
            "rt_capability_match": (1 - gi["rt_intensity"]) if g["gen"] in ["RTX_20", "RTX_30", "RTX_40", "RTX_50"] else 0,
            "generational_performance": pm / (2020 + (int(g["gen"].split("_")[1]) if "_" in g["gen"] else 30) - 10),
            "is_720p": int(resolution == "720p"), "is_1080p": int(resolution == "1080p"),
            "is_1440p": int(resolution == "1440p"), "is_4K": int(resolution == "4K"),
            "is_flagship": int(g["tier"] == "Flagship"), "is_high_end": int(g["tier"] == "High-End"),
            "is_mid_high": int(g["tier"] == "Mid-High"), "is_mid_range": int(g["tier"] == "Mid-Range"),
            "is_blackwell": int(g["arch"] == "Blackwell"), "is_ada_lovelace": int(g["arch"] == "Ada_Lovelace"),
            "is_ampere": int(g["arch"] == "Ampere"), "is_turing": int(g["arch"] == "Turing"),
        }
        f.update({
            "passmark_complexity_interaction": f["passmark_normalized"] * f["complexity_score"],
            "memory_resolution_interaction": mem * f["resolution_factor"],
            "cores_pixel_density": cores / px * 1e6,
            "clock_performance_ratio": base_clock / f["passmark_normalized"],
            "rt_memory_requirement": f["rt_intensity"] * mem,
            "rt_performance_balance": f["rt_intensity"] * f["passmark_normalized"],
            "complexity_vram_stress": f["complexity_score"] * f["vram_usage_encoded"],
            "memory_pixel_ratio": mem * 1024 ** 3 / px,
            "performance_per_pixel": pm / px * 1e6,
            "arch_efficiency": pm / max(1, g["year"] - 2015 + 1),
            "modern_game_compatibility": int(f["generation_age"] <= 5) * f["complexity_score"],
            "quality_performance_ratio": 4 / f["passmark_normalized"],
            "bottleneck_indicator": int(f["vram_resolution_ratio"] < 1),
        })
        core = ["passmark_normalized", "memory_gb", "resolution_factor", "complexity_score"]
        for i, a in enumerate(core):
            for b in core[i + 1:]:
                f[f"poly_{a}_{b}"] = f[a] * f[b]
        for a in ["passmark_normalized", "memory_gb", "complexity_score"]:
            f[f"poly_{a}_squared"] = f[a] ** 2
        for col in self.features:
            if col.startswith("game_"):
                f[col] = int(known is not None and col == f"game_{_safe(known)}")
            elif col.startswith("engine_"):
                f[col] = int(col == f"engine_{_safe(gi['engine']).replace('.', '_')}")
        return f

    def predict_rows(self, rows: list[dict]) -> np.ndarray:
        df = pd.DataFrame(rows)[self.features]
        scaled = pd.DataFrame(self.scaler.transform(df), columns=self.features)
        raw = self.model.predict(scaled)
        return self.calibrator.predict(raw) * self.domain_correction

    def predict_fps(self, *, gpu_name, gpu_passmark, gpu_vram_gb=8.0, resolution="1080p", game_name="", genre="",
                    **_ignored) -> float:
        if resolution not in PIXELS:
            resolution = "1080p"
        f = self.features_for(gpu_name, gpu_passmark, gpu_vram_gb, resolution, game_name, genre)
        return float(max(5.0, min(500.0, self.predict_rows([f])[0])))


def _v8_importlib(model: V8Model) -> types.SimpleNamespace:
    """Stand-in for importlib in the benchmark scripts, so their model.predict_fps calls reach v8."""
    module = types.SimpleNamespace(CanRunFPSModel=types.SimpleNamespace(from_pretrained=lambda *_a, **_k: model))
    util = types.SimpleNamespace(
        spec_from_file_location=lambda *_a, **_k: types.SimpleNamespace(
            loader=types.SimpleNamespace(exec_module=lambda _m: None)
        ),
        module_from_spec=lambda _s: module,
    )
    return types.SimpleNamespace(util=util)


def main() -> int:
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    os.chdir(ROOT_DIR)
    model = V8Model(gpu_hierarchy_path=DATA / "gpu_hierarchy.json")
    scripts = sys.argv[1:] or [
        "evaluate_external_benchmarks", "evaluate_techspot_benchmarks", "evaluate_openml_benchmarks",
    ]
    for name in scripts:
        module = importlib.import_module(f"scripts.ds_pipeline.{name}")
        module.importlib = _v8_importlib(model)
        buffer = io.StringIO()
        argv = sys.argv
        sys.argv = [name]
        try:
            with contextlib.redirect_stdout(buffer):
                module.main()
        finally:
            sys.argv = argv
        print(f"===== {name}, model column = legacy v8.0")
        for line in buffer.getvalue().splitlines():
            if line.lstrip().startswith("model ") or line.endswith(":"):
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
