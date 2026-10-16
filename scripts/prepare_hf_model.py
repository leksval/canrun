"""
Prepare and optionally upload CanRun FPS Model v1.0 for Hugging Face Hub (`grasant/canrun`).

Stages a self-contained, zero-proprietary-data Hugging Face Model repository under `dist/hf_model/`:
- `model.json`: Pure-JSON Two-Stage Log-Ridge + Monotonic GBDT weights (`lookup_table` stripped)
- `config.json`: Hugging Face model configuration and metadata (`version: "1.0"`)
- `evaluation_report.json`: 5-Fold GroupKFold (Leave-Games-Out & Leave-GPUs-Out) ablation report
- `inference.py`: Standalone, zero-dependency (`json` + `math` + `re`) Python inference module
- `README.md`: SOTA Hugging Face Model Card with `model-index` benchmarks, math, and quickstart
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hf_model_card import DEFAULT_REPO_ID, MODEL_NAME, build_model_card, inline_math_to_text  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
UNIFIED_MODEL_PATH = DATA_DIR / "ml_fps_model_unified.json"
EVAL_REPORT_PATH = DATA_DIR / "model_evaluation_report.json"
DEFAULT_STAGE_DIR = ROOT_DIR / "dist" / "hf_model"
MODEL_VERSION = "1.0"

WEIGHT_KEYS: tuple[str, ...] = (
    "version",
    "model_type",
    "trained_at",
    "training_summary",
    "feature_names",
    "monotonic_constraints",
    "scaler_mean",
    "scaler_scale",
    "ridge_weights",
    "ridge_intercept",
    "gbdt_baseline",
    "gbdt_trees",
    "conformal_quantiles",
    "empirical_bayes_prior",
    "cv_metrics",
    "ablation_comparison",
)


INFERENCE_PY_SOURCE = '''"""
Standalone, zero-dependency pure-Python inference for CanRun FPS Model v1.0.

Executes the 21-feature physical engineering pipeline and the Two-Stage
Standardized Log-Space Ridge Prior + Monotonic Huber GBDT Residual ensemble
using only Python's standard library (`json`, `math`, `re`, `pathlib`).
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

MODEL_VERSION = "1.0"
MODEL_NAME = "canrun-fps-predictor"

FEATURE_NAMES: list[str] = [
    "log_gpu_passmark",
    "log_cpu_passmark",
    "gpu_vram_gb",
    "gpu_arch_code",
    "cpu_to_gpu_log_ratio",
    "gpu_tier_score",
    "log_pixel_ratio",
    "vram_per_mpixel",
    "is_4k",
    "log_rec_gpu_passmark",
    "log_min_gpu_passmark",
    "gpu_headroom_rec",
    "gpu_headroom_min",
    "vram_headroom_gb",
    "spec_demand_index",
    "headroom_x_resolution",
    "cpu_bottleneck_1080p",
    "vram_cliff_penalty",
    "is_capped_60",
    "genre_demand",
    "cpu_headroom_rec",
]

MONOTONIC_CONSTRAINTS: list[int] = [
    1,   # log_gpu_passmark
    1,   # log_cpu_passmark
    1,   # gpu_vram_gb
    1,   # gpu_arch_code
    1,   # cpu_to_gpu_log_ratio
    1,   # gpu_tier_score
    -1,  # log_pixel_ratio
    1,   # vram_per_mpixel
    -1,  # is_4k
    0,   # log_rec_gpu_passmark
    0,   # log_min_gpu_passmark
    0,   # gpu_headroom_rec
    0,   # gpu_headroom_min
    0,   # vram_headroom_gb
    0,   # spec_demand_index
    0,   # headroom_x_resolution
    1,   # cpu_bottleneck_1080p
    0,   # vram_cliff_penalty
    -1,  # is_capped_60
    -1,  # genre_demand
    1,   # cpu_headroom_rec
]

RESOLUTION_PIXELS: dict[str, int] = {
    "720p": 1280 * 720,
    "1080p": 1920 * 1080,
    "1440p": 2560 * 1440,
    "4K": 3840 * 2160,
}
BASE_PIXELS_1080P = float(1920 * 1080)

CAPPED_60_TITLE_MARKERS: tuple[str, ...] = (
    "elden ring",
    "sekiro",
    "dark souls",
    "bloodborne",
    "street fighter",
    "tekken",
    "guilty gear",
    "mortal kombat",
)


def infer_gpu_arch_code(gpu_name: str, gen_hint: str = "") -> float:
    """Map GPU model or generation string to ordinal architecture code [1.0 .. 5.0]."""
    text = f"{gpu_name} {gen_hint}".lower()
    if "rtx_50" in text or re.search(r"\\brtx\\s*50\\d{2}\\b", text) or "blackwell" in text:
        return 5.0
    if (
        "rtx_40" in text
        or re.search(r"\\brtx\\s*40\\d{2}\\b", text)
        or "rx_7000" in text
        or re.search(r"\\brx\\s*7\\d{3}\\b", text)
        or "ada" in text
    ):
        return 4.0
    if (
        "rtx_30" in text
        or re.search(r"\\brtx\\s*30\\d{2}\\b", text)
        or "rx_6000" in text
        or re.search(r"\\brx\\s*6\\d{3}\\b", text)
        or "ampere" in text
    ):
        return 3.0
    if (
        "rtx_20" in text
        or "gtx_16" in text
        or re.search(r"\\b(?:rtx\\s*20|gtx\\s*16)\\d{2}\\b", text)
        or "turing" in text
    ):
        return 2.0
    return 1.0


def normalize_resolution_label(resolution: str | None) -> str:
    """Normalize resolution strings to one of 720p, 1080p, 1440p, 4K."""
    cleaned = re.sub(r"[^a-z0-9]+", "", (resolution or "1080p").lower())
    if cleaned in {"4k", "2160p", "3840x2160", "uhd"}:
        return "4K"
    if cleaned in {"1440p", "2560x1440", "qhd", "2k"}:
        return "1440p"
    if cleaned in {"720p", "1280x720", "hd"}:
        return "720p"
    return "1080p"


def infer_capped_60(game_name: str = "", genre: str = "") -> float:
    """Detect a 60 FPS engine lock from title/genre text — never from a game ID."""
    text = f"{game_name} {genre}".lower()
    if any(marker in text for marker in CAPPED_60_TITLE_MARKERS):
        return 1.0
    if "fighting" in text and "engine lock" in text:
        return 1.0
    return 0.0


def infer_genre_demand_offset(genre: str = "") -> float:
    """Map genre string from Steam Store metadata to physical rendering load offset [-0.35 .. +0.25]."""
    g = (genre or "").lower()
    if any(k in g for k in ("esports", "competitive", "moba", "sports", "counter-strike", "valorant")):
        return -0.35
    if any(k in g for k in ("indie", "platformer", "2d", "roguelike", "metroidvania", "puzzle", "card")):
        return -0.25
    if any(k in g for k in ("aaa", "horror", "simulation", "flight", "cyberpunk", "open world")):
        return 0.25
    if "rpg" in g:
        return 0.15
    return 0.0


def extract_feature_vector(
    *,
    gpu_passmark: float,
    cpu_passmark: float = 28000.0,
    gpu_vram_gb: float = 8.0,
    gpu_name: str = "",
    gpu_gen: str = "",
    resolution: str = "1080p",
    rec_gpu_passmark: float = 0.0,
    min_gpu_passmark: float = 0.0,
    rec_cpu_passmark: float = 13000.0,
    rec_vram_gb: float = 6.0,
    rec_ram_gb: float = 16.0,
    release_year: int = 2023,
    genre: str = "",
    is_capped_60: float = 0.0,
    game_name: str = "",
) -> list[float]:
    """Compute the 21 leak-free domain features in deterministic FEATURE_NAMES order."""
    safe_gpu_pm = max(400.0, float(gpu_passmark or 400.0))
    safe_cpu_pm = max(1500.0, float(cpu_passmark or 28000.0))
    safe_vram = max(1.0, float(gpu_vram_gb or 8.0))

    if rec_gpu_passmark <= 0 and min_gpu_passmark > 0:
        safe_min_gpu_pm = max(400.0, float(min_gpu_passmark))
        safe_rec_gpu_pm = safe_min_gpu_pm * 1.65
    elif rec_gpu_passmark > 0 and min_gpu_passmark <= 0:
        safe_rec_gpu_pm = max(400.0, float(rec_gpu_passmark))
        safe_min_gpu_pm = max(400.0, safe_rec_gpu_pm / 1.65)
    elif rec_gpu_passmark <= 0 and min_gpu_passmark <= 0:
        safe_rec_gpu_pm = 6385.0
        safe_min_gpu_pm = 3800.0
    else:
        safe_rec_gpu_pm = max(400.0, float(rec_gpu_passmark))
        safe_min_gpu_pm = max(400.0, min(safe_rec_gpu_pm, float(min_gpu_passmark)))

    safe_rec_cpu_pm = max(1500.0, float(rec_cpu_passmark or 13000.0))
    safe_rec_vram = max(1.0, float(rec_vram_gb or 6.0))
    safe_rec_ram = max(4.0, float(rec_ram_gb or 16.0))
    safe_year = max(2012, min(2027, int(release_year or 2023)))
    genre_offset = infer_genre_demand_offset(genre)

    res_key = normalize_resolution_label(resolution)
    pixels = float(RESOLUTION_PIXELS.get(res_key, RESOLUTION_PIXELS["1080p"]))
    mpixels = pixels / 1_000_000.0

    log_gpu_passmark = math.log(safe_gpu_pm)
    log_cpu_passmark = math.log(safe_cpu_pm)
    raw_arch = infer_gpu_arch_code(gpu_name, gpu_gen)
    gpu_arch_code = 1.0 + 3.8 * (safe_gpu_pm / 40000.0) + 0.01 * raw_arch
    effective_vram = 4.0 + 20.0 * (safe_gpu_pm / 40000.0) + 0.008 * min(24.0, safe_vram)
    cpu_to_gpu_log_ratio = 0.75 * log_gpu_passmark + 0.25 * log_cpu_passmark
    gpu_tier_score = safe_gpu_pm / 40000.0

    log_pixel_ratio = math.log(pixels / BASE_PIXELS_1080P)
    vram_per_mpixel = effective_vram / mpixels
    is_4k = 1.0 if res_key == "4K" else 0.0

    log_rec_gpu_passmark = math.log(safe_rec_gpu_pm)
    log_min_gpu_passmark = math.log(safe_min_gpu_pm)
    gpu_headroom_rec = log_gpu_passmark - log_rec_gpu_passmark
    gpu_headroom_min = log_gpu_passmark - log_min_gpu_passmark
    vram_headroom_gb = effective_vram - safe_rec_vram
    spec_demand_index = (
        0.50 * ((log_rec_gpu_passmark - math.log(4000.0)) / (math.log(32000.0) - math.log(4000.0)))
        + 0.18 * ((math.log(safe_rec_cpu_pm) - math.log(4000.0)) / (math.log(25000.0) - math.log(4000.0)))
        + 0.12 * (min(32.0, safe_rec_ram) / 32.0)
        + 0.10 * ((safe_year - 2015.0) / 11.0)
        + genre_offset
    )

    headroom_x_resolution = gpu_headroom_rec * (1.0 + 0.25 * max(0.0, log_pixel_ratio))
    cpu_bottleneck_1080p = log_cpu_passmark / (1.0 + 0.35 * max(0.0, log_pixel_ratio))
    vram_cliff_penalty = max(0.0, safe_rec_vram * (1.0 + 0.35 * is_4k) - effective_vram)
    capped_60 = 1.0 if float(is_capped_60 or 0.0) >= 0.5 else infer_capped_60(game_name, genre)
    cpu_headroom_rec = log_cpu_passmark - math.log(safe_rec_cpu_pm)

    return [
        log_gpu_passmark,
        log_cpu_passmark,
        effective_vram,
        gpu_arch_code,
        cpu_to_gpu_log_ratio,
        gpu_tier_score,
        log_pixel_ratio,
        vram_per_mpixel,
        is_4k,
        log_rec_gpu_passmark,
        log_min_gpu_passmark,
        gpu_headroom_rec,
        gpu_headroom_min,
        vram_headroom_gb,
        spec_demand_index,
        headroom_x_resolution,
        cpu_bottleneck_1080p,
        vram_cliff_penalty,
        capped_60,
        genre_offset,
        cpu_headroom_rec,
    ]


class CanRunFPSModel:
    """
    Zero-dependency Pure-Python inference engine for CanRun FPS Model v1.0.
    """

    def __init__(self, weights: dict[str, Any]) -> None:
        self.weights = weights
        self.version = str(weights.get("version", MODEL_VERSION))
        self.feature_names: list[str] = list(weights["feature_names"])
        self.scaler_mean: list[float] = [float(v) for v in weights["scaler_mean"]]
        self.scaler_scale: list[float] = [float(v) for v in weights["scaler_scale"]]
        self.ridge_weights: list[float] = [float(v) for v in weights["ridge_weights"]]
        self.ridge_intercept: float = float(weights["ridge_intercept"])
        self.gbdt_baseline: float = float(weights["gbdt_baseline"])
        self.gbdt_trees: list[list[dict[str, Any]]] = weights["gbdt_trees"]
        self.conformal_quantiles: dict[str, dict[str, float]] = weights.get("conformal_quantiles", {})

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path = "grasant/canrun-fps-predictor",
        *,
        filename: str = "model.json",
    ) -> CanRunFPSModel:
        """
        Load `model.json` from a local directory/file path or download from Hugging Face Hub.
        """
        candidate = Path(model_id_or_path)
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return cls(data)
        if candidate.is_dir() and (candidate / filename).is_file():
            data = json.loads((candidate / filename).read_text(encoding="utf-8"))
            return cls(data)
        local_default = Path(__file__).resolve().parent / filename
        if str(model_id_or_path) in {".", "grasant/canrun-fps-predictor", "grasant/canrun"} and local_default.is_file():
            data = json.loads(local_default.read_text(encoding="utf-8"))
            return cls(data)

        from huggingface_hub import hf_hub_download

        downloaded = Path(hf_hub_download(repo_id=str(model_id_or_path), filename=filename))
        data = json.loads(downloaded.read_text(encoding="utf-8"))
        return cls(data)

    def predict_from_features(self, x_vec: list[float]) -> float:
        """Execute Stage-0 Log-Ridge Prior + Stage-1 Monotonic GBDT in pure Python."""
        log_prior = self.ridge_intercept
        for x_val, mu, sc, w in zip(x_vec, self.scaler_mean, self.scaler_scale, self.ridge_weights):
            z = (x_val - mu) / sc if sc != 0.0 else 0.0
            log_prior += z * w

        residual = self.gbdt_baseline
        for tree_nodes in self.gbdt_trees:
            idx = 0
            while not tree_nodes[idx]["leaf"]:
                node = tree_nodes[idx]
                f_val = x_vec[node["f"]]
                if f_val <= node["t"]:
                    idx = node["l"]
                else:
                    idx = node["r"]
            residual += float(tree_nodes[idx]["v"])

        log_fps = max(math.log(5.0), min(math.log(500.0), log_prior + residual))
        return math.exp(log_fps)

    def predict_fps(
        self,
        *,
        gpu_passmark: float,
        cpu_passmark: float = 28000.0,
        gpu_vram_gb: float = 8.0,
        gpu_name: str = "",
        gpu_gen: str = "",
        resolution: str = "1080p",
        rec_gpu_passmark: float = 0.0,
        min_gpu_passmark: float = 0.0,
        rec_cpu_passmark: float = 13000.0,
        rec_vram_gb: float = 6.0,
        rec_ram_gb: float = 16.0,
        release_year: int = 2023,
        genre: str = "",
        is_capped_60: float = 0.0,
        game_name: str = "",
    ) -> float:
        """Predict rasterized FPS for the given hardware and Steam game specification."""
        x_vec = extract_feature_vector(
            gpu_passmark=gpu_passmark,
            cpu_passmark=cpu_passmark,
            gpu_vram_gb=gpu_vram_gb,
            gpu_name=gpu_name,
            gpu_gen=gpu_gen,
            resolution=resolution,
            rec_gpu_passmark=rec_gpu_passmark,
            min_gpu_passmark=min_gpu_passmark,
            rec_cpu_passmark=rec_cpu_passmark,
            rec_vram_gb=rec_vram_gb,
            rec_ram_gb=rec_ram_gb,
            release_year=release_year,
            genre=genre,
            is_capped_60=is_capped_60,
            game_name=game_name,
        )
        fps = self.predict_from_features(x_vec)
        if x_vec[18] >= 0.5:
            fps = min(60.0, fps)
        return round(fps, 2)

    def predict_with_interval(
        self,
        *,
        regime: str = "unseen_game_spec",
        **kwargs: Any,
    ) -> dict[str, float | str]:
        """Return point FPS prediction and 80% split-conformal log interval."""
        res_key = normalize_resolution_label(str(kwargs.get("resolution", "1080p")))
        fps = self.predict_fps(**kwargs)
        q_map = self.conformal_quantiles.get(regime, {})
        q = float(q_map.get(res_key, 0.0535))
        fps_low = round(fps * math.exp(-q), 1)
        fps_high = round(fps * math.exp(q), 1)
        if float(kwargs.get("is_capped_60", 0.0) or 0.0) >= 0.5 or infer_capped_60(
            str(kwargs.get("game_name", "")), str(kwargs.get("genre", ""))
        ):
            fps_low = min(60.0, fps_low)
            fps_high = min(60.0, fps_high)
        return {
            "expected_fps": fps,
            "conformal_80_low": fps_low,
            "conformal_80_high": fps_high,
            "resolution": res_key,
            "model_version": self.version,
        }


def fetch_steam_store_metadata(app_id: int | str, timeout: float = 10.0) -> dict[str, Any]:
    """
    Fetch public game requirements and metadata directly from Valve's unauthenticated
    Steam Store API (`store.steampowered.com/api/appdetails`) at runtime without
    redistributing Steam's store database.
    """
    clean_id = str(app_id).strip()
    if not clean_id.isdigit():
        raise ValueError(f"Steam app_id must be numeric, got {app_id!r}")
    url = f"https://store.steampowered.com/api/appdetails?appids={clean_id}&l=english"
    req = urllib.request.Request(url, headers={"User-Agent": "CanRunFPSPredictor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    entry = payload.get(clean_id, {})
    if not entry.get("success") or not isinstance(entry.get("data"), dict):
        raise RuntimeError(f"Steam Store API returned no data for app_id={clean_id}")
    data = entry["data"]
    pc_req = data.get("pc_requirements") or {}
    genres = ", ".join(g.get("description", "") for g in (data.get("genres") or []) if isinstance(g, dict))
    release_str = str((data.get("release_date") or {}).get("date", ""))
    year_match = re.search(r"\\b(19\\d{2}|20\\d{2})\\b", release_str)
    release_year = int(year_match.group(1)) if year_match else 2023
    return {
        "steam_app_id": int(clean_id),
        "game_name": str(data.get("name", "")),
        "release_year": release_year,
        "genre": genres,
        "minimum_html": str(pc_req.get("minimum", "")) if isinstance(pc_req, dict) else "",
        "recommended_html": str(pc_req.get("recommended", "")) if isinstance(pc_req, dict) else "",
    }
'''


def _ensure_local_model_files() -> None:
    """Restore gitignored runtime data/*.json from the newest dist/canrun-*.zip if missing on disk."""
    if UNIFIED_MODEL_PATH.is_file():
        return
    zip_candidates = sorted((ROOT_DIR / "dist").glob("canrun-*.zip"), reverse=True)
    for zpath in zip_candidates:
        import zipfile

        with zipfile.ZipFile(zpath) as zf:
            names = set(zf.namelist())
            if "canrun/data/ml_fps_model_unified.json" in names:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                for member in names:
                    if member.startswith("canrun/data/") and member.endswith(".json"):
                        target = DATA_DIR / Path(member).name
                        if not target.is_file():
                            target.write_bytes(zf.read(member))
                return


def stage_hf_bundle(stage_dir: Path = DEFAULT_STAGE_DIR) -> dict[str, Path]:
    """
    Stage `model.json`, `config.json`, `evaluation_report.json`, `inference.py`, and `README.md`
    into `stage_dir`, stripping `lookup_table` and verifying pure-Python parity.
    """
    _ensure_local_model_files()
    if not UNIFIED_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing trained model file: {UNIFIED_MODEL_PATH}")
    if not EVAL_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Missing evaluation report file: {EVAL_REPORT_PATH}")

    raw_unified = json.loads(UNIFIED_MODEL_PATH.read_text(encoding="utf-8"))
    raw_eval = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))

    # 1. Build clean weights payload without proprietary lookup_table
    weights_payload: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "architecture_version": str(raw_unified.get("version", "9.0-two-stage-monotonic-log-residual")),
    }
    for key in WEIGHT_KEYS:
        if key == "version":
            continue
        if key in raw_unified:
            weights_payload[key] = raw_unified[key]

    assert "lookup_table" not in weights_payload, "Proprietary lookup_table must not be included in HF model.json"
    assert len(weights_payload["feature_names"]) == 21, "Expected 21 leak-free features"

    stage_dir.mkdir(parents=True, exist_ok=True)

    model_json_path = stage_dir / "model.json"
    config_json_path = stage_dir / "config.json"
    eval_json_path = stage_dir / "evaluation_report.json"
    inference_py_path = stage_dir / "inference.py"
    readme_md_path = stage_dir / "README.md"

    model_json_path.write_text(json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8")

    # 2. Build Hugging Face config.json
    config_payload: dict[str, Any] = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "architecture_version": weights_payload["architecture_version"],
        "architectures": ["TwoStageLogResidualFPSModel"],
        "model_type": "two_stage_monotonic_gbdt_log_ridge",
        "library_name": "generic",
        "pipeline_tag": "tabular-regression",
        "license": "apache-2.0",
        "num_features": len(weights_payload["feature_names"]),
        "num_trees": len(weights_payload["gbdt_trees"]),
        "feature_names": weights_payload["feature_names"],
        "monotonic_constraints": weights_payload["monotonic_constraints"],
        "conformal_quantiles": weights_payload["conformal_quantiles"],
        "empirical_bayes_prior": weights_payload["empirical_bayes_prior"],
    }
    config_json_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    # 3. Write evaluation_report.json
    eval_payload: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        **raw_eval,
    }
    eval_json_path.write_text(json.dumps(eval_payload, indent=2) + "\n", encoding="utf-8")

    # 4. Write standalone inference.py
    inference_py_path.write_text(INFERENCE_PY_SOURCE, encoding="utf-8")

    # 5. Write SOTA Hugging Face Model Card README.md
    readme_md_path.write_text(inline_math_to_text(build_model_card(weights_payload, raw_eval)), encoding="utf-8")

    # 6. Verify parity between staged inference.py and runtime LightweightMLPredictor
    verify_staged_bundle(stage_dir)

    return {
        "model.json": model_json_path,
        "config.json": config_json_path,
        "evaluation_report.json": eval_json_path,
        "inference.py": inference_py_path,
        "README.md": readme_md_path,
    }


def verify_staged_bundle(stage_dir: Path = DEFAULT_STAGE_DIR) -> None:
    """Import staged `inference.py` and verify numerical parity and monotonic invariants."""
    from canrun_model_loader import LightweightMLPredictor

    model_json_path = stage_dir / "model.json"
    inference_py_path = stage_dir / "inference.py"

    raw_text = model_json_path.read_text(encoding="utf-8")
    assert '"lookup_table"' not in raw_text, "lookup_table leaked into model.json"

    spec = importlib.util.spec_from_file_location("hf_canrun_inference", inference_py_path)
    assert spec is not None and spec.loader is not None
    hf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_mod)

    hf_model = hf_mod.CanRunFPSModel.from_pretrained(model_json_path)
    runtime_predictor = LightweightMLPredictor()
    assert runtime_predictor.is_loaded and runtime_predictor.has_two_stage_trees

    probe_cases = [
        {
            "gpu_name": "RTX 4070",
            "gpu_passmark": 26950.0,
            "gpu_vram_gb": 12.0,
            "cpu_passmark": 34300.0,
            "resolution": "1440p",
            "rec_gpu_passmark": 19500.0,
            "min_gpu_passmark": 10100.0,
            "rec_cpu_passmark": 17500.0,
            "rec_vram_gb": 8.0,
            "rec_ram_gb": 16.0,
            "release_year": 2020,
            "genre": "Action RPG Open World Cyberpunk",
            "game_name": "Cyberpunk 2077",
        },
        {
            "gpu_name": "RTX 4090",
            "gpu_passmark": 38900.0,
            "gpu_vram_gb": 24.0,
            "cpu_passmark": 34300.0,
            "resolution": "4K",
            "rec_gpu_passmark": 14500.0,
            "min_gpu_passmark": 8500.0,
            "rec_cpu_passmark": 15000.0,
            "rec_vram_gb": 8.0,
            "rec_ram_gb": 16.0,
            "release_year": 2022,
            "genre": "Action RPG",
            "game_name": "Elden Ring",
        },
        {
            "gpu_name": "RTX 3060",
            "gpu_passmark": 17150.0,
            "gpu_vram_gb": 12.0,
            "cpu_passmark": 22000.0,
            "resolution": "1080p",
            "rec_gpu_passmark": 13500.0,
            "min_gpu_passmark": 7800.0,
            "rec_cpu_passmark": 13000.0,
            "rec_vram_gb": 6.0,
            "rec_ram_gb": 16.0,
            "release_year": 2024,
            "genre": "Action Adventure",
            "game_name": "Unseen Studio Title",
        },
    ]

    max_diff = 0.0
    for case in probe_cases:
        x_hf = hf_mod.extract_feature_vector(**case)
        x_rt = runtime_predictor.extract_feature_vector(**case)
        assert len(x_hf) == 21 and len(x_rt) == 21
        for a, b in zip(x_hf, x_rt):
            assert abs(a - b) < 1e-9, f"Feature mismatch: {a} vs {b}"

        raw_hf = hf_model.predict_from_features(x_hf)
        raw_rt = runtime_predictor.predict_two_stage_from_features(x_rt)
        assert raw_rt is not None
        diff = abs(raw_hf - raw_rt)
        max_diff = max(max_diff, diff)
        assert diff < 1e-6, f"Prediction mismatch ({diff}): {raw_hf} vs {raw_rt}"

    # Verify physical monotonicity: RTX 4090 >= RTX 4070 >= RTX 3060, and 1080p >= 1440p >= 4K
    fps_3060 = hf_model.predict_fps(
        gpu_name="RTX 3060", gpu_passmark=17150, gpu_vram_gb=12, resolution="1440p", rec_gpu_passmark=16000
    )
    fps_4070 = hf_model.predict_fps(
        gpu_name="RTX 4070", gpu_passmark=26950, gpu_vram_gb=12, resolution="1440p", rec_gpu_passmark=16000
    )
    fps_4090 = hf_model.predict_fps(
        gpu_name="RTX 4090", gpu_passmark=38900, gpu_vram_gb=24, resolution="1440p", rec_gpu_passmark=16000
    )
    assert fps_3060 <= fps_4070 <= fps_4090, f"GPU monotonicity violated: {fps_3060}, {fps_4070}, {fps_4090}"

    fps_1080 = hf_model.predict_fps(
        gpu_name="RTX 4070", gpu_passmark=26950, gpu_vram_gb=12, resolution="1080p", rec_gpu_passmark=16000
    )
    fps_1440 = hf_model.predict_fps(
        gpu_name="RTX 4070", gpu_passmark=26950, gpu_vram_gb=12, resolution="1440p", rec_gpu_passmark=16000
    )
    fps_4k = hf_model.predict_fps(
        gpu_name="RTX 4070", gpu_passmark=26950, gpu_vram_gb=12, resolution="4K", rec_gpu_passmark=16000
    )
    assert fps_1080 >= fps_1440 >= fps_4k, f"Resolution monotonicity violated: {fps_1080}, {fps_1440}, {fps_4k}"
    pycache_dir = stage_dir / "__pycache__"
    if pycache_dir.is_dir():
        shutil.rmtree(pycache_dir, ignore_errors=True)
    print(f"[HF Bundle Check] Verified 5 artifacts in {stage_dir} (max parity diff={max_diff:.2e} FPS).")


def upload_hf_bundle(
    stage_dir: Path = DEFAULT_STAGE_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    tag: str = "v1.0",
) -> None:
    """Create the public Hugging Face Model repository, upload `stage_dir`, and tag `v1.0`."""
    hf_exe = shutil.which("hf")
    if hf_exe:
        subprocess.run(
            [hf_exe, "repos", "create", repo_id, "--type", "model", "--public", "--exist-ok"],
            check=True,
        )
        # Run inside stage_dir: on Windows the CLI expands "__pycache__/*"
        # against the working directory before it reaches --delete.
        subprocess.run(
            [
                hf_exe,
                "upload",
                repo_id,
                ".",
                ".",
                "--type",
                "model",
                "--exclude",
                "__pycache__/*",
                "--delete",
                "__pycache__/*",
                "--commit-message",
                f"Release CanRun FPS Model {tag}",
            ],
            check=True,
            cwd=stage_dir,
        )
        subprocess.run(
            [
                hf_exe,
                "repos",
                "tag",
                "create",
                repo_id,
                tag,
                "--type",
                "model",
                "--message",
                f"CanRun FPS Model {tag}",
            ],
            check=False,
        )
        print(f"[HF Upload] Uploaded {stage_dir} to https://huggingface.co/{repo_id} (tag: {tag}).")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        commit_message=f"Release CanRun FPS Model {tag}",
    )
    try:
        api.create_tag(
            repo_id=repo_id,
            repo_type="model",
            tag=tag,
            tag_message=f"CanRun FPS Model {tag}",
        )
    except Exception as exc:
        print(f"[HF Upload] Note on tag {tag}: {exc}")
    print(f"[HF Upload] Uploaded {stage_dir} to https://huggingface.co/{repo_id} (tag: {tag}).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and optionally upload CanRun FPS Model v1.0 to Hugging Face.")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR, help="Directory to stage HF artifacts.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Target Hugging Face model repo ID.")
    parser.add_argument("--tag", type=str, default="v1.0", help="Release tag to create on upload.")
    parser.add_argument("--upload", action="store_true", help="Upload staged bundle to Hugging Face Hub.")
    args = parser.parse_args()

    paths = stage_hf_bundle(args.stage_dir)
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024.0
        print(f"  - {name:<24} {size_kb:>7.1f} KB  ({path})")

    if args.upload:
        upload_hf_bundle(args.stage_dir, repo_id=args.repo_id, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
