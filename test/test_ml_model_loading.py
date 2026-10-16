"""
Unit tests for the staged Hugging Face Model v1.0 bundle (`grasant/canrun-fps-predictor`)
and the zero-dependency standalone `inference.py` module.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from canrun_model_loader import LightweightMLPredictor
from scripts.prepare_hf_model import MODEL_NAME, MODEL_VERSION, stage_hf_bundle


def test_stage_hf_bundle_excludes_lookup_table_and_matches_runtime(tmp_path: Path) -> None:
    """Staging produces all 5 HF files, strips lookup_table, and matches runtime tree predictions."""
    staged = stage_hf_bundle(tmp_path)
    assert set(staged.keys()) == {
        "model.json",
        "config.json",
        "evaluation_report.json",
        "inference.py",
        "README.md",
    }
    for path in staged.values():
        assert path.is_file() and path.stat().st_size > 0

    weights = json.loads(staged["model.json"].read_text(encoding="utf-8"))
    assert weights["model_name"] == MODEL_NAME
    assert weights["version"] == MODEL_VERSION
    assert "lookup_table" not in weights
    assert len(weights["feature_names"]) == 21
    assert len(weights["gbdt_trees"]) == 100

    config = json.loads(staged["config.json"].read_text(encoding="utf-8"))
    assert config["name"] == MODEL_NAME
    assert config["version"] == MODEL_VERSION
    assert config["num_features"] == 21

    readme_text = staged["README.md"].read_text(encoding="utf-8")
    assert "model-index:" in readme_text
    assert "grasant/canrun-fps-predictor" in readme_text
    assert "## Real-world accuracy" in readme_text
    assert "harvested Steam store data stay private" in readme_text
    assert "fetch_steam_store_metadata" in readme_text

    spec = importlib.util.spec_from_file_location("hf_inference_test", staged["inference.py"])
    assert spec is not None and spec.loader is not None
    hf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_mod)

    hf_model = hf_mod.CanRunFPSModel.from_pretrained(staged["model.json"])
    runtime = LightweightMLPredictor()
    assert runtime.is_loaded and runtime.has_two_stage_trees

    kwargs = {
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
    }
    x_hf = hf_mod.extract_feature_vector(**kwargs)
    x_rt = runtime.extract_feature_vector(**kwargs)
    assert x_hf == x_rt

    pred_hf = hf_model.predict_from_features(x_hf)
    pred_rt = runtime.predict_two_stage_from_features(x_rt)
    assert pred_rt is not None
    assert abs(pred_hf - pred_rt) < 1e-6

    interval = hf_model.predict_with_interval(**kwargs)
    assert interval["conformal_80_low"] < interval["expected_fps"] < interval["conformal_80_high"]
    assert interval["model_version"] == "1.0"

    # 60 FPS engine lock enforcement
    capped = hf_model.predict_fps(
        gpu_name="RTX 4090",
        gpu_passmark=38900.0,
        gpu_vram_gb=24.0,
        cpu_passmark=34300.0,
        resolution="1080p",
        rec_gpu_passmark=14500.0,
        min_gpu_passmark=8500.0,
        game_name="Elden Ring",
        genre="Action RPG",
    )
    assert capped <= 60.0
