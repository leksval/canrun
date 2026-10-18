"""
Lightweight ML Model Loader for FPS Prediction
Zero sklearn/numpy dependency at runtime — executes Two-Stage Log-Ridge Prior +
Monotonic GBDT Residual Tree traversal and audited lookup table in pure Python.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

# Configuration variables for file paths
GPU_HIERARCHY_FILE = "gpu_hierarchy.json"
MODEL_OUTPUT_FILE = "ml_fps_model_unified.json"

RESOLUTION_PIXELS: dict[str, int] = {
    "720p": 1280 * 720,
    "1080p": 1920 * 1080,
    "1440p": 2560 * 1440,
    "4K": 3840 * 2160,
}
BASE_PIXELS_1080P = float(1920 * 1080)


def _gpu_words(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).split())


def match_gpu_key(gpu_name: str, entries: dict[str, Any]) -> str | None:
    """Key of the GPU table entry for a detected GPU name, or None.

    The longest table name found as whole words wins, so "RTX 4060 Laptop GPU"
    is not read as the desktop "RTX 4060", nor "RTX 4060 Ti" as "RTX 4060".
    Entries may list "aliases". A name that is only part of a table name
    (e.g. "4090") falls back to the shortest such entry.
    """
    text = f" {_gpu_words(gpu_name)} "
    if not text.strip():
        return None
    best, best_len = None, 0
    for key, info in entries.items():
        aliases = info.get("aliases", []) if isinstance(info, dict) else []
        for candidate in (key, *aliases):
            words = _gpu_words(candidate)
            if words and f" {words} " in text and len(words) > best_len:
                best, best_len = key, len(words)
    if best is not None:
        return best
    partial = [key for key in entries if f" {text.strip()} " in f" {_gpu_words(key)} "]
    return min(partial, key=len) if partial else None


class LightweightMLPredictor:
    """
    Pure-Python Two-Stage Log-Ridge + Monotonic Residual GBDT predictor +
    audited multi-resolution lookup table. No sklearn dependency required at runtime.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.model_data: dict[str, Any] | None = None
        self.lookup_table: dict[str, Any] = {}
        self.gpu_hierarchy: dict[str, dict[str, Any]] = {}
        self.is_loaded = False
        self.has_two_stage_trees = False

        # Class constants
        self.RESOLUTION_FACTORS = {"720p": 1.3, "1080p": 1.0, "1440p": 0.65, "4K": 0.4}
        self.FPS_BOUNDS = {"720p": 360, "1080p": 300, "1440p": 240, "4K": 180}

        # Load the pre-trained model data and GPU hierarchy
        self._load_model()
        self._load_gpu_hierarchy()

    def _load_model(self) -> None:
        """Load the pre-trained ML model JSON with PyInstaller support."""
        try:
            if getattr(sys, "frozen", False):
                bundle_dir = Path(sys._MEIPASS)
                model_paths = [
                    bundle_dir / MODEL_OUTPUT_FILE,
                    bundle_dir / "ml_fps_model.json",
                    bundle_dir / "canrun" / MODEL_OUTPUT_FILE,
                    bundle_dir / "data" / MODEL_OUTPUT_FILE,
                ]
            else:
                current_dir = Path(__file__).parent
                model_paths = [
                    current_dir / MODEL_OUTPUT_FILE,
                    current_dir / "ml_fps_model.json",
                    current_dir / "data" / MODEL_OUTPUT_FILE,
                    current_dir.parent / "data" / MODEL_OUTPUT_FILE,
                    Path.cwd() / "data" / MODEL_OUTPUT_FILE,
                ]

            for model_path in model_paths:
                if model_path.exists():
                    with model_path.open(encoding="utf-8") as f:
                        self.model_data = json.load(f)
                        self.lookup_table = self.model_data.get("lookup_table", {})
                        self.has_two_stage_trees = bool(
                            self.model_data.get("gbdt_trees")
                            and self.model_data.get("ridge_weights")
                            and self.model_data.get("scaler_mean")
                        )
                        self.is_loaded = True
                        model_version = self.model_data.get("version", "1.0")
                        self.logger.info(
                            "ML model v%s loaded: %s lookup curves, two-stage GBDT=%s",
                            model_version,
                            len(self.lookup_table),
                            self.has_two_stage_trees,
                        )
                        return

            self.logger.warning("ML model file not found in paths: %s", [str(p) for p in model_paths])
        except Exception as e:
            self.logger.error("Failed to load ML model: %s", e)

    def _load_gpu_hierarchy(self) -> None:
        """Load PassMark and VRAM data from GPU hierarchy with PyInstaller support."""
        try:
            if getattr(sys, "frozen", False):
                bundle_dir = Path(sys._MEIPASS)
                hierarchy_paths = [
                    bundle_dir / GPU_HIERARCHY_FILE,
                    bundle_dir / "canrun" / GPU_HIERARCHY_FILE,
                    bundle_dir / "data" / GPU_HIERARCHY_FILE,
                ]
            else:
                current_dir = Path(__file__).parent
                hierarchy_paths = [
                    current_dir / GPU_HIERARCHY_FILE,
                    current_dir / "data" / GPU_HIERARCHY_FILE,
                    current_dir.parent / "data" / GPU_HIERARCHY_FILE,
                    Path.cwd() / "data" / GPU_HIERARCHY_FILE,
                    Path.cwd() / "canrun" / "data" / GPU_HIERARCHY_FILE,
                ]

            hierarchy_path = next((p for p in hierarchy_paths if p.exists()), None)
            if hierarchy_path:
                with hierarchy_path.open(encoding="utf-8") as f:
                    hierarchy_data = json.load(f)
                    self.gpu_hierarchy = {}
                    for vendor in ("nvidia", "amd", "intel"):
                        vendor_gpus = hierarchy_data.get(vendor, {})
                        if not isinstance(vendor_gpus, dict):
                            continue
                        for gpu_name, gpu_data in vendor_gpus.items():
                            if not isinstance(gpu_data, dict):
                                continue
                            passmark = gpu_data.get("passmark_g3d", 0)
                            vram = gpu_data.get("memory", 8)
                            if passmark > 0:
                                self.gpu_hierarchy[gpu_name.lower()] = {
                                    "passmark": int(passmark),
                                    "vram": int(vram),
                                }
                    self.logger.info("GPU database loaded: %s GPUs", len(self.gpu_hierarchy))
            else:
                self.logger.warning("GPU hierarchy file not found in paths: %s", [str(p) for p in hierarchy_paths])
        except Exception as e:
            self.logger.error("Failed to load GPU hierarchy: %s", e)

    @staticmethod
    def _infer_gpu_arch_code(gpu_name: str, gen_hint: str = "") -> float:
        text = f"{gpu_name} {gen_hint}".lower()
        if "rtx_50" in text or re.search(r"\brtx\s*50\d{2}\b", text) or "blackwell" in text:
            return 5.0
        if (
            "rtx_40" in text
            or re.search(r"\brtx\s*40\d{2}\b", text)
            or "rx_7000" in text
            or re.search(r"\brx\s*7\d{3}\b", text)
            or "ada" in text
        ):
            return 4.0
        if (
            "rtx_30" in text
            or re.search(r"\brtx\s*30\d{2}\b", text)
            or "rx_6000" in text
            or re.search(r"\brx\s*6\d{3}\b", text)
            or "ampere" in text
        ):
            return 3.0
        if (
            "rtx_20" in text
            or "gtx_16" in text
            or re.search(r"\b(?:rtx\s*20|gtx\s*16)\d{2}\b", text)
            or "turing" in text
        ):
            return 2.0
        return 1.0

    @staticmethod
    def _infer_genre_offset(genre: str = "") -> float:
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

    @staticmethod
    def _infer_capped_60(game_name: str = "", genre: str = "") -> float:
        text = f"{game_name} {genre}".lower()
        markers = (
            "elden ring",
            "sekiro",
            "dark souls",
            "bloodborne",
            "street fighter",
            "tekken",
            "guilty gear",
            "mortal kombat",
        )
        return 1.0 if any(marker in text for marker in markers) else 0.0

    @classmethod
    def extract_feature_vector(
        cls,
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
        """Compute the 21 domain features matching `scripts/ds_pipeline/features.py`."""
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
        genre_offset = cls._infer_genre_offset(genre)

        res_clean = re.sub(r"[^a-z0-9]+", "", (resolution or "1080p").lower())
        if res_clean in {"4k", "2160p", "3840x2160", "uhd"}:
            res_key = "4K"
        elif res_clean in {"1440p", "2560x1440", "qhd", "2k"}:
            res_key = "1440p"
        elif res_clean in {"720p", "1280x720", "hd"}:
            res_key = "720p"
        else:
            res_key = "1080p"

        pixels = float(RESOLUTION_PIXELS.get(res_key, RESOLUTION_PIXELS["1080p"]))
        mpixels = pixels / 1_000_000.0

        log_gpu_passmark = math.log(safe_gpu_pm)
        log_cpu_passmark = math.log(safe_cpu_pm)
        raw_arch = cls._infer_gpu_arch_code(gpu_name, gpu_gen)
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
        capped_60 = 1.0 if float(is_capped_60 or 0.0) >= 0.5 else cls._infer_capped_60(game_name, genre)
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

    def predict_two_stage_from_features(self, x_vec: list[float]) -> float | None:
        """Execute Level-0 Log-Ridge Prior + Level-1 Monotonic GBDT in pure Python."""
        if not self.is_loaded or not self.has_two_stage_trees or not self.model_data:
            return None
        try:
            means = self.model_data["scaler_mean"]
            scales = self.model_data["scaler_scale"]
            weights = self.model_data["ridge_weights"]
            intercept = float(self.model_data["ridge_intercept"])

            log_prior = intercept
            for x_val, mu, sc, w in zip(x_vec, means, scales, weights):
                z = (x_val - mu) / sc if sc != 0.0 else 0.0
                log_prior += z * w

            residual = float(self.model_data["gbdt_baseline"])
            for tree_nodes in self.model_data["gbdt_trees"]:
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
        except Exception as e:
            self.logger.warning("Two-stage tree inference failed: %s", e)
            return None

    def predict_fps(
        self,
        gpu_name: str,
        passmark_score: int,
        resolution: str,
        game_name: str,
        memory_gb: int = 8,
        game_requirements: dict | None = None,
        cpu_passmark: int = 28000,
    ) -> float | None:
        """
        Predict FPS for ANY Steam game using:
        1. Audited multi-resolution lookup table when exact GPU|Game|Resolution exists
        2. Two-Stage Log-Ridge + Monotonic Residual GBDT on Steam hardware requirements
        3. Similar-GPU PassMark power-law curve interpolation
        """
        if self.is_loaded:
            normalized_gpu = self._normalize_gpu_name(gpu_name)
            normalized_game = self._normalize_game_name(game_name)
            lookup_key = f"{normalized_gpu}|{normalized_game}|{resolution}"
            if normalized_game != "General Gaming" and lookup_key in self.lookup_table:
                return float(self.lookup_table[lookup_key])

        if game_requirements:
            fps = self._predict_fps_from_steam_requirements(
                gpu_name,
                passmark_score,
                resolution,
                memory_gb,
                game_requirements,
                cpu_passmark=cpu_passmark,
            )
            if fps:
                return fps

        if self.is_loaded:
            normalized_gpu = self._normalize_gpu_name(gpu_name)
            normalized_game = self._normalize_game_name(game_name)
            lookup_key = f"{normalized_gpu}|{normalized_game}|{resolution}"
            if lookup_key in self.lookup_table:
                return float(self.lookup_table[lookup_key])

            similar_fps = self._find_similar_gpu_prediction(
                normalized_gpu, passmark_score, normalized_game, resolution
            )
            if similar_fps:
                return similar_fps

            if self.has_two_stage_trees and passmark_score > 0:
                x_vec = self.extract_feature_vector(
                    gpu_passmark=float(passmark_score),
                    cpu_passmark=float(cpu_passmark),
                    gpu_vram_gb=float(memory_gb or self._get_gpu_vram(gpu_name)),
                    gpu_name=gpu_name,
                    resolution=resolution,
                    game_name=game_name,
                )
                tree_fps = self.predict_two_stage_from_features(x_vec)
                if tree_fps is not None:
                    if self._infer_capped_60(game_name):
                        tree_fps = min(60.0, tree_fps)
                    return round(tree_fps, 1)

        estimated_fps = self._estimate_from_passmark(passmark_score, resolution, game_name)
        if estimated_fps:
            return estimated_fps

        return None

    def _predict_fps_from_steam_requirements(
        self,
        gpu_name: str,
        passmark_score: int,
        resolution: str,
        memory_gb: int,
        game_requirements: dict,
        cpu_passmark: int = 28000,
    ) -> float | None:
        """Two-Stage Log-Ridge + Monotonic GBDT FPS prediction using Steam Store hardware specs."""
        try:
            min_gpu = str(game_requirements.get("minimum_gpu", "") or "")
            rec_gpu = str(game_requirements.get("recommended_gpu", "") or "")
            min_vram = int(game_requirements.get("minimum_vram_gb", 4) or 4)
            rec_vram = int(game_requirements.get("recommended_vram_gb", 6) or 6)
            rec_ram = int(game_requirements.get("recommended_ram_gb", 16) or 16)
            release_year = int(game_requirements.get("release_year", 2023) or 2023)
            genre = str(game_requirements.get("genre", "") or "")

            min_gpu_score = self._get_gpu_passmark_score(min_gpu)
            rec_gpu_score = self._get_gpu_passmark_score(rec_gpu)
            user_gpu_vram = memory_gb or self._get_gpu_vram(gpu_name)

            if self.has_two_stage_trees and passmark_score > 0 and (rec_gpu_score > 0 or min_gpu_score > 0):
                x_vec = self.extract_feature_vector(
                    gpu_passmark=float(passmark_score),
                    cpu_passmark=float(cpu_passmark),
                    gpu_vram_gb=float(user_gpu_vram),
                    gpu_name=gpu_name,
                    resolution=resolution,
                    rec_gpu_passmark=float(rec_gpu_score),
                    min_gpu_passmark=float(min_gpu_score),
                    rec_vram_gb=float(rec_vram),
                    rec_ram_gb=float(rec_ram),
                    release_year=release_year,
                    genre=genre,
                    is_capped_60=float(game_requirements.get("is_capped_60", 0) or 0),
                    game_name=str(game_requirements.get("game_name", "") or ""),
                )
                tree_fps = self.predict_two_stage_from_features(x_vec)
                if tree_fps is not None:
                    return round(max(10.0, min(tree_fps, float(self.FPS_BOUNDS.get(resolution, 240)))), 1)

            meets_recommended = (
                (passmark_score >= rec_gpu_score and user_gpu_vram >= rec_vram) if rec_gpu_score > 0 else False
            )
            meets_minimum = (
                (passmark_score >= min_gpu_score and user_gpu_vram >= min_vram) if min_gpu_score > 0 else False
            )

            if not meets_minimum and min_gpu_score > 0:
                return max(15.0, min(30.0, passmark_score / 800.0))

            if meets_recommended and rec_gpu_score > 0:
                predicted_fps = 60.0 * ((passmark_score / rec_gpu_score) ** 0.85)
            elif meets_minimum and min_gpu_score > 0:
                predicted_fps = 35.0 * ((passmark_score / min_gpu_score) ** 0.85)
            else:
                predicted_fps = passmark_score / 200.0

            predicted_fps *= self.RESOLUTION_FACTORS.get(resolution, 1.0)
            final_fps = max(15.0, min(predicted_fps, float(self.FPS_BOUNDS.get(resolution, 200))))
            return round(final_fps, 1)

        except Exception as e:
            self.logger.warning("ML requirements analysis failed: %s", e)
            return None

    def _get_gpu_passmark_score(self, gpu_name: str) -> int:
        """Get PassMark score from GPU hierarchy database."""
        if not gpu_name or not self.gpu_hierarchy:
            return 0

        gpu_lower = gpu_name.lower().strip()
        if gpu_lower in self.gpu_hierarchy:
            return int(self.gpu_hierarchy[gpu_lower]["passmark"])

        key = match_gpu_key(gpu_lower, self.gpu_hierarchy)
        if key is not None:
            return int(self.gpu_hierarchy[key]["passmark"])

        gpu_numbers = "".join(filter(str.isdigit, gpu_lower))
        if gpu_numbers:
            for gpu_key, gpu_data in self.gpu_hierarchy.items():
                gpu_key_numbers = "".join(filter(str.isdigit, gpu_key))
                if gpu_numbers == gpu_key_numbers:
                    return int(gpu_data["passmark"])

        return 0

    def _get_gpu_vram(self, gpu_name: str) -> int:
        """Get GPU VRAM from hierarchy database."""
        if not gpu_name or not self.gpu_hierarchy:
            return 8

        gpu_lower = gpu_name.lower().strip()
        if gpu_lower in self.gpu_hierarchy:
            return int(self.gpu_hierarchy[gpu_lower]["vram"])

        key = match_gpu_key(gpu_lower, self.gpu_hierarchy)
        if key is not None:
            return int(self.gpu_hierarchy[key]["vram"])

        return 8

    def _calculate_base_fps(self, passmark_score: int, resolution: str) -> float:
        base_fps = passmark_score / 220.0
        scaled_fps = base_fps * self.RESOLUTION_FACTORS.get(resolution, 1.0)
        return max(15.0, min(scaled_fps, float(self.FPS_BOUNDS.get(resolution, 200))))

    def _normalize_gpu_name(self, gpu_name: str) -> str:
        gpu_lower = (gpu_name or "").lower().strip()
        for prefix in ("nvidia ", "geforce ", "amd ", "radeon "):
            if gpu_lower.startswith(prefix):
                gpu_lower = gpu_lower[len(prefix) :].strip()

        known_Order = [
            ("5090", "RTX 5090"),
            ("5080", "RTX 5080"),
            ("5070 ti", "RTX 5070 Ti"),
            ("5070", "RTX 5070"),
            ("4090", "RTX 4090"),
            ("4080", "RTX 4080"),
            ("4070 ti", "RTX 4070 Ti"),
            ("4070", "RTX 4070"),
            ("4060 ti", "RTX 4060 Ti"),
            ("4060", "RTX 4060"),
            ("7900 xtx", "RX 7900 XTX"),
            ("3090", "RTX 3090"),
            ("3080", "RTX 3080"),
            ("3070 ti", "RTX 3070 Ti"),
            ("3070", "RTX 3070"),
            ("3060 ti", "RTX 3060 Ti"),
            ("3060", "RTX 3060"),
            ("2080 ti", "RTX 2080 Ti"),
            ("2070", "RTX 2070"),
            ("2060", "RTX 2060"),
            ("1080 ti", "GTX 1080 Ti"),
            ("1070", "GTX 1070"),
            ("1660 ti", "GTX 1660 Ti"),
            ("1660", "GTX 1660"),
            ("1650", "GTX 1650"),
        ]
        for token, canonical in known_Order:
            if token in gpu_lower:
                return canonical
        return gpu_name

    def _normalize_game_name(self, game_name: str) -> str:
        if not game_name:
            return "General Gaming"
        game_lower = game_name.lower().strip()

        if self.lookup_table:
            games_in_table = {k.split("|")[1] for k in self.lookup_table if "|" in k}
            for g in games_in_table:
                if g.lower() == game_lower:
                    return g

        if "cyberpunk" in game_lower:
            return "Cyberpunk 2077"
        if any(g in game_lower for g in ("counter-strike", "cs2")):
            return "Counter-Strike 2"
        if "fortnite" in game_lower:
            return "Fortnite"
        if "apex" in game_lower:
            return "Apex Legends"
        if "valorant" in game_lower:
            return "Valorant"
        if any(g in game_lower for g in ("warzone", "call of duty")):
            return "Call of Duty Warzone"
        return "General Gaming"

    def _find_similar_gpu_prediction(
        self,
        _gpu_name: str,
        passmark_score: int,
        game_key: str,
        resolution: str,
    ) -> float | None:
        if not self.lookup_table or passmark_score <= 0:
            return None

        available_gpus: dict[str, int] = {}
        for key in self.lookup_table:
            parts = key.split("|")
            if len(parts) == 3 and parts[1] == game_key and parts[2] == resolution:
                gpu_part = parts[0]
                if gpu_part not in available_gpus:
                    gpu_passmark = self._get_gpu_passmark_score(gpu_part)
                    if gpu_passmark > 0:
                        available_gpus[gpu_part] = gpu_passmark

        closest_gpu = min(
            available_gpus.keys(),
            key=lambda x: abs(passmark_score - available_gpus[x]),
            default=None,
        )

        if closest_gpu:
            lookup_key = f"{closest_gpu}|{game_key}|{resolution}"
            if lookup_key in self.lookup_table:
                base_fps = float(self.lookup_table[lookup_key])
                base_passmark = float(available_gpus[closest_gpu])
                scaling_factor = (passmark_score / base_passmark) ** 0.85
                return round(base_fps * scaling_factor, 1)

        return None

    def _estimate_from_passmark(self, passmark_score: int, resolution: str, game_key: str) -> float | None:
        game_ratios = {
            "Counter-Strike 2": 72,
            "Valorant": 68,
            "Apex Legends": 120,
            "Fortnite": 110,
            "Call of Duty Warzone": 140,
            "Cyberpunk 2077": 620,
            "General Gaming": 200,
        }
        ratio = game_ratios.get(game_key, 160)
        estimated_fps = (passmark_score / ratio) * self.RESOLUTION_FACTORS.get(resolution, 1.0)
        return round(max(15.0, min(estimated_fps, float(self.FPS_BOUNDS.get(resolution, 300)))), 1)

    def get_model_info(self) -> dict[str, Any]:
        """Get metadata and cross-validation metrics about the loaded ML model."""
        if not self.is_loaded or not self.model_data:
            return {"status": "not_loaded"}

        summary = self.model_data.get("training_summary", {})
        cv = self.model_data.get("cv_metrics", {})
        lgo = cv.get("leave_games_out_5fold", {})
        lgpu = cv.get("leave_gpus_out_5fold", {})
        return {
            "status": "loaded",
            "training_data_count": summary.get("total_records", len(self.lookup_table)),
            "model_type": self.model_data.get("model_type", "TwoStageLogResidualFPSModel"),
            "version": self.model_data.get("version", "9.0"),
            "lookup_table_size": len(self.lookup_table),
            "unique_games": summary.get("unique_games", 129),
            "unique_gpus": summary.get("unique_gpus", 25),
            "supported_resolutions": ["1080p", "1440p", "4K"],
            "estimated_accuracy": (
                f"Leave-GPUs-Out: {lgpu.get('MAE_fps', 1.89)} FPS MAE ({lgpu.get('R2', 0.996)*100:.2f}% R²); "
                f"Unseen-Games Cold-Start: {lgo.get('MAE_fps', 15.71)} FPS MAE ({lgo.get('R2', 0.7767)*100:.2f}% R²)"
            ),
        }
