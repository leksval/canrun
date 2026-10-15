"""
Leak-free, domain-grounded feature engineering for CanRun FPS prediction.

Engineers 18 compact, physically meaningful continuous features (p << n) with
zero game-name one-hot encoding so the model generalizes to unseen Steam titles.
Pure Python + math implementation so identical feature extraction runs during
offline training and keyless runtime inference.
"""

from __future__ import annotations

import math
import re
from typing import Any

FEATURE_NAMES: list[str] = [
    # Hardware features (6)
    "log_gpu_passmark",
    "log_cpu_passmark",
    "gpu_vram_gb",
    "gpu_arch_code",
    "cpu_to_gpu_log_ratio",
    "gpu_tier_score",
    # Resolution & pixel load features (3)
    "log_pixel_ratio",
    "vram_per_mpixel",
    "is_4k",
    # Game demand features from Steam Store specs (6)
    "log_rec_gpu_passmark",
    "log_min_gpu_passmark",
    "gpu_headroom_rec",
    "gpu_headroom_min",
    "vram_headroom_gb",
    "spec_demand_index",
    # Physical interaction features (3)
    "headroom_x_resolution",
    "cpu_bottleneck_1080p",
    "vram_cliff_penalty",
    # Leak-free engine-regime features (3) — no game-name / AppID / GPU-string IDs
    "is_capped_60",
    "genre_demand",
    "cpu_headroom_rec",
]

# Monotonic constraints aligned with FEATURE_NAMES:
# +1: higher feature value must never decrease predicted FPS
# -1: higher feature value must never increase predicted FPS
#  0: unconstrained interaction/auxiliary feature
MONOTONIC_CONSTRAINTS: list[int] = [
    1,   # log_gpu_passmark (+1: stronger GPU -> higher FPS)
    1,   # log_cpu_passmark (+1: stronger CPU -> higher FPS)
    1,   # gpu_vram_gb (+1: more VRAM -> higher/equal FPS)
    1,   # gpu_arch_code (+1: newer architecture -> higher/equal FPS)
    1,   # cpu_to_gpu_log_ratio (+1: combined GPU+CPU log synergy)
    1,   # gpu_tier_score (+1: strictly monotonic in GPU PassMark)
    -1,  # log_pixel_ratio (-1: more pixels -> lower FPS)
    1,   # vram_per_mpixel (+1: more VRAM per megapixel -> higher FPS)
    -1,  # is_4k (-1: 4K -> lower FPS)
    0,   # log_rec_gpu_passmark (unconstrained across game regimes)
    0,   # log_min_gpu_passmark (unconstrained across game regimes)
    0,   # gpu_headroom_rec (unconstrained across game regimes; Ridge prior carries linear headroom slope)
    0,   # gpu_headroom_min (unconstrained across game regimes; Ridge prior carries linear headroom slope)
    0,   # vram_headroom_gb (unconstrained across game regimes)
    0,   # spec_demand_index (unconstrained across game regimes)
    0,   # headroom_x_resolution (unconstrained across game regimes)
    1,   # cpu_bottleneck_1080p (+1: stronger CPU at 1080p -> higher FPS)
    0,   # vram_cliff_penalty (unconstrained across game regimes)
    -1,  # is_capped_60 (-1: engine lock lowers the FPS ceiling)
    -1,  # genre_demand (-1: heavier genre -> lower FPS)
    1,   # cpu_headroom_rec (+1: CPU above the recommended spec -> higher FPS)
]

RESOLUTION_PIXELS: dict[str, int] = {
    "720p": 1280 * 720,
    "1080p": 1920 * 1080,
    "1440p": 2560 * 1440,
    "4K": 3840 * 2160,
}

BASE_PIXELS_1080P = float(1920 * 1080)


def infer_gpu_arch_code(gpu_name: str, gen_hint: str = "") -> float:
    """Map GPU model or generation string to ordinal architecture code [1.0 .. 5.0]."""
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


def infer_capped_60(game_name: str = "", genre: str = "") -> float:
    """Detect a 60 FPS engine lock from title/genre text — never from a game ID."""
    text = f"{game_name} {genre}".lower()
    if any(marker in text for marker in CAPPED_60_TITLE_MARKERS):
        return 1.0
    if "fighting" in text and "engine lock" in text:
        return 1.0
    return 0.0


def infer_genre_demand_offset(genre: str = "") -> float:
    """Map genre string from Steam Store metadata to physical rendering load offset [-0.35 .. +0.30]."""
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
) -> list[float]:
    """Compute the 21 domain features in deterministic FEATURE_NAMES order."""
    safe_gpu_pm = max(400.0, float(gpu_passmark or 400.0))
    safe_cpu_pm = max(1500.0, float(cpu_passmark or 28000.0))
    safe_vram = max(1.0, float(gpu_vram_gb or 8.0))

    # Impute missing minimum / recommended GPU PassMark scores consistently
    if rec_gpu_passmark <= 0 and min_gpu_passmark > 0:
        safe_min_gpu_pm = max(400.0, float(min_gpu_passmark))
        safe_rec_gpu_pm = safe_min_gpu_pm * 1.65
    elif rec_gpu_passmark > 0 and min_gpu_passmark <= 0:
        safe_rec_gpu_pm = max(400.0, float(rec_gpu_passmark))
        safe_min_gpu_pm = max(400.0, safe_rec_gpu_pm / 1.65)
    elif rec_gpu_passmark <= 0 and min_gpu_passmark <= 0:
        # Unspecified game requirements default to GTX 1050 Ti / GTX 750 Ti baseline
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

    # 1-6: Hardware features (smoothly & monotonically aligned with GPU/CPU performance)
    log_gpu_passmark = math.log(safe_gpu_pm)
    log_cpu_passmark = math.log(safe_cpu_pm)
    raw_arch = infer_gpu_arch_code(gpu_name, gpu_gen)
    gpu_arch_code = 1.0 + 3.8 * (safe_gpu_pm / 40000.0) + 0.01 * raw_arch
    effective_vram = 4.0 + 20.0 * (safe_gpu_pm / 40000.0) + 0.008 * min(24.0, safe_vram)
    cpu_to_gpu_log_ratio = 0.75 * log_gpu_passmark + 0.25 * log_cpu_passmark
    gpu_tier_score = safe_gpu_pm / 40000.0

    # 7-9: Resolution & pixel load features
    log_pixel_ratio = math.log(pixels / BASE_PIXELS_1080P)
    vram_per_mpixel = effective_vram / mpixels
    is_4k = 1.0 if res_key == "4K" else 0.0

    # 10-15: Game demand features from Steam Store specs
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

    # 16-18: Physical interaction features (monotonically aligned with hardware & resolution)
    headroom_x_resolution = gpu_headroom_rec * (1.0 + 0.25 * max(0.0, log_pixel_ratio))
    cpu_bottleneck_1080p = log_cpu_passmark / (1.0 + 0.35 * max(0.0, log_pixel_ratio))
    vram_cliff_penalty = max(0.0, safe_rec_vram * (1.0 + 0.35 * is_4k) - effective_vram)
    capped_60 = 1.0 if float(is_capped_60 or 0.0) >= 0.5 else 0.0
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


def extract_features_from_record(record: dict[str, Any]) -> list[float]:
    """Extract the 18-element feature vector from a clean dataset row."""
    return extract_feature_vector(
        gpu_passmark=float(record["gpu_passmark"]),
        cpu_passmark=float(record.get("cpu_passmark", 28000.0)),
        gpu_vram_gb=float(record.get("gpu_vram_gb", 8.0)),
        gpu_name=str(record.get("gpu_model", "")),
        gpu_gen=str(record.get("gpu_gen", "")),
        resolution=str(record.get("resolution", "1080p")),
        rec_gpu_passmark=float(record.get("rec_gpu_passmark", 0.0)),
        min_gpu_passmark=float(record.get("min_gpu_passmark", 0.0)),
        rec_cpu_passmark=float(record.get("rec_cpu_passmark", 13000.0)),
        rec_vram_gb=float(record.get("rec_vram_gb", 6.0)),
        rec_ram_gb=float(record.get("rec_ram_gb", 16.0)),
        release_year=int(record.get("release_year", 2023)),
        genre=str(record.get("genre", "")),
        is_capped_60=float(record.get("is_capped_60", 0) or 0),
    )


def records_to_matrix(records: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
    """Convert dataset records into numpy arrays (X, y_log, y_fps)."""
    import numpy as np

    X = np.array([extract_features_from_record(r) for r in records], dtype=np.float64)
    y_fps = np.array([float(r["fps"]) for r in records], dtype=np.float64)
    y_log = np.array([float(r["log_fps"]) for r in records], dtype=np.float64)
    return X, y_log, y_fps

