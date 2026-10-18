"""
Steam frame-rate lookup for CanRun.

The read API is IGamePerformanceStatsService/GetGameFrameRateStats. It is not in
Valve's public Web API reference, and it returns HTTP 401 to a valid outside key,
a fake key and no key alike (October 2026). Until Valve opens it, every Steam
player path below falls through to the benchmark curve or the store-spec model.
The published protobuf (June 2026) returns one bucket per hardware cluster:

    clusterid, report_count, report_days, mean_frame_rate,
    mean_frame_rate_stddev, frame_rate_histogram

Hardware names are not in that protobuf. The parser accepts them when the live
JSON includes gpu/cpu fields, and otherwise uses canrun/data/steam_hardware_clusters.json.
Only the Steam app id is sent. Matching stays local.
"""

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

STATS_URL = "https://api.steampowered.com/IGamePerformanceStatsService/GetGameFrameRateStats/v1/"
MIN_REPORTS = 10
STEAM_MIN_RELATIVE_SPREAD = 0.08
# Range half-widths that held about 90% of held-out rows in
# data/training_benchmarks.json, one per kind of estimate.
CURVE_RELATIVE_SPREAD = {"1080p": 0.20, "1440p": 0.20, "4K": 0.25}
SCALED_RELATIVE_SPREAD = 0.30
# Store-spec estimates come from the v9 model alone when it is loaded; the
# spec heuristic is only the fallback. On 182 measured TechPowerUp results for
# unseen games the model's median error was about 15% (heuristic about 19%),
# so the shown range stays far wider than the model's synthetic ±4.5–6% band.
REQUIREMENTS_RELATIVE_SPREAD = {"1080p": 0.20, "1440p": 0.22, "4K": 0.25}
FPS_FLOOR = 1
FPS_CEILING = 1000
PASSMARK_BAND = (0.5, 1.5)

SOURCE_STEAM = "steam"
SOURCE_SIMILAR = "steam_similar"
# Another GPU's Steam players, scaled by PassMark. Their settings are mixed.
SOURCE_SCALED = "steam_scaled"
SOURCE_REQUIREMENTS = "requirements"
SOURCE_REQUIREMENTS_MINIMUM = "requirements_minimum"
SOURCE_REQUIREMENTS_UNSPECIFIED = "requirements_unspecified"
# Requirement estimates stop at a high-refresh ceiling. A 1000 FPS figure
# would only mean "this PC is far past a weak spec", not a measured rate.
_REQUIREMENTS_FPS_CAP = 240
# Cards Steam pages name that are not in the GPU hierarchy. Scores are
# PassMark G3D averages, the same scale as that file.
_EXTRA_GPU_PASSMARKS = {
    "gtx 1060": 10016,
    "gtx 750 ti": 3900,
    "gtx 750": 3281,
    "gtx 660": 4043,
    "gtx 560": 2775,
    "gts 450": 1353,
    "geforce 7600 gs": 400,
    "rx 580": 8781,
    "rx 480": 8525,
    "rx 460": 4096,
    "rx 550": 2702,
    "r7 370": 4483,
    "hd 5750": 1174,
    "mx150": 2246,
    "vega 8": 2200,
    "iris xe": 2800,
    "uhd graphics 630": 1400,
    "uhd graphics 620": 1100,
    "uhd graphics": 1400,
    "hd graphics 5000": 700,
    "hd graphics 4000": 500,
    "hd graphics 3000": 400,
    "hd graphics 2500": 350,
    "hd graphics": 800,
}
# A recommended spec is treated as 60 FPS at 1080p. A minimum spec is 30.
_REQUIREMENTS_TARGET_FPS = 60
_MINIMUM_TARGET_FPS = 30
_RESOLUTION_FACTOR = {"1080p": 1.0, "1440p": 0.72, "4K": 0.42, "720p": 1.4}
SOURCE_UNKNOWN = "unknown"


@dataclass
class FpsEstimate:
    """FPS figure shown to the user."""

    expected_fps: int
    fps_min: int
    fps_max: int
    fps_source: str


@dataclass
class FrameRateBucket:
    """One hardware cluster from GetGameFrameRateStats."""

    cluster_id: str
    report_count: int
    mean_fps: float
    stddev: float
    gpu_name: str = ""
    cpu_name: str = ""


def valid_steam_app_id(appid: str) -> str:
    """Steam app ids are numeric. Anything else is refused before a request."""
    text = str(appid or "").strip()
    if text.isdigit() and len(text) <= 12:
        return text
    return ""


def redact_secrets(message: str, secret: str = "") -> str:
    """Remove API keys from text that might be written to a log."""
    text = str(message)
    if secret:
        text = text.replace(secret, "[redacted]")
    return re.sub(r"(?i)(key=)[^&\s\"']+", r"\1[redacted]", text)


def unknown_estimate() -> FpsEstimate:
    return FpsEstimate(0, 0, 0, SOURCE_UNKNOWN)


def _round_fps(value: float) -> int:
    """Round half away from zero for non-negative FPS values."""
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


def _clamp_fps(value: int) -> int:
    return max(FPS_FLOOR, min(FPS_CEILING, value))


def _ordered_range(expected: int, fps_min: int, fps_max: int) -> FpsEstimate:
    if fps_min > expected:
        fps_min = expected
    if fps_max < expected:
        fps_max = expected
    return FpsEstimate(expected, fps_min, fps_max, SOURCE_STEAM)


def _spread_band(center: float, spread: float) -> FpsEstimate:
    expected = _clamp_fps(_round_fps(center))
    fps_min = _clamp_fps(_round_fps(center - spread))
    fps_max = _clamp_fps(_round_fps(center + spread))
    return _ordered_range(expected, fps_min, fps_max)


def measured_range(mean_fps: float, stddev: float) -> FpsEstimate:
    """Steam sample. Use the player spread, and never narrower than ±8%."""
    spread = max(abs(stddev), abs(mean_fps) * STEAM_MIN_RELATIVE_SPREAD)
    return _spread_band(mean_fps, spread)


def cpu_bottleneck_factor(cpu_passmark: int | None) -> float:
    """Same CPU factor the formula predictor uses for a weak CPU."""
    if cpu_passmark is None:
        return 1.0
    if cpu_passmark < 8000:
        return 0.8
    if cpu_passmark < 15000:
        return 0.9
    return 1.0


def scale_from_reference(
    mean_fps: float,
    stddev: float,  # noqa: ARG001
    user_passmark: int,
    reference_passmark: int,
    cpu_factor: float,
    spread: float = SCALED_RELATIVE_SPREAD,
) -> FpsEstimate:
    """Scale a measured game result by relative GPU strength, then CPU factor.

    Callers pass the measured stddev. The band is a relative spread instead,
    because the PassMark step adds more error than the player spread.
    """
    if reference_passmark <= 0:
        return unknown_estimate()
    multiplier = ((user_passmark / reference_passmark) ** 0.85) * cpu_factor
    scaled = mean_fps * multiplier
    estimate = _spread_band(scaled, abs(scaled) * spread)
    estimate.fps_source = SOURCE_SCALED
    return estimate


def _data_file(filename: str) -> Path:
    here = Path(__file__).parent
    candidates = [
        here / "data" / filename,
        here.parent / "data" / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / "canrun" / "data" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return here / "data" / filename


def _env_files() -> list[Path]:
    here = Path(__file__).parent
    return [
        here / ".env",
        here.parent / ".env",
        Path.cwd() / ".env",
        Path.cwd() / "canrun" / ".env",
    ]


def load_dotenv_vars(env_paths: list[Path] | None = None) -> dict[str, str]:
    """Parse KEY=VALUE pairs from .env files without third-party dependencies."""
    paths = _env_files() if env_paths is None else env_paths
    parsed: dict[str, str] = {}
    for path in paths:
        try:
            if not path.is_file():
                continue
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                if key and key not in parsed:
                    parsed[key] = value
        except OSError:
            continue
    return parsed


def load_steam_webapi_key() -> str:
    """Developer key from STEAM_WEBAPI_KEY env var or local .env file."""
    env_key = os.environ.get("STEAM_WEBAPI_KEY", "").strip()
    if env_key:
        return env_key
    dotenv_vars = load_dotenv_vars()
    dotenv_key = dotenv_vars.get("STEAM_WEBAPI_KEY", "").strip()
    if dotenv_key:
        return dotenv_key
    return ""


@lru_cache(maxsize=1)
def load_cluster_map() -> dict[str, Any]:
    path = _data_file("steam_hardware_clusters.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("clusters", {})
            data.setdefault("reference_gpu", "RTX 5070")
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read Steam hardware cluster map: %s", exc)
    return {"clusters": {}, "reference_gpu": "RTX 5070"}


@lru_cache(maxsize=1)
def load_gpu_passmarks() -> dict[str, int]:
    """Normalized GPU name -> PassMark G3D from gpu_hierarchy.json."""
    path = _data_file("gpu_hierarchy.json")
    passmarks: dict[str, int] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read GPU hierarchy: %s", exc)
        return passmarks
    for section in ("nvidia", "amd", "intel"):
        for name, info in data.get(section, {}).items():
            if isinstance(info, dict) and "passmark_g3d" in info:
                passmarks[normalize_gpu(name)] = int(info["passmark_g3d"])
    return passmarks


@lru_cache(maxsize=1)
def load_cpu_patterns() -> list[tuple]:
    path = _data_file("cpu_benchmarks.json")
    patterns: list[tuple] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read CPU benchmarks: %s", exc)
        return patterns
    for mapping in data.get("cpu_patterns", {}).values():
        if not isinstance(mapping, dict):
            continue
        for pattern, score in mapping.items():
            patterns.append((pattern, int(score)))
    return patterns


def normalize_gpu(name: str) -> str:
    """Fold vendor prefixes so 'NVIDIA GeForce RTX 4070' matches 'RTX 4070'."""
    text = (name or "").lower().replace("(r)", " ").replace("(tm)", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    while tokens and tokens[0] in {"nvidia", "geforce", "amd", "radeon", "intel"}:
        tokens.pop(0)
    return " ".join(tokens)


def normalize_cpu(name: str) -> str:
    text = (name or "").lower().replace("(r)", " ").replace("(tm)", " ")
    text = re.sub(r"\d+\s*-?\s*core", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    drop = {"amd", "intel", "core", "processor", "cpu"}
    return " ".join(token for token in text.split() if token not in drop)


def gpu_vendor(name: str) -> str:
    text = f" {normalize_gpu(name)} "
    if " radeon " in text or text.strip().startswith("rx "):
        return "amd"
    if text.strip().startswith("arc ") or " arc " in text:
        return "intel"
    if any(token in text for token in (" rtx ", " gtx ", " geforce ")):
        return "nvidia"
    return "unknown"


def lookup_gpu_passmark(name: str, passmarks: dict[str, int]) -> int | None:
    """Exact name first, then without a memory size or a trailing 'graphics'.

    'RTX 5060 Ti 8GB' keeps its own entry, while 'RX 7600 XT 16GB' and
    'Intel Arc B580 Graphics' fall back to the base card.
    """
    key = normalize_gpu(name)
    if key in passmarks:
        return passmarks[key]
    base = re.sub(r"\s\d+\s?gb\b", "", key)
    base = re.sub(r"\sgraphics$", "", base).strip()
    return passmarks.get(base)


def requirement_passmarks(gpu_passmarks: dict[str, int] | None = None) -> dict[str, int]:
    """Hierarchy scores plus the extra cards that show up in Steam requirement lines."""
    combined = dict(load_gpu_passmarks() if gpu_passmarks is None else gpu_passmarks)
    for name, score in _EXTRA_GPU_PASSMARKS.items():
        combined.setdefault(name, score)
    return combined


def _requirement_haystack(text: str) -> str:
    """One lowercase line, with 'GTX 3060' and '1660Ti' written like the table.

    The Cyrillic abbreviation for gigabytes is rewritten to ``gb`` before
    non-ASCII characters are removed. Those two letters are not a Latin typo.
    """
    raw = (text or "").lower().replace("гб", "gb").replace("®", " ")  # noqa: RUF001
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    raw = re.sub(r"(\d)\s*gb\b", r"\1gb", raw)
    raw = re.sub(r"(\d{3,4})\s*(ti|super)\b", r"\1 \2", raw)
    raw = re.sub(r"\bgtx\s+((?:20|30|40|50)\d{2}(?:\s+ti|\s+super)?)\b", r"rtx \1", raw)
    raw = re.sub(r"\b(?:nvidia|geforce)\s+((?:20|30|40|50)\d{2})(?!\d)", r"rtx \1", raw)
    raw = re.sub(r"\b(?:nvidia|geforce)\s+([6-9]\d{2}|1[06]\d{2})(?!\d)", r"gtx \1", raw)
    return f" {re.sub(r'\s+', ' ', raw).strip()} "


def first_known_gpu(text: str, passmarks: dict[str, int]) -> tuple | None:
    """GPU named in a requirements line. A longer card name beats a shorter overlap."""
    haystack = _requirement_haystack(text)
    matches: list[tuple] = []
    for name, score in passmarks.items():
        needle = f" {name} "
        index = haystack.find(needle)
        if index < 0:
            continue
        matches.append((index, index + len(needle), name, int(score)))
    if not matches:
        return _closest_family_gpu(haystack, passmarks)
    matches.sort(key=lambda row: (row[1] - row[0]), reverse=True)
    kept: list[tuple] = []
    for match in matches:
        covered = any(other[0] <= match[0] and other[1] >= match[1] for other in kept)
        if not covered:
            kept.append(match)
    kept.sort(key=lambda row: row[0])
    return kept[0][2], kept[0][3]


def _closest_family_gpu(haystack: str, passmarks: dict[str, int]) -> tuple | None:
    """Map a near-miss model, such as GTX 1090 or RTX 8090, onto the closest real card."""
    families: dict[str, list[tuple]] = {}
    for name, score in passmarks.items():
        match = re.match(r"(rtx|gtx|rx)\s+(\d{3,4})\b", name)
        if not match:
            continue
        families.setdefault(match.group(1), []).append((int(match.group(2)), name, int(score)))
    found = re.findall(r"\b(rtx|gtx|rx)\s+(\d{3,4})\b", haystack)
    best: tuple | None = None
    for family, number_text in found:
        number = int(number_text)
        options = families.get(family) or []
        if not options:
            continue
        if number > max(row[0] for row in options):
            chosen = max(options, key=lambda row: row[0])
        else:
            chosen = min(options, key=lambda row: (abs(row[0] - number), -row[0]))
            if abs(chosen[0] - number) > 150:
                continue
        rank = abs(chosen[0] - number)
        if best is None or rank < best[0]:
            best = (rank, chosen[1], chosen[2])
    if best is None:
        return None
    return best[1], best[2]


@lru_cache(maxsize=1)
def _load_two_stage_ml_predictor():
    try:
        from canrun_model_loader import LightweightMLPredictor

        predictor = LightweightMLPredictor()
        if predictor.is_loaded and predictor.has_two_stage_trees:
            return predictor
    except Exception:
        pass
    return None


def estimate_from_requirements(
    user_gpu: str,
    user_cpu: str,
    recommended_gpu: str = "",
    minimum_gpu: str = "",
    resolution: str | None = None,
    gpu_passmarks: dict[str, int] | None = None,
    cpu_patterns: list[tuple] | None = None,
    game_name: str = "",
    user_vram_gb: float = 0.0,
    recommended_cpu: str = "",
    recommended_ram_gb: float = 0.0,
    release_year: int = 0,
    genre: str = "",
    steam_app_id: str = "",
) -> FpsEstimate:
    """FPS for a game with no player sample, scaled from its published GPU spec.

    Uses the Two-Stage Log-Ridge + Monotonic GBDT model when it is loaded and the
    page names a GPU, times the engine factor for the game's engine. Otherwise
    falls back to the power-law prior (recommended card = 60 FPS at 1080p,
    minimum card = 30 FPS). Missing store fields get the defaults training used.
    """
    if gpu_passmarks is None:
        gpu_passmarks = load_gpu_passmarks()
    if cpu_patterns is None:
        cpu_patterns = load_cpu_patterns()
    cards = requirement_passmarks(gpu_passmarks)
    user_passmark = lookup_gpu_passmark(user_gpu, cards)
    if not user_passmark:
        return unknown_estimate()
    spec = first_known_gpu(recommended_gpu, cards)
    min_spec = first_known_gpu(minimum_gpu, cards)
    target = _REQUIREMENTS_TARGET_FPS
    source = SOURCE_REQUIREMENTS
    if spec is None:
        spec = min_spec
        target = _MINIMUM_TARGET_FPS
        source = SOURCE_REQUIREMENTS_MINIMUM
    if spec is None:
        modest = lookup_gpu_passmark("GTX 1050 Ti", cards)
        if not modest:
            return unknown_estimate()
        spec = ("gtx 1050 ti", modest)
        target = _REQUIREMENTS_TARGET_FPS
        source = SOURCE_REQUIREMENTS_UNSPECIFIED
    if spec[1] <= 0:
        return unknown_estimate()
    resolution_key = _normalize_resolution(resolution) or "1080p"
    anchor = target * _RESOLUTION_FACTOR.get(resolution_key, 1.0)
    spread = REQUIREMENTS_RELATIVE_SPREAD.get(resolution_key, REQUIREMENTS_RELATIVE_SPREAD["1080p"])
    cpu_pm = lookup_cpu_passmark(user_cpu, cpu_patterns)
    cpu_factor = cpu_bottleneck_factor(cpu_pm)
    estimate = scale_from_reference(anchor, 0, user_passmark, spec[1], cpu_factor, spread)

    predictor = _load_two_stage_ml_predictor()
    if predictor is not None and source in {SOURCE_REQUIREMENTS, SOURCE_REQUIREMENTS_MINIMUM}:
        rec_pm = float(spec[1]) if source == SOURCE_REQUIREMENTS else float(spec[1]) * 1.65
        min_pm = float(min_spec[1]) if min_spec else (rec_pm / 1.65)
        min_vram = 4 if min_pm < 10000 else (6 if min_pm < 17000 else 8)
        rec_vram = max(min_vram, 6 if rec_pm < 14500 else 8)
        rec_cpu_pm = lookup_cpu_passmark(recommended_cpu, cpu_patterns) if recommended_cpu else None
        x_vec = predictor.extract_feature_vector(
            gpu_passmark=float(user_passmark),
            cpu_passmark=float(cpu_pm or 28000),
            gpu_vram_gb=float(user_vram_gb or 8.0),
            gpu_name=normalize_gpu(user_gpu),
            resolution=resolution_key,
            rec_gpu_passmark=rec_pm,
            min_gpu_passmark=min_pm,
            rec_cpu_passmark=float(rec_cpu_pm or 16000),
            rec_vram_gb=float(rec_vram),
            rec_ram_gb=float(recommended_ram_gb or 16),
            release_year=int(release_year or 2023),
            genre=str(genre or ""),
            game_name=str(game_name or ""),
        )
        tree_fps = predictor.predict_two_stage_from_features(x_vec)
        if tree_fps is not None and tree_fps > 0:
            tree_fps = float(tree_fps) * engine_factor(steam_app_id)
            estimate = _spread_band(tree_fps, tree_fps * spread)

    if estimate.expected_fps > _REQUIREMENTS_FPS_CAP:
        estimate = _spread_band(_REQUIREMENTS_FPS_CAP, _REQUIREMENTS_FPS_CAP * spread)
    estimate.fps_source = source
    return estimate


def lookup_cpu_passmark(name: str, patterns: list[tuple]) -> int | None:
    if not name or not name.strip() or len(name) > 200:
        return None
    for pattern, score in patterns:
        compiled = _cpu_regex(pattern)
        if compiled is not None and compiled.search(name):
            return score
    return None


@lru_cache(maxsize=512)
def _cpu_regex(pattern: str):
    if not isinstance(pattern, str) or len(pattern) > 80:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        logger.warning("Skipping invalid CPU benchmark pattern")
        return None


def _as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def parse_frame_rate_payload(payload: dict[str, Any] | None) -> list[FrameRateBucket]:
    """Parse GetGameFrameRateStats JSON, including optional hardware names."""
    if not isinstance(payload, dict):
        return []
    body = payload.get("response", payload)
    if not isinstance(body, dict):
        return []
    rows = body.get("frame_rates")
    if rows is None:
        rows = body.get("frameRates")
    if not isinstance(rows, list):
        return []

    buckets: list[FrameRateBucket] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mean = _first(row, "mean_frame_rate", "meanFrameRate")
        try:
            mean_fps = float(mean)
        except (TypeError, ValueError):
            continue
        gpu_name = _first(row, "gpu_name", "gpu", "adapter_description", "adapterDescription") or ""
        cpu_name = _first(row, "cpu_name", "cpu") or ""
        cluster_id = _first(row, "clusterid", "cluster_id", "clusterId") or ""
        buckets.append(
            FrameRateBucket(
                cluster_id=str(cluster_id),
                report_count=_as_int(_first(row, "report_count", "reportCount")),
                mean_fps=mean_fps,
                stddev=_as_float(_first(row, "mean_frame_rate_stddev", "meanFrameRateStddev")),
                gpu_name=str(gpu_name),
                cpu_name=str(cpu_name),
            )
        )
    return buckets


def apply_cluster_labels(buckets: list[FrameRateBucket], cluster_map: dict[str, Any]) -> None:
    clusters = cluster_map.get("clusters", {}) if isinstance(cluster_map, dict) else {}
    if not isinstance(clusters, dict):
        return
    for bucket in buckets:
        if bucket.gpu_name or not bucket.cluster_id:
            continue
        info = clusters.get(bucket.cluster_id)
        if isinstance(info, dict):
            bucket.gpu_name = str(info.get("gpu", "") or "")
            bucket.cpu_name = str(info.get("cpu", "") or "")


def _gpus_match(left: str, right: str) -> bool:
    normalized = normalize_gpu(left)
    return bool(normalized) and normalized == normalize_gpu(right)


def _cpus_match(bucket_cpu: str, user_cpu: str) -> bool:
    if not bucket_cpu.strip():
        return True
    if not user_cpu.strip():
        return False
    return normalize_cpu(bucket_cpu) == normalize_cpu(user_cpu)


def _in_band(user_passmark: int, reference_passmark: int) -> bool:
    low, high = PASSMARK_BAND
    return low * user_passmark <= reference_passmark <= high * user_passmark


def _passmark_distance(user_passmark: int, reference_passmark: int) -> int:
    return abs(user_passmark - reference_passmark)


def steam_client_roots() -> list[Path]:
    """Steam client directories on this PC."""
    roots: list[Path] = []
    program_files = os.environ.get("PROGRAMFILES(X86)")
    if program_files:
        roots.append(Path(program_files) / "Steam")
    local = Path.home() / "AppData" / "Local" / "Steam"
    if local not in roots:
        roots.append(local)
    return roots


def read_local_cluster_id() -> str | None:
    """Read this PC's hardware cluster id from the Steam client config, if present."""
    candidates: list[Path] = []
    for root in steam_client_roots():
        candidates.append(root / "config" / "config.vdf")
        userdata = root / "userdata"
        if not userdata.is_dir():
            continue
        try:
            for child in userdata.iterdir():
                candidates.append(child / "config" / "localconfig.vdf")
                if len(candidates) > 24:
                    break
        except OSError:
            continue

    pattern = re.compile(r'hardware_cluster_id["\s:=]+"?(\d+)', re.IGNORECASE)
    for path in candidates:
        try:
            if not path.is_file() or path.stat().st_size > 5_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def fit_log_fps_curve(points: list[tuple]) -> tuple | None:
    """Fit log(fps) = a + b * log(passmark). Needs two positive rows."""
    usable = [(float(passmark), float(fps)) for passmark, fps in points if passmark and fps and float(passmark) > 0 and float(fps) > 0]
    count = len(usable)
    if count < 2:
        return None
    sum_x = sum_y = sum_xx = sum_xy = 0.0
    for passmark, fps in usable:
        log_passmark = math.log(passmark)
        log_fps = math.log(fps)
        sum_x += log_passmark
        sum_y += log_fps
        sum_xx += log_passmark * log_passmark
        sum_xy += log_passmark * log_fps
    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0:
        return None
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return intercept, slope


def load_benchmark_table() -> dict[tuple, list[tuple]]:
    """Benchmark-table FPS rows keyed by (game, resolution)."""
    return _table_from_file(_data_file("training_benchmarks.json"))


def load_measured_table() -> dict[tuple, list[tuple]]:
    """Review measurements keyed by (game, resolution). Checked before the benchmark table."""
    return _table_from_file(_data_file("measured_benchmarks.json"))


def game_engine(steam_app_id: str) -> str:
    """Engine family of a Steam game, or "" when unknown."""
    return _engine_data(*_file_stamp(_data_file("game_engines.json")))[1].get(str(steam_app_id or ""), "")


def engine_factor(steam_app_id: str) -> float:
    """Measured-vs-model FPS ratio for the game's engine; 1.0 for unknown or uncorrected engines."""
    factors, apps = _engine_data(*_file_stamp(_data_file("game_engines.json")))
    factor = factors.get(apps.get(str(steam_app_id or ""), ""), 1.0)
    return factor if 0.5 <= factor <= 2.0 else 1.0


def _file_stamp(path: Path) -> tuple[str, int]:
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return str(path), 0


@lru_cache(maxsize=2)
def _engine_data(path_text: str, _stamp: int) -> tuple[dict[str, float], dict[str, str]]:
    try:
        data = json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    factors = data.get("factors") if isinstance(data, dict) else None
    apps = data.get("apps") if isinstance(data, dict) else None
    if not isinstance(factors, dict) or not isinstance(apps, dict):
        return {}, {}
    return (
        {str(k): float(v) for k, v in factors.items() if isinstance(v, (int, float))},
        {str(k): str(v) for k, v in apps.items() if isinstance(v, str)},
    )


def _table_from_file(path: Path) -> dict[tuple, list[tuple]]:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return {}
    return _parse_table_file(str(path), stamp)


@lru_cache(maxsize=4)
def _parse_table_file(path_text: str, _stamp: int) -> dict[tuple, list[tuple]]:
    path = Path(path_text)
    table: dict[tuple, list[tuple]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return table
    games = data.get("games", {})
    if not isinstance(games, dict):
        return table
    for game, info in games.items():
        gpu_rows = info.get("gpus", {}) if isinstance(info, dict) else {}
        if not isinstance(gpu_rows, dict):
            continue
        for resolution in ("1080p", "1440p", "4K"):
            points = []
            for gpu, spec in gpu_rows.items():
                if not isinstance(spec, dict) or resolution not in spec:
                    continue
                passmark = spec.get("passmark")
                fps = spec.get(resolution)
                if not passmark or not fps:
                    continue
                points.append((str(gpu), float(passmark), float(fps)))
            if points:
                table[(str(game), resolution)] = points
    return table


def _normalize_resolution(resolution: str | None) -> str | None:
    text = re.sub(r"[^a-z0-9]+", "", (resolution or "").lower())
    if text in {"4k", "2160p", "3840x2160"}:
        return "4K"
    if text in {"1440p", "2560x1440", "qhd"}:
        return "1440p"
    if text in {"1080p", "1920x1080", "fhd"}:
        return "1080p"
    return None


_NATIVE_ESTIMATE_SOURCES = {
    SOURCE_SIMILAR,
    SOURCE_REQUIREMENTS,
    SOURCE_REQUIREMENTS_MINIMUM,
    SOURCE_REQUIREMENTS_UNSPECIFIED,
}
# NVIDIA's labels are the ceiling: frame generation is up to 2×, multi frame
# generation up to 4×, and up to 6× in select games.
# https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-generation-6x-mode-released/
# Generating frames costs GPU time, so the measured gain is below the label:
# about 0.8 of the base rate per generated frame (Tom's Hardware, Alan Wake 2:
# 2× gave 1.84× to 1.87×, 3× 2.66× to 2.71×, 4× 3.44× to 3.49×). 6× is that
# rate extended, not a measurement.
_FRAME_GEN_EXTRA = 0.8
_MULTI_FRAME_GEN_EXTRA = {"3x": 1.65, "4x": 2.45, "6x": 4.0}
_FRAME_GEN_NOTE = "usually 1.8x"
_MULTI_FRAME_GEN_NOTES = {
    "4x": "usually 2.7-3.5x",
    "6x": "usually 3.5-5x",
}
# Path tracing replaces raster lighting. Games that ship it (Cyberpunk
# Overdrive, Alan Wake 2, Black Myth) typically cost about 2× to 3× the
# raster rate with upscaling off. NVIDIA does not publish one multiplier.
_PATH_TRACE_COST = (2.0, 3.0)


def nvidia_rtx_generation(gpu_name: str) -> str:
    """Return 20, 30, 40, or 50 from an RTX model name. Empty when it is not one."""
    match = re.search(r"\brtx\s*([2345])\d{3}\b", (gpu_name or "").lower())
    if not match:
        return ""
    return {"2": "20", "3": "30", "4": "40", "5": "50"}[match.group(1)]


def dlss_label(gpu_name: str) -> str:
    """Newest practical Super Resolution on this GPU.

    DLSS 4.5 runs on every RTX card. RTX 20 and 30 lack FP8, so NVIDIA
    recommends Model K (DLSS 4) there. RTX 40 and 50 run DLSS 4.5.
    """
    generation = nvidia_rtx_generation(gpu_name)
    if generation in {"40", "50"}:
        return "DLSS 4.5"
    if generation in {"20", "30"}:
        return "DLSS 4"
    return "DLSS"


def _about_fps(value: float) -> int:
    """Nearest 5 FPS, so a side note does not look more precise than the sample."""
    if value <= 0:
        return 0
    return int(value / 5 + 0.5) * 5


def graphics_feature_lines(
    fps_source: str,
    expected_fps: int,
    fps_min: int,
    fps_max: int,
    resolution: str | None,
    gpu_name: str,
    vram_gb: int = 0,
    supports_dlss: bool = False,
    supports_rtx: bool = False,
    game_name: str | None = None,
    refresh_hz: int = 0,
    rtx_features: dict | None = None,
) -> list[str]:
    """Side notes for DLSS and path tracing. The main FPS figure stays native.

    Steam player samples, and estimates scaled from them, already mix
    settings, so they get no note. The DLSS extras use measured gains below
    NVIDIA's ceilings: frame generation is up to 2× (usually about 1.8×),
    multi frame generation up to 4× or 6×, and Dynamic mode aims at this
    screen's refresh. Path tracing is a typical 2× to 3×
    cost on an RTX card, not a measurement of this game.

    rtx_features is this game's row in NVIDIA's RTX games list. With it, only
    the features the game ships are shown; None keeps the generic notes.
    """
    if rtx_features is not None:
        return _listed_feature_lines(
            fps_source, expected_fps, fps_min, fps_max, resolution, gpu_name,
            supports_dlss, supports_rtx, refresh_hz, rtx_features,
        )
    del vram_gb, game_name
    if fps_source not in _NATIVE_ESTIMATE_SOURCES or not supports_dlss:
        return []
    native = float(expected_fps or 0)
    if native <= 0 and fps_min and fps_max:
        native = (float(fps_min) + float(fps_max)) / 2
    if native <= 0:
        return []
    if fps_source != SOURCE_SIMILAR and native >= _REQUIREMENTS_FPS_CAP:
        return []

    version = dlss_label(gpu_name)
    lines = [f"{version}: {_dlss_modes(resolution, dlaa=False)}"]
    generation = nvidia_rtx_generation(gpu_name)
    if generation == "40":
        lines.extend(_frame_gen_lines(native, ""))
    elif generation == "50":
        lines.extend(_frame_gen_lines(native, "6x"))
        if refresh_hz and native < refresh_hz:
            lines.append(f"Dynamic mode: targets this {int(refresh_hz)} Hz screen")
    if supports_rtx:
        path_line = _path_tracing_line(native)
        if path_line:
            lines.append(path_line)
    return lines


def _dlss_modes(resolution: str | None, dlaa: bool) -> str:
    """DLSS modes in one short line. Ultra Performance is offered at 4K."""
    lowest = "Ultra Performance" if _normalize_resolution(resolution) == "4K" else "Performance"
    return ("DLAA, " if dlaa else "") + f"Quality to {lowest}"


def _frame_gen_lines(native: float, top: str) -> list[str]:
    """Measured frame generation gain. top is "" for 2×, "4x" or "6x" for multi frame generation."""
    if not top:
        extra = _about_fps(native * _FRAME_GEN_EXTRA)
        return [f"Frame gen 2x: +{extra} FPS ({_FRAME_GEN_NOTE})"] if extra >= 5 else []
    low_mode = "3x" if top == "4x" else "4x"
    low = _about_fps(native * _MULTI_FRAME_GEN_EXTRA[low_mode])
    high = _about_fps(native * _MULTI_FRAME_GEN_EXTRA[top])
    if high < 5:
        return []
    amount = f"{high}" if low >= high else f"{low}-{high}"
    return [f"Frame gen {top}: +{amount} FPS ({_MULTI_FRAME_GEN_NOTES[top]})"]


def _native_for_notes(fps_source: str, expected_fps: int, fps_min: int, fps_max: int) -> float:
    """The native FPS the side notes scale from, or 0 when this source gets no notes."""
    if fps_source not in _NATIVE_ESTIMATE_SOURCES:
        return 0.0
    native = float(expected_fps or 0)
    if native <= 0 and fps_min and fps_max:
        native = (float(fps_min) + float(fps_max)) / 2
    if native <= 0 or (fps_source != SOURCE_SIMILAR and native >= _REQUIREMENTS_FPS_CAP):
        return 0.0
    return native


def _listed_feature_lines(
    fps_source: str,
    expected_fps: int,
    fps_min: int,
    fps_max: int,
    resolution: str | None,
    gpu_name: str,
    supports_dlss: bool,
    supports_rtx: bool,
    refresh_hz: int,
    features: dict,
) -> list[str]:
    """DLSS, frame generation and ray tracing notes for what this game ships and this GPU runs."""
    if not features.get("listed") or not supports_dlss:
        return []
    native = _native_for_notes(fps_source, expected_fps, fps_min, fps_max)
    if native <= 0:
        return []
    version = dlss_label(gpu_name)
    generation = nvidia_rtx_generation(gpu_name)
    lines = []
    dlaa = bool(features.get("dlaa"))
    if features.get("super_resolution"):
        lines.append(f"{version}: {_dlss_modes(resolution, dlaa)}")
    elif dlaa:
        lines.append(f"{version}: DLAA")
    multi = str(features.get("multi_frame_generation") or "").upper()
    if generation == "50" and multi:
        top = "6x" if "6X" in multi else "4x"
        lines.extend(_frame_gen_lines(native, top))
        if top == "6x" and refresh_hz and native < refresh_hz:
            lines.append(f"Dynamic mode: targets this {int(refresh_hz)} Hz screen")
    elif generation in {"40", "50"} and features.get("frame_generation"):
        lines.extend(_frame_gen_lines(native, ""))
    ray_tracing = str(features.get("ray_tracing") or "").lower()
    if supports_rtx and ray_tracing == "path tracing":
        path_line = _path_tracing_line(native)
        if path_line:
            lines.append(path_line)
    elif supports_rtx and ray_tracing:
        lines.append("Ray tracing: supported in this game")
    return lines


def _path_tracing_line(native: float) -> str:
    """FPS given up when path tracing makes the frame about 2× to 3× heavier."""
    less_low = _about_fps(native - native / _PATH_TRACE_COST[0])
    less_high = _about_fps(native - native / _PATH_TRACE_COST[1])
    if less_high < 5:
        return ""
    if less_low >= less_high:
        return f"Path tracing: -{less_high} FPS (2-3x heavier)"
    return f"Path tracing: -{less_low} to -{less_high} FPS (2-3x heavier)"


def _game_tokens(name: str) -> list[str]:
    """Lowercase title words with sequel numerals folded, so Diablo IV matches diablo 4."""
    cleaned = re.sub(r"['\u2019]", "", (name or "").lower())
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    numerals = {
        "i": "1",
        "ii": "2",
        "iii": "3",
        "iv": "4",
        "v": "5",
        "vi": "6",
        "vii": "7",
        "viii": "8",
        "ix": "9",
        "x": "10",
    }
    return [numerals.get(token, token) for token in cleaned.split() if token]


def _normalize_game_key(name: str) -> str:
    return "".join(_game_tokens(name))


def _contains_title(longer: list[str], shorter: list[str]) -> bool:
    """Whole-word run of ``shorter`` inside ``longer``, not followed by a sequel number."""
    size = len(shorter)
    for start in range(len(longer) - size + 1):
        if longer[start:start + size] == shorter:
            following = longer[start + size] if start + size < len(longer) else ""
            if not following.isdigit():
                return True
    return False


_EDITION_WORDS = frozenset({
    "edition", "remastered", "remaster", "definitive", "complete", "goty", "game", "of", "the", "year",
    "directors", "cut", "enhanced", "deluxe", "ultimate", "anniversary", "standard", "gold", "premium",
})


def _only_edition_words(longer: list[str], shorter: list[str]) -> bool:
    """A typed name longer than the table title may only add edition words.

    "Prince of Persia The Lost Crown" is a different game from "Prince of Persia";
    "Cyberpunk 2077 Ultimate Edition" is the same game as "Cyberpunk 2077".
    """
    extra = list(longer)
    for token in shorter:
        if token in extra:
            extra.remove(token)
    return all(token in _EDITION_WORDS for token in extra)


def match_benchmark_game(game_name: str | None, games: list[str]) -> str | None:
    if game_name in games:
        return game_name
    query_tokens = _game_tokens(game_name or "")
    query = "".join(query_tokens)
    if not query:
        return None
    keyed = [(_game_tokens(game), game) for game in games]
    for tokens, game in keyed:
        if "".join(tokens) == query:
            return game
    if len(query) < 4:
        return None
    hits = [
        game for tokens, game in keyed
        if len("".join(tokens)) >= 4
        and (_contains_title(tokens, query_tokens)
             or (_contains_title(query_tokens, tokens) and _only_edition_words(query_tokens, tokens)))
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _benchmark_points(game_name: str | None, resolution: str | None, table: dict[tuple, list[tuple]] | None = None) -> list[tuple]:
    resolved = _normalize_resolution(resolution)
    if not game_name or not resolved:
        return []
    if table is None:
        measured = _benchmark_points(game_name, resolution, load_measured_table())
        if measured:
            return measured
        table = load_benchmark_table()
    games = sorted({game for game, _ in table})
    matched = match_benchmark_game(game_name, games)
    if not matched:
        return []
    return list(table.get((matched, resolved), []))


def leave_one_out_fps(game_name: str, resolution: str, gpu_model: str) -> float | None:
    """Curve prediction with this GPU removed from the fit."""
    points = _benchmark_points(game_name, resolution)
    target_passmark = None
    training = []
    for gpu, passmark, fps in points:
        if _gpus_match(gpu, gpu_model):
            target_passmark = passmark
        else:
            training.append((passmark, fps))
    if target_passmark is None:
        return None
    fitted = fit_log_fps_curve(training)
    if fitted is None:
        return None
    intercept, slope = fitted
    prediction = math.exp(intercept + slope * math.log(target_passmark))
    return _cap_at_tied_leaders(prediction, points)


def gpu_vendor(gpu_name: str) -> str:
    text = (gpu_name or "").lower()
    if re.search(r"\b(rtx|gtx|geforce|quadro|titan)\b", text):
        return "nvidia"
    if re.search(r"\b(rx|radeon)\b", text):
        return "amd"
    if re.search(r"\barc\b", text):
        return "intel"
    return ""


MIN_VENDOR_POINTS = 4


def same_vendor_points(points: list[tuple], gpu_model: str) -> list[tuple]:
    """Keep the user's vendor when it has enough points. AMD cards get more FPS per PassMark point."""
    vendor = gpu_vendor(gpu_model)
    kept = [row for row in points if gpu_vendor(row[0]) == vendor] if vendor else []
    return kept if len(kept) >= MIN_VENDOR_POINTS else points


def _cap_at_tied_leaders(prediction: float, points: list[tuple]) -> float:
    """Keep a tied top score, such as Elden Ring's 60 FPS cap, from being extrapolated."""
    ranked = sorted(points, key=lambda row: row[1], reverse=True)
    if len(ranked) < 2:
        return prediction
    if abs(ranked[0][2] - ranked[1][2]) <= 2:
        return min(prediction, float(ranked[0][2]))
    return prediction


def _estimate_from_curve(prediction: float, resolution: str | None) -> FpsEstimate:
    spread = CURVE_RELATIVE_SPREAD.get(_normalize_resolution(resolution) or "", CURVE_RELATIVE_SPREAD["1080p"])
    estimate = _spread_band(prediction, abs(prediction) * spread)
    estimate.fps_source = SOURCE_SIMILAR
    return estimate


def calibrated_similar_fps(
    game_name: str | None,
    resolution: str | None,
    gpu_model: str,
    gpu_passmarks: dict[str, int],
    *,
    exclude_gpu: str | None = None,
) -> FpsEstimate | None:
    """Shipped similar-PC estimate. exclude_gpu drops that card from the fit."""
    points = same_vendor_points(_benchmark_points(game_name, resolution), gpu_model)
    if not points:
        return None
    training = []
    for gpu, passmark, fps in points:
        if exclude_gpu and _gpus_match(gpu, exclude_gpu):
            continue
        training.append((passmark, fps))
    fitted = fit_log_fps_curve(training)
    user_passmark = lookup_gpu_passmark(gpu_model, gpu_passmarks)
    if user_passmark is None:
        for gpu, passmark, _fps in points:
            if _gpus_match(gpu, gpu_model):
                user_passmark = int(passmark)
                break
    if fitted is None or not user_passmark or user_passmark <= 0:
        return None
    intercept, slope = fitted
    prediction = math.exp(intercept + slope * math.log(user_passmark))
    return _estimate_from_curve(_cap_at_tied_leaders(prediction, points), resolution)


def resolve_fps(
    gpu_model: str,
    cpu_model: str,
    payload: dict[str, Any] | None,
    *,
    api_available: bool = True,
    local_cluster_id: str | None = None,
    cluster_map: dict[str, Any] | None = None,
    reference_gpu: str | None = None,
    gpu_passmarks: dict[str, int] | None = None,
    cpu_patterns: list[tuple] | None = None,
    game_name: str | None = None,
    resolution: str | None = None,
    exclude_gpu: str | None = None,
) -> FpsEstimate:
    """
    Pick an FPS figure for this GPU and CPU.

    An exact Steam bucket with at least 10 reports wins. Otherwise a
    per-game, per-resolution curve is used when that game was measured.
    Games without that curve keep the 0.85 scale from the nearest Steam bucket.
    """
    if gpu_passmarks is None:
        gpu_passmarks = load_gpu_passmarks()

    if not api_available:
        curve = calibrated_similar_fps(
            game_name,
            resolution,
            gpu_model,
            gpu_passmarks,
            exclude_gpu=exclude_gpu,
        )
        return curve if curve is not None else unknown_estimate()

    if cluster_map is None:
        cluster_map = load_cluster_map()
    if reference_gpu is None:
        reference_gpu = str(cluster_map.get("reference_gpu") or "RTX 5070")
    if gpu_passmarks is None:
        gpu_passmarks = load_gpu_passmarks()
    if cpu_patterns is None:
        cpu_patterns = load_cpu_patterns()

    buckets = parse_frame_rate_payload(payload)
    apply_cluster_labels(buckets, cluster_map)

    user_cluster = str(local_cluster_id) if local_cluster_id else ""
    for bucket in buckets:
        if bucket.report_count < MIN_REPORTS:
            continue
        labeled_exact = bucket.gpu_name and _gpus_match(bucket.gpu_name, gpu_model) and _cpus_match(bucket.cpu_name, cpu_model)
        cluster_exact = user_cluster and bucket.cluster_id == user_cluster and not bucket.gpu_name
        if labeled_exact or cluster_exact:
            return measured_range(bucket.mean_fps, bucket.stddev)

    curve = calibrated_similar_fps(
        game_name,
        resolution,
        gpu_model,
        gpu_passmarks,
        exclude_gpu=exclude_gpu,
    )
    if curve is not None:
        return curve

    if not buckets:
        return unknown_estimate()

    user_passmark = lookup_gpu_passmark(gpu_model, gpu_passmarks)
    cpu_factor = cpu_bottleneck_factor(lookup_cpu_passmark(cpu_model, cpu_patterns))
    similar = _closest_similar_bucket(buckets, gpu_model, user_passmark, gpu_passmarks)
    if similar is not None and user_passmark:
        reference_passmark = lookup_gpu_passmark(similar.gpu_name, gpu_passmarks)
        if reference_passmark:
            return scale_from_reference(
                similar.mean_fps,
                similar.stddev,
                user_passmark,
                reference_passmark,
                cpu_factor,
            )

    return _survey_leader_estimate(
        buckets,
        user_passmark,
        reference_gpu,
        gpu_passmarks,
        cpu_factor,
    )


def _closest_similar_bucket(
    buckets: list[FrameRateBucket],
    gpu_model: str,
    user_passmark: int | None,
    gpu_passmarks: dict[str, int],
) -> FrameRateBucket | None:
    if not user_passmark:
        return None
    candidates: list[FrameRateBucket] = []
    for bucket in buckets:
        if bucket.report_count < MIN_REPORTS or not bucket.gpu_name:
            continue
        if _gpus_match(bucket.gpu_name, gpu_model):
            continue
        reference_passmark = lookup_gpu_passmark(bucket.gpu_name, gpu_passmarks)
        if reference_passmark is None or not _in_band(user_passmark, reference_passmark):
            continue
        candidates.append(bucket)
    if not candidates:
        return None

    user_vendor = gpu_vendor(gpu_model)
    same_vendor = [bucket for bucket in candidates if gpu_vendor(bucket.gpu_name) == user_vendor]
    pool = same_vendor or candidates
    pool.sort(
        key=lambda bucket: (
            _passmark_distance(user_passmark, lookup_gpu_passmark(bucket.gpu_name, gpu_passmarks) or 0),
            -bucket.report_count,
        )
    )
    return pool[0]


def _survey_leader_estimate(
    buckets: list[FrameRateBucket],
    user_passmark: int | None,
    reference_gpu: str,
    gpu_passmarks: dict[str, int],
    cpu_factor: float,
) -> FpsEstimate:
    """Scale the busiest unlabeled bucket as if it were the survey-leading GPU."""
    unlabeled = [bucket for bucket in buckets if bucket.report_count >= MIN_REPORTS and not bucket.gpu_name]
    if not unlabeled or not user_passmark:
        return unknown_estimate()
    reference_passmark = lookup_gpu_passmark(reference_gpu, gpu_passmarks)
    if not reference_passmark:
        return unknown_estimate()
    busiest = max(unlabeled, key=lambda bucket: bucket.report_count)
    return scale_from_reference(
        busiest.mean_fps,
        busiest.stddev,
        user_passmark,
        reference_passmark,
        cpu_factor,
    )


class SteamFpsClient:
    """Fetches Steam frame-rate stats and resolves them for the local PC."""

    def __init__(self, api_key: str | None = None):
        self.api_key = load_steam_webapi_key() if api_key is None else api_key.strip()

    async def estimate(
        self,
        appid: str,
        gpu_model: str,
        cpu_model: str,
        game_name: str | None = None,
        resolution: str | None = None,
    ) -> FpsEstimate:
        app_id = valid_steam_app_id(appid)
        payload = None
        api_available = False
        if app_id and self.api_key:
            payload = await self._fetch(app_id)
            api_available = payload is not None
        elif not app_id or not self.api_key:
            logger.info("Steam FPS lookup skipped: missing app id or API key")
        return resolve_fps(
            gpu_model,
            cpu_model,
            payload,
            api_available=api_available,
            local_cluster_id=read_local_cluster_id() if api_available else None,
            game_name=game_name,
            resolution=resolution,
        )

    async def _fetch(self, appid: str) -> dict[str, Any] | None:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        params = {"key": self.api_key, "appid": appid, "format": "json"}
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(STATS_URL, params=params) as response,
            ):
                if response.status != 200:
                    logger.warning("Steam frame-rate stats returned HTTP %s", response.status)
                    return None
                return await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning(
                "Steam frame-rate stats request failed: %s",
                redact_secrets(exc, self.api_key),
            )
            return None
