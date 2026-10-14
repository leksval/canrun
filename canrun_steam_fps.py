"""
Steam frame-rate lookup for CanRun.

The public read API is IGamePerformanceStatsService/GetGameFrameRateStats.
An unauthenticated call returns HTTP 401, so a Steam Web API key is required.
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
from typing import Any, Dict, List, Optional

import aiohttp


logger = logging.getLogger(__name__)

STATS_URL = "https://api.steampowered.com/IGamePerformanceStatsService/GetGameFrameRateStats/v1/"
MIN_REPORTS = 30
STEAM_MIN_RELATIVE_SPREAD = 0.08
SIMILAR_RELATIVE_SPREAD = 0.20
FPS_FLOOR = 1
FPS_CEILING = 1000
PASSMARK_BAND = (0.5, 1.5)

SOURCE_STEAM = "steam"
SOURCE_SIMILAR = "steam_similar"
SOURCE_REQUIREMENTS = "requirements"
SOURCE_REQUIREMENTS_MINIMUM = "requirements_minimum"
SOURCE_REQUIREMENTS_UNSPECIFIED = "requirements_unspecified"
# Requirement estimates stop at a high-refresh ceiling. A 1000 FPS figure
# would only mean "this PC is far past a weak spec", not a measured rate.
_REQUIREMENTS_FPS_CAP = 240
# Cards Steam pages name that are not in the NVIDIA hierarchy. Scores sit on
# the same PassMark scale as the cards already in that file.
_EXTRA_GPU_PASSMARKS = {
    "gtx 1060": 10080,
    "gtx 750 ti": 3800,
    "gtx 750": 3200,
    "gtx 660": 4800,
    "gtx 560": 2200,
    "gts 450": 1800,
    "geforce 7600 gs": 400,
    "rx 6800": 25086,
    "rx 6600": 14500,
    "rx 580": 10080,
    "rx 480": 9200,
    "rx 460": 5500,
    "rx 550": 4500,
    "r7 370": 4500,
    "hd 5750": 900,
    "mx150": 2500,
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


def cpu_bottleneck_factor(cpu_passmark: Optional[int]) -> float:
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
    stddev: float,
    user_passmark: int,
    reference_passmark: int,
    cpu_factor: float,
) -> FpsEstimate:
    """Scale a measured game result by relative GPU strength, then CPU factor."""
    if reference_passmark <= 0:
        return unknown_estimate()
    multiplier = ((user_passmark / reference_passmark) ** 0.85) * cpu_factor
    scaled = mean_fps * multiplier
    estimate = _spread_band(scaled, abs(scaled) * SIMILAR_RELATIVE_SPREAD)
    estimate.fps_source = SOURCE_SIMILAR
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


def _config_files() -> List[Path]:
    here = Path(__file__).parent
    return [
        here / "config.json",
        here.parent / "config.json",
        Path.cwd() / "config.json",
        Path.cwd() / "canrun" / "config.json",
    ]


def load_steam_webapi_key() -> str:
    """Key from STEAM_WEBAPI_KEY, otherwise canrun_config.steam_webapi_key."""
    env_key = os.environ.get("STEAM_WEBAPI_KEY", "").strip()
    if env_key:
        return env_key
    for path in _config_files():
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            key = data.get("canrun_config", {}).get("steam_webapi_key", "")
            if isinstance(key, str) and key.strip():
                return key.strip()
        except (OSError, json.JSONDecodeError):
            continue
    return ""


@lru_cache(maxsize=1)
def load_cluster_map() -> Dict[str, Any]:
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
def load_gpu_passmarks() -> Dict[str, int]:
    """Normalized GPU name -> PassMark G3D from gpu_hierarchy.json."""
    path = _data_file("gpu_hierarchy.json")
    passmarks: Dict[str, int] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read GPU hierarchy: %s", exc)
        return passmarks
    for section in ("nvidia", "amd"):
        for name, info in data.get(section, {}).items():
            if isinstance(info, dict) and "passmark_g3d" in info:
                passmarks[normalize_gpu(name)] = int(info["passmark_g3d"])
    return passmarks


@lru_cache(maxsize=1)
def load_cpu_patterns() -> List[tuple]:
    path = _data_file("cpu_benchmarks.json")
    patterns: List[tuple] = []
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


def lookup_gpu_passmark(name: str, passmarks: Dict[str, int]) -> Optional[int]:
    return passmarks.get(normalize_gpu(name))


def requirement_passmarks(gpu_passmarks: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Hierarchy scores plus the extra cards that show up in Steam requirement lines."""
    combined = dict(load_gpu_passmarks() if gpu_passmarks is None else gpu_passmarks)
    for name, score in _EXTRA_GPU_PASSMARKS.items():
        combined.setdefault(name, score)
    return combined


def _requirement_haystack(text: str) -> str:
    """One lowercase line, with 'GTX 3060' and '1660Ti' written like the table."""
    raw = (text or "").lower().replace("гб", "gb").replace("®", " ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    raw = re.sub(r"(\d)\s*gb\b", r"\1gb", raw)
    raw = re.sub(r"(\d{3,4})\s*(ti|super)\b", r"\1 \2", raw)
    raw = re.sub(r"\bgtx\s+((?:20|30|40|50)\d{2}(?:\s+ti|\s+super)?)\b", r"rtx \1", raw)
    raw = re.sub(r"\b(?:nvidia|geforce)\s+(\d{3,4})(?!\d)", r"rtx \1", raw)
    return f" {re.sub(r'\s+', ' ', raw).strip()} "


def first_known_gpu(text: str, passmarks: Dict[str, int]) -> Optional[tuple]:
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
        family = _closest_family_gpu(haystack, passmarks)
        return family
    matches.sort(key=lambda row: (row[1] - row[0]), reverse=True)
    kept: list[tuple] = []
    for match in matches:
        covered = any(other[0] <= match[0] and other[1] >= match[1] for other in kept)
        if not covered:
            kept.append(match)
    kept.sort(key=lambda row: row[0])
    return kept[0][2], kept[0][3]


def _closest_family_gpu(haystack: str, passmarks: Dict[str, int]) -> Optional[tuple]:
    """Map a near-miss model, such as GTX 1090 or RTX 8090, onto the closest real card."""
    families: Dict[str, list[tuple]] = {}
    for name, score in passmarks.items():
        match = re.match(r"(rtx|gtx|rx)\s+(\d{3,4})\b", name)
        if not match:
            continue
        families.setdefault(match.group(1), []).append((int(match.group(2)), name, int(score)))
    found = re.findall(r"\b(rtx|gtx|rx)\s+(\d{3,4})\b", haystack)
    best: Optional[tuple] = None
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


def estimate_from_requirements(
    user_gpu: str,
    user_cpu: str,
    recommended_gpu: str = "",
    minimum_gpu: str = "",
    resolution: Optional[str] = None,
    gpu_passmarks: Optional[Dict[str, int]] = None,
    cpu_patterns: Optional[List[tuple]] = None,
) -> FpsEstimate:
    """FPS for a game with no player sample, scaled from its published GPU spec.

    The recommended card is taken as 60 FPS at 1080p. The minimum card is 30.
    The user's GPU and the monitor class move that anchor. Ray tracing is off.
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
    target = _REQUIREMENTS_TARGET_FPS
    source = SOURCE_REQUIREMENTS
    if spec is None:
        spec = first_known_gpu(minimum_gpu, cards)
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
    cpu_factor = cpu_bottleneck_factor(lookup_cpu_passmark(user_cpu, cpu_patterns))
    estimate = scale_from_reference(anchor, 0, user_passmark, spec[1], cpu_factor)
    if estimate.expected_fps > _REQUIREMENTS_FPS_CAP:
        estimate = _spread_band(_REQUIREMENTS_FPS_CAP, _REQUIREMENTS_FPS_CAP * SIMILAR_RELATIVE_SPREAD)
    estimate.fps_source = source
    return estimate


def lookup_cpu_passmark(name: str, patterns: List[tuple]) -> Optional[int]:
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


def _first(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def parse_frame_rate_payload(payload: Optional[Dict[str, Any]]) -> List[FrameRateBucket]:
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

    buckets: List[FrameRateBucket] = []
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


def apply_cluster_labels(buckets: List[FrameRateBucket], cluster_map: Dict[str, Any]) -> None:
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


def steam_client_roots() -> List[Path]:
    """Steam client directories on this PC."""
    roots: List[Path] = []
    program_files = os.environ.get("PROGRAMFILES(X86)")
    if program_files:
        roots.append(Path(program_files) / "Steam")
    local = Path.home() / "AppData" / "Local" / "Steam"
    if local not in roots:
        roots.append(local)
    return roots


def read_local_cluster_id() -> Optional[str]:
    """Read this PC's hardware cluster id from the Steam client config, if present."""
    candidates: List[Path] = []
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


def fit_log_fps_curve(points: List[tuple]) -> Optional[tuple]:
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


def load_benchmark_table() -> Dict[tuple, List[tuple]]:
    """Measured FPS rows keyed by (game, resolution)."""
    path = _data_file("training_benchmarks.json")
    table: Dict[tuple, List[tuple]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read training benchmarks: %s", exc)
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


def _normalize_resolution(resolution: Optional[str]) -> Optional[str]:
    text = re.sub(r"[^a-z0-9]+", "", (resolution or "").lower())
    if text in {"4k", "2160p", "3840x2160"}:
        return "4K"
    if text in {"1440p", "2560x1440", "qhd"}:
        return "1440p"
    if text in {"1080p", "1920x1080", "fhd"}:
        return "1080p"
    return None


def _normalize_game_key(name: str) -> str:
    """Lowercase title with sequel numerals folded, so Diablo IV matches diablo 4."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").lower())
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
    tokens = [numerals.get(token, token) for token in cleaned.split() if token]
    return "".join(tokens)


def match_benchmark_game(game_name: Optional[str], games: List[str]) -> Optional[str]:
    query = _normalize_game_key(game_name or "")
    if not query:
        return None
    keyed = [(_normalize_game_key(game), game) for game in games]
    for norm, game in keyed:
        if norm == query:
            return game
    if len(query) < 4:
        return None
    hits = [game for norm, game in keyed if query in norm or norm in query]
    if len(hits) == 1:
        return hits[0]
    return None


def _benchmark_points(game_name: Optional[str], resolution: Optional[str], table: Optional[Dict[tuple, List[tuple]]] = None) -> List[tuple]:
    resolved = _normalize_resolution(resolution)
    if not game_name or not resolved:
        return []
    if table is None:
        table = load_benchmark_table()
    games = sorted({game for game, _ in table})
    matched = match_benchmark_game(game_name, games)
    if not matched:
        return []
    return list(table.get((matched, resolved), []))


def leave_one_out_fps(game_name: str, resolution: str, gpu_model: str) -> Optional[float]:
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


def _cap_at_tied_leaders(prediction: float, points: List[tuple]) -> float:
    """Keep a tied top score, such as Elden Ring's 60 FPS cap, from being extrapolated."""
    ranked = sorted(points, key=lambda row: row[1], reverse=True)
    if len(ranked) < 2:
        return prediction
    if abs(ranked[0][2] - ranked[1][2]) <= 2:
        return min(prediction, float(ranked[0][2]))
    return prediction


def _estimate_from_curve(prediction: float) -> FpsEstimate:
    estimate = _spread_band(prediction, abs(prediction) * SIMILAR_RELATIVE_SPREAD)
    estimate.fps_source = SOURCE_SIMILAR
    return estimate


def calibrated_similar_fps(
    game_name: Optional[str],
    resolution: Optional[str],
    gpu_model: str,
    gpu_passmarks: Dict[str, int],
    *,
    exclude_gpu: Optional[str] = None,
) -> Optional[FpsEstimate]:
    """Shipped similar-PC estimate. exclude_gpu drops that card from the fit."""
    points = _benchmark_points(game_name, resolution)
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
    return _estimate_from_curve(_cap_at_tied_leaders(prediction, points))


def resolve_fps(
    gpu_model: str,
    cpu_model: str,
    payload: Optional[Dict[str, Any]],
    *,
    api_available: bool = True,
    local_cluster_id: Optional[str] = None,
    cluster_map: Optional[Dict[str, Any]] = None,
    reference_gpu: Optional[str] = None,
    gpu_passmarks: Optional[Dict[str, int]] = None,
    cpu_patterns: Optional[List[tuple]] = None,
    game_name: Optional[str] = None,
    resolution: Optional[str] = None,
    exclude_gpu: Optional[str] = None,
) -> FpsEstimate:
    """
    Pick an FPS figure for this GPU and CPU.

    An exact Steam bucket with at least 30 reports wins. Otherwise a
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
    buckets: List[FrameRateBucket],
    gpu_model: str,
    user_passmark: Optional[int],
    gpu_passmarks: Dict[str, int],
) -> Optional[FrameRateBucket]:
    if not user_passmark:
        return None
    candidates: List[FrameRateBucket] = []
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
    buckets: List[FrameRateBucket],
    user_passmark: Optional[int],
    reference_gpu: str,
    gpu_passmarks: Dict[str, int],
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

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = load_steam_webapi_key() if api_key is None else api_key.strip()

    async def estimate(
        self,
        appid: str,
        gpu_model: str,
        cpu_model: str,
        game_name: Optional[str] = None,
        resolution: Optional[str] = None,
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

    async def _fetch(self, appid: str) -> Optional[Dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        params = {"key": self.api_key, "appid": appid, "format": "json"}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(STATS_URL, params=params) as response:
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
