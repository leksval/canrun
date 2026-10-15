#!/usr/bin/env python3
"""Build the official G-Assist / mod.io zip.

NVIDIA's Protocol V2 layout (plugin.py, manifest.json, optional config.json,
requirements.txt, libs/) does not require a training corpus. mod.io only
hosts that zip. CanRun adds the sibling modules and the JSON tables the
installed plugin loads. Offline training files stay on this machine.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PLUGIN_FOLDER = "canrun"

PLUGIN_PY = [
    "plugin.py",
    "canrun_engine.py",
    "canrun_hardware_detector.py",
    "canrun_game_fetcher.py",
    "canrun_game_matcher.py",
    "canrun_compatibility_analyzer.py",
    "canrun_ml_predictor.py",
    "canrun_model_loader.py",
    "canrun_steam_fps.py",
    "g_assist_response_fixer.py",
]
PLUGIN_META = [
    "manifest.json",
    "config.json",
    "requirements.txt",
    "LICENSE",
]
# Runtime tables the installed plugin opens. training_benchmarks.json is the
# per-game FPS curve table, not the training corpus.
DATA_JSON = [
    "cpu_benchmarks.json",
    "gpu_hierarchy.json",
    "ml_fps_model_unified.json",
    "steam_hardware_clusters.json",
    "training_benchmarks.json",
]
# Present under data/ for the offline pipeline. Never copy these.
EXCLUDED_DATA = {
    "clean_training_dataset.json",
    "game_requirements.json",
    "model_evaluation_report.json",
    "steam_api_harvest.json",
}
EXCLUDED_SUFFIXES = {".pkl", ".joblib", ".pickle"}
# pip --target leftovers that G-Assist never imports.
DROP_LIBS = {
    "adodbapi",
    "asyncio_throttle",
    "bin",
    "bs4",
    "cloudpickle",
    "isapi",
    "joblib",
    "numpy",
    "numpy.libs",
    "pythonwin",
    "soupsieve",
    "win32comext",
}
DROP_DIST_INFO = {
    "asyncio_throttle",
    "beautifulsoup4",
    "cloudpickle",
    "joblib",
    "numpy",
    "soupsieve",
}


def _plugin_version() -> str:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("version") or "12.0.0")


def _is_excluded_data(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return name in EXCLUDED_DATA or suffix in EXCLUDED_SUFFIXES


def _copy_sdk(dest_libs: Path) -> None:
    src = ROOT / "libs" / "gassist_sdk"
    if not src.is_dir():
        raise SystemExit(f"Missing vendored SDK: {src}")
    shutil.copytree(
        src,
        dest_libs / "gassist_sdk",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )


def _prune_libs(libs: Path) -> None:
    for child in list(libs.iterdir()):
        name = child.name
        drop = name in DROP_LIBS or name.startswith(tuple(f"{pkg}-" for pkg in DROP_LIBS))
        drop = drop or name.endswith(".dist-info") and name.split("-", 1)[0] in DROP_DIST_INFO
        drop = drop or _is_excluded_data(name)
        if drop:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        elif child.is_dir() and child.name != "gassist_sdk":
            for cache in child.rglob("__pycache__"):
                shutil.rmtree(cache, ignore_errors=True)


def _install_requirements(libs: Path) -> None:
    pip = shutil.which("py") or shutil.which("python")
    if not pip:
        raise SystemExit("Python is required to install plugin dependencies into libs/")
    cmd = [pip, "-3", "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "-t", str(libs)]
    if pip.endswith("python.exe") or Path(pip).name.lower() == "python":
        cmd = [pip, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "-t", str(libs)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"pip install failed with {result.returncode}")


def _assert_official_layout(staging: Path, zip_path: Path) -> None:
    staged = [path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()]
    leaked = [name for name in staged if _is_excluded_data(Path(name).name) or "/.env" in f"/{name}"]
    if leaked:
        raise SystemExit(f"Training or harvest files were staged: {leaked}")

    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    roots = {name.split("/", 1)[0] for name in names}
    if roots != {PLUGIN_FOLDER}:
        raise SystemExit(f"Zip must contain exactly one root folder {PLUGIN_FOLDER!r}, found {sorted(roots)}")
    leaked = [name for name in names if _is_excluded_data(Path(name).name)]
    if leaked:
        raise SystemExit(f"Training or harvest files were zipped: {leaked}")

    data_names = sorted(name for name in names if name.startswith(f"{PLUGIN_FOLDER}/data/"))
    expected = {f"{PLUGIN_FOLDER}/data/{name}" for name in DATA_JSON}
    if set(data_names) != expected:
        raise SystemExit(f"data/ contents {data_names} do not match the runtime tables {sorted(expected)}")


def stage(staging: Path) -> list[str]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in PLUGIN_PY + PLUGIN_META:
        source = ROOT / name
        if not source.is_file():
            raise SystemExit(f"Missing plugin file: {source}")
        shutil.copy2(source, staging / name)

    data_dir = staging / "data"
    data_dir.mkdir()
    for name in DATA_JSON:
        if _is_excluded_data(name):
            raise SystemExit(f"{name} is training data and cannot be packaged")
        source = ROOT / "data" / name
        if not source.is_file():
            raise SystemExit(f"Missing runtime table: {source}")
        shutil.copy2(source, data_dir / name)

    skipped = sorted(
        path.name
        for path in (ROOT / "data").iterdir()
        if path.is_file() and path.name not in DATA_JSON
    )

    libs = staging / "libs"
    libs.mkdir()
    _copy_sdk(libs)
    _install_requirements(libs)
    _prune_libs(libs)
    return skipped


def write_zip(staging: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    # G-Assist Hub requires exactly one root folder. A flat archive, or a
    # "./" prefix, installs to the wrong place.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, f"{PLUGIN_FOLDER}/{path.relative_to(staging).as_posix()}")


def main() -> None:
    version = _plugin_version()
    staging = DIST / f"canrun-{version}"
    zip_path = DIST / f"canrun-{version}.zip"
    DIST.mkdir(exist_ok=True)
    skipped = stage(staging)
    write_zip(staging, zip_path)
    _assert_official_layout(staging, zip_path)

    files = [path for path in staging.rglob("*") if path.is_file()]
    bytes_on_disk = sum(path.stat().st_size for path in files)
    print(f"Runtime data: {', '.join(DATA_JSON)}")
    if skipped:
        print(f"Left out of data/: {', '.join(skipped)}")
    print(f"Staged {len(files)} files, {bytes_on_disk / 1_048_576:.1f} MB")
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
