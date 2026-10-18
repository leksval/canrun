#!/usr/bin/env python3
"""Stage and optionally upload the CanRun Gradio demo to a Hugging Face Space.

The Space is public, so it ships no Steam-derived data: no measured benchmark
curves (training_benchmarks.json), no Steam hardware clusters, and the model
without its lookup_table. FPS comes from the store-spec model only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_g_assist_zip import PLUGIN_META, PLUGIN_PY  # noqa: E402
from hf_model_card import inline_math_to_text  # noqa: E402

DATA_DIR = ROOT / "data"
DEFAULT_STAGE = ROOT / "dist" / "hf_space"
DEFAULT_REPO = "grasant/canrun"
SPACE_FILES = ["app.py", "README.md", "logo.png", *PLUGIN_PY, *PLUGIN_META]
PUBLIC_DATA = ["cpu_benchmarks.json", "game_engines.json", "gpu_hierarchy.json"]
FORBIDDEN = {
    "training_benchmarks.json",
    "measured_benchmarks.json",
    "steam_hardware_clusters.json",
    "steam_api_harvest.json",
    "clean_training_dataset.json",
    "game_requirements.json",
    ".env",
}
SPACE_FRONTMATTER = {"sdk_version": "6.30.0", "python_version": '"3.12"'}


def _space_readme(text: str) -> str:
    """Pin the Gradio and Python versions in the README front matter."""
    if not text.startswith("---"):
        raise ValueError("README.md has no Space front matter")
    end = text.index("\n---", 3)
    header, body = text[:end], text[end:]
    for key, value in SPACE_FRONTMATTER.items():
        if f"\n{key}:" not in header:
            header = header.replace("\nsdk: gradio", f"\nsdk: gradio\n{key}: {value}", 1)
    return header + body


def stage(stage_dir: Path) -> list[str]:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "data").mkdir(parents=True)

    for name in SPACE_FILES:
        src = ROOT / name
        if not src.is_file():
            raise FileNotFoundError(src)
        if name == "README.md":
            readme = inline_math_to_text(_space_readme(src.read_text(encoding="utf-8")))
            (stage_dir / name).write_text(readme, encoding="utf-8")
        else:
            shutil.copy2(src, stage_dir / name)

    shutil.copytree(
        ROOT / "libs" / "gassist_sdk",
        stage_dir / "libs" / "gassist_sdk",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in PUBLIC_DATA:
        shutil.copy2(DATA_DIR / name, stage_dir / "data" / name)

    model = json.loads((DATA_DIR / "ml_fps_model_unified.json").read_text(encoding="utf-8"))
    model.pop("lookup_table", None)
    (stage_dir / "data" / "ml_fps_model_unified.json").write_text(json.dumps(model), encoding="utf-8")

    files = sorted(p.relative_to(stage_dir).as_posix() for p in stage_dir.rglob("*") if p.is_file())
    leaked = [f for f in files if Path(f).name in FORBIDDEN]
    if leaked:
        raise RuntimeError(f"Steam-derived files staged: {leaked}")
    model_text = (stage_dir / "data" / "ml_fps_model_unified.json").read_text(encoding="utf-8")
    if '"lookup_table"' in model_text:
        raise RuntimeError("lookup_table leaked into the Space model")
    return files


def _remote_files(repo_id: str) -> list[str]:
    url = f"https://huggingface.co/api/spaces/{repo_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return [s["rfilename"] for s in json.load(resp).get("siblings", [])]


def upload(stage_dir: Path, repo_id: str, files: list[str]) -> None:
    stale = [f for f in _remote_files(repo_id) if f not in files and f != ".gitattributes"]
    cmd = ["hf", "upload", repo_id, ".", ".", "--type", "space", "--commit-message", "CanRun v12.2.0 demo"]
    for path in stale:
        cmd += ["--delete", path]
    # Run from the stage dir: on Windows the hf CLI expands glob arguments against the cwd.
    subprocess.run(cmd, cwd=stage_dir, check=True)
    print(f"Removed {len(stale)} stale files: {stale}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    files = stage(args.stage_dir)
    print(f"Staged {len(files)} files in {args.stage_dir}")
    for f in files:
        print(f"  {f}")
    if args.upload:
        upload(args.stage_dir, args.repo_id, files)
        print(f"https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
