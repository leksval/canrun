#!/usr/bin/env python3
"""
Test Official NVIDIA G-Assist Protocol V2 compliance.
"""

import json
import subprocess
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PLUGIN_DIR / "manifest.json"
COMMANDS = ("check_compatibility", "detect_hardware", "auto_detect")


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_g_assist_protocol_compliance():
    """Manifest matches the current G-Assist Protocol V2 contract."""
    manifest = _manifest()

    assert manifest["manifestVersion"] == 1
    assert manifest["name"] == "canrun"
    assert manifest["version"] == "10.0"
    assert isinstance(manifest["description"], str) and manifest["description"]
    assert manifest["executable"] == "plugin.py"
    assert manifest["persistent"] is True
    assert manifest["protocol_version"] == "2.0"
    assert "passthrough" not in manifest
    assert "tethered" not in manifest

    functions = manifest["functions"]
    assert [func["name"] for func in functions] == list(COMMANDS)

    check_compatibility = functions[0]
    assert "description" in check_compatibility
    assert "tags" in check_compatibility
    assert len(check_compatibility["tags"]) >= 5
    assert "properties" in check_compatibility
    assert "game_name" in check_compatibility["required"]
    for expected_tag in ("canrun", "can run", "game", "compatibility"):
        assert expected_tag in check_compatibility["tags"]

    assert functions[1]["required"] == []
    assert "user_input" in functions[2]["required"]


def test_registered_commands_match_manifest():
    """SDK command names match the manifest functions."""
    sys.path.insert(0, str(PLUGIN_DIR))
    from plugin import get_plugin

    plugin = get_plugin()
    manifest_names = {func["name"] for func in _manifest()["functions"]}
    assert set(plugin._commands) == manifest_names


def test_sdk_version():
    """Vendored SDK is the 3.1.0 release pinned in the alignment plan."""
    sys.path.insert(0, str(PLUGIN_DIR / "libs"))
    import gassist_sdk

    assert gassist_sdk.__version__ == "3.1.0"


def test_plugin_directory_structure():
    """Plugin layout matches the Protocol V2 directory contract."""
    required = [
        "manifest.json",
        "plugin.py",
        "config.json",
        "requirements.txt",
        "libs/gassist_sdk/__init__.py",
        "libs/gassist_sdk/plugin.py",
        "libs/gassist_sdk/protocol.py",
        "libs/gassist_sdk/types.py",
        "libs/gassist_sdk/mcp.py",
        "libs/gassist_sdk/ATTRIBUTION.md",
    ]
    for name in required:
        assert (PLUGIN_DIR / name).is_file(), f"Required file missing: {name}"
    assert (PLUGIN_DIR / "data").is_dir()

    manifests = [path.name for path in PLUGIN_DIR.glob("*manifest.json")]
    assert manifests == ["manifest.json"]


def test_cli_initialize_json():
    """CLI mode still returns a JSON success payload without the V1 executable."""
    result = subprocess.run(
        [sys.executable, str(PLUGIN_DIR / "plugin.py"), "initialize", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PLUGIN_DIR),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["message"]
