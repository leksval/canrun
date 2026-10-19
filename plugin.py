# SPDX-License-Identifier: Apache-2.0

"""
CanRun G-Assist Plugin
NVIDIA G-Assist plugin for game compatibility checking.
Speaks Protocol V2 (JSON-RPC 2.0) through gassist_sdk 3.1.0, and keeps a CLI for testing.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import traceback

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBS_PATH = os.path.join(_PLUGIN_DIR, "libs")
# G-Assist adds only libs/ to PYTHONPATH. Sibling modules such as canrun_engine.py
# live next to plugin.py, so that directory has to be importable too.
# A user-owned vendor directory is preferred because some files under libs/ are
# unreadable when the installer was run as Administrator.
_USER_VENDOR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "canrun", "vendor")
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)
if _LIBS_PATH not in sys.path:
    sys.path.insert(0, _LIBS_PATH)
if _USER_VENDOR and os.path.isdir(_USER_VENDOR) and _USER_VENDOR not in sys.path:
    sys.path.insert(0, _USER_VENDOR)

# Data Types
type Response = dict[str, bool | str | None]

LOG_FILE = os.path.join(os.environ.get("USERPROFILE", "."), 'canrun_plugin.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Imported after sys.path includes this plugin and libs/.
from gassist_sdk import Plugin  # noqa: E402
from gassist_sdk.types import (  # noqa: E402
    Context,
    ErrorCode,
    JsonRpcResponse,
    SystemInfo,
)

# Import CanRun components - flat structure for G-Assist compliance
try:
    from canrun_engine import CanRunEngine
    CANRUN_AVAILABLE = True
except Exception as e:
    logging.warning('CanRun engine not available: %s', e)
    CANRUN_AVAILABLE = False

# Import G-Assist response fixer
try:
    from g_assist_response_fixer import (
        create_safe_g_assist_response,
    )
    RESPONSE_FIXER_AVAILABLE = True
except ImportError as e:
    logging.warning('G-Assist response fixer not available: %s', e)
    RESPONSE_FIXER_AVAILABLE = False

# Fake game name used by the Protocol V2 subprocess test. Honored only when
# CANRUN_PROTOCOL_PROBE=1 so a normal G-Assist lookup still hits the engine.
PROTOCOL_PROBE_GAME = "protocol-probe"


def protocol_probe_requested(game_name: str) -> bool:
    """Test hook. A game name alone never enables it."""
    return (
        os.environ.get("CANRUN_PROTOCOL_PROBE") == "1"
        and str(game_name or "").strip() == PROTOCOL_PROBE_GAME
    )

def generate_failure_response(message: str | None = None) -> Response:
    """Generates a response indicating failure - G-ASSIST SAFE VERSION."""
    if RESPONSE_FIXER_AVAILABLE:
        return create_safe_g_assist_response(False, message)
    response = {'success': False}
    if message:
        # Basic ASCII cleaning fallback
        clean_msg = ''.join(char for char in str(message) if ord(char) < 128)
        response['message'] = clean_msg[:500]  # Limit length
    return response

def generate_success_response(message: str | None = None) -> Response:
    """Generates a response indicating success - G-ASSIST SAFE VERSION."""
    if RESPONSE_FIXER_AVAILABLE:
        return create_safe_g_assist_response(True, message)
    response = {'success': True}
    if message:
        # Basic ASCII cleaning fallback
        clean_msg = ''.join(char for char in str(message) if ord(char) < 128)
        response['message'] = clean_msg[:500]  # Limit length
    return response

# Initialize CanRun engine globally
canrun_engine = None

def initialize_canrun_engine():
    """Initialize CanRun engine with complete feature set."""
    global canrun_engine
    if CANRUN_AVAILABLE and canrun_engine is None:
        try:
            canrun_engine = CanRunEngine(
                cache_dir=os.path.join(os.path.dirname(__file__), "cache"),
                enable_llm=True
            )
            logging.info("CanRun engine initialized with complete feature set")
            return True
        except Exception as e:
            logging.error('Failed to initialize CanRun engine: %s', e)
            canrun_engine = None
            return False
    return canrun_engine is not None

async def execute_canrun_command(
    params: dict | None = None,
    context: dict | None = None,  # noqa: ARG001
    system_info: dict | None = None,  # noqa: ARG001
) -> dict:
    """Command handler for canrun function - game compatibility analysis.

    context and system_info stay in the signature for G-Assist command callers.
    """
    if not canrun_engine:
        return generate_failure_response("CanRun engine not available")

    if not params:
        return generate_failure_response("Game name is required for CanRun analysis")

    raw_name = str(params.get("game_name", "") or "")
    game_name = extract_game_name(raw_name)
    if not game_name:
        return generate_failure_response("Game name is required for CanRun analysis")
    show_devices = bool(params.get("steam_devices")) or asks_about_steam_device(raw_name)

    force_refresh = params.get("force_refresh", False)
    if isinstance(force_refresh, str):
        force_refresh = force_refresh.lower() == "true"

    logging.info('Starting CanRun analysis for: %s (force_refresh: %s)', game_name, force_refresh)

    try:
        result = await canrun_engine.check_game_compatibility(game_name, use_cache=not force_refresh)

        if result:
            formatted_result = format_canrun_response(result, show_devices=show_devices)
            return generate_success_response(formatted_result)
        return generate_success_response(_missing_game_reply(game_name))

    except Exception as e:
        if "Requirements not found" in str(e):
            logging.info("No Steam requirements for %s", game_name)
            return generate_success_response(_missing_game_reply(game_name))
        logging.exception("Error in game compatibility analysis")
        return generate_success_response(
            f'CanRun could not finish the check for "{game_name}". '
            "Try the exact Steam title again in a moment."
        )

# create_safe_g_assist_response cuts at 1400 characters, which would drop the verdict,
# and its bullet replacement adds a character per line.
_DEVICE_BLOCK_BUDGET = 1340

_ESTIMATE_NOTICE = "Estimate only. Your FPS may vary."
# G-Assist shows text only, so the verdict mark is emoji.
_CAN_RUN_MARK = "✅🏃"
# G-Assist renders Markdown, which merges blank lines and shows HTML entities as text,
# and it drops answers with non-emoji Unicode. A Markdown rule is the ASCII section break.
_SECTION_BREAK = "---"

_VENDOR_WORDS = re.compile(
    r"\b(?:nvidia|geforce|amd|radeon(?=\s+(?:rx|r\d|hd|vii|pro))|intel(?=\s+(?:core|arc))|core(?=\s+i\d)"
    r"|processor|graphics card|equivalent)\b",
    re.IGNORECASE,
)
_HARDWARE_FILLER = re.compile(r"\bor\s+(?:better|higher|newer|above|equivalent|similar)\b", re.IGNORECASE)
_SHORT_HARDWARE_MAX = 36


def short_hardware_names(text) -> str:
    """Requirement hardware without brand words, "RTX 2070 / RX 5700 XT", so it fits on one line."""
    raw = str(text or "").strip()
    parts = []
    for part in re.split(r"\s+or\s+|\s*/\s*|\s*\|\s*", _HARDWARE_FILLER.sub(" ", raw), flags=re.IGNORECASE):
        part = " ".join(_VENDOR_WORDS.sub(" ", part).split()).strip(" ,;.()")
        if part and part.lower() not in {p.lower() for p in parts}:
            parts.append(part)
    if not parts:
        return raw
    shown = parts[0]
    for part in parts[1:]:
        if len(shown) + 3 + len(part) > _SHORT_HARDWARE_MAX:
            break
        shown += " / " + part
    return shown


def format_canrun_response(result, show_devices: bool = False):
    """Format CanRun result for G-Assist display with proper line breaks."""
    try:
        # Extract key information
        tier_value = getattr(result.performance_prediction, 'tier', 'Unknown')
        tier = tier_value.name if hasattr(tier_value, 'name') else tier_value
        score = int(result.performance_prediction.score) if hasattr(result.performance_prediction, 'score') else 0
        can_run = result.can_run_game()
        exceeds_recommended = result.exceeds_recommended_requirements()

        # Get game name
        original_query = result.game_name
        steam_api_name = result.game_requirements.steam_api_name if hasattr(result.game_requirements, 'steam_api_name') and result.game_requirements.steam_api_name else result.game_requirements.game_name

        # Get performance details with ranges
        fps_min = getattr(result.performance_prediction, 'fps_min', None)
        fps_max = getattr(result.performance_prediction, 'fps_max', None)
        expected_fps = getattr(result.performance_prediction, 'expected_fps', None)

        fps_source = getattr(result.performance_prediction, 'fps_source', '') or ''
        if fps_source == 'steam':
            source_note = " (Steam players)"
        elif fps_source == 'steam_similar':
            source_note = " (similar PC, native, RT off)"
        elif fps_source == 'steam_scaled':
            source_note = " (Steam players, similar GPU)"
        elif fps_source == 'requirements':
            source_note = " (from recommended GPU, RT off)"
        elif fps_source == 'requirements_minimum':
            source_note = " (from minimum GPU, RT off)"
        elif fps_source == 'requirements_unspecified':
            source_note = " (no GPU on Steam page, RT off)"
        else:
            source_note = ""

        refresh_hz = int(getattr(result.hardware_specs, "primary_monitor_refresh_hz", 0) or 0)
        # Display FPS as range if available, otherwise single value or unknown.
        # The high end stops at this screen's refresh rate.
        if fps_min and fps_max and fps_min != fps_max:
            fps = _fps_range_text(int(fps_min), int(fps_max), refresh_hz, source_note)
            fps_line = f"FPS: {fps}"
        elif expected_fps and isinstance(expected_fps, (int, float)) and expected_fps > 0:
            fps = _fps_range_text(int(expected_fps), int(expected_fps), refresh_hz, source_note)
            fps_line = f"FPS: {fps}"
        else:
            fps = "Unknown"
            fps_line = "FPS: **Unknown** (no benchmark for this GPU)"

        settings = getattr(result.performance_prediction, 'recommended_settings', 'Unknown')
        recommended_resolution = getattr(result.performance_prediction, 'recommended_resolution', 'Unknown')
        from canrun_steam_fps import graphics_feature_lines
        feature_lines = graphics_feature_lines(
            fps_source,
            int(expected_fps or 0),
            int(fps_min or 0),
            int(fps_max or 0),
            getattr(result, "detected_resolution_class", "") or "",
            getattr(result.hardware_specs, "gpu_model", "") or "",
            int(getattr(result.hardware_specs, "gpu_vram_gb", 0) or 0),
            bool(getattr(result.hardware_specs, "supports_dlss", False)),
            bool(getattr(result.hardware_specs, "supports_rtx", False)),
            steam_api_name or result.game_name,
            refresh_hz,
            getattr(result.game_requirements, "rtx_features", None),
        )
        feature_block = "".join(f"\n• {line}" for line in feature_lines)

        # Get actual resolution from hardware specs
        actual_resolution = result.hardware_specs.primary_monitor_resolution
        display_note = _heavier_display_note(
            actual_resolution,
            getattr(result, "detected_resolution_class", ""),
        )

        from canrun_hardware_detector import laptop_power_note
        power_note = laptop_power_note(
            getattr(result.hardware_specs, "gpu_model", "") or "",
            int(getattr(result.hardware_specs, "gpu_power_limit_w", 0) or 0),
        )
        power_line = f"\n• {power_note}" if power_note else ""

        from canrun_game_fetcher import steam_device_lines
        devices = steam_device_lines(
            getattr(result.game_requirements, "steam_device_ratings", {}) or {},
            bool(getattr(result.game_requirements, "vr_only", False)),
            getattr(result.game_requirements, "steamfps", {}) or {},
        ) if show_devices else []
        device_block = "\n\n🕹️ **STEAM DEVICES**" + "".join(f"\n• {line}" for line in devices) if devices else ""

        message = f"""🎮 **Game:** {steam_api_name}

💻 **YOUR SYSTEM**
• **GPU:** {result.hardware_specs.gpu_model} ({result.hardware_specs.gpu_vram_gb}GB){power_line}
• **CPU:** {result.hardware_specs.cpu_model}
• **RAM:** {result.hardware_specs.ram_total_gb}GB
• **Display:** {actual_resolution} @ {result.hardware_specs.primary_monitor_refresh_hz}Hz{display_note}

{_SECTION_BREAK}

🎯 **GAME REQUIREMENTS**"""

        if exceeds_recommended:
            # System exceeds recommended - only show recommended requirements
            message += f"""
**Recommended** (Your system exceeds these)
• **GPU:** {short_hardware_names(result.game_requirements.recommended_gpu)}"""
            if hasattr(result.game_requirements, 'recommended_cpu') and result.game_requirements.recommended_cpu:
                message += f"""
• **CPU:** {short_hardware_names(result.game_requirements.recommended_cpu)}"""
            message += f"""
• **RAM:** {result.game_requirements.recommended_ram_gb}GB"""
        else:
            # System doesn't exceed recommended - show both minimal and recommended
            message += f"""
**Minimum**
• **GPU:** {short_hardware_names(result.game_requirements.minimum_gpu)}"""
            if hasattr(result.game_requirements, 'minimum_cpu') and result.game_requirements.minimum_cpu:
                message += f"""
• **CPU:** {short_hardware_names(result.game_requirements.minimum_cpu)}"""
            message += f"""
• **RAM:** {result.game_requirements.minimum_ram_gb}GB

**Recommended**
• **GPU:** {short_hardware_names(result.game_requirements.recommended_gpu)}"""
            if hasattr(result.game_requirements, 'recommended_cpu') and result.game_requirements.recommended_cpu:
                message += f"""
• **CPU:** {short_hardware_names(result.game_requirements.recommended_cpu)}"""
            message += f"""
• **RAM:** {result.game_requirements.recommended_ram_gb}GB"""

        verdict = f"{_CAN_RUN_MARK} CAN RUN" if can_run else "❌ CANNOT RUN"
        message += f"""

{_SECTION_BREAK}

🎯 **VERDICT:** {verdict}

⚡ **PERFORMANCE**
• {fps_line}{feature_block}
• **Settings:** {settings}
• **Resolution:** {recommended_resolution}
• **Score:** {tier} Tier ({score}/100){device_block}

{_ESTIMATE_NOTICE}

🔧 **OPTIMIZE**"""

        if hasattr(result.performance_prediction, 'upgrade_suggestions') and result.performance_prediction.upgrade_suggestions:
            for suggestion in result.performance_prediction.upgrade_suggestions[:2]:
                message += f"""
• {suggestion}"""
        else:
            if can_run:
                if not feature_lines and result.hardware_specs.supports_dlss:
                    message += """
• Turn on DLSS Quality for more FPS"""
                if not feature_lines and result.hardware_specs.supports_rtx:
                    message += """
• Turn on ray tracing for better visuals"""
                message += """
• Update GPU drivers"""
            else:
                message += """
• Upgrade the GPU to the minimum
• Check RAM meets the minimum"""

        # Add note if different game was found
        if steam_api_name.lower() != original_query.lower():
            message += f"""

📝 Note: Showing results for {steam_api_name}"""

        if device_block and len(message) > _DEVICE_BLOCK_BUDGET:
            message = message.replace(device_block, "", 1)
        return message.strip()

    except Exception as e:
        logging.error('Error formatting CanRun response: %s', e)
        return f"""🎮 CANRUN ANALYSIS: {getattr(result, 'game_name', 'Unknown Game')}

✅ Analysis completed but formatting error occurred.
Raw result available in logs."""

def _run_async(coro):
    """Run one coroutine on a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def missing_steam_game_message(game_name: str, suggestions: list[str] | None = None) -> str:
    """Explain a missing Steam page without turning it into a plugin error."""
    from canrun_game_fetcher import _suggestion_titles, steam_title_tokens

    name = (game_name or "that game").strip()
    lines = [
        f'CanRun could not find "{name}" on Steam.',
        "Requirements and FPS come from the Steam store, so this name cannot be scored.",
        "Ask again with the exact title on the Steam page.",
    ]
    tokens = steam_title_tokens(name)
    if "diablo" in tokens and "3" in tokens:
        lines.append("Diablo III is on Battle.net, not Steam. Diablo IV is the Diablo game on Steam.")
    close = _suggestion_titles(name, list(suggestions or []))
    if close:
        lines.append("Steam pages that start with that name: " + "; ".join(close) + ".")
    return "\n".join(lines)


def _message_or_raise(response: dict) -> str:
    """Return the user-visible report. Expected failures become SDK errors."""
    if not isinstance(response, dict) or not response.get("success"):
        message = "Command failed"
        if isinstance(response, dict) and response.get("message"):
            message = response["message"]
        raise RuntimeError(message)
    return response.get("message") or ""


def format_hardware_report(hw_specs) -> str:
    """Format detected hardware for G-Assist display."""
    return f"""SYSTEM HARDWARE DETECTION:

GRAPHICS CARD:
- GPU: {hw_specs.gpu_model}
- VRAM: {hw_specs.gpu_vram_gb}GB
- RTX Features: {'Supported' if hw_specs.supports_rtx else 'Not Available'}
- DLSS Support: {'Available' if hw_specs.supports_dlss else 'Not Available'}
- Driver Status: {'Compatible' if hw_specs.nvidia_driver_version != 'Unknown' else 'Unknown Version'}

PROCESSOR:
- CPU: {hw_specs.cpu_model}
- Cores: {hw_specs.cpu_cores} Physical / {hw_specs.cpu_threads} Logical
- Performance: {'High-Performance' if hw_specs.cpu_cores >= 6 else 'Mid-Range'}

MEMORY:
- RAM: {hw_specs.ram_total_gb}GB Total
- Speed: {hw_specs.ram_speed_mhz}MHz
- Gaming Performance: {'Excellent' if hw_specs.ram_total_gb >= 16 else 'Adequate' if hw_specs.ram_total_gb >= 8 else 'Below Recommended'}

DISPLAY:
- Resolution: {hw_specs.primary_monitor_resolution}
- Refresh Rate: {hw_specs.primary_monitor_refresh_hz}Hz

STORAGE:
- Type: {hw_specs.storage_type}
- Performance: {'Fast Loading' if 'SSD' in hw_specs.storage_type else 'Standard'}

SYSTEM:
- OS: {hw_specs.os_version}
- DirectX: {hw_specs.directx_version}
- G-Assist: Compatible (Plugin Working)

Hardware detection completed successfully using CanRun's privacy-aware detection system."""


def collect_hardware_report() -> dict:
    """Detect hardware and return a CLI-style success or failure response."""
    if not canrun_engine and not initialize_canrun_engine():
        return generate_failure_response("CanRun engine not available")
    if not canrun_engine:
        return generate_failure_response("CanRun engine not available")
    hw_specs = _run_async(canrun_engine.hardware_detector.get_hardware_specs())
    return generate_success_response(format_hardware_report(hw_specs))


def _fps_range_text(fps_min: int, fps_max: int, refresh_hz: int, source_note: str) -> str:
    """Cap the shown maximum at this monitor. Say what a faster screen could show."""
    note = source_note or ""
    if refresh_hz > 0 and fps_max > refresh_hz:
        shown = str(refresh_hz) if fps_min >= refresh_hz else f"{fps_min}-{refresh_hz}"
        uncapped = f"{fps_min}-{fps_max}" if fps_min != fps_max else str(fps_max)
        return f"**{shown}**{note}\n• Faster screen: about {uncapped} FPS"
    if fps_min != fps_max:
        return f"**{fps_min}-{fps_max}**{note}"
    return f"**~{fps_min}**{note}"


def _heavier_display_note(resolution: str, resolution_class: str) -> str:
    """Say when the FPS class is lighter than the monitor, so the range is not read as exact."""
    match = re.match(r"(\d+)\s*x\s*(\d+)", str(resolution or ""), re.IGNORECASE)
    class_pixels = {
        "4K": 3840 * 2160,
        "1440p": 2560 * 1440,
        "1080p": 1920 * 1080,
        "720p": 1280 * 720,
    }.get(resolution_class or "")
    if not match or not class_pixels:
        return ""
    pixels = int(match.group(1)) * int(match.group(2))
    if pixels <= class_pixels * 1.15:
        return ""
    return f" (FPS at {resolution_class})"


_MY_PC = r"(?:\s+(?:on|with)\s+(?:my|this|the)\s+(?:pc|computer|laptop|rig|system|machine|setup))?"
_QUERY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^.*?\bcan\s?run\b[\s:,\-]*(?P<game>.+?)" + _MY_PC + r"$",
        r"^(?:can|could|will|would)\s+(?:i|we|it|you|my\s+(?:pc|computer|laptop|rig|system)|this\s+(?:pc|computer|laptop))"
        r"\s+(?:be\s+able\s+to\s+|even\s+)?(?:run|play|handle)\s+(?P<game>.+?)" + _MY_PC + r"$",
        r"^(?:is\s+it\s+possible\s+to|check\s+if\s+i\s+can|let\s+me\s+know\s+if\s+i\s+can)\s+(?:run|play)\s+(?P<game>.+?)"
        + _MY_PC + r"$",
        r"^how\s+(?:well\s+)?(?:will|would|does|do|can)\s+(?P<game>.+?)\s+(?:run|perform|play)(?:\s+well)?" + _MY_PC + r"$",
        r"^(?:will|would|does|can)\s+(?P<game>.+?)\s+run(?:\s+well|\s+smoothly)?" + _MY_PC + r"$",
        r"^(?:how\s+many\s+fps|what\s+fps|fps|frame\s?rate|performance)(?:\s+(?:will|would|can|do)\s+i\s+get)?"
        r"\s+(?:in|for|on)\s+(?P<game>.+?)" + _MY_PC + r"$",
    )
]


_STEAM_DEVICE = r"steam\s?(?:deck|machine|frame|os)"
_STEAM_DEVICE_SUFFIX = re.compile(
    r"\s+(?:on|with|for|in)\s+(?:(?:my|the|a|this)\s+)?" + _STEAM_DEVICE + r"(?:\s+(?:oled|lcd))?$", re.IGNORECASE
)


def asks_about_steam_device(user_input: str) -> bool:
    """True when the question names a Steam Deck, Steam Machine, Steam Frame or SteamOS."""
    return bool(re.search(r"\b" + _STEAM_DEVICE + r"\b", user_input or "", re.IGNORECASE))


def extract_game_name(user_input: str) -> str:
    """Pull a game name out of a natural-language canrun query."""
    text = " ".join((user_input or "").split()).strip(" ?!.,:;")
    text = _STEAM_DEVICE_SUFFIX.sub("", text)
    if re.fullmatch(r"can\s?run", text, re.IGNORECASE):
        return ""
    game_name = text
    for pattern in _QUERY_PATTERNS:
        match = pattern.match(text)
        if match and match.group("game").strip():
            game_name = match.group("game")
            break
    from canrun_game_fetcher import prefer_default_title

    return prefer_default_title(game_name.strip(" ?!.,:;\"'"))


def execute_initialize_command() -> dict:
    """Command handler for initialize function."""
    logging.info('Initializing CanRun plugin')
    success = initialize_canrun_engine()
    if success:
        return generate_success_response('CanRun plugin initialized successfully')
    return generate_failure_response('Failed to initialize CanRun engine')

def execute_shutdown_command() -> dict:
    """Command handler for shutdown function."""
    logging.info('Shutting down CanRun plugin')
    return generate_success_response('CanRun plugin shutdown complete')

def cli_mode():
    """Run the plugin in CLI mode for direct testing."""
    # Filter out PyInstaller injected arguments like parent_pid=xxxx, pipe_handle=xxxx, etc.
    filtered_argv = []
    pyinstaller_patterns = ['parent_pid=', 'pipe_handle=', 'handle=', 'process_id=']
    for arg in sys.argv[1:]:  # Skip the script name
        is_pyinstaller_arg = any(arg.startswith(pattern) for pattern in pyinstaller_patterns)
        if not is_pyinstaller_arg:
            filtered_argv.append(arg)

    parser = argparse.ArgumentParser(description='CanRun Plugin CLI Mode')
    parser.add_argument('command', choices=['canrun', 'detect_hardware', 'initialize', 'shutdown'],
                        help='Command to execute')
    parser.add_argument('game_name', nargs='?', default='',
                        help='Game name for canrun command')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh game data from Steam API')
    parser.add_argument('--json', action='store_true',
                        help='Output in JSON format')

    args = parser.parse_args(filtered_argv)

    # Initialize the engine
    if not initialize_canrun_engine():
        result = generate_failure_response("Failed to initialize CanRun engine")
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result['message']}")
        return 1

    # Execute the command
    if args.command == 'initialize':
        result = execute_initialize_command()
    elif args.command == 'shutdown':
        result = execute_shutdown_command()
    elif args.command == 'detect_hardware':
        result = collect_hardware_report()
    elif args.command == 'canrun':
        if not args.game_name:
            result = generate_failure_response("Game name is required for canrun command")
        else:
            params = {"game_name": args.game_name, "force_refresh": args.force_refresh}
            result = _run_async(execute_canrun_command(params))
    else:
        result = generate_failure_response(f"Unknown command: {args.command}")

    # Output the result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get('success'):
            print(result.get('message', 'Command executed successfully'))
        else:
            print(f"Error: {result.get('message', 'Command failed')}")

    return 0 if result.get('success') else 1

class CanRunPlugin(Plugin):
    """Protocol V2 plugin. Command work runs off the reader thread so pings still get answers."""

    def __init__(self, name: str, version: str = "13.4.0", description: str = ""):
        import signal
        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)
        super().__init__(name, version, description)
        self._execute_lock = threading.Lock()

        def _chain(previous):
            def handler(signum, frame):
                self._handle_signal(signum, frame)
                if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                    previous(signum, frame)
            return handler

        signal.signal(signal.SIGINT, _chain(previous_int))
        signal.signal(signal.SIGTERM, _chain(previous_term))

    def _handle_execute(self, request):
        params = request.params or {}
        function_name = params.get("function", "")
        arguments = params.get("arguments") or {}
        context_data = params.get("context") or []
        system_info_data = params.get("system_info") or ""

        logging.info('Executing command: %s', function_name)

        cmd = self._commands.get(function_name)
        if cmd is None:
            response = JsonRpcResponse.make_error(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                f"Unknown command: {function_name}",
            )
            self._protocol.send_response(response)
            return

        if not self._execute_lock.acquire(blocking=False):
            self._send_error(
                request.id,
                ErrorCode.PLUGIN_ERROR,
                "Another command is already running",
            )
            return

        self._current_request_id = request.id
        self._keep_session = False

        if not isinstance(context_data, list):
            context_data = []
        if not isinstance(system_info_data, str):
            system_info_data = json.dumps(system_info_data) if system_info_data else ""
        if not isinstance(arguments, dict):
            arguments = {}

        def worker():
            try:
                context = Context.from_list(context_data)
                system_info = SystemInfo.from_string(system_info_data)
                result = self._call_handler(cmd.handler, arguments, context, system_info)
                self._send_complete(request.id, True, result, self._keep_session)
            except Exception as exc:
                logging.error('Command execution error: %s\n%s', exc, traceback.format_exc())
                self._send_error(request.id, ErrorCode.PLUGIN_ERROR, str(exc))
            finally:
                self._current_request_id = None
                self._execute_lock.release()

        threading.Thread(target=worker, name="canrun-execute", daemon=True).start()


def steam_unavailable_message(game_name: str, status: int) -> str:
    """The Steam store refused or failed the lookup, so the title itself may be fine."""
    name = (game_name or "that game").strip()
    return "\n".join([
        f'CanRun could not look up "{name}" because the Steam store is not answering this PC (HTTP {status}).',
        "Steam does this for a while after many requests or during maintenance.",
        "Try again in a few minutes.",
    ])


def _missing_game_reply(game_name: str) -> str:
    suggestions = []
    status = 0
    if canrun_engine is not None:
        suggestions = getattr(canrun_engine.game_fetcher, "recent_suggestions", [])
        status = int(getattr(canrun_engine.game_fetcher, "steam_store_status", 0) or 0)
    if status in (403, 429) or status >= 500:
        return steam_unavailable_message(game_name, status)
    return missing_steam_game_message(game_name, suggestions)


def check_compatibility(game_name: str = "", force_refresh: bool = False) -> str:
    """Complete CanRun analysis for a game name."""
    raw_name = str(game_name or "").strip()
    name = extract_game_name(raw_name)
    if protocol_probe_requested(raw_name) or protocol_probe_requested(name):
        time.sleep(1.0)
        return "protocol-probe-ok"
    if not initialize_canrun_engine():
        return "CanRun could not start. Quit the NVIDIA App from the system tray, open it again, and ask once more."
    if not name:
        return "CanRun needs a game name. Try: Can I run Cyberpunk 2077?"
    params = {"game_name": name, "force_refresh": force_refresh, "steam_devices": asks_about_steam_device(raw_name)}
    return _message_or_raise(_run_async(execute_canrun_command(params)))


def detect_hardware_command() -> str:
    """Detect and analyze system hardware specifications."""
    return _message_or_raise(collect_hardware_report())


def auto_detect(user_input: str = "") -> str:
    """Parse a natural-language compatibility question and run the check."""
    logging.info("Auto-detect received %d characters", len(user_input or ""))
    if not extract_game_name(user_input):
        return "CanRun needs a game name. Try: Can I run Cyberpunk 2077?"
    return check_compatibility(user_input)


_plugin = None


def _manifest_identity() -> tuple[str, str, str]:
    """Name, version, and description from manifest.json.

    mod.io installs the plugin in a folder named with the version, and G-Assist
    uses that folder name. The manifest name has to be the same string.
    """
    path = os.path.join(_PLUGIN_DIR, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return (
        str(data.get("name") or "canrun-13.4.0"),
        str(data.get("version") or "13.4.0"),
        str(data.get("description") or ""),
    )


def get_plugin() -> CanRunPlugin:
    """Build the Protocol V2 plugin and register manifest commands once."""
    global _plugin
    if _plugin is None:
        name, version, description = _manifest_identity()
        _plugin = CanRunPlugin(
            name=name,
            version=version,
            description=description,
        )
        _plugin.command(
            "check_compatibility",
            description=(
                "Check one Steam game against this PC. The returned text is the full answer. "
                "Show that text to the user unchanged."
            ),
        )(check_compatibility)
        _plugin.command(
            "detect_hardware",
            description="Report the GPU, CPU, RAM, and display on this PC. Show the returned text unchanged.",
        )(detect_hardware_command)
        _plugin.command(
            "auto_detect",
            description=(
                "Read a question such as 'Can I run Diablo IV?' and answer with the check. "
                "Show the returned text unchanged."
            ),
        )(auto_detect)
    return _plugin


_CLI_COMMANDS = {"canrun", "detect_hardware", "initialize", "shutdown"}


def _cli_requested() -> bool:
    """True only for an explicit CLI command. Other arguments stay on the G-Assist pipe."""
    for arg in sys.argv[1:]:
        if arg.startswith(("parent_pid=", "pipe_handle=", "handle=", "process_id=", "-")):
            continue
        return arg in _CLI_COMMANDS
    return False


def main():
    """CLI for known commands, otherwise the G-Assist Protocol V2 loop."""
    logging.info("CanRun Plugin Started argv=%s", sys.argv[1:])
    if _cli_requested():
        return cli_mode()
    logging.info("Starting G-Assist Protocol V2")
    get_plugin().run()
    return 0


class CanRunGAssistPlugin:
    """G-Assist Plugin class for MCP server integration."""

    def __init__(self):
        """Initialize the plugin."""
        self.engine = None
        initialize_canrun_engine()

    async def check_game_compatibility(self, params: dict) -> dict:
        """Check game compatibility - wrapper for MCP server."""
        return await execute_canrun_command(params)

    def detect_hardware(self, params: dict) -> dict:  # noqa: ARG002
        """Detect hardware for the Gradio app. params is unused; the signature stays stable."""
        try:
            return collect_hardware_report()
        except Exception as e:
            return generate_failure_response(f"Hardware detection error: {e!s}")


if __name__ == '__main__':
    sys.exit(main())
