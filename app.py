"""
CanRun Game Compatibility Checker - Simple MCP Server Implementation
"""

import base64
import json
import logging
import os
import signal
import sys
import time

import gradio as gr

import plugin as canrun_plugin
from canrun_engine import sanitize_game_name
from canrun_hardware_detector import HardwareSpecs
from canrun_steam_fps import redact_secrets
from plugin import CanRunGAssistPlugin, format_canrun_response

LOCAL_ONLY_HOST = "127.0.0.1"
# Hugging Face sets SPACE_ID. A Space runs on a server, so the visitor picks
# their hardware instead of the server's being detected.
ON_SPACE = bool(os.environ.get("SPACE_ID"))
DETECT_CHOICE = "This PC (auto-detect)"

CPU_CHOICES = {
    "AMD Ryzen 7 7800X3D": (8, 16),
    "AMD Ryzen 5 7600X": (6, 12),
    "AMD Ryzen 7 5800X3D": (8, 16),
    "AMD Ryzen 5 5600X": (6, 12),
    "AMD Ryzen 5 3600": (6, 12),
    "Intel Core i9-14900K": (24, 32),
    "Intel Core i7-14700K": (20, 28),
    "Intel Core i5-14600K": (14, 20),
    "Intel Core i5-13400F": (10, 16),
    "Intel Core i5-12400F": (6, 12),
    "Intel Core i7-10700K": (8, 16),
    "Intel Core i7-8700K": (6, 12),
}
RESOLUTIONS = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4K": (3840, 2160)}


def _load_nvidia_gpus() -> dict[str, int]:
    """NVIDIA card name -> VRAM in GB, fastest first."""
    path = os.path.join(os.path.dirname(__file__), "data", "gpu_hierarchy.json")
    try:
        with open(path, encoding="utf-8") as f:
            cards = json.load(f).get("nvidia", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read GPU list: %s", exc)
        return {"RTX 4070": 12}
    ranked = sorted(cards.items(), key=lambda item: item[1].get("passmark_g3d", 0), reverse=True)
    return {name: int(info.get("memory", 8)) for name, info in ranked}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
NVIDIA_GPUS = _load_nvidia_gpus()

def signal_handler(signum, _frame):
    """Handle shutdown signals gracefully."""
    logger.info('Received signal %s, shutting down gracefully...', signum)
    sys.exit(0)

def _chosen_hardware(gpu, cpu, ram_gb, resolution, refresh_hz) -> tuple[HardwareSpecs, dict[str, int]]:
    width, height = RESOLUTIONS.get(resolution, RESOLUTIONS["1440p"])
    cores, threads = CPU_CHOICES.get(cpu, (6, 12))
    specs = HardwareSpecs(
        gpu_model=gpu,
        gpu_vram_gb=NVIDIA_GPUS.get(gpu, 8),
        cpu_cores=cores,
        cpu_threads=threads,
        ram_total_gb=int(ram_gb),
        ram_speed_mhz=3200,
        storage_type="NVMe SSD",
        primary_monitor_refresh_hz=int(refresh_hz),
        primary_monitor_resolution=f"{width}x{height}",
        os_version="Windows 11",
        directx_version="DirectX 12",
        cpu_model=cpu,
    )
    return specs, {"width": width, "height": height}


async def analyze_game(game_name, gpu=DETECT_CHOICE, cpu=None, ram_gb=16, resolution="1440p", refresh_hz=144):
    """Analyze game compatibility using the CanRun engine"""
    game_name = sanitize_game_name(game_name or "")
    if not game_name:
        return "Please enter a game name to begin the analysis."

    if gpu != DETECT_CHOICE:
        try:
            CanRunGAssistPlugin()
            engine = canrun_plugin.canrun_engine
            if engine is None:
                return "The CanRun engine could not start. Details were written to the log."
            specs, display = _chosen_hardware(gpu, cpu, ram_gb, resolution, refresh_hz)
            result = await engine.check_game_compatibility(
                game_name, use_cache=False, hardware_specs=specs, display_resolution=display
            )
            return format_canrun_response(result)
        except ValueError as e:
            if "not found" in str(e):
                return f'Could not find "{game_name}" on Steam. Try the exact Steam title.'
            logger.error("Error analyzing game: %s", redact_secrets(e))
            return "An error occurred during analysis. Details were written to the log."
        except Exception as e:
            logger.error("Error analyzing game: %s", redact_secrets(e))
            return "An error occurred during analysis. Details were written to the log."

    try:
        plugin = CanRunGAssistPlugin()
        params = {"game_name": game_name, "force_refresh": True}
        response = await plugin.check_game_compatibility(params)

        if response.get("success", False):
            return response.get("message", "Analysis completed successfully.")
        return response.get("message", "Could not analyze the game. Please check the game name and try again.")
    except Exception as e:
        logger.error("Error analyzing game: %s", redact_secrets(e))
        return "An error occurred during analysis. Details were written to the log."

def detect_hardware():
    """Detect hardware specifications"""
    try:
        plugin = CanRunGAssistPlugin()
        response = plugin.detect_hardware({})

        if response.get("success", False):
            return response.get("message", "Hardware detection successful.")
        return response.get("message", "Could not detect hardware specifications.")
    except Exception as e:
        logger.error("Error detecting hardware: %s", redact_secrets(e))
        return "An error occurred during hardware detection. Details were written to the log."

def _logo_path() -> str:
    """Path to the bundled logo, used as both the header image and the favicon."""
    return os.path.join(os.path.dirname(__file__), "logo.png")

def get_logo_html():
    """Get HTML that displays the logo"""
    logo_path = _logo_path()

    if os.path.exists(logo_path):
        # Read the logo file and encode it as base64
        with open(logo_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Return HTML that displays the logo
        return f"""
        <div style="display: flex; align-items: center; margin-bottom: 0.5em">
            <img src="data:image/png;base64,{encoded_image}" alt="CanRun Logo" style="height: 4em; margin-right: 1em;">
            <div>
                <h1 style="margin: 0; padding: 0">CanRun Game Compatibility Checker</h1>
                <p style="margin: 0; padding: 0">Check if your PC can run any game with an advanced tier system and Steam API integration</p>
            </div>
        </div>
        """
    logger.warning('Logo file not found at %s', logo_path)
    return """
        <div>
            <h1>CanRun Game Compatibility Checker</h1>
            <p>Check if your PC can run any game with an advanced tier system and Steam API integration</p>
        </div>
        """

def create_gradio_interface():
    """Create a simple Gradio interface with logo and favicon"""
    with gr.Blocks(title="CanRun - Game Compatibility Checker") as demo:
        # Header with logo
        gr.HTML(get_logo_html())

        gpu_choices = list(NVIDIA_GPUS) if ON_SPACE else [DETECT_CHOICE, *NVIDIA_GPUS]
        default_gpu = "RTX 4070" if ON_SPACE and "RTX 4070" in NVIDIA_GPUS else gpu_choices[0]

        # Main content
        with gr.Row():
            with gr.Column():
                game_input = gr.Textbox(label="Game Name", placeholder="Enter game name (e.g., Diablo 4)")
                gpu_input = gr.Dropdown(gpu_choices, value=default_gpu, label="GPU (NVIDIA)")
                cpu_input = gr.Dropdown(list(CPU_CHOICES), value="AMD Ryzen 5 7600X", label="CPU")
                with gr.Row():
                    ram_input = gr.Dropdown([8, 16, 32, 64], value=32, label="RAM (GB)")
                    res_input = gr.Dropdown(list(RESOLUTIONS), value="1440p", label="Resolution")
                    hz_input = gr.Dropdown([60, 144, 165, 240], value=144, label="Refresh rate (Hz)")
                check_btn = gr.Button("Check Compatibility", variant="primary")
                hw_btn = gr.Button("Detect Hardware", variant="secondary", visible=not ON_SPACE)
                if ON_SPACE:
                    gr.Markdown(
                        "This demo runs on a server, so pick your hardware above. "
                        "The [G-Assist plugin](https://mod.io/g/g-assist/m/canrun2#description) "
                        "detects your PC automatically. The online demo has no measured benchmark "
                        "curves, so it uses the store-spec model only."
                    )

            with gr.Column():
                result_output = gr.Textbox(label="Results", lines=20)

        # Footer
        gr.HTML("""
        <div style="margin-top: 20px; text-align: center; padding: 10px; border-top: 1px solid #ddd;">
            <p>CanRun - Advanced Game Compatibility Checker with MCP Server Support</p>
            <p>Powered by G-Assist Integration</p>
        </div>
        """)

        # For async functions, we need to use .click(fn=..., inputs=..., outputs=...)
        check_btn.click(
            fn=analyze_game,
            inputs=[game_input, gpu_input, cpu_input, ram_input, res_input, hz_input],
            outputs=result_output,
        )
        hw_btn.click(fn=detect_hardware, inputs=None, outputs=result_output)

    return demo

def is_mcp_available():
    """Check if the MCP package is available"""
    try:
        import mcp as mcp_sdk
    except ImportError:
        return False
    return mcp_sdk is not None

def main():
    """Main application entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting CanRun Game Compatibility Checker")

    # Create Gradio interface
    demo = create_gradio_interface()

    # Check if MCP support is available
    mcp_enabled = is_mcp_available()
    if mcp_enabled:
        logger.info("MCP server functionality is enabled")
    else:
        logger.info("MCP server functionality is disabled. Install with 'pip install \"gradio[mcp]\"' to enable")

    # Launch with auto port discovery
    launch_kwargs = {
        "share": False,
        "favicon_path": _logo_path(),
        # Brand color matching the logo
        "theme": gr.themes.Default(primary_hue="green", secondary_hue="gray"),
    }

    # On a Space, Hugging Face sets GRADIO_SERVER_NAME so the proxy can reach the app.
    if not ON_SPACE:
        launch_kwargs["server_name"] = LOCAL_ONLY_HOST

    # Only enable MCP server if the package is available
    if mcp_enabled:
        launch_kwargs["mcp_server"] = True

    # Launch the server
    demo.queue().launch(**launch_kwargs)

    # Keep the main thread alive - Platform-independent approach
    logger.info("Press Ctrl+C to stop the server")
    try:
        # This works on most systems and is more reliable than signal.pause()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
