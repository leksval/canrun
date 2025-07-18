"""
CanRun Game Compatibility Checker - Simple MCP Server Implementation
"""

import gradio as gr
import logging
import os
import signal
import sys
import time
import asyncio
import base64
import platform

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
from src.canrun_engine import CanRunEngine
from plugin import CanRunGAssistPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

async def analyze_game(game_name):
    """Analyze game compatibility using the CanRun engine"""
    if not game_name:
        return "Please enter a game name to begin the analysis."
    
    try:
        plugin = CanRunGAssistPlugin()
        params = {"game_name": game_name, "force_refresh": True}
        # Properly await the async method
        response = await plugin.check_game_compatibility(params)
        
        if response.get("success", False):
            return response.get("message", "Analysis completed successfully.")
        else:
            return response.get("message", "Could not analyze the game. Please check the game name and try again.")
    except Exception as e:
        logger.error(f"Error analyzing game: {e}")
        return f"An error occurred during analysis: {e}"

def detect_hardware():
    """Detect hardware specifications"""
    try:
        plugin = CanRunGAssistPlugin()
        response = plugin.detect_hardware({})
        
        if response.get("success", False):
            return response.get("message", "Hardware detection successful.")
        else:
            return response.get("message", "Could not detect hardware specifications.")
    except Exception as e:
        logger.error(f"Error detecting hardware: {e}")
        return f"An error occurred during hardware detection: {e}"

def get_logo_html():
    """Get HTML that displays the logo"""
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    
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
    else:
        logger.warning(f"Logo file not found at {logo_path}")
        return """
        <div>
            <h1>CanRun Game Compatibility Checker</h1>
            <p>Check if your PC can run any game with an advanced tier system and Steam API integration</p>
        </div>
        """

def create_gradio_interface():
    """Create a simple Gradio interface with logo and favicon"""
    # Set custom theme with brand color matching the logo
    theme = gr.themes.Default(
        primary_hue="green",
        secondary_hue="gray",
    )
    
    # Define file paths
    favicon_path = os.path.join(os.path.dirname(__file__), "logo.png")
    
    with gr.Blocks(theme=theme, title="CanRun - Game Compatibility Checker", css="") as demo:
        # Header with logo
        gr.HTML(get_logo_html())
        
        # Main content
        with gr.Row():
            with gr.Column():
                game_input = gr.Textbox(label="Game Name", placeholder="Enter game name (e.g., Diablo 4)")
                check_btn = gr.Button("Check Compatibility", variant="primary")
                hw_btn = gr.Button("Detect Hardware", variant="secondary")
            
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
        check_btn.click(fn=analyze_game, inputs=game_input, outputs=result_output)
        hw_btn.click(fn=detect_hardware, inputs=None, outputs=result_output)
    
    return demo

def is_mcp_available():
    """Check if the MCP package is available"""
    try:
        import mcp
        return True
    except ImportError:
        return False

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
        "server_name": "0.0.0.0", 
        "share": False,
        "favicon_path": os.path.join(os.path.dirname(__file__), "logo.png"),
    }
    
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