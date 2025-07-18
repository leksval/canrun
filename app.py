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

def create_gradio_interface():
    """Create a simple Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# CanRun Game Compatibility Checker")
        
        with gr.Row():
            with gr.Column():
                game_input = gr.Textbox(label="Game Name", placeholder="Enter game name (e.g., Diablo 4)")
                check_btn = gr.Button("Check Compatibility")
                hw_btn = gr.Button("Detect Hardware")
            
            with gr.Column():
                result_output = gr.Textbox(label="Results", lines=20)
        
        # For async functions, we need to use .click(fn=..., inputs=..., outputs=...)
        check_btn.click(fn=analyze_game, inputs=game_input, outputs=result_output)
        hw_btn.click(fn=detect_hardware, inputs=None, outputs=result_output)
    
    return demo

def main():
    """Main application entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting CanRun Game Compatibility Checker")
    
    # Create Gradio interface
    demo = create_gradio_interface()
    
    # Launch with auto port discovery
    demo.queue().launch(server_name="0.0.0.0", share=False)
    
    # Keep the main thread alive
    logger.info("Press Ctrl+C to stop the server")
    if hasattr(signal, 'pause'):
        # Unix systems
        signal.pause()
    else:
        # Windows systems
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()