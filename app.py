import gradio as gr
from src.canrun_engine import CanRunEngine, CanRunResult
from plugin import CanRunGAssistPlugin

# Store previously requested game names to detect changes
previous_game_requests = set()

async def analyze_game(game_name: str) -> str:
    """
    Analyzes the compatibility of a game and returns the formatted result
    using the official G-Assist plugin formatter.
    """
    global previous_game_requests
    
    if not game_name:
        return "Please enter a game name to begin the analysis."

    try:
        # Initialize the plugin - create a fresh instance for each request
        plugin = CanRunGAssistPlugin()
        
        # Check if this is a new game request or a repeat
        is_new_game = game_name not in previous_game_requests
        
        # For numbered game sequels (like "diablo 3"), always treat as new
        has_number = any(c.isdigit() for c in game_name)
        
        # If it's a new game or has numbers, clear the cache to force fresh analysis
        if is_new_game or has_number:
            # Force engine to look for latest Steam data
            plugin_params = {"game_name": game_name, "force_refresh": True}
            previous_game_requests.add(game_name)
        else:
            # Use normal caching for repeat requests of the same game
            plugin_params = {"game_name": game_name, "force_refresh": False}
        
        # Use the plugin to perform the analysis
        plugin_response = await plugin.check_game_compatibility(plugin_params)
        
        # Return the plugin response
        if plugin_response.get("success", False):
            return plugin_response.get("message", "Analysis completed successfully.")
        else:
            return plugin_response.get("message", "Could not analyze the game. Please check the game name and try again.")

    except Exception as e:
        return f"An error occurred during analysis: {e}"

iface = gr.Interface(
    fn=analyze_game,
    inputs=gr.Textbox(lines=1, placeholder="Enter Game Name..."),
    outputs=gr.Textbox(label="Analysis Result", lines=25, interactive=False),
    title="CanRun Game Compatibility Checker",
    description="Enter the name of a game to see if your PC can run it.",
    flagging_mode="never"  # Updated from allow_flagging to flagging_mode
)

if __name__ == "__main__":
    iface.launch()