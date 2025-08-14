#!/usr/bin/env python3
"""
CanRun G-Assist Plugin
Official NVIDIA G-Assist plugin following standard communication protocol
"""

import sys
import json
import os
import logging
import asyncio
import platform
from typing import Optional, Dict, Any

# Windows pipe communication - only import on Windows
if platform.system() == "Windows":
    from ctypes import byref, windll, wintypes

# Import CanRun components
try:
    from canrun.src.canrun_engine import CanRunEngine
    CANRUN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CanRun engine not available: {e}")
    CANRUN_AVAILABLE = False

# Constants
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
BUFFER_SIZE = 4096
IS_WINDOWS = platform.system() == "Windows"

def setup_logging():
    """Setup minimal logging for plugin operations."""
    log_file = os.path.join(os.environ.get("USERPROFILE" if IS_WINDOWS else "HOME", "."), 'canrun_plugin.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding='utf-8'
    )

def parse_json_command(data):
    """Parse JSON command from G-Assist communication."""
    if not data:
        return None
    
    try:
        return json.loads(data.strip())
    except json.JSONDecodeError:
        # Extract JSON from mixed content
        data = data.strip()
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(data):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(data[start_idx:i+1])
                    except json.JSONDecodeError:
                        continue
        return None

def read_command():
    """Read command from stdin - Official NVIDIA Implementation."""
    try:
        line = sys.stdin.readline()
        if not line:
            logging.info('EOF received, shutting down')
            return "EOF"
            
        line = line.strip()
        if not line:
            logging.warning('Empty line received, continuing')
            return None
            
        logging.info(f'Received command: {line}')
        return json.loads(line)
        
    except json.JSONDecodeError as e:
        logging.error(f'Invalid JSON received: {e}')
        return None
    except Exception as e:
        logging.error(f'Error in read_command: {e}')
        return None

def write_response(response):
    """Write response to stdout - Official NVIDIA Implementation."""
    try:
        message = json.dumps(response) + '<<END>>'
        sys.stdout.write(message)
        sys.stdout.flush()
        logging.info(f'Response sent: {len(message)} characters')
    except Exception as e:
        logging.error(f'Error writing response: {e}')

def is_g_assist_environment():
    """Check if running in G-Assist environment."""
    return not sys.stdin.isatty()

class CanRunGAssistPlugin:
    """Official G-Assist plugin for CanRun game compatibility checking."""
    
    def __init__(self):
        """Initialize CanRun G-Assist plugin with complete engine integration."""
        if CANRUN_AVAILABLE:
            try:
                self.canrun_engine = CanRunEngine(
                    cache_dir=os.path.join(os.path.dirname(__file__), "cache"),
                    enable_llm=True
                )
                logging.info("CanRun engine initialized with complete feature set")
            except Exception as e:
                logging.error(f"Failed to initialize CanRun engine: {e}")
                self.canrun_engine = None
        else:
            self.canrun_engine = None
    
    async def check_game_compatibility(self, params):
        """Perform CanRun analysis using the full CanRun engine."""
        game_name = params.get("game_name", "").strip()
        
        force_refresh_param = params.get("force_refresh", False)
        if isinstance(force_refresh_param, str):
            force_refresh = force_refresh_param.lower() == "true"
        else:
            force_refresh = bool(force_refresh_param)
        
        if not game_name:
            return {
                "success": False,
                "message": "Game name is required for CanRun analysis"
            }
        
        logging.info(f"Starting CanRun analysis for: {game_name} (force_refresh: {force_refresh})")
        
        try:
            if not self.canrun_engine:
                return {
                    "success": False,
                    "message": f"CanRun engine not available. Cannot analyze {game_name}."
                }
            
            result = await self.canrun_engine.check_game_compatibility(game_name, use_cache=not force_refresh)
            
            if result:
                formatted_result = self.format_canrun_response(result)
                return {
                    "success": True,
                    "message": formatted_result
                }
            else:
                return {
                    "success": False,
                    "message": f"Could not analyze game: {game_name}. Please check the game name and try again."
                }
                
        except Exception as e:
            logging.error(f"Error in game compatibility analysis: {e}")
            return {
                "success": False,
                "message": f"Error analyzing game: {str(e)}"
            }
    
    def detect_hardware(self, params):
        """Provide comprehensive hardware detection with real system information."""
        logging.info("Starting hardware detection with actual system data")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            hardware_specs = loop.run_until_complete(self.canrun_engine.hardware_detector.get_hardware_specs())
            loop.close()
            
            hardware_message = f"""💻 SYSTEM HARDWARE DETECTION:

🖥️ GRAPHICS CARD:
• GPU: {hardware_specs.gpu_model}
• VRAM: {hardware_specs.gpu_vram_gb}GB
• RTX Features: {'✅ Supported' if hardware_specs.supports_rtx else '❌ Not Available'}
• DLSS Support: {'✅ Available' if hardware_specs.supports_dlss else '❌ Not Available'}
• Driver Status: {'✅ Compatible' if hardware_specs.nvidia_driver_version != 'Unknown' else '⚠️ Unknown Version'}

🧠 PROCESSOR:
• CPU: {hardware_specs.cpu_model}
• Cores: {hardware_specs.cpu_cores} Physical / {hardware_specs.cpu_threads} Logical
• Performance: {'✅ High-Performance' if hardware_specs.cpu_cores >= 6 else '⚠️ Mid-Range'}

💾 MEMORY:
• RAM: {hardware_specs.ram_total_gb}GB Total
• Speed: {hardware_specs.ram_speed_mhz}MHz
• Gaming Performance: {'✅ Excellent' if hardware_specs.ram_total_gb >= 16 else '⚠️ Adequate' if hardware_specs.ram_total_gb >= 8 else '❌ Below Recommended'}

🖥️ DISPLAY:
• Resolution: {hardware_specs.primary_monitor_resolution}
• Refresh Rate: {hardware_specs.primary_monitor_refresh_hz}Hz
• G-Sync/FreeSync: {'✅ Likely Supported' if hardware_specs.supports_rtx else '⚠️ Check Monitor Settings'}

💾 STORAGE:
• Type: {hardware_specs.storage_type}
• Performance: {'✅ Fast Loading' if 'SSD' in hardware_specs.storage_type else '⚠️ Standard'}

🖥️ SYSTEM:
• OS: {hardware_specs.os_version}
• DirectX: {hardware_specs.directx_version}
• G-Assist: ✅ Compatible (Plugin Working)

Hardware detection completed successfully using CanRun's privacy-aware detection system."""

            return {
                "success": True,
                "message": hardware_message
            }
        except Exception as e:
            logging.error(f"Error in hardware detection: {e}")
            return {
                "success": False,
                "message": f"Hardware detection failed: {str(e)}\n\nPlease check system compatibility and try again."
            }
    
    def format_canrun_response(self, result):
        """Format CanRun result for G-Assist display with complete information."""
        try:
            # Extract performance tier and score
            tier = result.performance_prediction.tier.name if hasattr(result.performance_prediction, 'tier') else 'Unknown'
            score = int(result.performance_prediction.score) if hasattr(result.performance_prediction, 'score') else 0
            
            # Get compatibility status
            can_run = "✅ CAN RUN" if result.can_run_game() else "❌ CANNOT RUN"
            exceeds_recommended = result.exceeds_recommended_requirements()
            
            # Format comprehensive response
            original_query = result.game_name
            matched_name = result.game_requirements.game_name
            
            # Get actual Steam API game name if available
            steam_api_name = result.game_requirements.steam_api_name if hasattr(result.game_requirements, 'steam_api_name') and result.game_requirements.steam_api_name else matched_name
            
            # Determine if game name was matched differently from user query
            steam_api_info = ""
            if original_query.lower() != steam_api_name.lower():
                steam_api_info = f"(Steam found: {steam_api_name})"
            
            title_line = ""
            if result.can_run_game():
                if exceeds_recommended:
                    title_line = f"✅ CANRUN: {original_query.upper()} will run EXCELLENTLY {steam_api_info}!"
                else:
                    title_line = f"✅ CANRUN: {original_query.upper()} will run {steam_api_info}!"
            else:
                title_line = f"❌ CANNOT RUN {original_query.upper()} {steam_api_info}!"

            response = f"""{title_line}

🎮 YOUR SEARCH: {original_query}
🎮 STEAM MATCHED GAME: {steam_api_name}

🏆 PERFORMANCE TIER: {tier} ({score}/100)

💻 SYSTEM SPECIFICATIONS:
• CPU: {result.hardware_specs.cpu_model}
• GPU: {result.hardware_specs.gpu_model} ({result.hardware_specs.gpu_vram_gb}GB VRAM)
• RAM: {result.hardware_specs.ram_total_gb}GB
• RTX Features: {'✅ Supported' if result.hardware_specs.supports_rtx else '❌ Not Available'}
• DLSS Support: {'✅ Available' if result.hardware_specs.supports_dlss else '❌ Not Available'}

🎯 GAME REQUIREMENTS:
• Minimum GPU: {result.game_requirements.minimum_gpu}
• Recommended GPU: {result.game_requirements.recommended_gpu}
• RAM Required: {result.game_requirements.minimum_ram_gb}GB (Min) / {result.game_requirements.recommended_ram_gb}GB (Rec)
• VRAM Required: {result.game_requirements.minimum_vram_gb}GB (Min) / {result.game_requirements.recommended_vram_gb}GB (Rec)

⚡ PERFORMANCE PREDICTION:
• Expected FPS: {getattr(result.performance_prediction, 'expected_fps', 'Unknown')}
• Recommended Settings: {getattr(result.performance_prediction, 'recommended_settings', 'Unknown')}
• Optimal Resolution: {getattr(result.performance_prediction, 'recommended_resolution', 'Unknown')}
• Performance Level: {'Exceeds Recommended' if exceeds_recommended else 'Meets Minimum' if result.can_run_game() else 'Below Minimum'}

🔧 OPTIMIZATION SUGGESTIONS:"""

            # Add optimization suggestions
            if hasattr(result.performance_prediction, 'upgrade_suggestions'):
                suggestions = result.performance_prediction.upgrade_suggestions[:3]
                for suggestion in suggestions:
                    response += f"\n• {suggestion}"
            else:
                response += "\n• Update GPU drivers for optimal performance"
                if result.hardware_specs.supports_dlss:
                    response += "\n• Enable DLSS for significant performance boost"
                if result.hardware_specs.supports_rtx:
                    response += "\n• Consider RTX features for enhanced visuals"

            # Add compatibility analysis
            if hasattr(result, 'compatibility_analysis') and result.compatibility_analysis:
                if hasattr(result.compatibility_analysis, 'bottlenecks') and result.compatibility_analysis.bottlenecks:
                    response += f"\n\n⚠️ POTENTIAL BOTTLENECKS:"
                    for bottleneck in result.compatibility_analysis.bottlenecks[:2]:
                        response += f"\n• {bottleneck.value}"

            # Add final verdict
            response += f"\n\n🎯 CANRUN VERDICT: {can_run}"
            
            # Make it clear if the Steam API returned something different than what was requested
            if steam_api_name.lower() != original_query.lower():
                response += f"\n\n🎮 NOTE: Steam found '{steam_api_name}' instead of '{original_query}'"
                response += f"\n    Results shown are for '{steam_api_name}'"
            
            return response
            
        except Exception as e:
            logging.error(f"Error formatting CanRun response: {e}")
            return f"🎮 CANRUN ANALYSIS: {getattr(result, 'game_name', 'Unknown Game')}\n\n✅ Analysis completed but formatting error occurred.\nRaw result available in logs."

async def handle_natural_language_query(query):
    """Handle natural language queries like 'canrun game?' and return formatted result."""
    # Extract game name from query
    game_name = query.strip()
    
    # Remove leading command patterns
    patterns = ["canrun ", "can run ", "can i run "]
    for pattern in patterns:
        if game_name.lower().startswith(pattern):
            game_name = game_name[len(pattern):].strip()
            break
    
    # Remove trailing question mark if present
    if game_name and game_name.endswith("?"):
        game_name = game_name[:-1].strip()
    
    if not game_name:
        return "Please specify a game name after 'canrun'."
    
    # Initialize plugin
    plugin = CanRunGAssistPlugin()
    
    # Use the same logic as in app.py for fresh analysis
    has_number = any(c.isdigit() for c in game_name)
    force_refresh = has_number  # Force refresh for numbered games
    
    # Create params
    params = {"game_name": game_name, "force_refresh": force_refresh}
    
    # Execute compatibility check
    response = await plugin.check_game_compatibility(params)
    
    # Return the formatted message (same as what Gradio would display)
    if response.get("success", False):
        return response.get("message", "Analysis completed successfully.")
    else:
        return response.get("message", f"Could not analyze game: {game_name}. Please check the game name and try again.")

def main():
    """Main plugin execution loop - Official NVIDIA Implementation."""
    setup_logging()
    logging.info("CanRun Plugin Started")
    
    # Filter out multiprocessing arguments that G-Assist may pass
    filtered_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('--multiprocessing-fork'):
            skip_next = False  # This arg has the value embedded
            continue
        if arg in ['parent_pid', 'pipe_handle'] and i > 1:
            skip_next = True  # Skip the next argument which is the value
            continue
        if arg.startswith('parent_pid=') or arg.startswith('pipe_handle='):
            continue  # Skip embedded values
        filtered_args.append(arg)
    
    logging.info(f"Original args: {sys.argv[1:]}")
    logging.info(f"Filtered args: {filtered_args}")
    
    # Check if command line arguments were provided (after filtering)
    if len(filtered_args) > 0:
        # Force UTF-8 encoding for stdout to handle emojis
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        
        # Handle command-line arguments in "canrun game?" format
        args = filtered_args
        
        # Process query
        query = " ".join(args)
        game_query = ""
        
        # Check if the query matches our expected format "canrun game?"
        # This will handle both "canrun game?" and just "game?"
        if args[0].lower() == "canrun" and len(args) > 1:
            # Extract just the game name after "canrun"
            game_query = " ".join(args[1:])
        elif query.lower().startswith("canrun "):
            # Handle case where "canrun" might be part of a single argument
            game_query = query[7:].strip()
        else:
            # Assume the entire query is the game name
            game_query = query
        
        # Always remove question mark from the end for processing
        game_query = game_query.rstrip("?").strip()
        
        # Debugging output to help troubleshoot argument issues
        logging.info(f"Command line args: {args}")
        logging.info(f"Processed game query: {game_query}")
        
        # Create event loop for async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the query and print result directly to stdout
        result = loop.run_until_complete(handle_natural_language_query(game_query))
        print(result)
        loop.close()
        return
    
    # Check if running in G-Assist environment
    in_g_assist = is_g_assist_environment()
    logging.info(f"Running in G-Assist environment: {in_g_assist}")
    
    # Initialize plugin - CanRun engine always available
    plugin = CanRunGAssistPlugin()
    logging.info("CanRun plugin initialized successfully")
    
    # If not in G-Assist environment, exit - we only care about G-Assist mode
    if not in_g_assist:
        print("This is a G-Assist plugin. Please run through G-Assist.")
        return
    
    # G-Assist protocol mode
    while True:
        command = read_command()
        if command == "EOF":
            logging.info("EOF received, exiting")
            break
        if command is None:
            continue
        
        # Handle G-Assist input in different formats
        if "tool_calls" in command:
            # Standard G-Assist protocol format with tool_calls
            for tool_call in command.get("tool_calls", []):
                func = tool_call.get("func")
                params = tool_call.get("params", {})
                
                if func == "initialize":
                    # Handle plugin initialization
                    write_response({
                        "success": True,
                        "message": "CanRun G-Assist plugin initialized successfully"
                    })
                elif func == "canrun":
                    # Handle canrun function for game compatibility checks
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(plugin.check_game_compatibility(params))
                    write_response(response)
                    loop.close()
                elif func == "check_compatibility":
                    # For async function, we need to run in an event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(plugin.check_game_compatibility(params))
                    write_response(response)
                    loop.close()
                elif func == "detect_hardware":
                    response = plugin.detect_hardware(params)
                    write_response(response)
                elif func == "auto_detect":
                    # Handle natural language input like "canrun game?"
                    user_input = params.get("user_input", "")
                    logging.info(f"Auto-detect received: {user_input}")
                    
                    # Extract game name from queries like "canrun game?"
                    game_name = user_input
                    if "canrun" in user_input.lower():
                        # Remove "canrun" prefix and extract game name
                        parts = user_input.lower().split("canrun")
                        if len(parts) > 1:
                            game_name = parts[1].strip()
                    
                    # Remove question mark if present
                    game_name = game_name.rstrip("?").strip()
                    
                    if game_name:
                        # Create compatibility check params
                        compat_params = {"game_name": game_name}
                        
                        # For async function, we need to run in an event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(plugin.check_game_compatibility(compat_params))
                        write_response(response)
                        loop.close()
                    else:
                        write_response({
                            "success": False,
                            "message": "Could not identify a game name in your query. Please try 'Can I run <game name>?'"
                        })
                elif func == "shutdown":
                    logging.info("Shutdown command received. Exiting.")
                    write_response({
                        "success": True,
                        "message": "CanRun G-Assist plugin shutdown complete"
                    })
                    return
                else:
                    logging.warning(f"Unknown function: {func}")
                    write_response({
                        "success": False,
                        "message": f"Unknown function: {func}"
                    })
        elif "user_input" in command:
            # Alternative format with direct user_input field
            user_input = command.get("user_input", "")
            logging.info(f"Direct user input received: {user_input}")
            
            # Check if this is a game compatibility query
            if "canrun" in user_input.lower() or "can run" in user_input.lower() or "can i run" in user_input.lower():
                # Extract game name
                game_name = ""
                for prefix in ["canrun ", "can run ", "can i run "]:
                    if user_input.lower().startswith(prefix):
                        game_name = user_input[len(prefix):].strip()
                        break
                
                # If no prefix found but contains "canrun" somewhere
                if not game_name and "canrun" in user_input.lower():
                    parts = user_input.lower().split("canrun")
                    if len(parts) > 1:
                        game_name = parts[1].strip()
                
                # Remove question mark if present
                game_name = game_name.rstrip("?").strip()
                
                if game_name:
                    # Create compatibility check params
                    compat_params = {"game_name": game_name}
                    
                    # For async function, we need to run in an event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(plugin.check_game_compatibility(compat_params))
                    write_response(response)
                    loop.close()
                else:
                    write_response({
                        "success": False,
                        "message": "Could not identify a game name in your query. Please try 'Can I run <game name>?'"
                    })
            else:
                # Not a game compatibility query
                write_response({
                    "success": False,
                    "message": "I can check if your system can run games. Try asking 'Can I run <game name>?'"
                })

if __name__ == "__main__":
    main()