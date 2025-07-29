#!/usr/bin/env python3
"""
CanRun G-Assist Plugin - Clean Implementation
Based on working minimal plugin communication with full CanRun engine integration
Follows official NVIDIA G-Assist plugin standards
"""

import sys
import json
import os
import logging
import asyncio
import platform
from typing import Optional, Dict, Any
from pathlib import Path

# Windows pipe communication - only import on Windows
if platform.system() == "Windows":
    from ctypes import byref, windll, wintypes

# Add src directory to path for CanRun engine imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import existing tested CanRun components
try:
    from canrun_engine import CanRunEngine
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
    """Setup UTF-8 logging to prevent encoding errors."""
    # Force UTF-8 encoding for all streams
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    
    log_file = os.path.join(os.environ.get("USERPROFILE" if IS_WINDOWS else "HOME", "."), 'canrun_plugin.log')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding='utf-8'
    )

def robust_json_parse(data):
    """Ultra-robust JSON parser that handles G-Assist's complex JSON with extra data."""
    if not data:
        return None
    
    try:
        # First, try standard JSON parsing
        return json.loads(data)
    except json.JSONDecodeError:
        pass
    
    # If that fails, try to extract JSON from the data
    data = data.strip()
    
    # Look for JSON object boundaries
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
                # Found complete JSON object
                json_str = data[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
    
    # If no valid JSON found, return None
    logging.warning(f"Could not parse JSON from data: {data[:100]}...")
    return None

def read_command():
    """Read command from Windows named pipe with UTF-8 encoding."""
    if not IS_WINDOWS:
        # Fallback for non-Windows
        try:
            line = sys.stdin.readline()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    
    try:
        pipe = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
        chunks = []
        
        while True:
            message_bytes = wintypes.DWORD()
            buffer = bytes(BUFFER_SIZE)
            success = windll.kernel32.ReadFile(
                pipe,
                buffer,
                BUFFER_SIZE,
                byref(message_bytes),
                None
            )

            if not success:
                error = windll.kernel32.GetLastError()
                if error == 109:  # ERROR_BROKEN_PIPE
                    logging.info('Pipe closed by G-Assist, exiting gracefully')
                    return None
                logging.error(f'ReadFile failed with error: {error}')
                return None

            if message_bytes.value == 0:
                logging.info('No data read from pipe, assuming closed')
                return None

            chunk = buffer.decode('utf-8', errors='ignore')[:message_bytes.value]
            chunks.append(chunk)

            if message_bytes.value < BUFFER_SIZE:
                break

        retval = ''.join(chunks)
        logging.info(f'Raw Input: {retval[:200]}...')
        return robust_json_parse(retval)
        
    except Exception as e:
        logging.error(f'Exception in read_command(): {e}')
        return None

def write_response(response):
    """Write response to Windows named pipe with UTF-8 encoding."""
    if not IS_WINDOWS:
        # Fallback for non-Windows
        try:
            message = json.dumps(response) + '<<END>>'
            sys.stdout.write(message)
            sys.stdout.flush()
        except Exception as e:
            logging.error(f'Error writing response: {e}')
        return
    
    try:
        pipe = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        json_message = json.dumps(response) + '<<END>>'
        message_bytes = json_message.encode('utf-8')
        
        bytes_written = wintypes.DWORD()
        windll.kernel32.WriteFile(
            pipe,
            message_bytes,
            len(message_bytes),
            bytes_written,
            None
        )
        logging.info(f'Response sent: {len(message_bytes)} bytes')
    except Exception as e:
        logging.error(f'Error writing response: {e}')

class CanRunPlugin:
    """Clean CanRun G-Assist plugin implementation."""
    
    def __init__(self):
        """Initialize CanRun plugin with engine integration."""
        self.engine = None
        
        if CANRUN_AVAILABLE:
            try:
                # Check if running in G-Assist environment vs test mode
                is_g_assist_env = len(sys.argv) == 1  # G-Assist calls with no args
                
                # Initialize CanRun engine with conditional LLM based on environment
                self.engine = CanRunEngine(
                    cache_dir=os.path.join(os.path.dirname(__file__), "cache"),
                    enable_llm=is_g_assist_env  # Only enable LLM in G-Assist environment
                )
                logging.info("CanRun engine initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize CanRun engine: {e}")
                self.engine = None
        else:
            logging.warning("CanRun engine not available")
    
    async def canrun(self, params):
        """Main CanRun function - check game compatibility."""
        game_name = params.get("game_name", params.get("game", "")).strip()
        
        if not game_name:
            return {
                "success": False,
                "message": "Game name is required for CanRun analysis"
            }
        
        logging.info(f"CANRUN: Checking compatibility for {game_name}")
        
        try:
            if not self.engine:
                return {
                    "success": False,
                    "message": f"CanRun engine not available. Cannot analyze {game_name}."
                }
            
            # Use the same CanRun engine logic as the working root plugin
            result = await self.engine.check_game_compatibility(game_name, use_cache=True)
            
            if result:
                # Use the same formatting logic as the working root plugin
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
    
    def format_canrun_response(self, result):
        """Format CanRun result for G-Assist display - verdict at bottom for visibility."""
        try:
            # Extract key information
            tier = result.performance_prediction.tier.name if hasattr(result.performance_prediction, 'tier') else 'Unknown'
            score = int(result.performance_prediction.score) if hasattr(result.performance_prediction, 'score') else 0
            can_run = result.can_run_game()
            exceeds_recommended = result.exceeds_recommended_requirements()
            
            # Get game name
            original_query = result.game_name
            steam_api_name = result.game_requirements.steam_api_name if hasattr(result.game_requirements, 'steam_api_name') and result.game_requirements.steam_api_name else result.game_requirements.game_name
            
            # Get performance details
            fps = getattr(result.performance_prediction, 'expected_fps', 'Unknown')
            settings = getattr(result.performance_prediction, 'recommended_settings', 'Unknown')
            resolution = getattr(result.performance_prediction, 'recommended_resolution', 'Unknown')
            
            # Build response with verdict at bottom
            response_parts = []
            
            # Game name at top
            response_parts.append(f"\t") 
            response_parts.append(f"🎮 **Game:** **{steam_api_name}**\t")

            
            # 1. YOUR SYSTEM
            response_parts.append("💻 **YOUR SYSTEM**")
            response_parts.append(f"• GPU: {result.hardware_specs.gpu_model} ({result.hardware_specs.gpu_vram_gb}GB)")
            response_parts.append(f"• CPU: {result.hardware_specs.cpu_model}")
            response_parts.append(f"• RAM: {result.hardware_specs.ram_total_gb}GB\t")

            
            # 2. GAME REQUIREMENTS
            response_parts.append("🎯 **GAME REQUIREMENTS**")
            
            # Minimum Requirements
            response_parts.append("**Minimum**")
            response_parts.append(f"• GPU: {result.game_requirements.minimum_gpu}")
            if hasattr(result.game_requirements, 'minimum_cpu') and result.game_requirements.minimum_cpu:
                response_parts.append(f"• CPU: {result.game_requirements.minimum_cpu}")
            response_parts.append(f"• RAM: {result.game_requirements.minimum_ram_gb}GB")
            
            # Recommended Requirements
            response_parts.append("**Recommended**")
            response_parts.append(f"• GPU: {result.game_requirements.recommended_gpu}")
            if hasattr(result.game_requirements, 'recommended_cpu') and result.game_requirements.recommended_cpu:
                response_parts.append(f"• CPU: {result.game_requirements.recommended_cpu}")
            response_parts.append(f"• RAM: {result.game_requirements.recommended_ram_gb}GB\t")
     
            
            # 3. EXPECTED PERFORMANCE
            response_parts.append("⚡ **PERFORMANCE**")
            response_parts.append(f"• FPS: **{fps}**")
            response_parts.append(f"• Settings: **{settings}**")
            response_parts.append(f"• Resolution: **{resolution}**")
            response_parts.append(f"• Score: **{tier} Tier ({score}/100)**\t")

            
            # 4. OPTIMIZE
            response_parts.append("🔧 **OPTIMIZE**")
            if hasattr(result.performance_prediction, 'upgrade_suggestions') and result.performance_prediction.upgrade_suggestions:
                for suggestion in result.performance_prediction.upgrade_suggestions[:2]:
                    response_parts.append(f"• {suggestion}")
            else:
                if can_run:
                    if result.hardware_specs.supports_dlss:
                        response_parts.append("• Enable DLSS Quality mode for higher framerates")
                    if result.hardware_specs.supports_rtx:
                        response_parts.append("• Enable RTX ray tracing for enhanced visual quality")
                    response_parts.append("• Update GPU drivers")
                else:
                    response_parts.append("• Upgrade GPU to meet minimum requirements")
                    response_parts.append("• Check RAM meets minimum requirements")
            
            # Add note if different game was found
            if steam_api_name.lower() != original_query.lower():
                response_parts.append(f"📝 Note: Showing results for {steam_api_name}")
            
            # 5. VERDICT - At bottom for visibility when scrolling!
            response_parts.append("\t")
            if can_run:
                if exceeds_recommended:
                    response_parts.append("🎯 VERDICT: ✅ CAN RUN - EXCELLENT PERFORMANCE!")
                else:
                    response_parts.append("🎯 VERDICT: ✅ CAN RUN")
            else:
                response_parts.append("🎯 **VERDICT:** ❌ CANNOT RUN")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logging.error(f"Error formatting CanRun response: {e}")
            return f"🎮 CANRUN ANALYSIS: {getattr(result, 'game_name', 'Unknown Game')}\n\n✅ Analysis completed but formatting error occurred.\nRaw result available in logs."

def main():
    """Main plugin execution loop."""
    setup_logging()
    logging.info("CanRun Plugin Started (Clean Implementation)")
    
    # Initialize plugin
    plugin = CanRunPlugin()
    
    # Check if running in test mode with command line arguments
    if len(sys.argv) > 1:
        # Test mode - handle command line arguments
        if len(sys.argv) >= 3 and sys.argv[1] == "canrun":
            game_name = " ".join(sys.argv[2:])
            logging.info(f"Test mode: Checking {game_name}")
            
            # Run the canrun function directly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(plugin.canrun({"game_name": game_name}))
            loop.close()
            
            # Print response for testing
            print(f"CanRun Test Result: {response}")
            return
        else:
            print("Usage: python plugin.py canrun <game_name>")
            return
    
    # G-Assist mode - Main G-Assist communication loop
    while True:
        command = read_command()
        if command is None:
            logging.info("No command received, pipe likely closed. Exiting gracefully.")
            break
        
        # Handle G-Assist tool_calls format (old format)
        if "tool_calls" in command:
            for tool_call in command.get("tool_calls", []):
                func = tool_call.get("func")
                params = tool_call.get("params", {})
                
                logging.info(f"Processing function: {func} with params: {params}")
                
                if func == "canrun":
                    # For async function, run in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(plugin.canrun(params))
                    write_response(response)
                    loop.close()
                elif func == "initialize":
                    # G-Assist initialization - respond with success
                    logging.info("Plugin initialization requested by G-Assist")
                    write_response({
                        "success": True,
                        "message": "CanRun plugin initialized successfully"
                    })
                elif func == "shutdown":
                    logging.info("Shutdown command received. Exiting.")
                    return
                else:
                    logging.warning(f"Unknown function: {func}")
                    write_response({
                        "success": False,
                        "message": f"Unknown function: {func}"
                    })

if __name__ == "__main__":
    main()