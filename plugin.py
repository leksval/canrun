# SPDX-License-Identifier: Apache-2.0

"""
CanRun G-Assist Plugin
NVIDIA G-Assist plugin for game compatibility checking with Windows pipes communication
Supports both G-Assist pipe mode and CLI mode for testing
"""

import json
import logging
import os
import asyncio
import platform
import sys
import argparse
from ctypes import byref, windll, wintypes
from typing import Optional, Dict, Any

# Import CanRun components - flat structure for G-Assist compliance
try:
    from canrun_engine import CanRunEngine
    CANRUN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CanRun engine not available: {e}")
    CANRUN_AVAILABLE = False

# Import G-Assist response fixer
try:
    from g_assist_response_fixer import (
        clean_ascii_text,
        create_safe_g_assist_response,
        format_g_assist_message,
        validate_g_assist_response
    )
    RESPONSE_FIXER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"G-Assist response fixer not available: {e}")
    RESPONSE_FIXER_AVAILABLE = False

# Data Types
type Response = dict[bool, Optional[str]]

LOG_FILE = os.path.join(os.environ.get("USERPROFILE", "."), 'canrun_plugin.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_command() -> dict | None:
    """Reads a command from the communication pipe - ROBUST WITH ENHANCED LIMITS."""
    try:
        STD_INPUT_HANDLE = -10
        pipe = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
        chunks = []
        total_bytes = 0
        MAX_INPUT_SIZE = 1024 * 1024  # 1MB limit for safety

        while total_bytes < MAX_INPUT_SIZE:
            BUFFER_SIZE = 8192  # Increased buffer size for better performance
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
                logging.error('Error reading from command pipe')
                return None

            bytes_read = message_bytes.value
            if bytes_read == 0:
                break

            # Decode with error handling for robust character support
            try:
                chunk = buffer[:bytes_read].decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                # Fallback to latin-1 for maximum compatibility
                chunk = buffer[:bytes_read].decode('latin-1', errors='replace')
            
            chunks.append(chunk)
            total_bytes += bytes_read

            # If we read less than the buffer size, we're done
            if bytes_read < BUFFER_SIZE:
                break

        if total_bytes >= MAX_INPUT_SIZE:
            logging.warning(f'Input truncated at {MAX_INPUT_SIZE} bytes')

        retval = ''.join(chunks).strip()
        
        # Enhanced JSON parsing with robust validation
        if not retval:
            logging.debug('Empty input received from G-Assist')
            return None
        
        # Log input size for monitoring
        if len(retval) > 1000:
            logging.info(f'Large input received: {len(retval)} characters')
            
        # Handle potential G-Assist protocol variations with size limits
        if len(retval) > 100000:  # 100KB limit for JSON
            logging.error(f'Input too large for JSON parsing: {len(retval)} characters')
            return None
            
        if retval.startswith('{') and retval.endswith('}'):
            try:
                parsed = json.loads(retval)
                logging.debug(f'Successfully parsed JSON command: {list(parsed.keys())}')
                return parsed
            except json.JSONDecodeError as e:
                logging.error(f'JSON decode error at position {e.pos}: {e.msg}')
                logging.error(f'Input preview: {repr(retval[:500])}...')
                return None
            except Exception as e:
                logging.error(f'Unexpected JSON parsing error: {str(e)}')
                return None
        else:
            # Handle non-JSON input with expanded command detection
            logging.info(f'Non-JSON input received: {repr(retval[:200])}')
            lower_input = retval.lower()
            
            # Expanded shutdown command detection
            shutdown_keywords = ['shutdown', 'exit', 'quit', 'stop', 'terminate', 'close']
            if any(keyword in lower_input for keyword in shutdown_keywords):
                logging.info('Shutdown command detected')
                return {'command': 'shutdown'}
            
            # Handle potential binary or corrupted data
            if len(retval) > 0 and all(ord(c) < 32 or ord(c) > 126 for c in retval[:50]):
                logging.warning('Binary or corrupted data received, ignoring')
                return None
                
            return None

    except Exception as e:
        logging.error(f'Critical error in read_command: {str(e)}')
        return None

def write_response(response: Response) -> None:
    """Write response to communication pipe - IMPROVED WITH G-ASSIST FIXES"""
    try:
        STD_OUTPUT_HANDLE = -11
        pipe = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

        # Use the response fixer if available, otherwise fallback to old method
        if RESPONSE_FIXER_AVAILABLE:
            # Validate and format the response properly
            is_valid, error = validate_g_assist_response(response)
            if not is_valid:
                logging.error(f"Invalid response format: {error}")
                response = create_safe_g_assist_response(False, f"Format error: {error}")
            
            json_message = format_g_assist_message(response)
        else:
            # Fallback to old method with basic safety
            if isinstance(response, dict) and 'message' in response:
                # Basic ASCII cleaning
                clean_msg = ''.join(char for char in response['message'] if ord(char) < 128)
                response['message'] = clean_msg
            json_message = json.dumps(response) + "<<END>>"

        message_bytes = json_message.encode('utf-8')
        message_len = len(message_bytes)

        bytes_written = wintypes.DWORD()
        success = windll.kernel32.WriteFile(
            pipe,
            message_bytes,
            message_len,
            byref(bytes_written),
            None
        )
        
        # Force flush the pipe buffer - CRITICAL FOR G-ASSIST
        windll.kernel32.FlushFileBuffers(pipe)
        
        if success:
            logging.info(f'Response sent successfully: {len(json_message)} characters')
        else:
            logging.error(f'WriteFile failed, bytes written: {bytes_written.value}')

    except Exception as e:
        logging.error(f'Failed to write response: {str(e)}')
        # Try emergency fallback response
        try:
            emergency_response = '{"success":false,"message":"Communication error"}<<END>>'
            emergency_bytes = emergency_response.encode('ascii')
            windll.kernel32.WriteFile(
                pipe, emergency_bytes, len(emergency_bytes), byref(wintypes.DWORD()), None
            )
            windll.kernel32.FlushFileBuffers(pipe)
        except:
            pass  # Ultimate fallback - just log and continue

def generate_failure_response(message: str = None) -> Response:
    """Generates a response indicating failure - G-ASSIST SAFE VERSION."""
    if RESPONSE_FIXER_AVAILABLE:
        return create_safe_g_assist_response(False, message)
    else:
        response = {'success': False}
        if message:
            # Basic ASCII cleaning fallback
            clean_msg = ''.join(char for char in str(message) if ord(char) < 128)
            response['message'] = clean_msg[:500]  # Limit length
        return response

def generate_success_response(message: str = None) -> Response:
    """Generates a response indicating success - G-ASSIST SAFE VERSION."""
    if RESPONSE_FIXER_AVAILABLE:
        return create_safe_g_assist_response(True, message)
    else:
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
            logging.error(f"Failed to initialize CanRun engine: {e}")
            canrun_engine = None
            return False
    return canrun_engine is not None

async def execute_canrun_command(params: dict = None, context: dict = None, system_info: dict = None) -> dict:
    """Command handler for canrun function - game compatibility analysis."""
    if not canrun_engine:
        return generate_failure_response("CanRun engine not available")
    
    if not params:
        return generate_failure_response("Game name is required for CanRun analysis")
    
    game_name = params.get("game_name", "").strip()
    if not game_name:
        return generate_failure_response("Game name is required for CanRun analysis")
    
    force_refresh = params.get("force_refresh", False)
    if isinstance(force_refresh, str):
        force_refresh = force_refresh.lower() == "true"
    
    logging.info(f"Starting CanRun analysis for: {game_name} (force_refresh: {force_refresh})")
    
    try:
        result = await canrun_engine.check_game_compatibility(game_name, use_cache=not force_refresh)
        
        if result:
            formatted_result = format_canrun_response(result)
            return generate_success_response(formatted_result)
        else:
            return generate_failure_response(f"Could not analyze game: {game_name}. Please check the game name and try again.")
            
    except Exception as e:
        logging.error(f"Error in game compatibility analysis: {e}")
        return generate_failure_response(f"Error analyzing game: {str(e)}")

def format_canrun_response(result):
    """Format CanRun result for G-Assist display with proper line breaks."""
    try:
        # Extract key information
        tier = result.performance_prediction.tier.name if hasattr(result.performance_prediction, 'tier') else 'Unknown'
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
        
        # Display FPS as range if available, otherwise single value or unknown
        if fps_min and fps_max and fps_min != fps_max:
            fps = f"{fps_min}-{fps_max} FPS"
        elif expected_fps and isinstance(expected_fps, (int, float)) and expected_fps > 0:
            fps = f"~{int(expected_fps)} FPS"
        else:
            fps = "Unknown"
            
        settings = getattr(result.performance_prediction, 'recommended_settings', 'Unknown')
        recommended_resolution = getattr(result.performance_prediction, 'recommended_resolution', 'Unknown')
        
        # Get actual resolution from hardware specs
        actual_resolution = result.hardware_specs.primary_monitor_resolution
        
        # Build response with explicit line breaks - v7.5.0 approach with consistent bolding
        message = f"""
🎮 **Game:** {steam_api_name}

💻 **YOUR SYSTEM**
• **GPU:** {result.hardware_specs.gpu_model} ({result.hardware_specs.gpu_vram_gb}GB)
• **CPU:** {result.hardware_specs.cpu_model}
• **RAM:** {result.hardware_specs.ram_total_gb}GB
• **Display:** {actual_resolution} @ **{result.hardware_specs.primary_monitor_refresh_hz}Hz**

🎯 **GAME REQUIREMENTS**"""
        
        if exceeds_recommended:
            # System exceeds recommended - only show recommended requirements
            message += f"""
**Recommended** (Your system exceeds these)
• **GPU:** {result.game_requirements.recommended_gpu}"""
            if hasattr(result.game_requirements, 'recommended_cpu') and result.game_requirements.recommended_cpu:
                message += f"""
• **CPU:** {result.game_requirements.recommended_cpu}"""
            message += f"""
• **RAM:** {result.game_requirements.recommended_ram_gb}GB"""
        else:
            # System doesn't exceed recommended - show both minimal and recommended
            message += f"""
**Minimum**
• **GPU:** {result.game_requirements.minimum_gpu}"""
            if hasattr(result.game_requirements, 'minimum_cpu') and result.game_requirements.minimum_cpu:
                message += f"""
• **CPU:** {result.game_requirements.minimum_cpu}"""
            message += f"""
• **RAM:** {result.game_requirements.minimum_ram_gb}GB

**Recommended**
• **GPU:** {result.game_requirements.recommended_gpu}"""
            if hasattr(result.game_requirements, 'recommended_cpu') and result.game_requirements.recommended_cpu:
                message += f"""
• **CPU:** {result.game_requirements.recommended_cpu}"""
            message += f"""
• **RAM:** {result.game_requirements.recommended_ram_gb}GB"""
        
        # 3. EXPECTED PERFORMANCE
        message += f"""

⚡ **PERFORMANCE**
• **FPS:** {fps}
• **Settings:** {settings}
• **Resolution:** {recommended_resolution}
• **Score:** {tier} Tier ({score}/100)

🔧 **OPTIMIZE**"""
        
        if hasattr(result.performance_prediction, 'upgrade_suggestions') and result.performance_prediction.upgrade_suggestions:
            for suggestion in result.performance_prediction.upgrade_suggestions[:2]:
                message += f"""
• {suggestion}"""
        else:
            if can_run:
                if result.hardware_specs.supports_dlss:
                    message += """
• Enable DLSS Quality mode for higher framerates"""
                if result.hardware_specs.supports_rtx:
                    message += """
• Enable RTX ray tracing for enhanced visual quality"""
                message += """
• Update GPU drivers"""
            else:
                message += """
• Upgrade GPU to meet minimum requirements
• Check RAM meets minimum requirements"""
        
        # Add note if different game was found
        if steam_api_name.lower() != original_query.lower():
            message += f"""

📝 Note: Showing results for {steam_api_name}"""
        
        # 5. VERDICT - At bottom for visibility when scrolling!
        message += """

"""
        if can_run:
            if exceeds_recommended:
                message += "🎯 VERDICT: ✅ CAN RUN - EXCELLENT PERFORMANCE!"
            else:
                message += "🎯 VERDICT: ✅ CAN RUN"
        else:
            message += "🎯 **VERDICT:** ❌ CANNOT RUN"
        
        return message.strip()
        
    except Exception as e:
        logging.error(f"Error formatting CanRun response: {e}")
        return f"""🎮 CANRUN ANALYSIS: {getattr(result, 'game_name', 'Unknown Game')}

✅ Analysis completed but formatting error occurred.
Raw result available in logs."""

def execute_initialize_command() -> dict:
    """Command handler for initialize function."""
    logging.info('Initializing CanRun plugin')
    success = initialize_canrun_engine()
    if success:
        return generate_success_response('CanRun plugin initialized successfully')
    else:
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
        if canrun_engine:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            hw_specs = loop.run_until_complete(canrun_engine.hardware_detector.get_hardware_specs())
            loop.close()
            result = generate_success_response(f"Hardware detected: GPU={hw_specs.gpu_model}, CPU={hw_specs.cpu_model}, RAM={hw_specs.ram_total_gb}GB")
        else:
            result = generate_failure_response("CanRun engine not available")
    elif args.command == 'canrun':
        if not args.game_name:
            result = generate_failure_response("Game name is required for canrun command")
        else:
            params = {"game_name": args.game_name, "force_refresh": args.force_refresh}
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(execute_canrun_command(params))
            loop.close()
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

def main():
    """Main plugin execution loop - OFFICIAL NVIDIA IMPLEMENTATION"""
    LOG_FILE = os.path.join(os.environ.get("USERPROFILE", "."), 'canrun_plugin.log')
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("CanRun Plugin Started")
    
    # Check if command line arguments were provided (CLI mode)
    if len(sys.argv) > 1:
        return cli_mode()
    
    # Check if running in G-Assist environment
    in_g_assist = not sys.stdin.isatty()
    logging.info(f"Running in G-Assist environment: {in_g_assist}")
    
    # Initialize plugin
    if not initialize_canrun_engine():
        logging.error("Failed to initialize CanRun engine")
        return 1
    
    logging.info("CanRun plugin initialized successfully")
    
    # If not in G-Assist environment, exit - we only care about G-Assist mode
    if not in_g_assist:
        print("This is a G-Assist plugin. Please run through G-Assist.")
        return
    
    # G-Assist protocol mode
    consecutive_failures = 0
    max_failures = 5  # Prevent infinite loop on persistent errors
    
    while consecutive_failures < max_failures:
        command = read_command()
        if command is None:
            consecutive_failures += 1
            logging.warning(f"Failed to read command (attempt {consecutive_failures}/{max_failures})")
            
            # Brief pause to prevent CPU spinning
            import time
            time.sleep(0.1)
            continue
        
        # Reset failure counter on successful read
        consecutive_failures = 0
        
        # Handle shutdown command
        if isinstance(command, dict) and command.get('command') == 'shutdown':
            logging.info("Shutdown command received. Exiting gracefully.")
            return 0
        
        # Handle G-Assist input in different formats
        if "tool_calls" in command:
            # Standard G-Assist protocol format with tool_calls
            for tool_call in command.get("tool_calls", []):
                func = tool_call.get("func")
                params = tool_call.get("params", {})
                
                if func == "check_compatibility":
                    # For async function, we need to run in an event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(execute_canrun_command(params))
                    write_response(response)
                    loop.close()
                elif func == "detect_hardware":
                    # Handle hardware detection requests
                    if canrun_engine:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        hw_specs = loop.run_until_complete(canrun_engine.hardware_detector.get_hardware_specs())
                        loop.close()
                        
                        # Format hardware message similar to v7.0.0
                        hardware_message = f"""SYSTEM HARDWARE DETECTION:

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
                        
                        response = generate_success_response(hardware_message)
                        write_response(response)
                    else:
                        response = generate_failure_response("CanRun engine not available")
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
                        response = loop.run_until_complete(execute_canrun_command(compat_params))
                        write_response(response)
                        loop.close()
                    else:
                        write_response(generate_failure_response("Could not identify a game name in your query. Please try 'Can I run <game name>?'"))
                elif func == "initialize":
                    # Handle initialize command
                    response = execute_initialize_command()
                    write_response(response)
                elif func == "shutdown":
                    logging.info("Shutdown command received. Exiting.")
                    return
                else:
                    logging.warning(f"Unknown function: {func}")
                    write_response(generate_failure_response(f"Unknown function: {func}"))
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
                    response = loop.run_until_complete(execute_canrun_command(compat_params))
                    write_response(response)
                    loop.close()
                else:
                    write_response(generate_failure_response("Could not identify a game name in your query. Please try 'Can I run <game name>?'"))
            else:
                # Not a game compatibility query
                write_response(generate_failure_response("I can check if your system can run games. Try asking 'Can I run <game name>?'"))

class CanRunGAssistPlugin:
    """G-Assist Plugin class for MCP server integration."""
    
    def __init__(self):
        """Initialize the plugin."""
        self.engine = None
        initialize_canrun_engine()
        
    async def check_game_compatibility(self, params: dict) -> dict:
        """Check game compatibility - wrapper for MCP server."""
        return await execute_canrun_command(params)
    
    def detect_hardware(self, params: dict) -> dict:
        """Detect hardware - placeholder for MCP server."""
        try:
            if canrun_engine:
                # For now, return success message
                return generate_success_response("Hardware detection functionality available")
            else:
                return generate_failure_response("CanRun engine not available")
        except Exception as e:
            return generate_failure_response(f"Hardware detection error: {str(e)}")

if __name__ == '__main__':
    sys.exit(main())