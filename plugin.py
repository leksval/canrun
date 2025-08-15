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
    """Format CanRun result for G-Assist display - IMPROVED WITH RELEVANT DATA."""
    try:
        # Extract key information safely
        game_name = getattr(result, 'game_name', 'Unknown Game')
        can_run = result.can_run_game() if hasattr(result, 'can_run_game') else False
        exceeds_recommended = result.exceeds_recommended_requirements() if hasattr(result, 'exceeds_recommended_requirements') else False
        
        # Get performance info with ML model ranges
        tier = 'Unknown'
        score = 0
        fps_display = 'Unknown'
        settings = 'Unknown'
        
        if hasattr(result, 'performance_prediction'):
            pred = result.performance_prediction
            tier = pred.tier.name if hasattr(pred, 'tier') and hasattr(pred.tier, 'name') else 'Unknown'
            score = int(pred.score) if hasattr(pred, 'score') else 0
            settings = pred.recommended_settings if hasattr(pred, 'recommended_settings') else 'Medium'
            
            # Extract FPS range from ML model data
            if hasattr(pred, 'fps_range') and pred.fps_range:
                # If we have a proper range object
                fps_display = f"{pred.fps_range.min}-{pred.fps_range.max}"
            elif hasattr(pred, 'fps_min') and hasattr(pred, 'fps_max'):
                # If we have separate min/max values
                fps_display = f"{int(pred.fps_min)}-{int(pred.fps_max)}"
            elif hasattr(pred, 'expected_fps_range'):
                # Alternative range format
                fps_display = str(pred.expected_fps_range)
            elif hasattr(pred, 'ml_prediction') and hasattr(pred.ml_prediction, 'fps_range'):
                # Range from ML prediction sub-object
                ml_range = pred.ml_prediction.fps_range
                fps_display = f"{int(ml_range.min)}-{int(ml_range.max)}"
            elif hasattr(pred, 'expected_fps'):
                # Fallback to single value if no range available
                single_fps = int(pred.expected_fps)
                fps_display = f"{single_fps}"
            else:
                fps_display = 'Unknown'
        
        # Get comprehensive system and game info
        gpu_model = 'Unknown GPU'
        matched_game = game_name
        ram_gb = 0
        cpu_model = 'Unknown CPU'
        
        if hasattr(result, 'hardware_specs'):
            hw = result.hardware_specs
            if hasattr(hw, 'gpu_model'):
                gpu_model = str(hw.gpu_model)
            if hasattr(hw, 'ram_total_gb'):
                ram_gb = int(hw.ram_total_gb)
            if hasattr(hw, 'cpu_model'):
                cpu_model = str(hw.cpu_model)
            
        if hasattr(result, 'game_requirements') and hasattr(result.game_requirements, 'game_name'):
            matched_game = str(result.game_requirements.game_name)
        
        # Clean all text of unicode characters
        if RESPONSE_FIXER_AVAILABLE:
            game_name = clean_ascii_text(game_name)
            gpu_model = clean_ascii_text(gpu_model)
            matched_game = clean_ascii_text(matched_game)
            settings = clean_ascii_text(settings)
            cpu_model = clean_ascii_text(cpu_model)
        else:
            # Basic fallback cleaning
            game_name = ''.join(char for char in game_name if ord(char) < 128)
            gpu_model = ''.join(char for char in gpu_model if ord(char) < 128)
            matched_game = ''.join(char for char in matched_game if ord(char) < 128)
            settings = ''.join(char for char in settings if ord(char) < 128)
            cpu_model = ''.join(char for char in cpu_model if ord(char) < 128)
        
        # Extract numeric FPS for verdict assessment (use max of range if available)
        fps_numeric = 0
        if '-' in fps_display:
            # Extract max FPS from range for verdict
            try:
                fps_parts = fps_display.split('-')
                fps_numeric = int(fps_parts[-1])  # Use max value
            except (ValueError, IndexError):
                fps_numeric = 0
        elif fps_display.isdigit():
            fps_numeric = int(fps_display)
        
        # Create meaningful verdict with proper performance assessment
        if can_run:
            if exceeds_recommended and fps_numeric > 120:
                verdict = "EXCELLENT"
                performance_level = "Far exceeds recommended"
            elif exceeds_recommended:
                verdict = "VERY GOOD"
                performance_level = "Exceeds recommended"
            elif fps_numeric >= 60:
                verdict = "GOOD"
                performance_level = "Meets recommended"
            else:
                verdict = "PLAYABLE"
                performance_level = "Meets minimum"
        else:
            verdict = "INSUFFICIENT"
            performance_level = "Below minimum requirements"
        
        # Smart resolution recommendation based on actual performance
        if fps_numeric > 120 and tier in ['S', 'A']:
            resolution_tip = "Perfect for 4K gaming"
        elif fps_numeric > 80 and tier in ['S', 'A', 'B']:
            resolution_tip = "Excellent for 1440p"
        elif fps_numeric >= 60:
            resolution_tip = "Great for 1080p"
        elif fps_numeric >= 30:
            resolution_tip = "Playable at 1080p"
        else:
            resolution_tip = "Consider upgrading hardware"
        
        # Create the exact same working format with ML FPS range data
        response = f"""Game: {matched_game}
Verdict: {verdict} ({performance_level})
Performance: {fps_display} FPS at {settings} settings
GPU: {gpu_model} (Tier {tier})
Recommendation: {resolution_tip}"""
        
        # Ensure length compliance with better fallback
        if len(response) > 280:
            response = f"""Game: {matched_game}
Verdict: {verdict}
Performance: {fps_display} FPS at {settings}
GPU: {gpu_model} (Tier {tier})"""
        
        return response
        
    except Exception as e:
        logging.error(f"Error formatting CanRun response: {e}")
        # Ultra-safe fallback
        game_name = getattr(result, 'game_name', 'Unknown Game')
        if RESPONSE_FIXER_AVAILABLE:
            game_name = clean_ascii_text(game_name)
        else:
            game_name = ''.join(char for char in str(game_name) if ord(char) < 128)
        return f"Analysis complete: {game_name} compatibility checked"

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