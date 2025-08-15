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

# Data Types
type Response = dict[bool, Optional[str]]

LOG_FILE = os.path.join(os.environ.get("USERPROFILE", "."), 'canrun_plugin.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_command() -> dict | None:
    """Reads a command from the communication pipe - NVIDIA Standard Implementation."""
    try:
        STD_INPUT_HANDLE = -10
        pipe = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
        chunks = []

        while True:
            BUFFER_SIZE = 4096
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

            # Add the chunk we read
            chunk = buffer.decode('utf-8')[:message_bytes.value]
            chunks.append(chunk)

            # If we read less than the buffer size, we're done
            if message_bytes.value < BUFFER_SIZE:
                break

        retval = ''.join(chunks)
        return json.loads(retval)

    except json.JSONDecodeError:
        logging.error('Failed to decode JSON input')
        return None
    except Exception as e:
        logging.error(f'Unexpected error in read_command: {str(e)}')
        return None

def write_response(response: Response) -> None:
    """Writes a response to the communication pipe - NVIDIA Standard Implementation."""
    try:
        STD_OUTPUT_HANDLE = -11
        pipe = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

        json_message = json.dumps(response)
        message_bytes = json_message.encode('utf-8')
        message_len = len(message_bytes)

        bytes_written = wintypes.DWORD()
        windll.kernel32.WriteFile(
            pipe,
            message_bytes,
            message_len,
            bytes_written,
            None
        )

    except Exception as e:
        logging.error(f'Failed to write response: {str(e)}')
        pass

def generate_failure_response(message: str = None) -> Response:
    """Generates a response indicating failure."""
    response = {'success': False}
    if message:
        response['message'] = message
    return response

def generate_success_response(message: str = None) -> Response:
    """Generates a response indicating success."""
    response = {'success': True}
    if message:
        response['message'] = message
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
    """Format CanRun result for G-Assist display."""
    try:
        # Extract key information
        tier = result.performance_prediction.tier.name if hasattr(result.performance_prediction, 'tier') else 'Unknown'
        score = int(result.performance_prediction.score) if hasattr(result.performance_prediction, 'score') else 0
        fps = result.performance_prediction.expected_fps if hasattr(result.performance_prediction, 'expected_fps') else 0
        settings = result.performance_prediction.recommended_settings if hasattr(result.performance_prediction, 'recommended_settings') else 'Unknown'
        
        # Get compatibility status
        can_run = result.can_run_game()
        exceeds_recommended = result.exceeds_recommended_requirements()
        
        # Get game information
        game_name = result.game_name
        matched_name = result.game_requirements.game_name
        
        # Get resolution information safely - match v8.0.0 approach
        try:
            current_resolution = getattr(result.hardware_specs, 'primary_monitor_resolution', '1920x1080')
            # Validate that it's a proper resolution string
            if not current_resolution or 'x' not in current_resolution:
                current_resolution = "1920x1080"
        except:
            current_resolution = "1920x1080"
        
        # Determine optimal resolution based on GPU tier
        if tier in ['S', 'A']:
            optimal_resolution = "4K (3840x2160)"
        elif tier in ['B', 'C']:
            optimal_resolution = "1440p (2560x1440)"
        else:
            optimal_resolution = "1080p (1920x1080)"
        
        # Create response
        if can_run:
            header = f"CANRUN: {game_name.upper()} will run {'EXCELLENTLY' if exceeds_recommended else 'WELL'}"
            verdict = "CAN RUN"
        else:
            header = f"CANRUN: {game_name.upper()} CANNOT RUN"
            verdict = "CANNOT RUN"
        
        # Use actual ML prediction variance data
        fps_range_info = ""
        try:
            # Get variance data from ML predictor assessment
            fps_min = getattr(result.performance_prediction, 'fps_min', 0)
            fps_max = getattr(result.performance_prediction, 'fps_max', 0)
            variance_range = getattr(result.performance_prediction, 'fps_variance_range', 0)
            
            # If ML variance data is available, use it
            if fps_min > 0 and fps_max > 0 and variance_range > 0:
                fps_range_info = f"- FPS Range: {fps_min}-{fps_max} at {current_resolution}"
                
                # Add helpful tip based on actual predicted performance
                if fps_min >= 120:
                    fps_range_info += f" (Excellent for high refresh rate)"
                elif fps_min >= 60:
                    fps_range_info += f" (Smooth gameplay)"
                elif fps_max >= 60:
                    fps_range_info += f" (Good performance, some settings adjustments recommended)"
                else:
                    fps_range_info += f" (Consider lowering settings for better performance)"
            else:
                # Fallback to simple display if variance data not available
                fps_range_info = f"- Performance: {fps} FPS at {current_resolution}"
                
        except Exception as e:
            fps_range_info = f"- Performance: {fps} FPS at {current_resolution}"

        # Build detailed response without emojis for NVIDIA compliance
        response = f"""{header}

YOUR SEARCH: {game_name}
STEAM MATCHED GAME: {matched_name}

PERFORMANCE TIER: {tier} ({score}/100)

SYSTEM SPECIFICATIONS:
- CPU: {result.hardware_specs.cpu_model}
- GPU: {result.hardware_specs.gpu_model} ({result.hardware_specs.gpu_vram_gb}GB VRAM)
- RAM: {result.hardware_specs.ram_total_gb}GB
- RTX Features: {'Supported' if result.hardware_specs.supports_rtx else 'Not Available'}
- DLSS Support: {'Available' if result.hardware_specs.supports_dlss else 'Not Available'}

GAME REQUIREMENTS:
- Minimum GPU: {result.game_requirements.minimum_gpu.replace('®', '').replace('™', '')}
- Recommended GPU: {result.game_requirements.recommended_gpu.replace('®', '').replace('™', '')}
- RAM Required: {result.game_requirements.minimum_ram_gb}GB (Min) / {result.game_requirements.recommended_ram_gb}GB (Rec)
- VRAM Required: {result.game_requirements.minimum_vram_gb}GB (Min) / {result.game_requirements.recommended_vram_gb}GB (Rec)

PERFORMANCE PREDICTION:
- Recommended Settings: {settings}
- Current Resolution: {current_resolution}
- Optimal Resolution: {optimal_resolution}
- Performance Level: {'Exceeds Recommended' if exceeds_recommended else 'Meets Minimum'}
{fps_range_info}

CANRUN VERDICT: {verdict}"""
        
        return response
        
    except Exception as e:
        logging.error(f"Error formatting CanRun response: {e}")
        return f"Analysis failed for {getattr(result, 'game_name', 'Unknown Game')}. Please check system compatibility."

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
    """Main entry point - NVIDIA G-Assist Standard Implementation with CLI support."""
    
    # Check if running in CLI mode (has command line arguments)
    if len(sys.argv) > 1:
        # CLI mode - parse arguments and execute directly
        return cli_mode()
    
    # G-Assist pipe mode - original implementation
    TOOL_CALLS_PROPERTY = 'tool_calls'
    CONTEXT_PROPERTY = 'messages'
    SYSTEM_INFO_PROPERTY = 'system_info'
    FUNCTION_PROPERTY = 'func'
    PARAMS_PROPERTY = 'properties'
    INITIALIZE_COMMAND = 'initialize'
    SHUTDOWN_COMMAND = 'shutdown'

    ERROR_MESSAGE = 'CanRun Plugin Error!'

    # Generate command handler mapping
    commands = {
        'initialize': execute_initialize_command,
        'shutdown': execute_shutdown_command,
        'canrun': execute_canrun_command,
    }
    cmd = ''

    logging.info('CanRun Plugin started in G-Assist mode')
    while cmd != SHUTDOWN_COMMAND:
        response = None
        input_data = read_command()
        if input_data is None:
            logging.error('Error reading command')
            continue

        logging.info(f'Received input: {input_data}')
        
        if TOOL_CALLS_PROPERTY in input_data:
            tool_calls = input_data[TOOL_CALLS_PROPERTY]
            for tool_call in tool_calls:
                if FUNCTION_PROPERTY in tool_call:
                    cmd = tool_call[FUNCTION_PROPERTY]
                    logging.info(f'Processing command: {cmd}')
                    if cmd in commands:
                        if cmd == INITIALIZE_COMMAND or cmd == SHUTDOWN_COMMAND:
                            response = commands[cmd]()
                        else:
                            # For CanRun function, run async
                            params = input_data.get(PARAMS_PROPERTY, {})
                            context = input_data.get(CONTEXT_PROPERTY, {})
                            system_info = input_data.get(SYSTEM_INFO_PROPERTY, {})
                            
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            response = loop.run_until_complete(commands[cmd](params, context, system_info))
                            loop.close()
                    else:
                        logging.warning(f'Unknown command: {cmd}')
                        response = generate_failure_response(f'{ERROR_MESSAGE} Unknown command: {cmd}')
                else:
                    logging.warning('Malformed input: missing function property')
                    response = generate_failure_response(f'{ERROR_MESSAGE} Malformed input.')
        else:
            logging.warning('Malformed input: missing tool_calls property')
            response = generate_failure_response(f'{ERROR_MESSAGE} Malformed input.')

        logging.info(f'Sending response: {response}')
        write_response(response)

        if cmd == SHUTDOWN_COMMAND:
            logging.info('Shutdown command received, terminating plugin')
            break
    
    logging.info('CanRun G-Assist Plugin stopped.')
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