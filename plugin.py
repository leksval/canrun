#!/usr/bin/env python3
"""
CanRun G-Assist Plugin - Official NVIDIA G-Assist Integration
Privacy-focused game compatibility analysis for RTX/GTX systems with Steam API integration.

This plugin uses the official NVIDIA G-Assist pipe communication protocol.
It communicates with G-Assist through Windows named pipes using stdin/stdout.

Usage:
    G-Assist Mode (Production):
        python plugin.py
        
    Command Line Interface (Testing):
        python plugin.py --function check_compatibility --game "Diablo 4"
        python plugin.py --function detect_hardware
        python plugin.py --help

Communication Protocol:
    Input: JSON commands via stdin from G-Assist
    Output: JSON responses via stdout to G-Assist
    Format: {"tool_calls": [{"func": "function_name", "params": {...}}]}
    Response: {"success": true/false, "message": "response"}<<END>>
"""

import json
import sys
import logging
import os
import asyncio
import argparse
import re
from typing import Optional, Dict, Any
from pathlib import Path
from ctypes import byref, windll, wintypes

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.canrun_engine import CanRunEngine
from src.privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from src.dynamic_performance_predictor import DynamicPerformancePredictor, PerformanceTier

# Windows pipe communication constants
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
BUFFER_SIZE = 4096

# Configure logging with detailed format
LOG_FILE = os.path.join(os.environ.get('USERPROFILE', '.'), 'canrun-plugin.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# Global plugin components
canrun_engine = None
privacy_detector = None
performance_predictor = None

def initialize_plugin() -> dict:
    """
    Initialize the CanRun plugin components.
    
    Returns:
        dict: Success/failure response
    """
    global canrun_engine, privacy_detector, performance_predictor
    
    try:
        logging.info("🚀 Initializing CanRun G-Assist plugin...")
        
        # Initialize CanRun engine with Steam API integration
        canrun_engine = CanRunEngine(enable_llm=True)
        logging.info("✅ CanRun engine initialized with Steam API integration")
        
        # Initialize privacy-aware hardware detector
        privacy_detector = PrivacyAwareHardwareDetector()
        logging.info("✅ Privacy detector initialized")
        
        # Initialize dynamic performance predictor
        performance_predictor = DynamicPerformancePredictor()
        logging.info("✅ Dynamic performance predictor initialized")
        
        logging.info("🎯 CanRun G-Assist plugin initialized successfully")
        return {"success": True, "message": "CanRun plugin initialized successfully"}
        
    except Exception as e:
        logging.error(f"❌ Plugin initialization failed: {e}")
        return {"success": False, "message": f"Plugin initialization failed: {str(e)}"}

def check_compatibility(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a game can run on this system with performance analysis.
    
    Args:
        params (Dict[str, Any]): Parameters containing game_name
    
    Returns:
        Dict[str, Any]: G-Assist response format
    """
    try:
        game_name = params.get("game_name", "")
        if not game_name:
            return {"success": False, "message": "Game name is required"}
            
        logging.info(f"🎮 Checking compatibility for: {game_name}")
        
        # Initialize plugin if not already done
        if canrun_engine is None:
            init_result = initialize_plugin()
            if not init_result["success"]:
                return {"success": False, "message": f"Plugin initialization failed: {init_result['message']}"}
        
        # Run async compatibility check in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(canrun_engine.check_game_compatibility(game_name))
        finally:
            loop.close()
        
        if result:
            # Get hardware specs for dynamic performance prediction
            cpu_frequency = _get_cpu_frequency_from_hardware(result.hardware_specs)
            
            hardware_specs = {
                'gpu_model': result.hardware_specs.gpu_model,
                'cpu_model': result.hardware_specs.cpu_model,
                'ram_total_gb': result.hardware_specs.ram_total_gb,
                'cpu_cores': result.hardware_specs.cpu_cores,
                'cpu_threads': result.hardware_specs.cpu_threads,
                'cpu_frequency': cpu_frequency,
                'gpu_vram_gb': result.hardware_specs.gpu_vram_gb
            }
            
            # Convert game requirements to dict format for dynamic predictor
            game_requirements = {
                'recommended': {
                    'processor': result.game_requirements.recommended_cpu,
                    'graphics': result.game_requirements.recommended_gpu,
                    'memory': result.game_requirements.recommended_ram_gb
                }
            }
            
            # Get dynamic performance assessment
            performance_assessment = None
            if performance_predictor:
                try:
                    performance_assessment = performance_predictor.assess_performance(
                        hardware_specs=hardware_specs,
                        game_requirements=game_requirements
                    )
                except Exception as e:
                    logging.warning(f"Dynamic performance prediction failed: {e}")
            
            # Extract performance tier information
            performance_tier = "Unknown"
            tier_score = 0
            expected_fps = 0
            
            if performance_assessment:
                performance_tier = performance_assessment.tier.name
                tier_score = performance_assessment.score
                expected_fps = performance_assessment.expected_fps
            
            # Create compatibility verdict
            compatibility_verdict = "EXCELLENT" if result.exceeds_recommended_requirements() else "GOOD" if result.can_run_game() else "INSUFFICIENT"
            
            # ✅ Extract LLM Analysis
            llm_insights = ""
            if result.llm_analysis:
                try:
                    llm_analysis = result.llm_analysis.get('analysis')
                    if llm_analysis and hasattr(llm_analysis, 'analysis_text'):
                        llm_insights = f"\n\n🧠 G-Assist AI Analysis:\n{llm_analysis.analysis_text}"
                    elif llm_analysis and hasattr(llm_analysis, 'recommendations'):
                        llm_insights = f"\n\n🧠 G-Assist Recommendations:\n• {'; '.join(llm_analysis.recommendations[:3])}"
                except Exception as e:
                    logging.debug(f"LLM analysis extraction failed: {e}")
            
            # ✅ Extract Component Analysis
            component_breakdown = ""
            if hasattr(result.compatibility_analysis, 'component_analyses'):
                component_breakdown = "\n\n📊 Component Analysis:"
                for comp in result.compatibility_analysis.component_analyses:
                    try:
                        status = "✅" if comp.meets_recommended else "⚠️" if comp.meets_minimum else "❌"
                        score = int(comp.score * 100) if hasattr(comp, 'score') else 0
                        component_breakdown += f"\n   {status} {comp.component.value}: {score}/100"
                        if hasattr(comp, 'upgrade_suggestion') and comp.upgrade_suggestion:
                            component_breakdown += f" - {comp.upgrade_suggestion}"
                    except Exception as e:
                        logging.debug(f"Component analysis extraction failed for {comp}: {e}")
            
            # ✅ Extract Bottlenecks and Recommendations
            optimization_tips = ""
            if performance_assessment:
                if hasattr(performance_assessment, 'bottlenecks') and performance_assessment.bottlenecks:
                    optimization_tips += f"\n\n⚠️ Bottlenecks: {', '.join(performance_assessment.bottlenecks)}"
                if hasattr(performance_assessment, 'upgrade_suggestions') and performance_assessment.upgrade_suggestions:
                    optimization_tips += f"\n💡 Recommendations: {'; '.join(performance_assessment.upgrade_suggestions[:3])}"
            
            # ✅ Add Compatibility Analysis Recommendations
            if hasattr(result.compatibility_analysis, 'recommendations') and result.compatibility_analysis.recommendations:
                if not optimization_tips:
                    optimization_tips = "\n"
                optimization_tips += f"\n🔧 System Optimizations: {'; '.join(result.compatibility_analysis.recommendations[:2])}"
            
            # Create concise G-Assist response (like Twitch plugin format)
            if compatibility_verdict == "EXCELLENT":
                verdict_emoji = "✅"
                verdict_text = "YES! Your system can run this game excellently"
            elif compatibility_verdict == "GOOD":
                verdict_emoji = "✅"
                verdict_text = "YES! Your system can run this game well"
            else:
                verdict_emoji = "❌"
                verdict_text = "Your system may struggle with this game"
            
            # Simple, direct response like Twitch plugin
            response_message = f"""{verdict_emoji} {result.game_name} - {verdict_text}
Performance: {performance_tier} tier ({tier_score}/100)
Expected FPS: {expected_fps}
GPU: {result.hardware_specs.gpu_model}
Recommendation: {"Ultra settings at 4K" if tier_score >= 90 else "High settings at 1440p" if tier_score >= 70 else "Medium settings at 1080p"}"""
            
            logging.info(f"✅ Compatibility check successful for {game_name}")
            return {"success": True, "message": response_message}
            
        else:
            error_msg = f"❌ Could not analyze compatibility for {game_name}. Steam API may be unavailable or game not found."
            logging.error(error_msg)
            return {"success": False, "message": error_msg}
            
    except Exception as e:
        error_msg = f"❌ Compatibility check failed for {game_name}: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}

def detect_hardware(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect and analyze system hardware specifications.
    
    Args:
        params (Dict[str, Any]): Parameters (unused for hardware detection)
    
    Returns:
        Dict[str, Any]: G-Assist response format
    """
    try:
        logging.info("🔍 Detecting hardware specifications...")
        
        # Initialize plugin if not already done
        if privacy_detector is None:
            init_result = initialize_plugin()
            if not init_result["success"]:
                return {"success": False, "message": f"Plugin initialization failed: {init_result['message']}"}
        
        # Run async hardware detection in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            hardware_info = loop.run_until_complete(privacy_detector.get_hardware_specs())
        finally:
            loop.close()
        
        # Format hardware info for G-Assist (concise like Twitch plugin)
        hardware_summary = f"""💻 Your Gaming System:
GPU: {hardware_info.gpu_model} ({hardware_info.gpu_vram_gb} GB VRAM)
CPU: {hardware_info.cpu_model} ({hardware_info.cpu_cores} cores)
RAM: {hardware_info.ram_total_gb} GB
RTX Features: {'✅ Supported' if hardware_info.supports_rtx else '❌ Not Available'}
Gaming Performance: {'Excellent' if hardware_info.gpu_vram_gb >= 16 else 'Good' if hardware_info.gpu_vram_gb >= 8 else 'Basic'}"""
        
        logging.info("✅ Hardware detection successful")
        return {"success": True, "message": hardware_summary}
        
    except Exception as e:
        error_msg = f"❌ Hardware detection failed: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}

def _get_cpu_frequency_from_hardware(hardware_specs):
    """Extract CPU frequency from hardware specs, with intelligent fallback."""
    try:
        # Try to get actual CPU frequency from hardware specs
        if hasattr(hardware_specs, 'cpu_frequency'):
            return hardware_specs.cpu_frequency
        elif hasattr(hardware_specs, 'cpu_freq'):
            return hardware_specs.cpu_freq * 1000  # Convert GHz to MHz if needed
        else:
            # Intelligent fallback based on CPU model
            cpu_model = getattr(hardware_specs, 'cpu_model', '').lower()
            
            # Extract frequency from CPU model name if possible
            freq_match = re.search(r'(\d+\.?\d*)\s*ghz', cpu_model)
            if freq_match:
                return float(freq_match.group(1)) * 1000  # Convert GHz to MHz
            
            # Fallback estimates based on CPU generation/model
            if 'i9' in cpu_model or 'ryzen 9' in cpu_model:
                return 3500  # High-end CPUs typically 3.5+ GHz
            elif 'i7' in cpu_model or 'ryzen 7' in cpu_model:
                return 3200  # Mid-high CPUs typically 3.2+ GHz
            elif 'i5' in cpu_model or 'ryzen 5' in cpu_model:
                return 2800  # Mid-range CPUs typically 2.8+ GHz
            else:
                return 2400  # Conservative fallback for older/lower-end CPUs
    except Exception as e:
        logging.warning(f"Failed to determine CPU frequency: {e}")
        return 2800  # Safe fallback

def read_command() -> Optional[Dict[str, Any]]:
    """
    Read command from stdin using standard Python input.
    
    Returns:
        Optional[Dict[str, Any]]: Parsed command dictionary or None if failed
    """
    try:
        # Read from stdin
        line = sys.stdin.readline().strip()
        if not line:
            return None
            
        logging.info(f'Received command: {line}')
        return json.loads(line)
        
    except json.JSONDecodeError as e:
        logging.error(f'Invalid JSON received: {e}')
        return None
    except Exception as e:
        logging.error(f'Error in read_command: {e}')
        return None

def write_response(response: Dict[str, Any]) -> None:
    """
    Write response to stdout pipe using Windows API.
    
    Args:
        response: Dictionary containing 'success' and optional 'message'
    """
    try:
        pipe = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        # Add <<END>> marker for message termination
        message = json.dumps(response) + '<<END>>'
        message_bytes = message.encode('utf-8')
        
        bytes_written = wintypes.DWORD()
        windll.kernel32.WriteFile(
            pipe,
            message_bytes,
            len(message_bytes),
            byref(bytes_written),
            None
        )
        logging.info(f'Sent response: {json.dumps(response)}')
    except Exception as e:
        logging.error(f'Error writing response: {e}')

def run_command_line_interface():
    """
    Run the plugin in command-line interface mode.
    Supports direct function calls with arguments.
    """
    parser = argparse.ArgumentParser(
        description="CanRun G-Assist Plugin - Game Compatibility Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plugin.py --function check_compatibility --game "Diablo 4"
  python plugin.py --function detect_hardware
  
Functions:
  check_compatibility  Check if a game can run on this system
  detect_hardware      Detect system hardware specifications
        """
    )
    
    parser.add_argument(
        '--function', 
        required=True,
        choices=['check_compatibility', 'detect_hardware'],
        help='Function to execute'
    )
    
    parser.add_argument(
        '--game', 
        help='Game name for compatibility check (required for check_compatibility)'
    )
    
    args = parser.parse_args()
    
    # Initialize plugin
    init_result = initialize_plugin()
    if not init_result["success"]:
        result = {"success": False, "message": init_result["message"]}
        print(json.dumps(result))
        sys.exit(1)
    
    # Execute the requested function
    if args.function == 'check_compatibility':
        if not args.game:
            result = {"success": False, "message": "Game name is required for compatibility check"}
        else:
            result = check_compatibility({"game_name": args.game})
    elif args.function == 'detect_hardware':
        result = detect_hardware({})
    else:
        result = {"success": False, "message": f"Unknown function: {args.function}"}
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

def run_g_assist_interface():
    """
    Run the plugin in G-Assist interface mode.
    Uses Windows named pipes for communication with G-Assist.
    """
    # Initialize plugin
    init_result = initialize_plugin()
    if not init_result["success"]:
        logging.error(f"❌ Plugin initialization failed: {init_result['message']}")
        write_response({"success": False, "message": f"Plugin initialization failed: {init_result['message']}"})
        sys.exit(1)
    
    logging.info("🚀 CanRun G-Assist Plugin starting...")
    logging.info("🎮 Ready for game compatibility analysis!")
    
    # Main interaction loop
    while True:
        try:
            # Read command from G-Assist via stdin pipe
            command = read_command()
            if command is None:
                continue
            
            # Process tool calls
            for tool_call in command.get("tool_calls", []):
                func = tool_call.get("func")
                params = tool_call.get("params", {})
                
                if func == "check_compatibility":
                    response = check_compatibility(params)
                    write_response(response)
                elif func == "detect_hardware":
                    response = detect_hardware(params)
                    write_response(response)
                elif func == "shutdown":
                    logging.info("👋 CanRun plugin shutting down...")
                    return
                else:
                    response = {"success": False, "message": f"Unknown function: {func}"}
                    write_response(response)
            
        except KeyboardInterrupt:
            logging.info("\n👋 CanRun plugin shutting down...")
            break
        except Exception as e:
            logging.error(f"Error in G-Assist interface: {e}")
            write_response({"success": False, "message": f"Error: {e}"})

def main():
    """
    Main entry point for the CanRun G-Assist plugin.
    
    Determines whether to run in command-line interface mode or G-Assist mode
    based on the presence of command-line arguments and stdin availability.
    
    Command-line mode: Used when arguments are provided (e.g., --function, --game)
    G-Assist mode: Used when no arguments are provided or stdin has data
    """
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Run in command-line interface mode
        run_command_line_interface()
    else:
        # For Windows, we'll use a different approach
        try:
            # Try to read from stdin with a timeout
            if sys.stdin.isatty():
                # No piped input, run in G-Assist mode anyway
                run_g_assist_interface()
            else:
                # Piped input detected, run in G-Assist mode
                run_g_assist_interface()
        except:
            # Fallback to G-Assist mode
            run_g_assist_interface()

if __name__ == '__main__':
    main()