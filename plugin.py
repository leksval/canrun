#!/usr/bin/env python3
"""
CanRun G-Assist Plugin - Official NVIDIA G-Assist Integration
Privacy-focused game compatibility analysis for RTX/GTX systems with Steam API integration.

This plugin uses the official NVIDIA RISE library for G-Assist integration.
It supports both interactive RISE mode and command-line interface for testing.

Usage:
    Interactive RISE Mode (G-Assist):
        python plugin.py
        
    Command Line Interface (Testing):
        python plugin.py --function check_compatibility --game "Diablo 4"
        python plugin.py --function detect_hardware
        python plugin.py --help

Dependencies:
    - rise: Official NVIDIA G-Assist Python bindings (REQUIRED)
    - asyncio: For Steam API integration
    - argparse: For command-line argument parsing
    - All CanRun engine modules

RISE Integration:
    The plugin registers with G-Assist using rise.register_rise_client() and
    processes commands through rise.send_rise_command(). It provides intelligent
    game compatibility analysis with Steam API integration.
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.canrun_engine import CanRunEngine
from src.privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from src.dynamic_performance_predictor import DynamicPerformancePredictor, PerformanceTier

# Import RISE library for G-Assist integration
import rise

# Type definitions
Response = Dict[str, Any]

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

def check_compatibility(game_name: str) -> str:
    """
    Check if a game can run on this system with performance analysis.
    
    Args:
        game_name (str): Name of the game to check compatibility for
    
    Returns:
        str: Formatted compatibility analysis message
    """
    try:
        logging.info(f"🎮 Checking compatibility for: {game_name}")
        
        # Initialize plugin if not already done
        if canrun_engine is None:
            init_result = initialize_plugin()
            if not init_result["success"]:
                return f"❌ Plugin initialization failed: {init_result['message']}"
        
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
            
            # Create comprehensive response message
            response_lines = [
                f"🎮 CanRun Analysis: {result.game_name}",
                f"",
                f"🏆 Performance Tier: {performance_tier} ({tier_score}/100)",
                f"🎯 Compatibility: {compatibility_verdict}",
                f"⚡ Expected FPS: {expected_fps}" if expected_fps > 0 else "",
                f"",
                f"💻 Your System:",
                f"   GPU: {result.hardware_specs.gpu_model} ({result.hardware_specs.gpu_vram_gb} GB VRAM)",
                f"   CPU: {result.hardware_specs.cpu_model} ({result.hardware_specs.cpu_cores} cores)",
                f"   RAM: {result.hardware_specs.ram_total_gb} GB",
                f"   RTX Support: {'✅' if result.hardware_specs.supports_rtx else '❌'}",
                f"   DLSS Support: {'✅' if result.hardware_specs.supports_dlss else '❌'}",
                f"",
                f"📋 Game Requirements:",
                f"   CPU: {result.game_requirements.recommended_cpu}",
                f"   GPU: {result.game_requirements.recommended_gpu}",
                f"   RAM: {result.game_requirements.recommended_ram_gb} GB",
                f"",
                f"🔍 Data Source: {result.game_requirements.source}"
            ]
            
            # Filter out empty lines
            response_lines = [line for line in response_lines if line.strip()]
            response_message = "\n".join(response_lines)
            
            logging.info(f"✅ Compatibility check successful for {game_name}")
            return response_message
            
        else:
            error_msg = f"❌ Could not analyze compatibility for {game_name}. Steam API may be unavailable or game not found."
            logging.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ Compatibility check failed for {game_name}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def detect_hardware() -> str:
    """
    Detect and analyze system hardware specifications.
    
    Returns:
        str: Formatted hardware specifications message
    """
    try:
        logging.info("🔍 Detecting hardware specifications...")
        
        # Initialize plugin if not already done
        if privacy_detector is None:
            init_result = initialize_plugin()
            if not init_result["success"]:
                return f"❌ Plugin initialization failed: {init_result['message']}"
        
        # Run async hardware detection in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            hardware_info = loop.run_until_complete(privacy_detector.get_hardware_specs())
        finally:
            loop.close()
        
        # Format hardware info for G-Assist
        hardware_summary = f"""💻 System Hardware Specifications:

🎮 Graphics Card: {hardware_info.gpu_model} ({hardware_info.gpu_vram_gb} GB VRAM)
   RTX Support: {'✅' if hardware_info.supports_rtx else '❌'}
   DLSS Support: {'✅' if hardware_info.supports_dlss else '❌'}

🧠 Processor: {hardware_info.cpu_model} ({hardware_info.cpu_cores} cores, {hardware_info.cpu_threads} threads)

💾 Memory: {hardware_info.ram_total_gb} GB RAM ({hardware_info.ram_speed_mhz} MHz)

💽 Storage: {hardware_info.storage_type}

🖥️  Display: {hardware_info.primary_monitor_resolution} @ {hardware_info.primary_monitor_refresh_hz} Hz

🔧 System Info:
   OS: {hardware_info.os_version}
   DirectX: {hardware_info.directx_version}
   NVIDIA Driver: {hardware_info.nvidia_driver_version}"""
        
        logging.info("✅ Hardware detection successful")
        return hardware_summary
        
    except Exception as e:
        error_msg = f"❌ Hardware detection failed: {str(e)}"
        logging.error(error_msg)
        return error_msg

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

def process_user_query(user_input: str) -> str:
    """
    Process user queries and determine the appropriate action.
    
    Args:
        user_input (str): User's input query
        
    Returns:
        str: Response message
    """
    user_input_lower = user_input.lower()
    
    # Check for hardware detection queries
    hardware_keywords = ['hardware', 'specs', 'system', 'gpu', 'cpu', 'ram', 'my computer', 'my system']
    if any(keyword in user_input_lower for keyword in hardware_keywords):
        return detect_hardware()
    
    # Check for game compatibility queries
    compatibility_keywords = ['can run', 'canrun', 'compatibility', 'will work', 'run on', 'play']
    game_keywords = ['game', 'diablo', 'cyberpunk', 'witcher', 'gta', 'call of duty', 'battlefield', 'assassin']
    
    if any(keyword in user_input_lower for keyword in compatibility_keywords) or any(keyword in user_input_lower for keyword in game_keywords):
        # Try to extract game name from the query
        game_name = extract_game_name(user_input)
        if game_name:
            return check_compatibility(game_name)
        else:
            return """🎮 CanRun Game Compatibility Checker

I can help you check if games can run on your system! Try asking:
• "Can I run Diablo 4?"
• "Will Cyberpunk 2077 work on my system?"
• "Check compatibility for The Witcher 3"

Or ask about your hardware:
• "What are my system specs?"
• "Show me my hardware"

What game would you like me to check?"""
    
    # Default response for unclear queries
    return """🎮 CanRun Game Compatibility Checker

I'm your AI assistant for game compatibility analysis! I can help you:

🎯 Check Game Compatibility:
   • "Can I run [game name]?"
   • "Will [game] work on my system?"
   • "Check compatibility for [game]"

💻 Analyze Your Hardware:
   • "What are my system specs?"
   • "Show me my hardware"
   • "Detect my system"

I use Steam API integration and advanced performance prediction to give you accurate S-A-B-C-D-F tier ratings with expected FPS.

What would you like me to help you with?"""

def extract_game_name(user_input: str) -> Optional[str]:
    """
    Extract game name from user input using pattern matching.
    
    Args:
        user_input (str): User's input query
        
    Returns:
        Optional[str]: Extracted game name or None
    """
    # Common game name patterns
    patterns = [
        r'(?:can (?:i )?run|will|check|compatibility (?:for )?|canrun)\s+([^?]+?)(?:\?|$)',
        r'([a-zA-Z0-9\s:]+?)(?:\s+(?:compatibility|can run|will work))',
        r'(?:game|title)\s+([^?]+?)(?:\?|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            game_name = match.group(1).strip()
            # Clean up common words
            game_name = re.sub(r'\b(?:the|game|on my system|work|run)\b', '', game_name, flags=re.IGNORECASE).strip()
            if len(game_name) > 2:  # Minimum reasonable game name length
                return game_name
    
    # Fallback: look for known game names
    known_games = [
        'diablo 4', 'diablo iv', 'cyberpunk 2077', 'the witcher 3', 'gta 5', 'gta v',
        'call of duty', 'battlefield', 'assassin\'s creed', 'red dead redemption',
        'elden ring', 'hogwarts legacy', 'starfield', 'baldur\'s gate 3'
    ]
    
    user_input_lower = user_input.lower()
    for game in known_games:
        if game in user_input_lower:
            return game.title()
    
    return None

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
            message = check_compatibility(args.game)
            result = {"success": True, "message": message}
    elif args.function == 'detect_hardware':
        message = detect_hardware()
        result = {"success": True, "message": message}
    else:
        result = {"success": False, "message": f"Unknown function: {args.function}"}
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

def run_rise_interface():
    """
    Run the plugin in RISE interface mode for G-Assist integration.
    Uses the official NVIDIA RISE library for communication.
    """
    # Initialize plugin
    init_result = initialize_plugin()
    if not init_result["success"]:
        print(f"❌ Plugin initialization failed: {init_result['message']}")
        sys.exit(1)
    
    print("🚀 CanRun G-Assist Plugin starting...")
    print("🎮 Ready for game compatibility analysis!")
    
    # Register with G-Assist
    rise.register_rise_client()
    
    # Main interaction loop
    while True:
        try:
            # Get user input (this would come from G-Assist in real usage)
            user_prompt = input("Enter your query (or 'quit' to exit): ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 CanRun plugin shutting down...")
                break
            
            if not user_prompt:
                continue
            
            # Process the user query
            response = process_user_query(user_prompt)
            
            # Send response through RISE
            rise_response = rise.send_rise_command(response)
            print(f"🎯 Response: {rise_response}")
            
        except KeyboardInterrupt:
            print("\n👋 CanRun plugin shutting down...")
            break
        except Exception as e:
            logging.error(f"Error in RISE interface: {e}")
            print(f"❌ Error: {e}")

def main():
    """
    Main entry point for the CanRun G-Assist plugin.
    
    Determines whether to run in command-line interface mode or RISE mode
    based on the presence of command-line arguments.
    
    Command-line mode: Used when arguments are provided (e.g., --function, --game)
    RISE mode: Used when no arguments are provided (default G-Assist behavior)
    """
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Run in command-line interface mode
        run_command_line_interface()
    else:
        # Run in RISE interface mode for G-Assist
        run_rise_interface()

if __name__ == '__main__':
    main()