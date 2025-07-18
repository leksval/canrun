#!/usr/bin/env python3
"""
Enhanced CanRun G-Assist Plugin - NVIDIA G-Assist Integration
Enhanced frontend-backend communication with tool name detection and Steam Compare UI integration.

This plugin implements the NVIDIA G-Assist communication patterns:
- Tool name detection and routing
- Steam Compare UI integration
- Enhanced chat window response formatting
- Real-time Steam data fetching and display
"""

import json
import sys
import logging
import os
import asyncio
import argparse
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from ctypes import byref, windll, wintypes
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.canrun_engine import CanRunEngine
from src.privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from src.dynamic_performance_predictor import DynamicPerformancePredictor, PerformanceTier
from src.game_requirements_fetcher import GameRequirementsFetcher

# Windows pipe communication constants
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
BUFFER_SIZE = 4096

# Configure logging with detailed format
LOG_FILE = os.path.join(os.environ.get('USERPROFILE', '.'), 'canrun-enhanced-plugin.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# Global plugin components
canrun_engine = None
privacy_detector = None
performance_predictor = None
requirements_fetcher = None

class ToolNameDetector:
    """Detects tool names and game references from user input."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Tool detection patterns
        self.tool_patterns = {
            'check_compatibility': [
                r'can\s+(?:i\s+)?(?:run|play)',
                r'will\s+(?:this\s+)?(?:game\s+)?(?:run|work)',
                r'compatibility\s+(?:check|test)',
                r'system\s+requirements',
                r'can\s+my\s+(?:pc|computer|system)\s+(?:run|handle)',
                r'(?:game\s+)?performance\s+(?:check|test)',
                r'(?:will|can)\s+.*(?:work|run)\s+on\s+my'
            ],
            'detect_hardware': [
                r'(?:my\s+)?(?:system\s+)?(?:specs|specifications)',
                r'hardware\s+(?:info|information|details)',
                r'what\s+(?:are\s+)?my\s+(?:pc\s+)?specs',
                r'(?:show|tell)\s+me\s+my\s+(?:system|hardware)',
                r'computer\s+(?:specs|information)',
                r'gpu\s+(?:info|details)',
                r'cpu\s+(?:info|details)'
            ],
            'compare_games': [
                r'compare\s+(?:games?|performance)',
                r'(?:game\s+)?comparison',
                r'vs\s+',
                r'versus',
                r'which\s+(?:game\s+)?(?:runs\s+)?better',
                r'performance\s+comparison'
            ],
            'steam_compare': [
                r'steam\s+(?:compare|comparison)',
                r'steam\s+(?:store|page)',
                r'steam\s+(?:requirements|specs)',
                r'check\s+(?:on\s+)?steam'
            ]
        }
        
        # Game name extraction patterns
        self.game_patterns = [
            r'(?:can\s+i\s+run\s+|will\s+|check\s+)([A-Za-z0-9\s:\'\"]+?)(?:\s+on\s+my|\s+work|\s*\?|$)',
            r'(?:game\s+)?\"([^\"]+)\"',
            r'(?:game\s+)?\'([^\']+)\'',
            r'(?:compatibility\s+for\s+|requirements\s+for\s+)([A-Za-z0-9\s:\'\"]+?)(?:\s+on|\s*\?|$)',
            r'(?:performance\s+of\s+|how\s+well\s+does\s+)([A-Za-z0-9\s:\'\"]+?)(?:\s+run|\s+work|\s*\?|$)'
        ]
    
    def detect_tool_and_extract_params(self, user_input: str) -> Dict[str, Any]:
        """
        Detect tool name and extract parameters from user input.
        
        Args:
            user_input: Raw user input from G-Assist chat
            
        Returns:
            Dict containing detected tool and parameters
        """
        user_input_lower = user_input.lower().strip()
        
        # Detect primary tool
        detected_tool = self._detect_primary_tool(user_input_lower)
        
        # Extract game names if relevant
        game_names = self._extract_game_names(user_input)
        
        # Build parameters based on detected tool
        params = {}
        
        if detected_tool == 'check_compatibility' and game_names:
            params['game_name'] = game_names[0]  # Use first detected game
            if len(game_names) > 1:
                params['compare_with'] = game_names[1:]
        elif detected_tool == 'compare_games' and len(game_names) >= 2:
            params['games'] = game_names
        elif detected_tool == 'steam_compare' and game_names:
            params['game_name'] = game_names[0]
            params['show_steam_data'] = True
        
        result = {
            'tool': detected_tool,
            'params': params,
            'confidence': self._calculate_confidence(user_input_lower, detected_tool),
            'original_input': user_input
        }
        
        self.logger.info(f"Tool detection result: {result}")
        return result
    
    def _detect_primary_tool(self, user_input: str) -> str:
        """Detect the primary tool from user input."""
        tool_scores = {}
        
        for tool, patterns in self.tool_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, user_input, re.IGNORECASE))
                score += matches
            tool_scores[tool] = score
        
        # Return tool with highest score, default to check_compatibility
        if max(tool_scores.values()) > 0:
            return max(tool_scores, key=tool_scores.get)
        
        # Default fallback based on content
        if any(word in user_input for word in ['spec', 'hardware', 'system', 'gpu', 'cpu']):
            return 'detect_hardware'
        elif any(word in user_input for word in ['compare', 'vs', 'versus', 'better']):
            return 'compare_games'
        else:
            return 'check_compatibility'
    
    def _extract_game_names(self, user_input: str) -> List[str]:
        """Extract game names from user input."""
        games = []
        
        for pattern in self.game_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                game_name = match.strip().strip('.,!?')
                if len(game_name) > 2 and game_name not in games:
                    games.append(game_name)
        
        # Clean up extracted games
        cleaned_games = []
        for game in games:
            # Remove common false positives
            if not any(word in game.lower() for word in ['my pc', 'my system', 'my computer', 'this game']):
                cleaned_games.append(game)
        
        return cleaned_games[:3]  # Limit to 3 games max
    
    def _calculate_confidence(self, user_input: str, detected_tool: str) -> float:
        """Calculate confidence score for tool detection."""
        if detected_tool not in self.tool_patterns:
            return 0.5
        
        patterns = self.tool_patterns[detected_tool]
        matches = sum(len(re.findall(pattern, user_input, re.IGNORECASE)) for pattern in patterns)
        
        # Normalize confidence based on input length and matches
        confidence = min(0.9, 0.3 + (matches * 0.2))
        return confidence


class SteamCompareUI:
    """Handles Steam Compare UI functionality and data formatting."""
    
    def __init__(self, requirements_fetcher):
        self.requirements_fetcher = requirements_fetcher
        self.logger = logging.getLogger(__name__)
    
    async def get_steam_comparison_data(self, game_name: str) -> Dict[str, Any]:
        """Get Steam comparison data for a game."""
        try:
            # Fetch game requirements from Steam
            requirements = await self.requirements_fetcher.fetch_requirements(game_name)
            
            if not requirements:
                return {
                    'success': False,
                    'message': f'Steam data not found for {game_name}'
                }
            
            # Format Steam comparison data
            steam_data = {
                'game_name': requirements.game_name,
                'source': requirements.source,
                'minimum_requirements': {
                    'cpu': requirements.minimum_cpu,
                    'gpu': requirements.minimum_gpu,
                    'ram': f"{requirements.minimum_ram_gb} GB",
                    'storage': f"{requirements.minimum_storage_gb} GB",
                    'directx': requirements.minimum_directx,
                    'os': requirements.minimum_os
                },
                'recommended_requirements': {
                    'cpu': requirements.recommended_cpu,
                    'gpu': requirements.recommended_gpu,
                    'ram': f"{requirements.recommended_ram_gb} GB",
                    'storage': f"{requirements.recommended_storage_gb} GB",
                    'directx': requirements.recommended_directx,
                    'os': requirements.recommended_os
                },
                'last_updated': requirements.last_updated
            }
            
            return {
                'success': True,
                'steam_data': steam_data
            }
            
        except Exception as e:
            self.logger.error(f"Steam comparison data fetch failed: {e}")
            return {
                'success': False,
                'message': f'Failed to fetch Steam data: {str(e)}'
            }
    
    def format_steam_compare_response(self, steam_data: Dict[str, Any], compatibility_result: Dict[str, Any] = None) -> str:
        """Format Steam comparison data for G-Assist chat display."""
        if not steam_data.get('success'):
            return f"❌ {steam_data.get('message', 'Steam data unavailable')}"
        
        data = steam_data['steam_data']
        game_name = data['game_name']
        
        # Create Steam Compare UI style response
        response = f"""🎮 **{game_name}** - Steam Requirements Analysis

📊 **MINIMUM REQUIREMENTS**
• CPU: {data['minimum_requirements']['cpu']}
• GPU: {data['minimum_requirements']['gpu']}
• RAM: {data['minimum_requirements']['ram']}
• Storage: {data['minimum_requirements']['storage']}
• DirectX: {data['minimum_requirements']['directx']}
• OS: {data['minimum_requirements']['os']}

🚀 **RECOMMENDED REQUIREMENTS**
• CPU: {data['recommended_requirements']['cpu']}
• GPU: {data['recommended_requirements']['gpu']}
• RAM: {data['recommended_requirements']['ram']}
• Storage: {data['recommended_requirements']['storage']}
• DirectX: {data['recommended_requirements']['directx']}
• OS: {data['recommended_requirements']['os']}

📈 **SOURCE**: {data['source']}"""

        # Add compatibility analysis if available
        if compatibility_result:
            verdict = "✅ COMPATIBLE" if compatibility_result.get('can_run') else "❌ INCOMPATIBLE"
            performance_tier = compatibility_result.get('performance_tier', 'Unknown')
            expected_fps = compatibility_result.get('expected_fps', 0)
            
            response += f"""

🎯 **YOUR SYSTEM COMPATIBILITY**
{verdict} | Performance: {performance_tier} | Expected FPS: {expected_fps}"""
        
        return response


class EnhancedResponseFormatter:
    """Enhanced response formatter for G-Assist chat window."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_compatibility_response(self, result: Dict[str, Any], include_steam_ui: bool = False) -> str:
        """Format compatibility check response for G-Assist chat."""
        if not result.get('success'):
            return f"❌ {result.get('message', 'Compatibility check failed')}"
        
        # Extract key information
        game_name = result.get('game_name', 'Unknown Game')
        can_run = result.get('can_run', False)
        performance_tier = result.get('performance_tier', 'Unknown')
        expected_fps = result.get('expected_fps', 0)
        gpu_model = result.get('gpu_model', 'Unknown GPU')
        
        # Create main verdict
        verdict_emoji = "✅" if can_run else "❌"
        verdict_text = "YES! Your system can run this game" if can_run else "Your system may struggle with this game"
        
        # Performance recommendation
        if expected_fps >= 60:
            settings_rec = "Ultra settings at 4K"
        elif expected_fps >= 45:
            settings_rec = "High settings at 1440p"
        elif expected_fps >= 30:
            settings_rec = "Medium settings at 1080p"
        else:
            settings_rec = "Low settings recommended"
        
        # Build response
        response = f"""{verdict_emoji} **{game_name}** - {verdict_text}

🎯 **PERFORMANCE ANALYSIS**
• Performance Tier: {performance_tier}
• Expected FPS: {expected_fps}
• Your GPU: {gpu_model}
• Recommended: {settings_rec}"""
        
        # Add Steam Compare UI section if requested
        if include_steam_ui and result.get('steam_data'):
            response += f"""

📊 **STEAM REQUIREMENTS COMPARISON**
{self._format_steam_section(result['steam_data'])}"""
        
        # Add optimization tips if available
        if result.get('optimization_tips'):
            response += f"""

💡 **OPTIMIZATION TIPS**
{result['optimization_tips']}"""
        
        return response
    
    def format_hardware_response(self, hardware_info: Dict[str, Any]) -> str:
        """Format hardware detection response for G-Assist chat."""
        if not hardware_info.get('success'):
            return f"❌ {hardware_info.get('message', 'Hardware detection failed')}"
        
        hw = hardware_info.get('hardware_specs', {})
        
        # Determine gaming performance level
        vram_gb = hw.get('gpu_vram_gb', 0)
        if vram_gb >= 16:
            gaming_level = "Excellent (4K Gaming Ready)"
        elif vram_gb >= 8:
            gaming_level = "Good (1440p Gaming Ready)"
        elif vram_gb >= 4:
            gaming_level = "Basic (1080p Gaming)"
        else:
            gaming_level = "Limited (Entry Level)"
        
        response = f"""💻 **YOUR GAMING SYSTEM**

🖥️ **GRAPHICS**
• GPU: {hw.get('gpu_model', 'Unknown')}
• VRAM: {vram_gb} GB
• RTX Features: {'✅ Supported' if hw.get('supports_rtx') else '❌ Not Available'}
• DLSS: {'✅ Supported' if hw.get('supports_dlss') else '❌ Not Available'}

⚡ **PROCESSOR & MEMORY**
• CPU: {hw.get('cpu_model', 'Unknown')}
• Cores: {hw.get('cpu_cores', 'Unknown')}
• RAM: {hw.get('ram_total_gb', 'Unknown')} GB

🎮 **GAMING PERFORMANCE**: {gaming_level}"""
        
        return response
    
    def format_comparison_response(self, games_data: List[Dict[str, Any]]) -> str:
        """Format game comparison response for G-Assist chat."""
        if not games_data:
            return "❌ No games found for comparison"
        
        response = "🎮 **GAME PERFORMANCE COMPARISON**\n\n"
        
        for i, game_data in enumerate(games_data, 1):
            if game_data.get('success'):
                game_name = game_data.get('game_name', f'Game {i}')
                performance_tier = game_data.get('performance_tier', 'Unknown')
                expected_fps = game_data.get('expected_fps', 0)
                can_run = game_data.get('can_run', False)
                
                status_emoji = "✅" if can_run else "❌"
                
                response += f"""{status_emoji} **{game_name}**
• Performance: {performance_tier} tier
• Expected FPS: {expected_fps}
• Status: {'Compatible' if can_run else 'Incompatible'}

"""
        
        return response.strip()
    
    def _format_steam_section(self, steam_data: Dict[str, Any]) -> str:
        """Format Steam data section."""
        min_req = steam_data.get('minimum_requirements', {})
        rec_req = steam_data.get('recommended_requirements', {})
        
        return f"""**Minimum**: {min_req.get('cpu', 'Unknown')} | {min_req.get('gpu', 'Unknown')} | {min_req.get('ram', 'Unknown')}
**Recommended**: {rec_req.get('cpu', 'Unknown')} | {rec_req.get('gpu', 'Unknown')} | {rec_req.get('ram', 'Unknown')}"""


def initialize_enhanced_plugin() -> dict:
    """Initialize the enhanced CanRun plugin components."""
    global canrun_engine, privacy_detector, performance_predictor, requirements_fetcher
    
    try:
        logging.info("🚀 Initializing Enhanced CanRun G-Assist plugin...")
        
        # Initialize CanRun engine with Steam API integration
        canrun_engine = CanRunEngine(enable_llm=True)
        logging.info("✅ CanRun engine initialized with Steam API integration")
        
        # Initialize privacy-aware hardware detector
        privacy_detector = PrivacyAwareHardwareDetector()
        logging.info("✅ Privacy detector initialized")
        
        # Initialize dynamic performance predictor
        performance_predictor = DynamicPerformancePredictor()
        logging.info("✅ Dynamic performance predictor initialized")
        
        # Initialize requirements fetcher
        requirements_fetcher = GameRequirementsFetcher()
        logging.info("✅ Game requirements fetcher initialized")
        
        logging.info("🎯 Enhanced CanRun G-Assist plugin initialized successfully")
        return {"success": True, "message": "Enhanced CanRun plugin initialized successfully"}
        
    except Exception as e:
        logging.error(f"❌ Enhanced plugin initialization failed: {e}")
        return {"success": False, "message": f"Enhanced plugin initialization failed: {str(e)}"}


async def process_enhanced_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Process enhanced tool calls with Steam Compare UI integration."""
    func = tool_call.get("func")
    params = tool_call.get("params", {})
    
    # Initialize components if needed
    if canrun_engine is None:
        init_result = initialize_enhanced_plugin()
        if not init_result["success"]:
            return {"success": False, "message": f"Plugin initialization failed: {init_result['message']}"}
    
    # Initialize helper classes
    steam_compare = SteamCompareUI(requirements_fetcher)
    formatter = EnhancedResponseFormatter()
    
    try:
        if func == "check_compatibility":
            return await handle_enhanced_compatibility_check(params, steam_compare, formatter)
        elif func == "detect_hardware":
            return await handle_enhanced_hardware_detection(formatter)
        elif func == "compare_games":
            return await handle_enhanced_game_comparison(params, formatter)
        elif func == "steam_compare":
            return await handle_steam_compare_ui(params, steam_compare, formatter)
        elif func == "auto_detect":
            return await handle_auto_detection(params, steam_compare, formatter)
        else:
            return {"success": False, "message": f"Unknown enhanced function: {func}"}
            
    except Exception as e:
        logging.error(f"Enhanced tool call processing failed: {e}")
        return {"success": False, "message": f"Enhanced processing error: {str(e)}"}


async def handle_enhanced_compatibility_check(params: Dict[str, Any], steam_compare: SteamCompareUI, formatter: EnhancedResponseFormatter) -> Dict[str, Any]:
    """Handle enhanced compatibility check with Steam Compare UI."""
    try:
        game_name = params.get("game_name", "")
        include_steam_ui = params.get("show_steam_data", True)
        
        if not game_name:
            return {"success": False, "message": "Game name is required"}
        
        logging.info(f"🎮 Enhanced compatibility check for: {game_name}")
        
        # Run compatibility check
        try:
            # Try to get current loop
            loop = asyncio.get_running_loop()
            # If we're already in a loop, use await directly
            result = await canrun_engine.check_game_compatibility(game_name)
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(canrun_engine.check_game_compatibility(game_name))
            finally:
                loop.close()
        
        if result:
            # Get Steam comparison data if requested
            steam_data = None
            if include_steam_ui:
                steam_result = await steam_compare.get_steam_comparison_data(game_name)
                if steam_result.get('success'):
                    steam_data = steam_result['steam_data']
            
            # Format enhanced response
            response_data = {
                'success': True,
                'game_name': result.game_name,
                'can_run': result.can_run_game(),
                'performance_tier': result.performance_prediction.tier.name if hasattr(result.performance_prediction, 'tier') else 'Unknown',
                'expected_fps': getattr(result.performance_prediction, 'expected_fps', 0),
                'gpu_model': result.hardware_specs.gpu_model,
                'steam_data': steam_data
            }
            
            # Add optimization tips
            if hasattr(result.performance_prediction, 'upgrade_suggestions'):
                response_data['optimization_tips'] = '; '.join(result.performance_prediction.upgrade_suggestions[:3])
            
            formatted_response = formatter.format_compatibility_response(response_data, include_steam_ui)
            
            logging.info(f"✅ Enhanced compatibility check successful for {game_name}")
            return {"success": True, "message": formatted_response}
        else:
            error_msg = f"❌ Could not analyze compatibility for {game_name}. Steam API may be unavailable or game not found."
            logging.error(error_msg)
            return {"success": False, "message": error_msg}
            
    except Exception as e:
        error_msg = f"❌ Enhanced compatibility check failed for {game_name}: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}


async def handle_enhanced_hardware_detection(formatter: EnhancedResponseFormatter) -> Dict[str, Any]:
    """Handle enhanced hardware detection."""
    try:
        logging.info("🔍 Enhanced hardware detection...")
        
        # Use existing event loop or create new one if needed
        try:
            # Try to get current loop
            loop = asyncio.get_running_loop()
            # If we're already in a loop, use await directly
            hardware_info = await privacy_detector.get_hardware_specs()
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                hardware_info = loop.run_until_complete(privacy_detector.get_hardware_specs())
            finally:
                loop.close()
        
        # Format response data
        response_data = {
            'success': True,
            'hardware_specs': {
                'gpu_model': hardware_info.gpu_model,
                'gpu_vram_gb': hardware_info.gpu_vram_gb,
                'cpu_model': hardware_info.cpu_model,
                'cpu_cores': hardware_info.cpu_cores,
                'ram_total_gb': hardware_info.ram_total_gb,
                'supports_rtx': hardware_info.supports_rtx,
                'supports_dlss': hardware_info.supports_dlss
            }
        }
        
        formatted_response = formatter.format_hardware_response(response_data)
        
        logging.info("✅ Enhanced hardware detection successful")
        return {"success": True, "message": formatted_response}
        
    except Exception as e:
        error_msg = f"❌ Enhanced hardware detection failed: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}


async def handle_enhanced_game_comparison(params: Dict[str, Any], formatter: EnhancedResponseFormatter) -> Dict[str, Any]:
    """Handle enhanced game comparison."""
    try:
        games = params.get("games", [])
        if len(games) < 2:
            return {"success": False, "message": "At least 2 games required for comparison"}
        
        logging.info(f"🎮 Enhanced game comparison for: {games}")
        
        # Check compatibility for each game
        comparison_results = []
        for game in games[:3]:  # Limit to 3 games
            try:
                # Use proper async handling
                try:
                    # Try to get current loop
                    loop = asyncio.get_running_loop()
                    # If we're already in a loop, use await directly
                    result = await canrun_engine.check_game_compatibility(game)
                except RuntimeError:
                    # No running loop, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(canrun_engine.check_game_compatibility(game))
                    finally:
                        loop.close()
                
                if result:
                    comparison_results.append({
                        'success': True,
                        'game_name': result.game_name,
                        'can_run': result.can_run_game(),
                        'performance_tier': result.performance_prediction.tier.name if hasattr(result.performance_prediction, 'tier') else 'Unknown',
                        'expected_fps': getattr(result.performance_prediction, 'expected_fps', 0)
                    })
                else:
                    comparison_results.append({
                        'success': False,
                        'game_name': game,
                        'message': f"Could not analyze {game}"
                    })
            except Exception as e:
                comparison_results.append({
                    'success': False,
                    'game_name': game,
                    'message': f"Error analyzing {game}: {str(e)}"
                })
        
        formatted_response = formatter.format_comparison_response(comparison_results)
        
        logging.info(f"✅ Enhanced game comparison successful")
        return {"success": True, "message": formatted_response}
        
    except Exception as e:
        error_msg = f"❌ Enhanced game comparison failed: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}


async def handle_steam_compare_ui(params: Dict[str, Any], steam_compare: SteamCompareUI, formatter: EnhancedResponseFormatter) -> Dict[str, Any]:
    """Handle Steam Compare UI functionality."""
    try:
        game_name = params.get("game_name", "")
        if not game_name:
            return {"success": False, "message": "Game name is required for Steam comparison"}
        
        logging.info(f"🎮 Steam Compare UI for: {game_name}")
        
        # Get Steam comparison data
        steam_result = await steam_compare.get_steam_comparison_data(game_name)
        
        if steam_result.get('success'):
            # Also get compatibility analysis
            compatibility_result = None
            try:
                # Use proper async handling
                try:
                    # Try to get current loop
                    loop = asyncio.get_running_loop()
                    # If we're already in a loop, use await directly
                    compat_result = await canrun_engine.check_game_compatibility(game_name)
                except RuntimeError:
                    # No running loop, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        compat_result = loop.run_until_complete(canrun_engine.check_game_compatibility(game_name))
                    finally:
                        loop.close()
                
                if compat_result:
                    compatibility_result = {
                        'can_run': compat_result.can_run_game(),
                        'performance_tier': compat_result.performance_prediction.tier.name if hasattr(compat_result.performance_prediction, 'tier') else 'Unknown',
                        'expected_fps': getattr(compat_result.performance_prediction, 'expected_fps', 0)
                    }
            except Exception as e:
                logging.warning(f"Compatibility analysis failed for Steam Compare: {e}")
            
            formatted_response = steam_compare.format_steam_compare_response(steam_result, compatibility_result)
            
            logging.info(f"✅ Steam Compare UI successful for {game_name}")
            return {"success": True, "message": formatted_response}
        else:
            return {"success": False, "message": steam_result.get('message', 'Steam comparison failed')}
            
    except Exception as e:
        error_msg = f"❌ Steam Compare UI failed: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}


async def handle_auto_detection(params: Dict[str, Any], steam_compare: SteamCompareUI, formatter: EnhancedResponseFormatter) -> Dict[str, Any]:
    """Handle automatic tool detection from user input."""
    try:
        user_input = params.get("user_input", "")
        if not user_input:
            return {"success": False, "message": "User input is required for auto-detection"}
        
        logging.info(f"🤖 Auto-detecting tool from input: {user_input}")
        
        # Detect tool and extract parameters
        detector = ToolNameDetector()
        detection_result = detector.detect_tool_and_extract_params(user_input)
        
        detected_tool = detection_result['tool']
        detected_params = detection_result['params']
        confidence = detection_result['confidence']
        
        logging.info(f"🎯 Detected tool: {detected_tool} (confidence: {confidence:.2f})")
        
        # Route to appropriate handler
        if detected_tool == 'check_compatibility':
            return await handle_enhanced_compatibility_check(detected_params, steam_compare, formatter)
        elif detected_tool == 'detect_hardware':
            return await handle_enhanced_hardware_detection(formatter)
        elif detected_tool == 'compare_games':
            return await handle_enhanced_game_comparison(detected_params, formatter)
        elif detected_tool == 'steam_compare':
            return await handle_steam_compare_ui(detected_params, steam_compare, formatter)
        else:
            return {"success": False, "message": f"Could not determine appropriate action from: {user_input}"}
            
    except Exception as e:
        error_msg = f"❌ Auto-detection failed: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}


def read_enhanced_command() -> Optional[Dict[str, Any]]:
    """
    Read enhanced command from stdin using standard Python input.
    Supports both traditional G-Assist commands and auto-detection.
    
    Returns:
        Optional[Dict[str, Any]]: Parsed command dictionary or None if failed
    """
    try:
        # Read from stdin
        line = sys.stdin.readline().strip()
        if not line:
            return None
            
        logging.info(f'Received enhanced command: {line}')
        
        # Try to parse as JSON first (traditional G-Assist format)
        try:
            command = json.loads(line)
            return command
        except json.JSONDecodeError:
            # If not JSON, treat as raw user input for auto-detection
            return {
                "tool_calls": [
                    {
                        "func": "auto_detect",
                        "params": {"user_input": line}
                    }
                ]
            }
            
    except Exception as e:
        logging.error(f'Error in read_enhanced_command: {e}')
        return None


def write_enhanced_response(response: Dict[str, Any]) -> None:
    """
    Write enhanced response to stdout pipe using Windows API.
    
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
        logging.info(f'Sent enhanced response: {json.dumps(response)}')
    except Exception as e:
        logging.error(f'Error writing enhanced response: {e}')


def run_enhanced_command_line_interface():
    """
    Run the enhanced plugin in command-line interface mode.
    Supports direct function calls with arguments and auto-detection.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced CanRun G-Assist Plugin - Game Compatibility Analysis with Steam Compare UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plugin.py --function check_compatibility --game "Diablo 4"
  python plugin.py --function detect_hardware
  python plugin.py --function steam_compare --game "Cyberpunk 2077"
  python plugin.py --function auto_detect --input "Can I run Elden Ring?"
  
Functions:
  check_compatibility  Check if a game can run on this system with Steam Compare UI
  detect_hardware      Detect system hardware specifications
  steam_compare        Show Steam Compare UI for a specific game
  compare_games        Compare multiple games performance
  auto_detect          Automatically detect tool from natural language input
        """
    )
    
    parser.add_argument(
        '--function',
        required=True,
        choices=['check_compatibility', 'detect_hardware', 'steam_compare', 'compare_games', 'auto_detect'],
        help='Function to execute'
    )
    
    parser.add_argument(
        '--game',
        help='Game name for compatibility check or Steam compare (required for check_compatibility, steam_compare)'
    )
    
    parser.add_argument(
        '--games',
        nargs='+',
        help='Multiple game names for comparison (required for compare_games)'
    )
    
    parser.add_argument(
        '--input',
        help='Natural language input for auto-detection (required for auto_detect)'
    )
    
    parser.add_argument(
        '--show-steam',
        action='store_true',
        help='Include Steam Compare UI data in compatibility check'
    )
    
    # Temporarily redirect stderr to suppress duplicate error messages
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # Restore stderr
        sys.stderr = old_stderr
        # If argument parsing fails, exit with error
        if e.code != 0:
            sys.exit(e.code)
    finally:
        # Always restore stderr
        sys.stderr = old_stderr
    
    # Initialize enhanced plugin
    init_result = initialize_enhanced_plugin()
    if not init_result["success"]:
        result = {"success": False, "message": init_result["message"]}
        print(json.dumps(result))
        sys.exit(1)
    
    # Build tool call based on arguments
    tool_call = {"func": args.function, "params": {}}
    
    if args.function == 'check_compatibility':
        if not args.game:
            result = {"success": False, "message": "Game name is required for compatibility check"}
        else:
            tool_call["params"] = {
                "game_name": args.game,
                "show_steam_data": args.show_steam
            }
    elif args.function == 'steam_compare':
        if not args.game:
            result = {"success": False, "message": "Game name is required for Steam compare"}
        else:
            tool_call["params"] = {"game_name": args.game}
    elif args.function == 'compare_games':
        if not args.games or len(args.games) < 2:
            result = {"success": False, "message": "At least 2 games required for comparison"}
        else:
            tool_call["params"] = {"games": args.games}
    elif args.function == 'auto_detect':
        if not args.input:
            result = {"success": False, "message": "Input text is required for auto-detection"}
        else:
            tool_call["params"] = {"user_input": args.input}
    elif args.function == 'detect_hardware':
        tool_call["params"] = {}
    
    # Execute the function if no errors
    if 'result' not in locals():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_enhanced_tool_call(tool_call))
        finally:
            loop.close()
    
    # Output result as JSON
    print(json.dumps(result, indent=2))


async def run_enhanced_g_assist_interface():
    """
    Run the enhanced plugin in G-Assist interface mode.
    Uses Windows named pipes for communication with G-Assist.
    Supports both traditional commands and auto-detection.
    """
    # Initialize enhanced plugin
    init_result = initialize_enhanced_plugin()
    if not init_result["success"]:
        logging.error(f"❌ Enhanced plugin initialization failed: {init_result['message']}")
        write_enhanced_response({"success": False, "message": f"Enhanced plugin initialization failed: {init_result['message']}"})
        sys.exit(1)
    
    logging.info("🚀 Enhanced CanRun G-Assist Plugin starting...")
    logging.info("🎮 Ready for enhanced game compatibility analysis with Steam Compare UI!")
    
    # Main interaction loop
    while True:
        try:
            # Read command from G-Assist via stdin pipe
            command = read_enhanced_command()
            if command is None:
                continue
            
            # Process tool calls
            for tool_call in command.get("tool_calls", []):
                try:
                    response = await process_enhanced_tool_call(tool_call)
                    write_enhanced_response(response)
                except Exception as e:
                    logging.error(f"Tool call processing error: {e}")
                    write_enhanced_response({"success": False, "message": f"Processing error: {str(e)}"})
                
                # Handle shutdown
                if tool_call.get("func") == "shutdown":
                    logging.info("👋 Enhanced CanRun plugin shutting down...")
                    return
            
        except KeyboardInterrupt:
            logging.info("\n👋 Enhanced CanRun plugin shutting down...")
            break
        except Exception as e:
            logging.error(f"Error in enhanced G-Assist interface: {e}")
            write_enhanced_response({"success": False, "message": f"Enhanced error: {e}"})


def main():
    """
    Main entry point for the Enhanced CanRun G-Assist plugin.
    
    Determines whether to run in command-line interface mode or G-Assist mode
    based on the presence of command-line arguments and stdin availability.
    
    Command-line mode: Used when arguments are provided (e.g., --function, --game)
    G-Assist mode: Used when no arguments are provided or stdin has data
    """
    # Add plugin discovery logging
    logging.info("🚀 CanRun G-Assist Plugin starting...")
    logging.info(f"📁 Plugin started from: {os.path.abspath(__file__)}")
    logging.info(f"📂 Working directory: {os.getcwd()}")
    logging.info(f"🔧 Python executable: {sys.executable}")
    logging.info(f"📋 Command line args: {sys.argv}")
    
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Run in command-line interface mode
        run_enhanced_command_line_interface()
    else:
        # For Windows, we'll use a different approach
        try:
            # Try to read from stdin with a timeout
            if sys.stdin.isatty():
                # No piped input, run in G-Assist mode anyway
                asyncio.run(run_enhanced_g_assist_interface())
            else:
                # Piped input detected, run in G-Assist mode
                asyncio.run(run_enhanced_g_assist_interface())
        except:
            # Fallback to G-Assist mode
            asyncio.run(run_enhanced_g_assist_interface())


if __name__ == '__main__':
    main()