#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from ctypes import byref, windll, wintypes
from datetime import datetime

# CRITICAL FIX: Set UTF-8 encoding for Windows console (Official Microsoft solution)
# Based on: https://learn.microsoft.com/en-us/windows/console/console-application-issues
if sys.platform == "win32":
    try:
        # Set console code pages to UTF-8 (CP_UTF8 = 65001)
        windll.kernel32.SetConsoleCP(65001)
        windll.kernel32.SetConsoleOutputCP(65001)
        
        # Also configure Python's stdout/stderr encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Fallback: ignore encoding errors silently
        pass

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

# Configure logging with detailed format - FIXED: Prevent stdout contamination and Unicode issues
LOG_FILE = os.path.join(os.environ.get('USERPROFILE', '.'), 'canrun-enhanced-plugin.log')

# Only configure logging if not already configured (prevents duplicate handlers)
if not logging.getLogger().handlers:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        filemode='a',  # Append mode to prevent file conflicts
        encoding='utf-8'  # CRITICAL FIX: Handle Unicode characters properly
    )

# Ensure no console output during G-Assist mode
logging.getLogger().handlers = [h for h in logging.getLogger().handlers if not isinstance(h, logging.StreamHandler)]

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
        logging.info("Initializing Enhanced CanRun G-Assist plugin...")
        
        # Initialize CanRun engine with Steam API integration
        canrun_engine = CanRunEngine(enable_llm=True)
        logging.info("CanRun engine initialized with Steam API integration")
        
        # Initialize privacy-aware hardware detector
        privacy_detector = PrivacyAwareHardwareDetector()
        logging.info("Privacy detector initialized")
        
        # Initialize dynamic performance predictor
        performance_predictor = DynamicPerformancePredictor()
        logging.info("Dynamic performance predictor initialized")
        
        # Initialize requirements fetcher
        requirements_fetcher = GameRequirementsFetcher()
        logging.info("Game requirements fetcher initialized")
        
        logging.info("Enhanced CanRun G-Assist plugin initialized successfully")
        return {"success": True, "message": "Enhanced CanRun plugin initialized successfully"}
        
    except Exception as e:
        # CRITICAL FIX: Comprehensive Unicode error handling for compiled executable
        try:
            # Try to log with Unicode characters
            logging.error(f"Enhanced plugin initialization failed: {e}")
        except (UnicodeEncodeError, UnicodeDecodeError, OSError):
            # Fallback: Log without Unicode characters and handle all encoding issues
            try:
                error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                logging.error(f"Enhanced plugin initialization failed: {error_msg}")
            except:
                # Ultimate fallback: minimal logging
                logging.error("Enhanced plugin initialization failed: encoding error")
        
        # CRITICAL FIX: Ensure response message is completely safe for all environments
        try:
            safe_error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        except:
            safe_error_msg = "initialization error (encoding issue)"
        
        return {"success": False, "message": f"Enhanced plugin initialization failed: {safe_error_msg}"}


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
    """Handle enhanced compatibility check with Steam Compare UI and improved timeout handling."""
    try:
        game_name = params.get("game_name", "")
        include_steam_ui = params.get("show_steam_data", True)
        
        if not game_name:
            return {"success": False, "message": "Game name is required"}
        
        logging.info(f"Enhanced compatibility check for: {game_name}")
        logging.info(f"Step 1: Initializing CanRun engine for {game_name}")
        
        # CRITICAL FIX: Always fetch Steam data first to ensure it's available
        steam_data = None
        steam_requirements_text = ""
        
        # Get Steam comparison data with improved error handling
        try:
            logging.info(f"Step 1.5: Fetching Steam data for {game_name}")
            steam_result = await asyncio.wait_for(
                steam_compare.get_steam_comparison_data(game_name),
                timeout=20.0  # Increased timeout for Steam data
            )
            if steam_result.get('success'):
                steam_data = steam_result['steam_data']
                logging.info(f"Steam data retrieved for {game_name}")
                
                # Format Steam requirements for inclusion in response
                min_req = steam_data.get('minimum_requirements', {})
                rec_req = steam_data.get('recommended_requirements', {})
                steam_requirements_text = f"""
📊 **STEAM REQUIREMENTS COMPARISON**
**Minimum**: {min_req.get('cpu', 'Unknown')} | {min_req.get('gpu', 'Unknown')} | {min_req.get('ram', 'Unknown')}
**Recommended**: {rec_req.get('cpu', 'Unknown')} | {rec_req.get('gpu', 'Unknown')} | {rec_req.get('ram', 'Unknown')}"""
            else:
                logging.warning(f"Steam data unavailable for {game_name}: {steam_result.get('message', 'Unknown error')}")
                steam_requirements_text = f"\n📊 **STEAM DATA**: Not available for {game_name}"
        except asyncio.TimeoutError:
            logging.warning(f"Steam data timeout for {game_name}, proceeding without Steam data")
            steam_requirements_text = f"\n📊 **STEAM DATA**: Timeout fetching data for {game_name}"
        except Exception as e:
            logging.warning(f"Steam data error for {game_name}: {e}")
            steam_requirements_text = f"\n📊 **STEAM DATA**: Error fetching data - {str(e)}"
        
        # Run compatibility check with timeout protection
        try:
            # Use asyncio.wait_for with timeout to prevent hanging in G-Assist
            result = await asyncio.wait_for(
                canrun_engine.check_game_compatibility(game_name),
                timeout=25.0  # Reduced timeout since Steam data is already fetched
            )
            logging.info(f"Step 2: CanRun engine completed for {game_name}")
        except asyncio.TimeoutError:
            logging.error(f"❌ Timeout checking compatibility for {game_name}")
            # FALLBACK: Provide basic response with Steam data if available
            fallback_message = f"⏰ **{game_name}** - Analysis timeout, but here's what we found:{steam_requirements_text}"
            return {"success": True, "message": fallback_message}
        except RuntimeError as e:
            # Handle event loop issues in G-Assist environment
            logging.warning(f"Event loop issue for {game_name}: {e}")
            try:
                # Fallback: try without event loop management
                result = await canrun_engine.check_game_compatibility(game_name)
                logging.info(f"Step 2: CanRun engine completed (fallback) for {game_name}")
            except Exception as fallback_error:
                logging.error(f"❌ Fallback failed for {game_name}: {fallback_error}")
                # FALLBACK: Provide basic response with Steam data if available
                fallback_message = f"❌ **{game_name}** - Engine error, but here's Steam data:{steam_requirements_text}"
                return {"success": True, "message": fallback_message}
        
        if result:
            logging.info(f"Step 3: Processing results for {game_name}")
            
            # ENHANCED: Extract comprehensive performance data with LLM analysis
            try:
                can_run = result.can_run_game()
                performance_tier = "Unknown"
                expected_fps = 0
                gpu_model = "Unknown GPU"
                
                # Extract performance tier
                if hasattr(result.performance_prediction, 'tier'):
                    performance_tier = result.performance_prediction.tier.name
                elif hasattr(result, 'performance_tier'):
                    performance_tier = result.performance_tier
                
                # Extract expected FPS
                if hasattr(result.performance_prediction, 'expected_fps'):
                    expected_fps = result.performance_prediction.expected_fps
                elif hasattr(result, 'expected_fps'):
                    expected_fps = result.expected_fps
                
                # Extract GPU model
                if hasattr(result, 'hardware_specs') and hasattr(result.hardware_specs, 'gpu_model'):
                    gpu_model = result.hardware_specs.gpu_model
                
                # Enhanced LLM analysis using structured context
                llm_insights = ""
                if hasattr(canrun_engine, 'compatibility_analyzer') and canrun_engine.compatibility_analyzer.llm_analyzer:
                    try:
                        # Get comprehensive LLM analysis context
                        llm_context = await canrun_engine.compatibility_analyzer.get_llm_analysis_context(
                            game_name, result.hardware_specs, result.requirements, result.compatibility_analysis
                        )
                        
                        # Extract LLM insights if available
                        if 'llm_analysis' in llm_context and 'analysis_text' in llm_context['llm_analysis']:
                            llm_insights = f"\n\n🤖 **AI INSIGHTS**\n{llm_context['llm_analysis']['analysis_text'][:200]}..."
                            logging.info(f"LLM analysis integrated for {game_name}")
                        
                    except Exception as e:
                        logging.warning(f"LLM analysis integration failed: {e}")
                
                # Create comprehensive response
                verdict_emoji = "✅" if can_run else "❌"
                verdict_text = "YES! Your system can run this game" if can_run else "Your system may struggle with this game"
                
                # Performance recommendation based on tier and FPS
                if performance_tier in ['S', 'A'] and expected_fps >= 60:
                    settings_rec = "Ultra settings at 4K"
                elif performance_tier in ['A', 'B'] and expected_fps >= 45:
                    settings_rec = "High settings at 1440p"
                elif performance_tier in ['B', 'C'] and expected_fps >= 30:
                    settings_rec = "Medium settings at 1080p"
                else:
                    settings_rec = "Low settings recommended"
                
                # Build comprehensive G-Assist response
                comprehensive_response = f"""{verdict_emoji} **{game_name}** - {verdict_text}

🎯 **PERFORMANCE ANALYSIS**
• Performance Tier: {performance_tier}
• Expected FPS: {expected_fps}
• Your GPU: {gpu_model}
• Recommended: {settings_rec}{steam_requirements_text}{llm_insights}"""

                # Add optimization tips if available
                if hasattr(result.performance_prediction, 'upgrade_suggestions') and result.performance_prediction.upgrade_suggestions:
                    tips = '; '.join(result.performance_prediction.upgrade_suggestions[:3])
                    comprehensive_response += f"""

💡 **OPTIMIZATION TIPS**
{tips}"""
                
                logging.info(f"Enhanced compatibility check successful for {game_name}")
                return {"success": True, "message": comprehensive_response}
                
            except Exception as format_error:
                logging.error(f"Response formatting error for {game_name}: {format_error}")
                # FALLBACK: Provide basic response with available data
                fallback_message = f"✅ **{game_name}** - Analysis completed with some formatting issues{steam_requirements_text}"
                return {"success": True, "message": fallback_message}
        else:
            logging.error(f"❌ CanRun engine returned None for {game_name}")
            # FALLBACK: Provide Steam data if available
            fallback_message = f"❌ **{game_name}** - Could not complete full analysis{steam_requirements_text}"
            return {"success": True, "message": fallback_message}
            
    except Exception as e:
        logging.error(f"❌ Enhanced compatibility check exception for {game_name}: {str(e)}", exc_info=True)
        return {"success": False, "message": f"Analysis failed for {game_name}: {str(e)}"}


async def handle_enhanced_hardware_detection(formatter: EnhancedResponseFormatter) -> Dict[str, Any]:
    """Handle enhanced hardware detection."""
    try:
        logging.info("Enhanced hardware detection...")
        
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
        
        logging.info("Enhanced hardware detection successful")
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
        
        logging.info(f"Enhanced game comparison for: {games}")
        
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
        
        logging.info(f"Enhanced game comparison successful")
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
        
        logging.info(f"Steam Compare UI for: {game_name}")
        
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
            
            logging.info(f"Steam Compare UI successful for {game_name}")
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
        
        logging.info(f"Auto-detecting tool from input: {user_input}")
        
        # Detect tool and extract parameters
        detector = ToolNameDetector()
        detection_result = detector.detect_tool_and_extract_params(user_input)
        
        detected_tool = detection_result['tool']
        detected_params = detection_result['params']
        confidence = detection_result['confidence']
        
        logging.info(f"Detected tool: {detected_tool} (confidence: {confidence:.2f})")
        
        # ENHANCED: Ensure Steam data is always included for compatibility checks
        if detected_tool == 'check_compatibility':
            # Force Steam data inclusion for G-Assist compatibility checks
            detected_params['show_steam_data'] = True
            logging.info(f" Routing to compatibility check with Steam data for: {detected_params.get('game_name', 'unknown game')}")
            return await handle_enhanced_compatibility_check(detected_params, steam_compare, formatter)
        elif detected_tool == 'detect_hardware':
            return await handle_enhanced_hardware_detection(formatter)
        elif detected_tool == 'compare_games':
            return await handle_enhanced_game_comparison(detected_params, formatter)
        elif detected_tool == 'steam_compare':
            return await handle_steam_compare_ui(detected_params, steam_compare, formatter)
        else:
            # FALLBACK: If no specific tool detected but game name found, default to compatibility check
            if detected_params.get('game_name'):
                logging.info(f" Fallback: Defaulting to compatibility check for: {detected_params['game_name']}")
                detected_params['show_steam_data'] = True
                return await handle_enhanced_compatibility_check(detected_params, steam_compare, formatter)
            else:
                return {"success": False, "message": f"Could not determine appropriate action from: {user_input}"}
            
    except Exception as e:
        error_msg = f" Auto-detection failed: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "message": error_msg}


def read_g_assist_command() -> Optional[Dict[str, Any]]:
    """
    Read G-Assist command using official NVIDIA G-Assist protocol.
    Based on official documentation from https://github.com/NVIDIA/G-Assist
    
    Returns:
        Optional[Dict[str, Any]]: Parsed command dictionary or None if failed
    """
    try:
        # OFFICIAL PROTOCOL: Read from stdin using standard input
        # G-Assist communicates via standard input/output, not Windows pipes
        import sys
        
        # Read line from stdin (G-Assist sends commands line by line)
        line = sys.stdin.readline()
        if not line:
            return None
            
        message = line.strip()
        if not message:
            return None
            
        logging.info(f' G-Assist command received: {message[:100]}...' if len(message) > 100 else f' G-Assist command: {message}')
        
        # Parse G-Assist JSON command
        try:
            command = json.loads(message)
            
            # Validate G-Assist command structure
            if not isinstance(command, dict):
                logging.error(f"Invalid command format: not a dictionary")
                return None
                
            # Official G-Assist format: {"tool_calls": [{"func": "function_name", "params": {...}}]}
            if "tool_calls" not in command:
                # Convert simple command to tool_calls format
                if "function" in command:
                    command = {
                        "tool_calls": [
                            {
                                "func": command["function"],
                                "params": command.get("params", {})
                            }
                        ]
                    }
                else:
                    # Treat as natural language input
                    command = {
                        "tool_calls": [
                            {
                                "func": "auto_detect",
                                "params": {"user_input": message}
                            }
                        ]
                    }
            
            return command
            
        except json.JSONDecodeError as e:
            logging.warning(f'JSON decode error: {e}, treating as natural language')
            # If not JSON, treat as raw user input for auto-detection
            return {
                "tool_calls": [
                    {
                        "func": "auto_detect",
                        "params": {"user_input": message}
                    }
                ]
            }
            
    except Exception as e:
        logging.error(f' Error reading G-Assist command: {e}')
        return None


def write_g_assist_response(response: Dict[str, Any]) -> None:
    """
    Write G-Assist response using official NVIDIA G-Assist protocol.
    Based on official documentation from https://github.com/NVIDIA/G-Assist
    
    Args:
        response: Dictionary containing 'success' and optional 'message'
    """
    try:
        # Ensure response has required G-Assist format
        if not isinstance(response, dict):
            response = {"success": False, "message": "Invalid response format"}
        
        # OFFICIAL G-ASSIST PROTOCOL: Simple response structure
        g_assist_response = {
            "success": response.get("success", False),
            "message": response.get("message", "")
        }
        
        # Convert to JSON with ASCII-safe formatting for better compatibility
        json_response = json.dumps(g_assist_response, ensure_ascii=True)
        
        # CRITICAL: Use official G-Assist termination marker
        # Based on official documentation: responses must end with '<<END>>'
        message = json_response + '<<END>>'
        
        # OFFICIAL PROTOCOL: Write to stdout (G-Assist reads from plugin's stdout)
        import sys
        sys.stdout.write(message)
        sys.stdout.flush()  # Ensure immediate delivery to G-Assist
        
        # Log the response (to file only, never to stdout after the response)
        logging.info(f' G-Assist response sent: {json_response[:200]}...' if len(json_response) > 200 else f' G-Assist response: {json_response}')
        
    except Exception as e:
        logging.error(f' Error writing G-Assist response: {e}')
        # Fallback response to stdout
        try:
            fallback = json.dumps({"success": False, "message": f"Response error: {str(e)}"}) + '<<END>>'
            import sys
            sys.stdout.write(fallback)
            sys.stdout.flush()
        except:
            pass  # If even fallback fails, log it but don't crash


# CLI mode removed - this plugin is designed exclusively for NVIDIA G-Assist
# CLI functionality has been removed to eliminate Unicode encoding crashes
# and focus on the core G-Assist integration which works perfectly


async def run_g_assist_interface():
    """
    Run the plugin in G-Assist interface mode with official NVIDIA protocol.
    Based on official documentation from https://github.com/NVIDIA/G-Assist
    """
    try:
        # Initialize plugin components (logging only, no stdout)
        init_result = initialize_enhanced_plugin()
        
        if not init_result["success"]:
            logging.error(f" Plugin initialization failed: {init_result['message']}")
            write_g_assist_response({
                "success": False,
                "message": f"Plugin initialization failed: {init_result['message']}"
            })
            return
    
        logging.info(" CanRun G-Assist Plugin ready for communication")
        logging.info(" Waiting for G-Assist commands...")
        
        # Main G-Assist communication loop
        while True:
            try:
                # Read command from G-Assist using stdin
                command = read_g_assist_command()
                if command is None:
                    # EOF or no command received, exit gracefully
                    logging.info(" No more commands, exiting...")
                    break
                
                tool_calls = command.get("tool_calls", [])
                logging.info(f" Processing {len(tool_calls)} tool call(s)")
                
                # Process each tool call
                for tool_call in tool_calls:
                    func_name = tool_call.get('func', 'unknown')
                    
                    try:
                        logging.info(f" Executing: {func_name}")
                        
                        # Handle shutdown command immediately
                        if func_name == "shutdown":
                            logging.info(" G-Assist plugin shutting down...")
                            write_g_assist_response({
                                "success": True,
                                "message": "Plugin shutdown complete"
                            })
                            return
                        
                        # Execute with timeout to prevent G-Assist hanging
                        response = await asyncio.wait_for(
                            process_enhanced_tool_call(tool_call),
                            timeout=30.0  # 30 second timeout per function
                        )
                        
                        logging.info(f" {func_name} completed successfully")
                        write_g_assist_response(response)
                        
                    except asyncio.TimeoutError:
                        timeout_msg = f"Function '{func_name}' timed out after 30 seconds"
                        logging.error(f" {timeout_msg}")
                        write_g_assist_response({
                            "success": False,
                            "message": timeout_msg
                        })
                        
                    except Exception as e:
                        error_msg = f"Error executing '{func_name}': {str(e)}"
                        logging.error(f" {error_msg}", exc_info=True)
                        write_g_assist_response({
                            "success": False,
                            "message": error_msg
                        })
                        
            except KeyboardInterrupt:
                logging.info(" Plugin interrupted, shutting down...")
                break
                
            except EOFError:
                logging.info(" G-Assist closed connection, shutting down...")
                break
                
            except Exception as e:
                logging.error(f" G-Assist interface error: {e}", exc_info=True)
                write_g_assist_response({
                    "success": False,
                    "message": f"Interface error: {str(e)}"
                })
                
    except Exception as e:
        logging.error(f" Fatal G-Assist interface error: {e}", exc_info=True)
        try:
            write_g_assist_response({
                "success": False,
                "message": f"Fatal error: {str(e)}"
            })
        except:
            pass  # If we can't even send error response, just log and exit


def main():
    """
    Main entry point for the CanRun G-Assist plugin.
    
    This plugin is designed exclusively for NVIDIA G-Assist integration.
    CLI mode has been removed to eliminate Unicode encoding issues and
    focus on the core G-Assist functionality which works perfectly.
    """
    # Log startup information (to file only)
    logging.info("🎮 CanRun G-Assist Plugin starting...")
    logging.info(f"📁 Plugin path: {os.path.abspath(__file__)}")
    logging.info(f"📂 Working directory: {os.getcwd()}")
    logging.info(f"🐍 Python executable: {sys.executable}")
    logging.info(f"⚙️ Command line args: {sys.argv}")
    
    # This plugin runs exclusively in G-Assist mode
    logging.info("🚀 Running in G-Assist mode (CLI mode removed)")
    
    try:
        asyncio.run(run_g_assist_interface())
    except KeyboardInterrupt:
        logging.info("⏹️ Plugin interrupted")
    except Exception as e:
        logging.error(f"❌ G-Assist mode error: {e}", exc_info=True)
        # Try to send error response to G-Assist
        try:
            write_g_assist_response({
                "success": False,
                "message": f"Plugin error: {str(e)}"
            })
        except:
            pass


if __name__ == '__main__':
    main()