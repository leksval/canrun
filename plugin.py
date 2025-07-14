"""
CanRun G-Assist Plugin - Official G-Assist Plugin Implementation
Standalone plugin for NVIDIA G-Assist platform.
Privacy-focused game compatibility analysis for RTX/GTX systems.
"""

import sys
import json
import asyncio
import logging
import os
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, TypedDict

# Add src to path for imports - PyInstaller compatible
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        base_path = sys._MEIPASS
    else:
        # Running as normal Python script
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

# Add src to path
sys.path.insert(0, get_resource_path('src'))

from canrun_engine import CanRunEngine
from privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from service_container import get_container


class Response(TypedDict, total=False):
    """Type definition for plugin response objects."""
    success: bool
    message: str


class GAssistCommunication:
    """
    G-Assist communication protocol handler.
    Handles Windows named pipes communication with proper JSON format and <<END>> markers.
    """
    
    def __init__(self):
        """Initialize G-Assist communication."""
        self.logger = logging.getLogger(__name__)
        self.input_stream = sys.stdin
        self.output_stream = sys.stdout
        
    def read_command(self) -> Optional[Dict[str, Any]]:
        """
        Read a command from G-Assist via stdin with comprehensive debugging.
        
        Returns:
            Command dictionary or None if no command available
        """
        try:
            # Log that we're waiting for G-Assist input
            self.logger.debug("🔍 Waiting for G-Assist command via stdin...")
            
            # Read input from stdin (G-Assist uses stdin for communication)
            line = self.input_stream.readline().strip()
            
            # Log raw input received
            if not line:
                self.logger.debug("📭 No input received from G-Assist (empty line)")
                return None
            
            self.logger.info(f"📨 RAW G-ASSIST INPUT: '{line}'")
            self.logger.info(f"📏 Input length: {len(line)} characters")
            
            # Parse JSON command
            try:
                command = json.loads(line)
                self.logger.info(f"✅ PARSED G-ASSIST COMMAND: {json.dumps(command, indent=2)}")
                
                # Log command structure analysis
                if isinstance(command, dict):
                    self.logger.info(f"📋 Command keys: {list(command.keys())}")
                    
                    # Check for tool_calls which is what we expect
                    if "tool_calls" in command:
                        tool_calls = command["tool_calls"]
                        self.logger.info(f"🔧 Found {len(tool_calls)} tool calls:")
                        for i, tool_call in enumerate(tool_calls):
                            func_name = tool_call.get("func", "UNKNOWN")
                            params = tool_call.get("params", {})
                            self.logger.info(f"   {i+1}. Function: {func_name}, Params: {params}")
                    else:
                        self.logger.warning("⚠️  No 'tool_calls' found in command - this may be why functions aren't triggered!")
                        
                return command
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ JSON DECODE ERROR: {e}")
                self.logger.error(f"❌ Failed to parse: '{line}'")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR reading G-Assist command: {e}")
            self.logger.error(f"❌ Exception type: {type(e).__name__}")
            return None
    
    def write_response(self, response: Dict[str, Any]) -> None:
        """
        Write a response to G-Assist via stdout with comprehensive debugging.
        
        Args:
            response: Response dictionary containing success flag and message
        """
        try:
            # Ensure response has required fields
            if "success" not in response:
                response["success"] = True
            if "message" not in response:
                response["message"] = "Command executed successfully"
            
            # Log response analysis before sending
            self.logger.info("📤 PREPARING RESPONSE FOR G-ASSIST:")
            self.logger.info(f"📤 Response success: {response.get('success', 'UNKNOWN')}")
            self.logger.info(f"📤 Response keys: {list(response.keys())}")
            
            # Log response size and structure
            response_json = json.dumps(response, ensure_ascii=False)
            self.logger.info(f"📤 Response JSON length: {len(response_json)} characters")
            self.logger.info(f"📤 FULL RESPONSE JSON: {response_json}")
            
            # Write response with G-Assist required format
            full_output = response_json + "<<END>>\n"
            self.logger.info(f"📤 SENDING TO G-ASSIST: '{full_output.strip()}'")
            
            self.output_stream.write(full_output)
            self.output_stream.flush()
            
            self.logger.info("✅ Response sent to G-Assist successfully")
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR writing response to G-Assist: {e}")
            self.logger.error(f"❌ Failed response data: {response}")
            self.logger.error(f"❌ Exception type: {type(e).__name__}")
            # Send error response
            error_response = {
                "success": False,
                "message": f"Internal error: {str(e)}"
            }
            try:
                error_json = json.dumps(error_response, ensure_ascii=False)
                self.output_stream.write(error_json + "<<END>>\n")
                self.output_stream.flush()
            except:
                pass
    
    def write_error(self, error_message: str) -> None:
        """
        Write an error response to G-Assist.
        
        Args:
            error_message: Error message to send
        """
        error_response = {
            "success": False,
            "message": error_message
        }
        self.write_response(error_response)
    
    def is_shutdown_command(self, command: Dict[str, Any]) -> bool:
        """
        Check if the command is a shutdown command.
        
        Args:
            command: Command dictionary
            
        Returns:
            True if this is a shutdown command
        """
        if not command:
            return False
        
        # Check for shutdown in tool_calls
        tool_calls = command.get("tool_calls", [])
        for tool_call in tool_calls:
            if tool_call.get("func") == "shutdown":
                return True
        
        # Check for direct shutdown command
        return command.get("command") == "shutdown"


class CanRunGAssistPlugin:
    """CanRun plugin for NVIDIA G-Assist platform."""
    
    def __init__(self):
        """Initialize the plugin with robust error handling."""
        try:
            self.setup_logging()
            self.logger = logging.getLogger(__name__)
            self.logger.info("Starting CanRun G-Assist plugin initialization...")
            
            # Initialize G-Assist communication
            self.comm = GAssistCommunication()
            
            # Initialize with fallback values in case of errors
            self.container = None
            self.engine = None
            self.privacy_detector = None
            
            try:
                # Initialize service container
                self.container = get_container()
                self._setup_services()
                self.logger.info("Service container initialized")
            except Exception as e:
                self.logger.warning(f"Service container initialization failed: {e}")
            
            try:
                # Initialize engine with cache cleared on first run
                self.engine = CanRunEngine(enable_llm=True)
                self.logger.info("CanRun engine initialized")
            except Exception as e:
                self.logger.error(f"CanRun engine initialization failed: {e}")
                # Create minimal fallback engine
                self.engine = None
            
            try:
                self.privacy_detector = PrivacyAwareHardwareDetector()
                self.logger.info("Privacy detector initialized")
            except Exception as e:
                self.logger.error(f"Privacy detector initialization failed: {e}")
                self.privacy_detector = None
            
            # Clear cache on first run to avoid corrupted JSON issues
            try:
                self._clear_cache_on_first_run()
            except Exception as e:
                self.logger.warning(f"Cache clearing failed: {e}")
            
            # Plugin metadata
            self.plugin_info = {
                'name': 'CanRun Universal Game Compatibility Checker',
                'version': '3.0.0',
                'description': 'Privacy-focused game compatibility analysis for RTX/GTX systems',
                'requirements': ['NVIDIA RTX/GTX GPU', 'G-Assist Compatible'],
                'privacy_policy': 'Privacy-by-design with anonymized hardware detection'
            }
            
            # Validate GPU requirement (non-critical)
            try:
                self._validate_system()
            except Exception as e:
                self.logger.warning(f"System validation failed: {e}")
            
            self.logger.info("CanRun G-Assist plugin initialized successfully")
            
        except Exception as e:
            # Critical initialization failure - log but don't crash
            try:
                if hasattr(self, 'logger'):
                    self.logger.critical(f"Plugin initialization failed: {e}")
                else:
                    print(f"CRITICAL: Plugin initialization failed: {e}")
            except:
                print(f"CRITICAL: Plugin initialization failed: {e}")
    
    def setup_logging(self):
        """Setup comprehensive logging configuration for G-Assist debugging."""
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create detailed logging configuration
        logging.basicConfig(
            level=logging.DEBUG,  # Changed to DEBUG for comprehensive G-Assist tracking
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('canrun_plugin.log', mode='w'),  # Overwrite each run for clarity
                logging.StreamHandler()  # Also output to console for G-Assist visibility
            ]
        )
        
        # Log startup and discovery information
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("🚀 CANRUN G-ASSIST PLUGIN STARTUP - DEBUG MODE ENABLED")
        logger.info("=" * 80)
        
        # Log command line arguments for G-Assist discovery debugging
        logger.info(f"📋 Command line arguments: {sys.argv}")
        
        # Log environment information
        logger.info(f"🏠 Working directory: {os.getcwd()}")
        logger.info(f"🐍 Python executable: {sys.executable}")
        logger.info(f"📂 Plugin file location: {__file__}")
        
        # Check if running as PyInstaller executable
        if getattr(sys, 'frozen', False):
            logger.info(f"📦 Running as PyInstaller executable from: {sys._MEIPASS}")
        else:
            logger.info("🐍 Running as Python script")
            
        # Log manifest.json existence and readability
        manifest_path = os.path.join(os.path.dirname(__file__), 'manifest.json')
        if os.path.exists(manifest_path):
            logger.info(f"✅ Manifest found at: {manifest_path}")
            try:
                with open(manifest_path, 'r') as f:
                    manifest_content = f.read()
                    logger.info(f"📜 Manifest size: {len(manifest_content)} bytes")
                    logger.debug(f"📜 Manifest content preview: {manifest_content[:200]}...")
            except Exception as e:
                logger.error(f"❌ Failed to read manifest: {e}")
        else:
            logger.error(f"❌ Manifest NOT found at: {manifest_path}")
    
    def _setup_services(self):
        """Setup dependency injection services."""
        try:
            # Register services in container
            self.container.register_singleton('hardware_detector', 
                                            lambda: PrivacyAwareHardwareDetector())
            self.logger.debug("Services registered successfully")
        except Exception as e:
            self.logger.error(f"Error setting up services: {e}")
    
    def _clear_cache_on_first_run(self):
        """Clear cache on first run to avoid corrupted JSON issues."""
        try:
            import glob
            cache_files = glob.glob("cache/*.json")
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                    self.logger.info(f"Cleared cache file: {cache_file}")
                except Exception as e:
                    self.logger.warning(f"Could not clear cache file {cache_file}: {e}")
        except Exception as e:
            self.logger.warning(f"Cache clearing failed: {e}")
    
    def _validate_system(self):
        """Validate that the system meets requirements."""
        try:
            # Basic system validation
            if not self.privacy_detector.has_nvidia_gpu():
                self.logger.warning("No NVIDIA GPU detected - some features may be limited")
            
            self.logger.info("System validation completed")
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
    
    async def handle_detect_hardware(self, params: dict) -> Response:
        """Handle hardware detection request."""
        try:
            hardware_info = await self.privacy_detector.get_hardware_specs()
            
            # Format hardware info for G-Assist
            hardware_summary = f"""
# System Hardware Specifications

## Graphics Card
- **GPU**: {hardware_info.gpu_model}
- **VRAM**: {hardware_info.gpu_vram_gb} GB
- **Vendor**: {hardware_info.gpu_vendor}
- **RTX Support**: {hardware_info.supports_rtx}
- **DLSS Support**: {hardware_info.supports_dlss}

## Processor
- **CPU**: {hardware_info.cpu_model}
- **Cores**: {hardware_info.cpu_cores}
- **Threads**: {hardware_info.cpu_threads}

## Memory
- **RAM**: {hardware_info.ram_total_gb} GB
- **Speed**: {hardware_info.ram_speed_mhz} MHz

## Storage
- **Primary**: {hardware_info.storage_type}

## Display
- **Resolution**: {hardware_info.primary_monitor_resolution}
- **Refresh Rate**: {hardware_info.primary_monitor_refresh_hz} Hz

## Operating System
- **Version**: {hardware_info.os_version}
- **DirectX**: {hardware_info.directx_version}
- **NVIDIA Driver**: {hardware_info.nvidia_driver_version}
"""
            
            return {
                "success": True,
                "message": hardware_summary
            }
            
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            return {
                "success": False,
                "message": f"Hardware detection failed: {str(e)}"
            }

    async def handle_check_compatibility(self, params: dict) -> Response:
        """Handle game compatibility check with new minimum requirements logic."""
        try:
            game_name = params.get('game_name', '')
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Use the correct method name and get full CanRunResult
            result = await self.engine.check_game_compatibility(game_name)
            
            if result:
                # Return structured data for G-Assist LLM processing
                structured_data = {
                    "game_name": result.game_name,
                    "timestamp": result.timestamp,
                    "canrun_status": {
                        "can_run_minimum": result.can_run_game(),
                        "can_run_recommended": result.exceeds_recommended_requirements(),
                        "runnable_message": result.get_runnable_status_message(),
                        "minimum_requirements_status": result.get_minimum_requirements_status()
                    },
                    "hardware_specs": {
                        "gpu_model": result.hardware_specs.gpu_model,
                        "gpu_vram_gb": result.hardware_specs.gpu_vram_gb,
                        "cpu_model": result.hardware_specs.cpu_model,
                        "cpu_cores": result.hardware_specs.cpu_cores,
                        "ram_total_gb": result.hardware_specs.ram_total_gb,
                        "supports_rtx": result.hardware_specs.supports_rtx,
                        "supports_dlss": result.hardware_specs.supports_dlss,
                        "os_version": result.hardware_specs.os_version,
                        "nvidia_driver_version": result.hardware_specs.nvidia_driver_version
                    },
                    "game_requirements": {
                        "minimum": {
                            "cpu": result.game_requirements.minimum_cpu,
                            "gpu": result.game_requirements.minimum_gpu,
                            "ram_gb": result.game_requirements.minimum_ram_gb,
                            "storage_gb": result.game_requirements.minimum_storage_gb,
                            "directx": result.game_requirements.minimum_directx,
                            "os": result.game_requirements.minimum_os
                        },
                        "recommended": {
                            "cpu": result.game_requirements.recommended_cpu,
                            "gpu": result.game_requirements.recommended_gpu,
                            "ram_gb": result.game_requirements.recommended_ram_gb,
                            "storage_gb": result.game_requirements.recommended_storage_gb,
                            "directx": result.game_requirements.recommended_directx,
                            "os": result.game_requirements.recommended_os
                        }
                    },
                    "compatibility_analysis": {
                        "overall_compatibility": result.compatibility_analysis.overall_compatibility.value,
                        "overall_score": result.compatibility_analysis.overall_score,
                        "summary": result.compatibility_analysis.summary,
                        "bottlenecks": result.compatibility_analysis.bottlenecks,
                        "can_run_minimum": result.compatibility_analysis.can_run_minimum,
                        "can_run_recommended": result.compatibility_analysis.can_run_recommended
                    },
                    "performance_prediction": {
                        "predictions": [
                            {
                                "quality_preset": pred.quality_preset.value,
                                "resolution": pred.resolution.value,
                                "expected_fps": pred.expected_fps,
                                "frame_time_ms": pred.frame_time_ms,
                                "gpu_utilization": pred.gpu_utilization,
                                "vram_usage_gb": pred.vram_usage_gb
                            } for pred in result.performance_prediction.predictions
                        ],
                        "optimization_suggestions": result.performance_prediction.optimization_suggestions
                    },
                    "analysis_metadata": {
                        "cache_used": result.cache_used,
                        "analysis_time_ms": result.analysis_time_ms,
                        "llm_analysis_available": result.llm_analysis is not None
                    }
                }
                
                # Extract S-A-B-C-D-F tier information for prominent display
                performance_tier = "Unknown"
                tier_description = "Analysis pending"
                tier_score = 0
                expected_fps = 0
                
                if result.performance_prediction and hasattr(result.performance_prediction, 'tier'):
                    performance_tier = result.performance_prediction.tier.name
                    tier_description = result.performance_prediction.tier_description
                    tier_score = result.performance_prediction.score
                    expected_fps = result.performance_prediction.expected_fps
                
                # Create G-Assist optimized response with prominent tier display
                compatibility_verdict = "EXCELLENT" if result.exceeds_recommended_requirements() else "GOOD" if result.can_run_game() else "INSUFFICIENT"
                
                # Format comprehensive response for G-Assist LLM with prominent tier display
                g_assist_summary = {
                    "performance_tier": performance_tier,
                    "tier_description": tier_description,
                    "tier_score": tier_score,
                    "expected_fps": expected_fps,
                    "compatibility_verdict": compatibility_verdict,
                    "can_run": result.can_run_game(),
                    "exceeds_recommended": result.exceeds_recommended_requirements(),
                    "overall_score": result.compatibility_analysis.overall_score,
                    "rtx_features": result.hardware_specs.supports_rtx,
                    "dlss_support": result.hardware_specs.supports_dlss,
                    "tier_summary": f"Tier {performance_tier} ({tier_score}/100) - {tier_description} - Expected {expected_fps} FPS",
                    "hardware_verdict": {
                        "gpu": f"{result.hardware_specs.gpu_model} vs {result.game_requirements.minimum_gpu}",
                        "cpu": f"{result.hardware_specs.cpu_model} vs {result.game_requirements.minimum_cpu}",
                        "ram": f"{result.hardware_specs.ram_total_gb}GB vs {result.game_requirements.minimum_ram_gb}GB"
                    }
                }
                
                return {
                    "success": True,
                    "data": structured_data,
                    "g_assist_summary": g_assist_summary,
                    "message": f"CanRun Analysis: {result.game_name} - Tier {performance_tier} - {compatibility_verdict}"
                }
            else:
                return {
                    "success": False,
                    "message": f"Could not analyze compatibility for {game_name}"
                }
                
        except Exception as e:
            self.logger.error(f"Compatibility check failed: {e}")
            return {
                "success": False,
                "message": f"Compatibility check failed: {str(e)}"
            }
    
    async def handle_predict_performance(self, params: dict) -> Response:
        """Handle performance prediction request."""
        try:
            game_name = params.get('game_name', '')
            settings = params.get('settings', 'High')
            resolution = params.get('resolution', '1080p')
            
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Get performance prediction using analyze_game_compatibility
            result = await self.engine.analyze_game_compatibility(game_name, settings, resolution)
            
            if result:
                performance = result.get('performance', {})
                
                response_text = f"""
# Performance Prediction: {game_name}

## Settings
- **Graphics**: {settings}
- **Resolution**: {resolution}

## Performance Metrics
- **Expected FPS**: {performance.get('fps', 'Unknown')}
- **Performance Level**: {performance.get('performance_level', 'Unknown')}
- **Frame Stability**: {performance.get('stability', 'Unknown')}

## RTX Features
- **DLSS Available**: {performance.get('dlss_available', 'Unknown')}
- **Ray Tracing**: {performance.get('ray_tracing_support', 'Unknown')}
"""
                
                # Add optimization suggestions
                optimizations = result.get('optimization_suggestions', [])
                if optimizations:
                    response_text += f"\n## Optimization Recommendations\n"
                    for opt in optimizations:
                        response_text += f"- {opt}\n"
                
                return {
                    "success": True,
                    "message": response_text
                }
            else:
                return {
                    "success": False,
                    "message": f"Could not predict performance for {game_name}"
                }
                
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            return {
                "success": False,
                "message": f"Performance prediction failed: {str(e)}"
            }
    
    async def handle_predict_advanced_performance(self, params: dict) -> Response:
        """Handle advanced tiered performance assessment request."""
        try:
            game_name = params.get('game_name', '')
            
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Get hardware specifications
            hardware_specs = await self.engine.hardware_detector.get_hardware_specs()
            hardware_dict = hardware_specs.to_dict()
            
            # Get advanced performance assessment
            assessment = await self.engine._predict_advanced_performance(hardware_dict)
            
            # Check if minimal requirements are met (tier D or better)
            can_run = assessment['tier'] in ['S', 'A', 'B', 'C', 'D']
            canrun_status = "🎮 **CANRUN!** ✅" if can_run else "❌ **Cannot Run** - Below minimum requirements"
            
            response_text = f"""
# Advanced Performance Assessment: {game_name}

{canrun_status}

## Performance Tier: {assessment['tier']}
**Score**: {assessment['score']}/100
**Description**: {assessment['tier_description']}

## Performance Metrics
- **Expected FPS**: {assessment['expected_fps']}
- **Recommended Settings**: {assessment['recommended_settings']}
- **Recommended Resolution**: {assessment['recommended_resolution']}

## System Analysis
"""
            
            if assessment['bottlenecks']:
                response_text += f"**Bottlenecks**: {', '.join(assessment['bottlenecks'])}\n"
            else:
                response_text += "**Bottlenecks**: None detected\n"
            
            if assessment['upgrade_suggestions']:
                response_text += f"\n## Upgrade Recommendations\n"
                for suggestion in assessment['upgrade_suggestions']:
                    response_text += f"- {suggestion}\n"
            
            return {
                "success": True,
                "message": response_text
            }
            
        except Exception as e:
            self.logger.error(f"Advanced performance assessment failed: {e}")
            return {
                "success": False,
                "message": f"Advanced performance assessment failed: {str(e)}"
            }
    
    async def handle_get_optimization_suggestions(self, params: dict) -> Response:
        """Handle optimization suggestions request."""
        try:
            game_name = params.get('game_name', '')
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Get optimization suggestions with settings and resolution
            settings = params.get('settings', 'High')
            resolution = params.get('resolution', '1080p')
            suggestions = await self.engine.get_optimization_suggestions(game_name, settings, resolution)
            
            if suggestions:
                response_text = f"""
# Optimization Recommendations: {game_name}

## Performance Optimizations
"""
                
                for suggestion in suggestions:
                    response_text += f"- **{suggestion.get('category', 'General')}**: {suggestion.get('description', '')}\n"
                
                return {
                    "success": True,
                    "message": response_text
                }
            else:
                return {
                    "success": False,
                    "message": f"No optimization suggestions available for {game_name}"
                }
                
        except Exception as e:
            self.logger.error(f"Optimization suggestions failed: {e}")
            return {
                "success": False,
                "message": f"Optimization suggestions failed: {str(e)}"
            }
    
    async def handle_get_intelligent_analysis(self, params: dict) -> Response:
        """Handle intelligent LLM analysis request."""
        try:
            game_name = params.get('game_name', '')
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Get intelligent analysis using G-Assist LLM
            analysis = await self.engine.get_intelligent_analysis(game_name)
            
            if analysis:
                response_text = f"""
# Intelligent Analysis: {game_name}

{analysis.get('analysis', 'Analysis not available')}

## Key Insights
{analysis.get('insights', 'No insights available')}

## Recommendations
{analysis.get('recommendations', 'No recommendations available')}
"""
                
                return {
                    "success": True,
                    "message": response_text
                }
            else:
                return {
                    "success": False,
                    "message": f"Could not generate intelligent analysis for {game_name}"
                }
                
        except Exception as e:
            self.logger.error(f"Intelligent analysis failed: {e}")
            return {
                "success": False,
                "message": f"Intelligent analysis failed: {str(e)}"
            }
    
    async def handle_ask_intelligent_question(self, params: dict) -> Response:
        """Handle intelligent question request."""
        try:
            query = params.get('query', '')
            game_name = params.get('game_name', '')
            
            if not query:
                return {
                    "success": False,
                    "message": "Query is required"
                }
            
            # Process intelligent question using G-Assist LLM
            answer = await self.engine.ask_intelligent_question(query, game_name)
            
            if answer:
                response_text = f"""
# Gaming Performance Query

**Question**: {query}

**Answer**: {answer}
"""
                
                return {
                    "success": True,
                    "message": response_text
                }
            else:
                return {
                    "success": False,
                    "message": "Could not process the question"
                }
                
        except Exception as e:
            self.logger.error(f"Intelligent question failed: {e}")
            return {
                "success": False,
                "message": f"Question processing failed: {str(e)}"
            }
    
    async def process_command(self, command: dict) -> dict:
        """Process a command from G-Assist."""
        try:
            # Extract tool calls from command
            tool_calls = command.get("tool_calls", [])
            
            if not tool_calls:
                return {
                    "success": False,
                    "message": "No tool calls found in command"
                }
            
            # Process first tool call
            tool_call = tool_calls[0]
            func_name = tool_call.get("func", "")
            params = tool_call.get("params", {})
            
            # Route to appropriate handler
            if func_name == "detect_hardware":
                return await self.handle_detect_hardware(params)
            elif func_name == "check_compatibility":
                return await self.handle_check_compatibility(params)
            elif func_name == "predict_performance":
                return await self.handle_predict_performance(params)
            elif func_name == "get_optimization_suggestions":
                return await self.handle_get_optimization_suggestions(params)
            elif func_name == "get_intelligent_analysis":
                return await self.handle_get_intelligent_analysis(params)
            elif func_name == "ask_intelligent_question":
                return await self.handle_ask_intelligent_question(params)
            else:
                return {
                    "success": False,
                    "message": f"Unknown function: {func_name}"
                }
                
        except Exception as e:
            # Enhanced error handling with detailed logging
            self.logger.error(f"Command processing failed: {e}")
            self.logger.error(f"Command data: {command}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return structured error response for G-Assist
            return {
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "context": "command_processing"
                },
                "message": f"Command processing failed: {str(e)}"
            }
    
    async def run(self):
        """Main plugin loop with comprehensive G-Assist communication tracking."""
        self.logger.info("🚀 STARTING G-ASSIST COMMUNICATION LOOP")
        self.logger.info("🔍 Plugin ready to receive G-Assist commands via stdin...")
        
        # Track communication statistics
        command_count = 0
        successful_responses = 0
        failed_responses = 0
        
        try:
            while True:
                self.logger.debug("🔄 Waiting for next G-Assist command...")
                
                # Read command from G-Assist
                command = self.comm.read_command()
                
                if command is None:
                    # No command received, continue listening
                    await asyncio.sleep(0.1)
                    continue
                
                # Track received commands
                command_count += 1
                self.logger.info(f"🎯 PROCESSING COMMAND #{command_count}")
                
                # Check for shutdown command
                if self.comm.is_shutdown_command(command):
                    self.logger.info("🛑 Shutdown command received from G-Assist")
                    break
                
                # Process command with enhanced error handling
                try:
                    self.logger.info("⚙️  Processing G-Assist command...")
                    response = await self.process_command(command)
                    
                    # Send response back to G-Assist with communication error handling
                    try:
                        self.comm.write_response(response)
                        successful_responses += 1
                        self.logger.info(f"✅ Successfully processed command #{command_count} (Success rate: {successful_responses}/{command_count})")
                    except Exception as comm_err:
                        failed_responses += 1
                        self.logger.error(f"❌ Failed to send response to G-Assist for command #{command_count}: {comm_err}")
                        self.logger.error(f"❌ Response data: {response}")
                        self.logger.error(f"❌ Communication failure rate: {failed_responses}/{command_count}")
                        
                        # Try to send error notification to G-Assist
                        try:
                            self.comm.write_error(f"Communication error: {str(comm_err)}")
                        except:
                            self.logger.critical("💥 CRITICAL: Unable to communicate with G-Assist - plugin may be disconnected")
                
                except Exception as cmd_err:
                    failed_responses += 1
                    self.logger.error(f"❌ Command processing error for command #{command_count}: {cmd_err}")
                    self.logger.error(f"❌ Command data: {command}")
                    self.logger.error(f"❌ Processing failure rate: {failed_responses}/{command_count}")
                    
                    # Try to send error response to G-Assist
                    error_response = {
                        "success": False,
                        "error": {
                            "type": type(cmd_err).__name__,
                            "message": str(cmd_err),
                            "context": "command_execution"
                        },
                        "message": f"Command execution failed: {str(cmd_err)}"
                    }
                    
                    try:
                        self.comm.write_response(error_response)
                    except:
                        self.logger.critical("💥 CRITICAL: Unable to send error response to G-Assist")
                
        except KeyboardInterrupt:
            self.logger.info("Plugin interrupted by user")
        except Exception as e:
            self.logger.error(f"Plugin fatal error: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            try:
                self.comm.write_error(f"Plugin fatal error: {str(e)}")
            except:
                self.logger.critical("Unable to notify G-Assist of fatal error")
        finally:
            self.logger.info("CanRun G-Assist plugin stopped")


async def handle_cli_command(plugin, function_name, params=None):
    """Handle command line interface commands."""
    if params is None:
        params = {}
    
    # Create a mock command for CLI execution with correct G-Assist format
    command = {
        "tool_calls": [
            {
                "func": function_name,
                "params": params
            }
        ]
    }
    
    # Process the command
    response = await plugin.process_command(command)
    
    # Print the response in a clean format for CLI
    if response.get("success"):
        # For CLI, just print the formatted message instead of JSON
        message = response.get("message", "")
        print(message)
    else:
        print(f"Error: {response.get('message', 'Unknown error')}")
        sys.exit(1)


def main():
    """Main entry point for G-Assist plugin with comprehensive argument debugging."""
    import argparse
    
    # Create logger before plugin initialization for early debugging
    logger = logging.getLogger(__name__)
    
    # Log startup information
    logger.info("🎯 MAIN ENTRY POINT - G-ASSIST PLUGIN STARTING")
    logger.info(f"🔧 Python version: {sys.version}")
    logger.info(f"📋 Raw sys.argv: {sys.argv}")
    
    parser = argparse.ArgumentParser(description='CanRun G-Assist Plugin')
    parser.add_argument('--function', help='Function to execute (for CLI testing)')
    parser.add_argument('--game', help='Game name parameter')
    parser.add_argument('--settings', help='Graphics settings (Low, Medium, High, Ultra)')
    parser.add_argument('--resolution', help='Resolution (1080p, 1440p, 4K)')
    parser.add_argument('--query', help='Query for intelligent questions')
    
    # G-Assist multiprocessing arguments (critical for G-Assist discovery)
    parser.add_argument('--multiprocessing-fork', nargs='*', help='G-Assist multiprocessing arguments')
    
    # Use parse_known_args to handle any unknown G-Assist arguments gracefully
    args, unknown = parser.parse_known_args()
    
    # Comprehensive argument logging for G-Assist debugging
    logger.info("📋 PARSED ARGUMENTS:")
    logger.info(f"   --function: {args.function}")
    logger.info(f"   --game: {args.game}")
    logger.info(f"   --settings: {args.settings}")
    logger.info(f"   --resolution: {args.resolution}")
    logger.info(f"   --query: {args.query}")
    logger.info(f"   --multiprocessing-fork: {args.multiprocessing_fork}")
    
    # Log unknown arguments - critical for G-Assist integration debugging
    if unknown:
        logger.warning(f"⚠️  UNKNOWN ARGUMENTS from G-Assist: {unknown}")
        logger.warning("⚠️  This might indicate G-Assist is sending unexpected parameters")
    else:
        logger.info("✅ No unknown arguments - G-Assist communication format looks correct")
    
    # Detect execution mode
    if args.function:
        logger.info(f"🧪 RUNNING IN CLI TEST MODE - Function: {args.function}")
    elif args.multiprocessing_fork is not None:
        logger.info("🔗 RUNNING IN G-ASSIST MULTIPROCESSING MODE")
        logger.info("🔗 This indicates G-Assist has successfully launched our plugin!")
    else:
        logger.info("🚀 RUNNING IN STANDARD G-ASSIST MODE")
    
    try:
        logger.info("🏗️  Initializing CanRun G-Assist plugin...")
        plugin = CanRunGAssistPlugin()
        
        # Handle CLI mode
        if args.function:
            logger.info(f"🧪 Executing CLI function: {args.function}")
            params = {}
            if args.game:
                params['game_name'] = args.game
            if args.settings:
                params['settings'] = args.settings
            if args.resolution:
                params['resolution'] = args.resolution
            if args.query:
                params['query'] = args.query
            
            logger.info(f"🧪 CLI parameters: {params}")
            asyncio.run(handle_cli_command(plugin, args.function, params))
        else:
            # Run in G-Assist mode
            logger.info("🚀 Entering G-Assist communication mode...")
            logger.info("🔍 Plugin will now listen for G-Assist commands via stdin")
            asyncio.run(plugin.run())
    except Exception as e:
        logger.critical(f"💥 CRITICAL FAILURE starting plugin: {e}")
        logger.critical(f"💥 Exception type: {type(e).__name__}")
        logger.critical(f"💥 This failure prevents G-Assist from using our plugin!")
        import traceback
        logger.critical(f"💥 Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()