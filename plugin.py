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
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from canrun_engine import CanRunEngine
from privacy_aware_hardware_detector import PrivacyAwareHardwareDetector
from service_container import get_container


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
        Read a command from G-Assist via stdin.
        
        Returns:
            Command dictionary or None if no command available
        """
        try:
            # Read input from stdin (G-Assist uses stdin for communication)
            line = self.input_stream.readline().strip()
            if not line:
                return None
            
            # Parse JSON command
            command = json.loads(line)
            self.logger.debug(f"Received command: {command}")
            return command
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading command: {e}")
            return None
    
    def write_response(self, response: Dict[str, Any]) -> None:
        """
        Write a response to G-Assist via stdout.
        
        Args:
            response: Response dictionary containing success flag and message
        """
        try:
            # Ensure response has required fields
            if "success" not in response:
                response["success"] = True
            if "message" not in response:
                response["message"] = "Command executed successfully"
            
            # Convert to JSON and write with <<END>> marker
            response_json = json.dumps(response, ensure_ascii=False)
            self.output_stream.write(response_json + "<<END>>\n")
            self.output_stream.flush()
            
            self.logger.debug(f"Sent response: {response}")
            
        except Exception as e:
            self.logger.error(f"Error writing response: {e}")
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
        """Initialize the plugin."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize G-Assist communication
        self.comm = GAssistCommunication()
        
        # Initialize service container
        self.container = get_container()
        self._setup_services()
        
        # Initialize engine
        self.engine = CanRunEngine(enable_llm=True)
        self.privacy_detector = PrivacyAwareHardwareDetector()
        
        # Plugin metadata
        self.plugin_info = {
            'name': 'CanRun Universal Game Compatibility Checker',
            'version': '1.1.0',
            'description': 'Privacy-focused game compatibility analysis for RTX/GTX systems',
            'requirements': ['NVIDIA RTX/GTX GPU', 'G-Assist Compatible'],
            'privacy_policy': 'Privacy-by-design with anonymized hardware detection'
        }
        
        # Validate GPU requirement
        self._validate_system()
        
        self.logger.info("CanRun G-Assist plugin initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('canrun_plugin.log'),
                logging.StreamHandler()
            ]
        )
    
    def _setup_services(self):
        """Setup dependency injection services."""
        try:
            # Register services in container
            self.container.register_singleton('hardware_detector', 
                                            lambda: PrivacyAwareHardwareDetector())
            self.logger.debug("Services registered successfully")
        except Exception as e:
            self.logger.error(f"Error setting up services: {e}")
    
    def _validate_system(self):
        """Validate that the system meets requirements."""
        try:
            # Basic system validation
            if not self.privacy_detector.has_nvidia_gpu():
                self.logger.warning("No NVIDIA GPU detected - some features may be limited")
            
            self.logger.info("System validation completed")
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
    
    async def handle_detect_hardware(self, params: dict) -> dict:
        """Handle hardware detection request."""
        try:
            hardware_info = await self.privacy_detector.get_hardware_specs()
            
            # Format hardware info for G-Assist
            hardware_summary = f"""
# System Hardware Specifications

## Graphics Card
- **GPU**: {hardware_info.gpu_name}
- **VRAM**: {hardware_info.gpu_memory_gb} GB
- **Architecture**: {hardware_info.gpu_architecture}

## Processor
- **CPU**: {hardware_info.cpu_name}
- **Cores**: {hardware_info.cpu_cores}
- **Threads**: {hardware_info.cpu_threads}

## Memory
- **RAM**: {hardware_info.memory_gb} GB
- **Type**: {hardware_info.memory_type}

## Storage
- **Primary**: {hardware_info.storage_type}
- **Available**: {hardware_info.storage_free_gb} GB free

## Operating System
- **OS**: {hardware_info.os_name}
- **Version**: {hardware_info.os_version}
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
    
    async def handle_check_compatibility(self, params: dict) -> dict:
        """Handle game compatibility check."""
        try:
            game_name = params.get('game_name', '')
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Get compatibility analysis
            result = await self.engine.analyze_game_compatibility(game_name)
            
            if result:
                compatibility = result.get('compatibility', {})
                performance = result.get('performance', {})
                
                response_text = f"""
# Game Compatibility Analysis: {game_name}

## Compatibility Level
**{compatibility.get('compatibility_level', 'Unknown')}** ({compatibility.get('overall_score', 0)}/100)

## Performance Prediction
- **Expected FPS**: {performance.get('fps', 'Unknown')}
- **Performance Level**: {performance.get('performance_level', 'Unknown')}
- **Stability**: {performance.get('stability', 'Unknown')}

## System Analysis
"""
                
                # Add component analysis
                components = compatibility.get('component_analysis', {})
                for component, analysis in components.items():
                    response_text += f"- **{component.upper()}**: {analysis.get('status', 'Unknown')} (Score: {analysis.get('score', 0)})\n"
                
                # Add bottlenecks if any
                bottlenecks = compatibility.get('bottlenecks', [])
                if bottlenecks:
                    response_text += f"\n## Potential Bottlenecks\n"
                    for bottleneck in bottlenecks:
                        response_text += f"- {bottleneck}\n"
                
                return {
                    "success": True,
                    "message": response_text
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
    
    async def handle_predict_performance(self, params: dict) -> dict:
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
            
            # Get performance prediction
            result = await self.engine.predict_performance(game_name, settings, resolution)
            
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
    
    async def handle_get_optimization_suggestions(self, params: dict) -> dict:
        """Handle optimization suggestions request."""
        try:
            game_name = params.get('game_name', '')
            if not game_name:
                return {
                    "success": False,
                    "message": "Game name is required"
                }
            
            # Get optimization suggestions
            suggestions = await self.engine.get_optimization_suggestions(game_name)
            
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
    
    async def handle_get_intelligent_analysis(self, params: dict) -> dict:
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
    
    async def handle_ask_intelligent_question(self, params: dict) -> dict:
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
            self.logger.error(f"Command processing failed: {e}")
            return {
                "success": False,
                "message": f"Command processing failed: {str(e)}"
            }
    
    async def run(self):
        """Main plugin loop."""
        self.logger.info("CanRun G-Assist plugin started")
        
        try:
            while True:
                # Read command from G-Assist
                command = self.comm.read_command()
                
                if command is None:
                    # No command received, continue listening
                    await asyncio.sleep(0.1)
                    continue
                
                # Check for shutdown command
                if self.comm.is_shutdown_command(command):
                    self.logger.info("Shutdown command received")
                    break
                
                # Process command
                response = await self.process_command(command)
                
                # Send response back to G-Assist
                self.comm.write_response(response)
                
        except KeyboardInterrupt:
            self.logger.info("Plugin interrupted by user")
        except Exception as e:
            self.logger.error(f"Plugin error: {e}")
            self.comm.write_error(f"Plugin error: {str(e)}")
        finally:
            self.logger.info("CanRun G-Assist plugin stopped")


def main():
    """Main entry point for G-Assist plugin."""
    try:
        plugin = CanRunGAssistPlugin()
        asyncio.run(plugin.run())
    except Exception as e:
        logging.error(f"Failed to start plugin: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()