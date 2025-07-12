"""
CanRun G-Assist Plugin Entry Point
Python-based plugin for NVIDIA G-Assist platform integration.
"""

import sys
import json
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.insert(0, '../src')

from canrun_engine import CanRunAPI
from hardware_detector import HardwareDetector


class GAssistPlugin:
    """Main G-Assist plugin class for CanRun."""
    
    def __init__(self):
        """Initialize G-Assist plugin."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.api = CanRunAPI()
        self.hardware_detector = HardwareDetector()
        
        # Plugin metadata
        self.plugin_info = {
            'name': 'CanRun Universal Game Compatibility Checker',
            'version': '1.0.0',
            'author': 'CanRun Development Team',
            'description': 'Check game compatibility and performance on NVIDIA RTX/GTX systems',
            'supported_commands': [
                'can_run_game',
                'performance_estimate',
                'hardware_info',
                'game_requirements',
                'optimization_suggestions'
            ]
        }
        
        # Voice command patterns
        self.command_patterns = {
            'can_run': [
                r'can\s+(?:i\s+)?(?:run|play)\s+(.+)',
                r'(?:will|can)\s+(.+)\s+(?:run|work)\s+(?:on\s+)?(?:my\s+)?(?:system|computer|pc)',
                r'(?:is\s+)?(.+)\s+compatible\s+(?:with\s+)?(?:my\s+)?(?:system|computer|pc)',
                r'check\s+(?:game\s+)?compatibility\s+(?:for\s+)?(.+)'
            ],
            'performance': [
                r'(?:what\s+)?(?:fps|framerate|performance)\s+(?:will\s+)?(?:i\s+)?(?:get\s+)?(?:in\s+|for\s+|with\s+)?(.+)',
                r'how\s+(?:well\s+)?(?:will\s+)?(.+)\s+(?:run|perform)',
                r'(?:what\s+)?(?:settings|graphics)\s+(?:should\s+)?(?:i\s+)?(?:use\s+)?(?:for\s+)?(.+)',
                r'performance\s+(?:estimate|prediction)\s+(?:for\s+)?(.+)'
            ],
            'hardware': [
                r'(?:what\s+)?(?:is\s+)?(?:my\s+)?(?:system|hardware|computer|pc)\s+(?:specs|specifications|info)',
                r'(?:show|tell)\s+(?:me\s+)?(?:my\s+)?(?:system|hardware|computer|pc)\s+(?:specs|specifications|info)',
                r'(?:hardware|system)\s+(?:information|info|summary)'
            ],
            'requirements': [
                r'(?:what\s+)?(?:are\s+)?(?:the\s+)?(?:system\s+)?requirements\s+(?:for\s+)?(.+)',
                r'(?:minimum|recommended)\s+(?:system\s+)?requirements\s+(?:for\s+)?(.+)',
                r'(.+)\s+(?:system\s+)?requirements'
            ],
            'optimization': [
                r'(?:how\s+)?(?:can\s+)?(?:i\s+)?(?:improve|optimize|boost)\s+(?:performance|fps)\s+(?:in\s+|for\s+)?(.+)',
                r'(?:optimization|performance)\s+(?:tips|suggestions|advice)\s+(?:for\s+)?(.+)',
                r'(?:what\s+)?(?:settings|options)\s+(?:should\s+)?(?:i\s+)?(?:change|adjust)\s+(?:in\s+|for\s+)?(.+)'
            ]
        }
        
        self.logger.info("CanRun G-Assist plugin initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('plugin.log'),
                logging.StreamHandler()
            ]
        )
    
    async def process_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process G-Assist command and return response.
        
        Args:
            command_data: Command data from G-Assist
            
        Returns:
            Response dictionary for G-Assist
        """
        try:
            self.logger.info(f"Processing command: {command_data}")
            
            # Extract text from command
            text = command_data.get('text', '').strip().lower()
            if not text:
                return self._create_error_response("No command text provided")
            
            # Parse command intent and extract game name
            intent, game_name = self._parse_command(text)
            
            if not intent:
                return self._create_error_response("Command not understood")
            
            # Route to appropriate handler
            if intent == 'can_run':
                return await self._handle_can_run(game_name)
            elif intent == 'performance':
                return await self._handle_performance(game_name)
            elif intent == 'hardware':
                return await self._handle_hardware()
            elif intent == 'requirements':
                return await self._handle_requirements(game_name)
            elif intent == 'optimization':
                return await self._handle_optimization(game_name)
            else:
                return self._create_error_response("Unknown command intent")
                
        except Exception as e:
            self.logger.error(f"Command processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return self._create_error_response(f"Processing error: {str(e)}")
    
    def _parse_command(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse command text to extract intent and game name."""
        text = text.strip().lower()
        
        # Check each command pattern
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    game_name = match.group(1).strip() if match.groups() else None
                    if game_name:
                        # Clean up game name
                        game_name = self._clean_game_name(game_name)
                    return intent, game_name
        
        return None, None
    
    def _clean_game_name(self, game_name: str) -> str:
        """Clean up extracted game name."""
        # Remove common words that might interfere
        stop_words = ['the', 'a', 'an', 'on', 'my', 'system', 'computer', 'pc']
        
        # Split into words and filter
        words = game_name.split()
        cleaned_words = [word for word in words if word not in stop_words]
        
        # Rejoin and capitalize
        cleaned_name = ' '.join(cleaned_words)
        return cleaned_name.title()
    
    async def _handle_can_run(self, game_name: str) -> Dict[str, Any]:
        """Handle can run compatibility check."""
        if not game_name:
            return self._create_error_response("Please specify a game name")
        
        try:
            result = await self.api.can_run_game(game_name)
            
            if not result['is_nvidia_gpu']:
                return self._create_response(
                    f"CanRun requires an NVIDIA RTX or GTX GPU. Your current system uses a non-NVIDIA GPU and is not compatible with G-Assist.",
                    result
                )
            
            # Create response based on compatibility
            if result['can_run_recommended']:
                response_text = f"✅ {game_name} will run excellently on your RTX/GTX system! You exceed the recommended requirements."
            elif result['can_run']:
                response_text = f"✅ {game_name} can run on your RTX/GTX system, but may require lower settings for optimal performance."
            else:
                response_text = f"❌ {game_name} may not run well on your RTX/GTX system. Consider upgrading your hardware."
            
            response_text += f"\n\nCompatibility Level: {result['compatibility_level']}"
            response_text += f"\nSummary: {result['summary']}"
            
            return self._create_response(response_text, result)
            
        except Exception as e:
            self.logger.error(f"Can run check failed: {e}")
            return self._create_error_response(f"Failed to check compatibility for {game_name}")
    
    async def _handle_performance(self, game_name: str) -> Dict[str, Any]:
        """Handle performance estimate request."""
        if not game_name:
            return self._create_error_response("Please specify a game name")
        
        try:
            result = await self.api.get_performance_estimate(game_name)
            
            if 'error' in result:
                return self._create_error_response(result['error'])
            
            # Format performance response
            response_text = f"🎮 Performance Estimate for {game_name}:\n\n"
            
            if result['best_expected_fps'] > 0:
                response_text += f"Expected FPS: {result['best_expected_fps']} at {result['best_settings']['resolution']} {result['best_settings']['quality']}\n"
                
                if result['best_settings']['dlss_enabled']:
                    response_text += "🚀 DLSS: Enabled for performance boost\n"
                if result['best_settings']['rtx_enabled']:
                    response_text += "✨ RTX: Ray tracing enabled\n"
                
                response_text += f"\nOverall Performance Score: {result['overall_score']:.1%}\n"
                
                if result['optimization_suggestions']:
                    response_text += "\n💡 Optimization Tips:\n"
                    for suggestion in result['optimization_suggestions'][:3]:  # Top 3 suggestions
                        response_text += f"• {suggestion}\n"
            else:
                response_text += "Performance analysis not available"
            
            return self._create_response(response_text, result)
            
        except Exception as e:
            self.logger.error(f"Performance estimate failed: {e}")
            return self._create_error_response(f"Failed to get performance estimate for {game_name}")
    
    async def _handle_hardware(self) -> Dict[str, Any]:
        """Handle hardware information request."""
        try:
            hardware = await self.api.get_hardware_summary()
            
            response_text = "🖥️ Your System Specifications:\n\n"
            response_text += f"GPU: {hardware['gpu']} ({hardware['gpu_memory_gb']} GB)\n"
            response_text += f"CPU: {hardware['cpu']} ({hardware['cpu_cores']} cores)\n"
            response_text += f"RAM: {hardware['ram_gb']} GB\n"
            response_text += f"OS: {hardware['os']}\n"
            response_text += f"DirectX: {hardware['directx']}\n"
            
            if hardware['g_assist_compatible']:
                response_text += "\n✅ G-Assist Compatible: Your NVIDIA RTX/GTX system is fully supported by CanRun!"
            else:
                response_text += "\n❌ G-Assist Incompatible: CanRun requires an NVIDIA RTX or GTX GPU"
            
            return self._create_response(response_text, hardware)
            
        except Exception as e:
            self.logger.error(f"Hardware info failed: {e}")
            return self._create_error_response("Failed to get hardware information")
    
    async def _handle_requirements(self, game_name: str) -> Dict[str, Any]:
        """Handle game requirements request."""
        if not game_name:
            return self._create_error_response("Please specify a game name")
        
        try:
            # Get full compatibility analysis to include requirements
            result = await self.api.engine.check_game_compatibility(game_name)
            
            if not result.game_requirements:
                return self._create_error_response(f"Requirements not found for {game_name}")
            
            requirements = result.game_requirements
            response_text = f"📋 System Requirements for {game_name}:\n\n"
            
            # Minimum requirements
            if requirements.minimum_requirements:
                response_text += "🔹 Minimum Requirements:\n"
                for component, requirement in requirements.minimum_requirements.items():
                    response_text += f"• {component}: {requirement}\n"
            
            # Recommended requirements
            if requirements.recommended_requirements:
                response_text += "\n🔸 Recommended Requirements:\n"
                for component, requirement in requirements.recommended_requirements.items():
                    response_text += f"• {component}: {requirement}\n"
            
            response_text += f"\nSource: {requirements.source}"
            
            return self._create_response(response_text, {
                'requirements': requirements.__dict__,
                'game_name': game_name
            })
            
        except Exception as e:
            self.logger.error(f"Requirements check failed: {e}")
            return self._create_error_response(f"Failed to get requirements for {game_name}")
    
    async def _handle_optimization(self, game_name: str) -> Dict[str, Any]:
        """Handle optimization suggestions request."""
        if not game_name:
            return self._create_error_response("Please specify a game name")
        
        try:
            result = await self.api.get_performance_estimate(game_name)
            
            if 'error' in result:
                return self._create_error_response(result['error'])
            
            response_text = f"⚡ Optimization Tips for {game_name}:\n\n"
            
            if result['optimization_suggestions']:
                for i, suggestion in enumerate(result['optimization_suggestions'], 1):
                    response_text += f"{i}. {suggestion}\n"
            else:
                response_text += "No specific optimization suggestions available"
            
            # Add general RTX/GTX tips
            if result['dlss_support']:
                response_text += "\n🚀 DLSS Available: Enable for significant performance boost"
            if result['rtx_support']:
                response_text += "\n✨ RTX Features: Ray tracing available for enhanced visuals"
            
            return self._create_response(response_text, result)
            
        except Exception as e:
            self.logger.error(f"Optimization suggestions failed: {e}")
            return self._create_error_response(f"Failed to get optimization tips for {game_name}")
    
    def _create_response(self, text: str, data: Any = None) -> Dict[str, Any]:
        """Create successful response for G-Assist."""
        return {
            'success': True,
            'response': text,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'plugin': self.plugin_info['name']
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response for G-Assist."""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'plugin': self.plugin_info['name']
        }
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return self.plugin_info


# Main plugin interface for G-Assist
class CanRunGAssistPlugin:
    """Main plugin interface for G-Assist platform."""
    
    def __init__(self):
        self.plugin = GAssistPlugin()
        self.logger = logging.getLogger(__name__)
    
    async def handle_request(self, request_data: str) -> str:
        """
        Handle G-Assist request via pipe communication.
        
        Args:
            request_data: JSON string from G-Assist
            
        Returns:
            JSON response string
        """
        try:
            # Parse request
            request = json.loads(request_data)
            
            # Process command
            response = await self.plugin.process_command(request)
            
            # Return JSON response
            return json.dumps(response, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"Request handling failed: {e}")
            error_response = {
                'success': False,
                'error': f"Request processing failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)


# Main entry point for G-Assist
def main():
    """Main entry point for G-Assist plugin."""
    plugin = CanRunGAssistPlugin()
    
    # Test mode - process command line arguments
    if len(sys.argv) > 1:
        test_command = ' '.join(sys.argv[1:])
        test_request = {'text': test_command}
        
        async def test_plugin():
            response = await plugin.handle_request(json.dumps(test_request))
            print(response)
        
        asyncio.run(test_plugin())
    else:
        # Plugin mode - wait for G-Assist communication
        print("CanRun G-Assist Plugin initialized and waiting for commands...")
        
        # In a real implementation, this would listen for pipe communication
        # For now, we'll simulate with stdin/stdout
        async def plugin_loop():
            while True:
                try:
                    # Read from stdin (simulating pipe communication)
                    line = input()
                    if line.strip():
                        response = await plugin.handle_request(line)
                        print(response)
                        sys.stdout.flush()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        
        asyncio.run(plugin_loop())


if __name__ == "__main__":
    main()