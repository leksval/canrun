"""
CanRun G-Assist Plugin - Official G-Assist Plugin Implementation
Standalone executable plugin for NVIDIA G-Assist platform.
"""

import sys
import json
import asyncio
import logging
import os
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from rise import rise
    RISE_AVAILABLE = True
except ImportError:
    RISE_AVAILABLE = False
    print("Warning: rise module not available. Plugin will run in standalone mode.")

from canrun_engine import CanRunAPI


class CanRunGAssistPlugin:
    """CanRun plugin for G-Assist platform."""
    
    def __init__(self):
        """Initialize the plugin."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.api = CanRunAPI()
        
        # Plugin metadata
        self.plugin_info = {
            'name': 'CanRun Universal Game Compatibility Checker',
            'version': '1.0.0',
            'description': 'Check game compatibility and performance on NVIDIA RTX/GTX systems'
        }
        
        # Initialize G-Assist connection if available
        if RISE_AVAILABLE:
            try:
                rise.register_rise_client()
                self.logger.info("G-Assist connection established")
            except Exception as e:
                self.logger.error(f"Failed to connect to G-Assist: {e}")
        
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
    
    async def check_game_compatibility(self, game_name: str) -> dict:
        """Check if a game can run on the current system."""
        try:
            result = await self.api.can_run_game(game_name)
            
            # Format response for G-Assist
            if not result['is_nvidia_gpu']:
                return {
                    'success': False,
                    'message': 'CanRun requires an NVIDIA RTX or GTX GPU. Your system is not compatible with G-Assist.',
                    'data': result
                }
            
            if result['can_run_recommended']:
                message = f"✅ {game_name} will run excellently on your RTX/GTX system! You exceed the recommended requirements."
            elif result['can_run']:
                message = f"✅ {game_name} can run on your RTX/GTX system, but may require lower settings for optimal performance."
            else:
                message = f"❌ {game_name} may not run well on your RTX/GTX system. Consider upgrading your hardware."
            
            message += f"\n\nCompatibility Level: {result['compatibility_level']}"
            message += f"\nSummary: {result['summary']}"
            
            return {
                'success': True,
                'message': message,
                'data': result
            }
            
        except Exception as e:
            self.logger.error(f"Compatibility check failed: {e}")
            return {
                'success': False,
                'message': f"Failed to check compatibility for {game_name}: {str(e)}",
                'data': None
            }
    
    async def get_performance_estimate(self, game_name: str) -> dict:
        """Get performance estimate for a game."""
        try:
            result = await self.api.get_performance_estimate(game_name)
            
            if 'error' in result:
                return {
                    'success': False,
                    'message': result['error'],
                    'data': None
                }
            
            # Format performance response
            message = f"🎮 Performance Estimate for {game_name}:\n\n"
            
            if result['best_expected_fps'] > 0:
                message += f"Expected FPS: {result['best_expected_fps']} at {result['best_settings']['resolution']} {result['best_settings']['quality']}\n"
                
                if result['best_settings']['dlss_enabled']:
                    message += "🚀 DLSS: Enabled for performance boost\n"
                if result['best_settings']['rtx_enabled']:
                    message += "✨ RTX: Ray tracing enabled\n"
                
                message += f"\nOverall Performance Score: {result['overall_score']:.1%}\n"
                
                if result['optimization_suggestions']:
                    message += "\n💡 Optimization Tips:\n"
                    for suggestion in result['optimization_suggestions'][:3]:
                        message += f"• {suggestion}\n"
            else:
                message += "Performance analysis not available"
            
            return {
                'success': True,
                'message': message,
                'data': result
            }
            
        except Exception as e:
            self.logger.error(f"Performance estimate failed: {e}")
            return {
                'success': False,
                'message': f"Failed to get performance estimate for {game_name}: {str(e)}",
                'data': None
            }
    
    async def get_system_info(self) -> dict:
        """Get system hardware information."""
        try:
            hardware = await self.api.get_hardware_summary()
            
            message = "🖥️ Your System Specifications:\n\n"
            message += f"GPU: {hardware['gpu']} ({hardware['gpu_memory_gb']} GB)\n"
            message += f"CPU: {hardware['cpu']} ({hardware['cpu_cores']} cores)\n"
            message += f"RAM: {hardware['ram_gb']} GB\n"
            message += f"OS: {hardware['os']}\n"
            message += f"DirectX: {hardware['directx']}\n"
            
            if hardware['g_assist_compatible']:
                message += "\n✅ G-Assist Compatible: Your NVIDIA RTX/GTX system is fully supported by CanRun!"
            else:
                message += "\n❌ G-Assist Incompatible: CanRun requires an NVIDIA RTX or GTX GPU"
            
            return {
                'success': True,
                'message': message,
                'data': hardware
            }
            
        except Exception as e:
            self.logger.error(f"System info failed: {e}")
            return {
                'success': False,
                'message': f"Failed to get system information: {str(e)}",
                'data': None
            }
    
    async def get_game_requirements(self, game_name: str) -> dict:
        """Get game system requirements."""
        try:
            # Get full compatibility analysis to include requirements
            result = await self.api.engine.check_game_compatibility(game_name)
            
            if not result.game_requirements:
                return {
                    'success': False,
                    'message': f"Requirements not found for {game_name}",
                    'data': None
                }
            
            requirements = result.game_requirements
            message = f"📋 System Requirements for {game_name}:\n\n"
            
            # Minimum requirements
            if requirements.minimum_requirements:
                message += "🔹 Minimum Requirements:\n"
                for component, requirement in requirements.minimum_requirements.items():
                    message += f"• {component}: {requirement}\n"
            
            # Recommended requirements
            if requirements.recommended_requirements:
                message += "\n🔸 Recommended Requirements:\n"
                for component, requirement in requirements.recommended_requirements.items():
                    message += f"• {component}: {requirement}\n"
            
            message += f"\nSource: {requirements.source}"
            
            return {
                'success': True,
                'message': message,
                'data': requirements.__dict__
            }
            
        except Exception as e:
            self.logger.error(f"Requirements check failed: {e}")
            return {
                'success': False,
                'message': f"Failed to get requirements for {game_name}: {str(e)}",
                'data': None
            }
    
    async def get_optimization_tips(self, game_name: str) -> dict:
        """Get optimization suggestions for a game."""
        try:
            result = await self.api.get_performance_estimate(game_name)
            
            if 'error' in result:
                return {
                    'success': False,
                    'message': result['error'],
                    'data': None
                }
            
            message = f"⚡ Optimization Tips for {game_name}:\n\n"
            
            if result['optimization_suggestions']:
                for i, suggestion in enumerate(result['optimization_suggestions'], 1):
                    message += f"{i}. {suggestion}\n"
            else:
                message += "No specific optimization suggestions available"
            
            # Add general RTX/GTX tips
            if result['dlss_support']:
                message += "\n🚀 DLSS Available: Enable for significant performance boost"
            if result['rtx_support']:
                message += "\n✨ RTX Features: Ray tracing available for enhanced visuals"
            
            return {
                'success': True,
                'message': message,
                'data': result
            }
            
        except Exception as e:
            self.logger.error(f"Optimization tips failed: {e}")
            return {
                'success': False,
                'message': f"Failed to get optimization tips for {game_name}: {str(e)}",
                'data': None
            }


def main():
    """Main entry point for G-Assist plugin."""
    plugin = CanRunGAssistPlugin()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python g-assist-plugin-canrun.py <function> [arguments]")
        print("Functions: check_compatibility, performance_estimate, system_info, game_requirements, optimization_tips")
        sys.exit(1)
    
    function = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    async def run_plugin():
        try:
            if function == "check_compatibility":
                if not args:
                    print("Error: Game name required")
                    sys.exit(1)
                game_name = " ".join(args)
                result = await plugin.check_game_compatibility(game_name)
                
            elif function == "performance_estimate":
                if not args:
                    print("Error: Game name required")
                    sys.exit(1)
                game_name = " ".join(args)
                result = await plugin.get_performance_estimate(game_name)
                
            elif function == "system_info":
                result = await plugin.get_system_info()
                
            elif function == "game_requirements":
                if not args:
                    print("Error: Game name required")
                    sys.exit(1)
                game_name = " ".join(args)
                result = await plugin.get_game_requirements(game_name)
                
            elif function == "optimization_tips":
                if not args:
                    print("Error: Game name required")
                    sys.exit(1)
                game_name = " ".join(args)
                result = await plugin.get_optimization_tips(game_name)
                
            else:
                print(f"Error: Unknown function '{function}'")
                sys.exit(1)
            
            # Output result
            if result['success']:
                print(result['message'])
            else:
                print(f"Error: {result['message']}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Plugin error: {str(e)}")
            sys.exit(1)
    
    # Run the plugin
    asyncio.run(run_plugin())


if __name__ == "__main__":
    main()